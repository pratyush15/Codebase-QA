
"""
Tree-sitter based AST chunking.

Replaces the heuristic separator-based splitting in ingestion.py's default
path for languages we have a grammar for. Instead of guessing chunk
boundaries from separator strings (which can and does split a function
mid-body), this walks the actual syntax tree and cuts along real
function/class/method boundaries, so a chunk is always a whole,
syntactically complete unit.

Class/impl blocks are further split into a header chunk (signature, fields,
docstring) plus one chunk per method — each method chunk is prefixed with a
short comment naming its enclosing class, so that context isn't lost once
methods are split apart from their class body.

Returns None (caller falls back to the old heuristic splitter) when:
- the language has no tree-sitter grammar wired up here
- tree-sitter itself isn't installed (optional dependency)
- parsing raises — tree-sitter is normally error-tolerant, but we don't
  want a bug in our own byte-slicing to silently corrupt a chunk
"""
import logging
from typing import Dict, List, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

try:
    from tree_sitter_language_pack import get_parser
    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False
    logger.warning(
        "tree-sitter-language-pack not installed — AST chunking disabled, "
        "falling back to heuristic separator-based splitting for all files."
    )

# Our internal language name (from utils.language_detect) -> tree-sitter-language-pack identifier.
# Languages not listed here always fall back to the heuristic splitter.
LANGUAGE_TO_TS_NAME = {
    "Python": "python",
    "JavaScript": "javascript",
    "JavaScript (React)": "javascript",
    "TypeScript": "typescript",
    "TypeScript (React)": "tsx",
    "Java": "java",
    "Go": "go",
    "Rust": "rust",
    "C": "c",
    "C++": "cpp",
}

# Top-level node types treated as one standalone, whole chunk.
TOP_LEVEL_DEFINITION_TYPES = {
    "python": {"function_definition", "class_definition", "decorated_definition"},
    "javascript": {"function_declaration", "class_declaration", "lexical_declaration"},
    "typescript": {"function_declaration", "class_declaration", "interface_declaration", "lexical_declaration"},
    "tsx": {"function_declaration", "class_declaration", "interface_declaration", "lexical_declaration"},
    "java": {"class_declaration", "interface_declaration"},
    "go": {"function_declaration", "method_declaration", "type_declaration"},
    "rust": {"function_item", "struct_item", "impl_item", "enum_item", "trait_item"},
    "c": {"function_definition", "struct_specifier"},
    "cpp": {"function_definition", "class_specifier", "struct_specifier"},
}

# Of those, the ones we recurse into to split out individual methods rather
# than keeping the whole class/impl block as a single (potentially huge) chunk.
CLASS_LIKE_TYPES = {
    "python": {"class_definition"},
    "javascript": {"class_declaration"},
    "typescript": {"class_declaration"},
    "tsx": {"class_declaration"},
    "java": {"class_declaration", "interface_declaration"},
    "cpp": {"class_specifier", "struct_specifier"},
    "rust": {"impl_item"},
}

# The child node type that holds a class/impl's body, per language grammar.
BODY_CHILD_TYPE = {
    "python": "block",
    "javascript": "class_body",
    "typescript": "class_body",
    "tsx": "class_body",
    "java": "class_body",
    "cpp": "field_declaration_list",
    "rust": "declaration_list",
}

# Node types inside a class/impl body that count as one method each.
METHOD_TYPES = {
    "python": {"function_definition", "decorated_definition"},
    "javascript": {"method_definition"},
    "typescript": {"method_definition"},
    "tsx": {"method_definition"},
    "java": {"method_declaration", "constructor_declaration"},
    "cpp": {"function_definition"},
    "rust": {"function_item"},
}

# node.type -> a short human-readable symbol kind, for the symbol_type
# metadata field. Anything not listed here falls back to "other" (or, for
# "decorated_definition", to whatever the wrapped inner node resolves to).
SYMBOL_TYPE_BY_NODE_TYPE = {
    "function_definition": "function",
    "function_declaration": "function",
    "function_item": "function",
    "method_declaration": "method",
    "method_definition": "method",
    "constructor_declaration": "method",
    "class_definition": "class",
    "class_declaration": "class",
    "class_specifier": "class",
    "interface_declaration": "interface",
    "struct_item": "struct",
    "struct_specifier": "struct",
    "enum_item": "enum",
    "trait_item": "trait",
    "impl_item": "impl",
    "type_declaration": "type",
    "lexical_declaration": "variable",
}

# Identifier-shaped node types across the grammars we support — used both
# as a direct match and as the target of the depth-limited fallback search
# in _find_first_identifier.
_IDENTIFIER_NODE_TYPES = {"identifier", "type_identifier", "field_identifier"}

_parser_cache = {}


def _get_cached_parser(ts_name: str):
    if ts_name not in _parser_cache:
        _parser_cache[ts_name] = get_parser(ts_name)
    return _parser_cache[ts_name]


def supports_ast_chunking(language: str) -> bool:
    return TREE_SITTER_AVAILABLE and language in LANGUAGE_TO_TS_NAME


def _decode(src_bytes: bytes, node) -> str:
    return src_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


def _find_first_identifier(node, max_depth: int = 4):
    """
    Depth-limited, document-order search for the first identifier-shaped
    node in a subtree. Used as a fallback for grammars — mainly C/C++ —
    where the symbol name isn't exposed as a simple 'name' field but is
    nested inside a declarator; the first identifier encountered while
    walking in is, in practice, almost always the symbol being defined.
    """
    if max_depth < 0:
        return None
    if node.type in _IDENTIFIER_NODE_TYPES:
        return node
    for child in node.children:
        found = _find_first_identifier(child, max_depth - 1)
        if found is not None:
            return found
    return None


def _resolve_inner_definition(node):
    """Unwraps a Python decorated_definition to the function/class it decorates."""
    if node.type != "decorated_definition":
        return node
    inner = next(
        (c for c in node.children if c.type in ("function_definition", "class_definition")),
        None,
    )
    return inner if inner is not None else node


def _extract_symbol_name(src_bytes: bytes, node) -> Optional[str]:
    """
    Best-effort symbol name for a definition node (function, class,
    method, etc.). Tries the grammar's own 'name' field first — accurate
    for Python/JS/TS/Java/Go/Rust — then falls back to a depth-limited
    search for the first identifier in the subtree, which covers
    grammars (mainly C/C++) where the name is nested inside a declarator
    rather than exposed as a simple field. Returns None if nothing
    identifier-shaped can be found at all (e.g. an anonymous construct).
    """
    node = _resolve_inner_definition(node)

    name_node = node.child_by_field_name("name")
    if name_node is None:
        name_node = _find_first_identifier(node)
    if name_node is None:
        return None
    return _decode(src_bytes, name_node)


def _symbol_type_for(node) -> str:
    """symbol_type for a definition node, resolving decorated_definition to its inner kind."""
    resolved = _resolve_inner_definition(node)
    return SYMBOL_TYPE_BY_NODE_TYPE.get(resolved.type, "other")


def _split_class_node(src_bytes: bytes, node, ts_name: str) -> List[Dict]:
    body_type = BODY_CHILD_TYPE.get(ts_name)
    method_types = METHOD_TYPES.get(ts_name, set())
    body = next((c for c in node.children if c.type == body_type), None)

    class_name = _extract_symbol_name(src_bytes, node)
    class_symbol_type = _symbol_type_for(node)

    if body is None or not method_types:
        return [{
            "text": _decode(src_bytes, node),
            "symbol_name": class_name,
            "symbol_type": class_symbol_type,
            "parent_symbol": None,
        }]

    methods = [c for c in body.children if c.type in method_types]
    if not methods:
        # A class with no recognized methods (e.g. fields-only, or an empty
        # interface) — keep it as a single chunk, nothing to split out.
        return [{
            "text": _decode(src_bytes, node),
            "symbol_name": class_name,
            "symbol_type": class_symbol_type,
            "parent_symbol": None,
        }]

    chunks: List[Dict] = []

    # Header: class signature + fields/docstring, up to the first method.
    header_text = src_bytes[node.start_byte:methods[0].start_byte].decode("utf-8", errors="replace").rstrip()
    if header_text.strip():
        chunks.append({
            "text": header_text,
            "symbol_name": class_name,
            "symbol_type": class_symbol_type,
            "parent_symbol": None,
        })

    display_class_name = class_name or "?"

    for m in methods:
        chunks.append({
            "text": f"# (method of {display_class_name})\n{_decode(src_bytes, m)}",
            "symbol_name": _extract_symbol_name(src_bytes, m),
            # Always "method" here, never delegated to _symbol_type_for(m):
            # a method's underlying node.type (e.g. "function_definition"
            # in Python/C++, "function_item" in Rust) is identical to a
            # free-standing function's — only this call site's context
            # (inside a class/impl body) tells them apart.
            "symbol_type": "method",
            "parent_symbol": class_name,
        })

    return chunks


def _chunk_source_ast_detailed(content: str, language: str, max_chunk_size: int) -> Optional[List[Dict]]:
    """
    Shared implementation behind both chunk_source_ast (plain text chunks,
    for backward compatibility) and chunk_source_ast_with_symbols (chunks
    plus extracted symbol metadata). Splits source code along real AST
    boundaries and returns None if this language isn't supported here or
    parsing fails, signaling the caller to fall back to the heuristic
    splitter.

    Each returned dict has:
      - text: the chunk's source text
      - symbol_name: the function/class/method name this chunk defines,
        or None for chunks that aren't a single named definition (e.g. a
        block of top-level imports/constants)
      - symbol_type: "function" / "class" / "method" / "interface" /
        "struct" / "enum" / "trait" / "impl" / "type" / "variable" /
        "other", or None alongside a None symbol_name
      - parent_symbol: the enclosing class/impl name, set only for method
        chunks; None otherwise
    """
    ts_name = LANGUAGE_TO_TS_NAME.get(language)
    if not TREE_SITTER_AVAILABLE or ts_name is None:
        return None

    try:
        parser = _get_cached_parser(ts_name)
        src_bytes = content.encode("utf-8", errors="replace")
        root = parser.parse(src_bytes).root_node
    except Exception:
        logger.exception(
            "Tree-sitter parse failed for language '%s' — falling back to heuristic splitter", language
        )
        return None

    definition_types = TOP_LEVEL_DEFINITION_TYPES.get(ts_name, set())
    class_like_types = CLASS_LIKE_TYPES.get(ts_name, set())

    chunks: List[Dict] = []
    buffer_start: Optional[int] = None

    def flush_buffer(end_byte: int):
        nonlocal buffer_start
        if buffer_start is not None and end_byte > buffer_start:
            text = src_bytes[buffer_start:end_byte].decode("utf-8", errors="replace").strip()
            if text:
                chunks.append({"text": text, "symbol_name": None, "symbol_type": None, "parent_symbol": None})
        buffer_start = None

    try:
        for child in root.children:
            if child.type in class_like_types:
                flush_buffer(child.start_byte)
                chunks.extend(_split_class_node(src_bytes, child, ts_name))
            elif child.type in definition_types:
                flush_buffer(child.start_byte)
                chunks.append({
                    "text": _decode(src_bytes, child),
                    "symbol_name": _extract_symbol_name(src_bytes, child),
                    "symbol_type": _symbol_type_for(child),
                    "parent_symbol": None,
                })
            elif buffer_start is None:
                buffer_start = child.start_byte
        flush_buffer(len(src_bytes))
    except Exception:
        logger.exception(
            "Error walking AST for language '%s' — falling back to heuristic splitter", language
        )
        return None

    if not chunks:
        return None

    # A handful of AST chunks (a giant generated function, a huge class
    # header) can still exceed a sane size — re-split just those with the
    # ordinary recursive splitter, rather than forcing every chunk through
    # it and losing the whole-function guarantee for everything else. Each
    # resulting piece keeps its parent chunk's symbol metadata: they're
    # all still part of that one oversized definition, even if the text
    # itself gets cut into smaller pieces here.
    size_ceiling = max_chunk_size * 3
    oversized_splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_chunk_size,
        chunk_overlap=min(150, max_chunk_size // 4),
    )

    final_chunks: List[Dict] = []
    for chunk in chunks:
        if len(chunk["text"]) > size_ceiling:
            for piece in oversized_splitter.split_text(chunk["text"]):
                final_chunks.append({**chunk, "text": piece})
        else:
            final_chunks.append(chunk)

    return final_chunks


def chunk_source_ast(content: str, language: str, max_chunk_size: int) -> Optional[List[str]]:
    """
    Split source code along real AST boundaries. Returns None if this
    language isn't supported here or parsing fails, signaling the caller
    to fall back to the heuristic splitter.

    Plain-text variant, kept for backward compatibility — 
    chunk_source_ast_with_symbols for chunks plus extracted symbol
    metadata (function/class/method names).
    """
    detailed = _chunk_source_ast_detailed(content, language, max_chunk_size)
    if detailed is None:
        return None
    return [c["text"] for c in detailed]


def chunk_source_ast_with_symbols(content: str, language: str, max_chunk_size: int) -> Optional[List[Dict]]:
    """
    Same chunking as chunk_source_ast, but each chunk also carries its
    extracted symbol info — in _chunk_source_ast_detailed's docstring
    for the exact dict shape. Used by core.ingestion.ingest_files to
    populate the symbol_name/symbol_type/parent_symbol metadata fields
    that power "show me the `retrieve_documents` function"-style
    filtering, on top of the existing file/language filters.
    """
    return _chunk_source_ast_detailed(content, language, max_chunk_size)