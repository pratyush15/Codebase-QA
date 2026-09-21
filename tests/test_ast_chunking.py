
from core.ast_chunking import chunk_source_ast, chunk_source_ast_with_symbols, supports_ast_chunking


def test_supports_ast_chunking_for_known_languages():
    assert supports_ast_chunking("Python") is True
    assert supports_ast_chunking("JavaScript") is True
    assert supports_ast_chunking("Go") is True


def test_supports_ast_chunking_false_for_unknown_language():
    assert supports_ast_chunking("SQL") is False
    assert supports_ast_chunking("Markdown") is False
    assert supports_ast_chunking("Unknown") is False


def test_chunk_source_ast_returns_none_for_unsupported_language():
    assert chunk_source_ast("SELECT * FROM t;", "SQL", max_chunk_size=1000) is None


def test_chunk_source_ast_splits_top_level_functions_whole():
    src = "def foo(a, b):\n    return a + b\n\n\ndef bar():\n    return 1\n"
    chunks = chunk_source_ast(src, "Python", max_chunk_size=1000)

    assert chunks is not None
    assert any("def foo(a, b):" in c and "return a + b" in c for c in chunks)
    assert any("def bar():" in c and "return 1" in c for c in chunks)
    # Each function should be a single, complete chunk — not split mid-body.
    for c in chunks:
        if "def foo" in c:
            assert "return a + b" in c


def test_chunk_source_ast_never_splits_a_function_body_across_chunks():
    # A function long enough that the OLD heuristic splitter would have cut
    # it mid-body — AST chunking must still keep it whole (or push it
    # through the oversized-fallback splitter as a single unit, not
    # silently truncate it).
    body_lines = "\n".join(f"    x{i} = {i}" for i in range(80))
    src = f"def big_function():\n{body_lines}\n    return x0\n"

    chunks = chunk_source_ast(src, "Python", max_chunk_size=1000)

    assert chunks is not None
    joined = "\n".join(chunks)
    assert "def big_function():" in joined
    assert "return x0" in joined


def test_chunk_source_ast_splits_class_into_header_and_methods():
    src = (
        "class Bar:\n"
        "    \"\"\"A class.\"\"\"\n\n"
        "    def __init__(self):\n"
        "        self.x = 1\n\n"
        "    def method(self):\n"
        "        return self.x\n"
    )
    chunks = chunk_source_ast(src, "Python", max_chunk_size=1000)

    assert chunks is not None
    assert any("class Bar:" in c for c in chunks)
    assert any("__init__" in c and "(method of Bar)" in c for c in chunks)
    assert any("def method(self):" in c and "(method of Bar)" in c for c in chunks)
    # methods should be separate chunks, not merged into one
    method_chunks = [c for c in chunks if "(method of Bar)" in c]
    assert len(method_chunks) == 2


def test_chunk_source_ast_class_with_no_methods_stays_one_chunk():
    src = "class DataOnly:\n    x: int\n    y: int\n"
    chunks = chunk_source_ast(src, "Python", max_chunk_size=1000)
    assert chunks is not None
    assert len(chunks) == 1
    assert "class DataOnly:" in chunks[0]


def test_chunk_source_ast_javascript_functions_and_classes():
    src = (
        "function add(a, b) { return a + b; }\n\n"
        "class Widget {\n"
        "  render() { return 1; }\n"
        "}\n"
    )
    chunks = chunk_source_ast(src, "JavaScript", max_chunk_size=1000)

    assert chunks is not None
    assert any("function add(a, b)" in c for c in chunks)
    assert any("class Widget" in c for c in chunks)
    assert any("render()" in c and "(method of Widget)" in c for c in chunks)


def test_chunk_source_ast_handles_empty_content():
    chunks = chunk_source_ast("", "Python", max_chunk_size=1000)
    assert chunks is None


def test_chunk_source_ast_oversized_function_gets_resplit():
    # A single function so large it exceeds the 3x-chunk_size ceiling should
    # still come back split (via the fallback splitter) rather than as one
    # enormous chunk.
    body_lines = "\n".join(f"    x{i} = {i}" for i in range(2000))
    src = f"def huge():\n{body_lines}\n    return x0\n"

    chunks = chunk_source_ast(src, "Python", max_chunk_size=200)

    assert chunks is not None
    assert len(chunks) > 1
    assert all(len(c) <= 200 * 3 + 500 for c in chunks)  # generous slack for splitter overlap

# ---------- chunk_source_ast_with_symbols ----------

def test_with_symbols_none_for_unsupported_language():
    assert chunk_source_ast_with_symbols("SELECT * FROM t;", "SQL", max_chunk_size=1000) is None


def test_with_symbols_matches_chunk_source_ast_text_exactly():
    # Same underlying walk — the plain-text variant should be a strict
    # projection of the detailed one, never a different chunking.
    src = "def foo(a, b):\n    return a + b\n\n\ndef bar():\n    return 1\n"
    plain = chunk_source_ast(src, "Python", max_chunk_size=1000)
    detailed = chunk_source_ast_with_symbols(src, "Python", max_chunk_size=1000)

    assert [c["text"] for c in detailed] == plain


def test_with_symbols_top_level_function_tagged_correctly():
    src = "def foo(a, b):\n    return a + b\n"
    chunks = chunk_source_ast_with_symbols(src, "Python", max_chunk_size=1000)

    assert len(chunks) == 1
    assert chunks[0]["symbol_name"] == "foo"
    assert chunks[0]["symbol_type"] == "function"
    assert chunks[0]["parent_symbol"] is None


def test_with_symbols_class_header_and_methods_tagged_correctly():
    src = (
        "class Bar:\n"
        "    \"\"\"A class.\"\"\"\n\n"
        "    def __init__(self):\n"
        "        self.x = 1\n\n"
        "    def method(self):\n"
        "        return self.x\n"
    )
    chunks = chunk_source_ast_with_symbols(src, "Python", max_chunk_size=1000)

    header = next(c for c in chunks if c["text"].startswith("class Bar"))
    assert header["symbol_name"] == "Bar"
    assert header["symbol_type"] == "class"
    assert header["parent_symbol"] is None

    init_chunk = next(c for c in chunks if "__init__" in c["text"])
    assert init_chunk["symbol_name"] == "__init__"
    assert init_chunk["symbol_type"] == "method"
    assert init_chunk["parent_symbol"] == "Bar"

    method_chunk = next(c for c in chunks if "def method" in c["text"])
    assert method_chunk["symbol_name"] == "method"
    assert method_chunk["symbol_type"] == "method"
    assert method_chunk["parent_symbol"] == "Bar"


def test_with_symbols_decorated_method_unwraps_to_inner_function_name():
    src = (
        "class Bar:\n"
        "    @property\n"
        "    def value(self):\n"
        "        return 1\n"
    )
    chunks = chunk_source_ast_with_symbols(src, "Python", max_chunk_size=1000)
    method_chunk = next(c for c in chunks if "def value" in c["text"])

    assert method_chunk["symbol_name"] == "value"
    assert method_chunk["symbol_type"] == "method"
    assert method_chunk["parent_symbol"] == "Bar"


def test_with_symbols_decorated_top_level_function_unwraps_correctly():
    src = "@staticmethod\ndef helper():\n    return 1\n"
    chunks = chunk_source_ast_with_symbols(src, "Python", max_chunk_size=1000)

    assert len(chunks) == 1
    assert chunks[0]["symbol_name"] == "helper"
    assert chunks[0]["symbol_type"] == "function"


def test_with_symbols_non_definition_leftover_has_no_symbol():
    src = "import os\nimport sys\n\nX = 1\n"
    chunks = chunk_source_ast_with_symbols(src, "Python", max_chunk_size=1000)

    assert len(chunks) == 1
    assert chunks[0]["symbol_name"] is None
    assert chunks[0]["symbol_type"] is None
    assert chunks[0]["parent_symbol"] is None


def test_with_symbols_class_with_no_methods_still_tagged_as_class():
    src = "class DataOnly:\n    x: int\n    y: int\n"
    chunks = chunk_source_ast_with_symbols(src, "Python", max_chunk_size=1000)

    assert len(chunks) == 1
    assert chunks[0]["symbol_name"] == "DataOnly"
    assert chunks[0]["symbol_type"] == "class"


def test_with_symbols_javascript_function_and_class_and_method():
    src = (
        "function add(a, b) { return a + b; }\n\n"
        "class Widget {\n"
        "  render() { return 1; }\n"
        "}\n"
    )
    chunks = chunk_source_ast_with_symbols(src, "JavaScript", max_chunk_size=1000)

    func_chunk = next(c for c in chunks if "function add" in c["text"])
    assert func_chunk["symbol_name"] == "add"
    assert func_chunk["symbol_type"] == "function"

    class_chunk = next(c for c in chunks if c["text"].startswith("class Widget"))
    assert class_chunk["symbol_name"] == "Widget"
    assert class_chunk["symbol_type"] == "class"

    method_chunk = next(c for c in chunks if "render()" in c["text"])
    assert method_chunk["symbol_name"] == "render"
    assert method_chunk["symbol_type"] == "method"
    assert method_chunk["parent_symbol"] == "Widget"


def test_with_symbols_go_function_and_type():
    src = (
        "package main\n\n"
        "func Add(a, b int) int {\n    return a + b\n}\n\n"
        "type Widget struct {\n    X int\n}\n"
    )
    chunks = chunk_source_ast_with_symbols(src, "Go", max_chunk_size=1000)

    func_chunk = next(c for c in chunks if "func Add" in c["text"])
    assert func_chunk["symbol_name"] == "Add"
    assert func_chunk["symbol_type"] == "function"

    type_chunk = next(c for c in chunks if "type Widget" in c["text"])
    assert type_chunk["symbol_name"] == "Widget"
    assert type_chunk["symbol_type"] == "type"


def test_with_symbols_cpp_function_name_found_via_fallback_search():
    # C++ doesn't expose the function name as a simple 'name' field the
    # way Python/JS do — this exercises the _find_first_identifier fallback.
    src = "int add(int a, int b) { return a + b; }\n"
    chunks = chunk_source_ast_with_symbols(src, "C++", max_chunk_size=1000)

    assert len(chunks) == 1
    assert chunks[0]["symbol_name"] == "add"
    assert chunks[0]["symbol_type"] == "function"


def test_with_symbols_oversized_chunk_split_pieces_keep_parent_symbol_metadata():
    body_lines = "\n".join(f"    x{i} = {i}" for i in range(200))
    src = f"def big_function():\n{body_lines}\n    return x0\n"

    chunks = chunk_source_ast_with_symbols(src, "Python", max_chunk_size=200)

    assert len(chunks) > 1  # confirms the oversized-resplit path actually triggered
    assert all(c["symbol_name"] == "big_function" for c in chunks)
    assert all(c["symbol_type"] == "function" for c in chunks)


def test_with_symbols_empty_content_returns_none():
    assert chunk_source_ast_with_symbols("", "Python", max_chunk_size=1000) is None