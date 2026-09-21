"""
.gitignore-aware filtering for codebase indexing.

Without this, node_modules/, dist/, coverage/, vendored dependencies, and
similar generated or vendored trees only get skipped if they happen to
match the hardcoded SKIP_DIRS list in config.py. Anything project-specific
— a custom build output directory, a fixtures/ folder full of golden test
data, a generated protobuf/ tree — sails straight through, gets chunked,
embedded, and then dilutes retrieval with noise nobody wanted searched in
the first place.

This reads the project's own root .gitignore (if present) and applies it
with the same matching engine `git` itself uses (gitwildmatch syntax, via
the `pathspec` library) — so "what git tracks" and "what gets indexed"
stay in sync automatically, with no project-specific config needed on our
side.

Scope: only the root .gitignore is honored. Nested per-directory
.gitignore files (e.g. a monorepo subproject with its own rules) are a
known, deliberate limitation — correctly resolving override precedence
across directory levels is easy to get subtly wrong, and a single root
file already covers the overwhelming majority of real repos.
"""
import logging
from pathlib import Path
from typing import Optional, Tuple

import pathspec

logger = logging.getLogger(__name__)


def find_root_gitignore(root_dir: Path) -> Optional[Path]:
    """
    Looks for a .gitignore directly under root_dir, or one level down if
    root_dir contains exactly one entry and it's a directory — the shape
    produced by "Download ZIP" on GitHub/GitLab, which wraps the whole
    repo in a single "reponame-branch/" folder.
    """
    if not root_dir.is_dir():
        return None

    direct = root_dir / ".gitignore"
    if direct.is_file():
        return direct

    try:
        entries = list(root_dir.iterdir())
    except OSError:
        return None

    if len(entries) == 1 and entries[0].is_dir():
        wrapped = entries[0] / ".gitignore"
        if wrapped.is_file():
            return wrapped

    return None


def load_gitignore_spec(gitignore_path: Path) -> Optional[pathspec.PathSpec]:
    try:
        content = gitignore_path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        logger.exception("Failed to read .gitignore at %s", gitignore_path)
        return None

    try:
        return pathspec.PathSpec.from_lines("gitignore", content.splitlines())
    except Exception:
        logger.exception("Failed to parse .gitignore at %s", gitignore_path)
        return None


def build_gitignore_spec(root_dir: Path) -> Tuple[Optional[pathspec.PathSpec], Optional[Path]]:
    """
    Convenience wrapper: find + load a project's root .gitignore in one
    call.

    Returns (spec, spec_root). spec_root is the directory the spec's
    patterns are relative to — root_dir itself, or its single wrapping
    subdirectory — callers must resolve paths relative to *spec_root*,
    not necessarily root_dir, before matching. Returns (None, None) when
    there's no .gitignore to apply (spec-less callers should treat every
    path as not ignored).
    """
    gitignore_path = find_root_gitignore(root_dir)
    if gitignore_path is None:
        return None, None

    spec = load_gitignore_spec(gitignore_path)
    if spec is None:
        return None, None

    logger.info("Applying .gitignore rules from %s", gitignore_path)
    return spec, gitignore_path.parent


def is_path_ignored(
    spec: Optional[pathspec.PathSpec],
    spec_root: Optional[Path],
    full_path: Path,
    is_dir: bool = False,
) -> bool:
    """
    True if full_path should be excluded per spec. is_dir matters:
    gitignore patterns ending in '/' only match directories, and
    pathspec's matcher only honors that when the tested path itself ends
    in '/'.

    Always False when spec/spec_root are None (no .gitignore in play), so
    callers can call this unconditionally without an extra None check.
    """
    if spec is None or spec_root is None:
        return False

    try:
        rel = full_path.relative_to(spec_root).as_posix()
    except ValueError:
        # full_path isn't under spec_root at all — e.g. spec_root is a
        # single wrapping subfolder and full_path is a sibling of it.
        # Nothing to ignore in that case.
        return False

    if is_dir:
        rel += "/"

    return spec.match_file(rel)