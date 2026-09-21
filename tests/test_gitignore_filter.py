from pathlib import Path

from utils.gitignore_filter import (
    find_root_gitignore,
    load_gitignore_spec,
    build_gitignore_spec,
    is_path_ignored,
)


# ---------- find_root_gitignore ----------

def test_finds_gitignore_directly_under_root(tmp_path):
    (tmp_path / ".gitignore").write_text("node_modules/\n")
    found = find_root_gitignore(tmp_path)
    assert found == tmp_path / ".gitignore"


def test_returns_none_when_no_gitignore_anywhere(tmp_path):
    (tmp_path / "src").mkdir()
    assert find_root_gitignore(tmp_path) is None


def test_finds_gitignore_inside_single_wrapping_folder(tmp_path):
    # Shape produced by GitHub/GitLab "Download ZIP": everything nested
    # under one "reponame-branch/" folder.
    wrapper = tmp_path / "myrepo-main"
    wrapper.mkdir()
    (wrapper / ".gitignore").write_text("dist/\n")
    found = find_root_gitignore(tmp_path)
    assert found == wrapper / ".gitignore"


def test_does_not_look_inside_wrapper_when_multiple_top_level_entries(tmp_path):
    # Two top-level entries -> not the single-wrapper-folder shape, so we
    # should NOT go hunting inside either of them.
    (tmp_path / "folder_a").mkdir()
    (tmp_path / "folder_a" / ".gitignore").write_text("ignored/\n")
    (tmp_path / "folder_b").mkdir()
    assert find_root_gitignore(tmp_path) is None


def test_returns_none_for_nonexistent_root():
    assert find_root_gitignore(Path("/definitely/does/not/exist")) is None


# ---------- load_gitignore_spec ----------

def test_load_gitignore_spec_parses_patterns(tmp_path):
    gi = tmp_path / ".gitignore"
    gi.write_text("*.pyc\nnode_modules/\n")
    spec = load_gitignore_spec(gi)
    assert spec is not None
    assert spec.match_file("foo.pyc") is True
    assert spec.match_file("node_modules/") is True
    assert spec.match_file("src/app.py") is False


def test_load_gitignore_spec_returns_none_on_unreadable_file(tmp_path):
    missing = tmp_path / "does_not_exist" / ".gitignore"
    assert load_gitignore_spec(missing) is None


# ---------- build_gitignore_spec ----------

def test_build_gitignore_spec_returns_none_none_when_absent(tmp_path):
    spec, root = build_gitignore_spec(tmp_path)
    assert spec is None
    assert root is None


def test_build_gitignore_spec_returns_spec_and_correct_root(tmp_path):
    (tmp_path / ".gitignore").write_text("build/\n")
    spec, root = build_gitignore_spec(tmp_path)
    assert spec is not None
    assert root == tmp_path


def test_build_gitignore_spec_root_is_wrapper_folder_when_nested(tmp_path):
    wrapper = tmp_path / "myrepo-main"
    wrapper.mkdir()
    (wrapper / ".gitignore").write_text("build/\n")
    spec, root = build_gitignore_spec(tmp_path)
    assert spec is not None
    assert root == wrapper


# ---------- is_path_ignored ----------

def test_is_path_ignored_false_when_no_spec():
    assert is_path_ignored(None, None, Path("anything/goes.py")) is False


def test_is_path_ignored_matches_file_pattern(tmp_path):
    (tmp_path / ".gitignore").write_text("*.log\n")
    spec, root = build_gitignore_spec(tmp_path)

    assert is_path_ignored(spec, root, tmp_path / "server.log") is True
    assert is_path_ignored(spec, root, tmp_path / "server.py") is False


def test_is_path_ignored_directory_only_pattern_requires_is_dir_flag(tmp_path):
    (tmp_path / ".gitignore").write_text("build/\n")
    spec, root = build_gitignore_spec(tmp_path)

    # As a directory: ignored.
    assert is_path_ignored(spec, root, tmp_path / "build", is_dir=True) is True
    # A *file* literally named "build" (no trailing slash) should NOT match
    # a directory-only gitignore pattern.
    assert is_path_ignored(spec, root, tmp_path / "build", is_dir=False) is False


def test_is_path_ignored_unanchored_pattern_matches_at_any_depth(tmp_path):
    (tmp_path / ".gitignore").write_text("node_modules/\n")
    spec, root = build_gitignore_spec(tmp_path)

    assert is_path_ignored(spec, root, tmp_path / "node_modules", is_dir=True) is True
    assert is_path_ignored(spec, root, tmp_path / "packages" / "app" / "node_modules", is_dir=True) is True


def test_is_path_ignored_anchored_pattern_only_matches_at_root(tmp_path):
    (tmp_path / ".gitignore").write_text("/coverage\n")
    spec, root = build_gitignore_spec(tmp_path)

    assert is_path_ignored(spec, root, tmp_path / "coverage", is_dir=True) is True
    assert is_path_ignored(spec, root, tmp_path / "src" / "coverage", is_dir=True) is False


def test_is_path_ignored_respects_negation(tmp_path):
    (tmp_path / ".gitignore").write_text("*.pyc\n!important.pyc\n")
    spec, root = build_gitignore_spec(tmp_path)

    assert is_path_ignored(spec, root, tmp_path / "throwaway.pyc") is True
    assert is_path_ignored(spec, root, tmp_path / "important.pyc") is False


def test_is_path_ignored_returns_false_for_path_outside_spec_root(tmp_path):
    (tmp_path / ".gitignore").write_text("*.log\n")
    spec, _ = build_gitignore_spec(tmp_path)
    outside = tmp_path.parent / "sibling" / "file.log"

    assert is_path_ignored(spec, tmp_path, outside) is False