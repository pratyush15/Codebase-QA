
import io
import zipfile
import tempfile

import pytest

from utils.file_handler import (
    extract_files_from_zip,
    extract_files_from_directory,
    _safe_extract,
    _validate_zip_safety,
    UnsafeZipError,
    is_readable_file,
    get_summary,
)
from pathlib import Path


def _make_zip(entries: dict) -> bytes:
    """entries: {filename_in_zip: content_str}"""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, content in entries.items():
            zf.writestr(name, content)
    return buf.getvalue()


# ---------- zip-slip protection ----------

def test_safe_extract_rejects_parent_directory_traversal():
    zip_bytes = _make_zip({"../../evil.py": "print('pwned')"})
    tmp_dir = tempfile.mkdtemp()

    with pytest.raises(UnsafeZipError):
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            _safe_extract(zf, tmp_dir)


def test_safe_extract_rejects_absolute_path_entry():
    # Some zip tools store absolute paths directly; these must be rejected too.
    zip_bytes = _make_zip({"/etc/evil.py": "print('pwned')"})
    tmp_dir = tempfile.mkdtemp()

    with pytest.raises(UnsafeZipError):
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            _safe_extract(zf, tmp_dir)


def test_extract_files_from_zip_rejects_malicious_archive():
    zip_bytes = _make_zip({"../outside.py": "print('escape')"})
    with pytest.raises(UnsafeZipError):
        extract_files_from_zip(zip_bytes)


def test_safe_extract_allows_normal_nested_paths():
    zip_bytes = _make_zip({"src/app/main.py": "print('hello')", "README.md": "# hi"})
    tmp_dir = tempfile.mkdtemp()

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        _safe_extract(zf, tmp_dir)  # should not raise

    assert (Path(tmp_dir) / "src" / "app" / "main.py").exists()
    assert (Path(tmp_dir) / "README.md").exists()


# ---------- normal extraction behavior ----------

def test_extract_files_from_zip_returns_readable_files():
    zip_bytes = _make_zip({
        "src/app.py": "print('hello')",
        "assets/logo.png": "not real png bytes but irrelevant",
        "README.md": "# Project",
    })

    files = extract_files_from_zip(zip_bytes)
    paths = {f["path"] for f in files}

    assert "src/app.py" in paths
    assert "README.md" in paths
    # .png is in SKIP_EXTENSIONS, should never come through
    assert not any(p.endswith(".png") for p in paths)


def test_extract_files_from_zip_skips_empty_files():
    zip_bytes = _make_zip({"empty.py": "", "whitespace_only.py": "   \n  \n"})
    files = extract_files_from_zip(zip_bytes)
    assert files == []


def test_extract_files_from_zip_skips_configured_skip_dirs():
    zip_bytes = _make_zip({
        "node_modules/pkg/index.js": "console.log('dep')",
        "src/index.js": "console.log('mine')",
    })
    files = extract_files_from_zip(zip_bytes)
    paths = {f["path"] for f in files}
    assert not any("node_modules" in p for p in paths)
    assert any(p.endswith("src/index.js") or p.endswith("index.js") for p in paths)


# ---------- is_readable_file ----------

def test_is_readable_file_accepts_supported_extension():
    assert is_readable_file(Path("app/main.py")) is True


def test_is_readable_file_rejects_skip_extension():
    assert is_readable_file(Path("logo.png")) is False


def test_is_readable_file_rejects_unsupported_extension():
    assert is_readable_file(Path("data.bin")) is False


def test_is_readable_file_rejects_files_in_skip_dirs():
    assert is_readable_file(Path("node_modules/pkg/index.js")) is False
    assert is_readable_file(Path("venv/lib/site.py")) is False


# ---------- get_summary ----------

def test_get_summary_counts_files_and_extensions():
    files = [
        {"path": "a.py", "extension": ".py", "size_kb": 1.0},
        {"path": "b.py", "extension": ".py", "size_kb": 2.0},
        {"path": "c.js", "extension": ".js", "size_kb": 0.5},
    ]
    summary = get_summary(files)

    assert summary["total_files"] == 3
    assert summary["total_size_kb"] == 3.5
    assert summary["by_extension"][".py"] == 2
    assert summary["by_extension"][".js"] == 1


def test_get_summary_handles_empty_list():
    summary = get_summary([])
    assert summary == {"total_files": 0, "total_size_kb": 0, "by_extension": {}}


# ---------- extract_files_from_directory ----------

def test_extract_files_from_directory_reads_readable_files(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("print('hello')")
    (tmp_path / "README.md").write_text("# Project")
    (tmp_path / "logo.png").write_bytes(b"\x89PNG fake")

    files = extract_files_from_directory(str(tmp_path))
    paths = {f["path"] for f in files}

    assert "src/app.py" in paths or str(Path("src") / "app.py") in paths
    assert "README.md" in paths
    assert not any(p.endswith(".png") for p in paths)


def test_extract_files_from_directory_skips_configured_skip_dirs(tmp_path):
    (tmp_path / "node_modules" / "pkg").mkdir(parents=True)
    (tmp_path / "node_modules" / "pkg" / "index.js").write_text("console.log('dep')")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "index.js").write_text("console.log('mine')")

    files = extract_files_from_directory(str(tmp_path))
    paths = {f["path"] for f in files}

    assert not any("node_modules" in p for p in paths)
    assert any("index.js" in p for p in paths)


def test_extract_files_from_directory_skips_empty_files(tmp_path):
    (tmp_path / "empty.py").write_text("")
    (tmp_path / "real.py").write_text("x = 1")

    files = extract_files_from_directory(str(tmp_path))
    paths = {f["path"] for f in files}

    assert "empty.py" not in paths
    assert "real.py" in paths

# ---------- .gitignore awareness ----------

def test_extract_files_from_zip_respects_root_gitignore():
    zip_bytes = _make_zip({
        ".gitignore": "build/\n*.generated.js\n",
        "src/app.py": "print('hello')",
        "build/bundle.js": "// generated output",
        "src/schema.generated.js": "// generated",
    })
    files = extract_files_from_zip(zip_bytes)
    paths = {f["path"] for f in files}

    assert "src/app.py" in paths
    assert not any("build/" in p for p in paths)
    assert not any(p.endswith(".generated.js") for p in paths)


def test_extract_files_from_zip_with_no_gitignore_is_unaffected():
    zip_bytes = _make_zip({
        "src/app.py": "print('hello')",
        "out/bundle.js": "// not ignored, no .gitignore present",
    })
    files = extract_files_from_zip(zip_bytes)
    paths = {f["path"] for f in files}

    assert "src/app.py" in paths
    assert "out/bundle.js" in paths


def test_extract_files_from_zip_gitignore_disabled_via_config(monkeypatch):
    import utils.file_handler as fh
    monkeypatch.setattr(fh, "RESPECT_GITIGNORE", False)

    zip_bytes = _make_zip({
        ".gitignore": "out/\n",
        "out/bundle.js": "// would normally be ignored",
    })
    files = extract_files_from_zip(zip_bytes)
    paths = {f["path"] for f in files}

    assert "out/bundle.js" in paths


def test_extract_files_from_directory_respects_root_gitignore(tmp_path):
    (tmp_path / ".gitignore").write_text("vendor/\n*.log\n")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("print('hello')")
    (tmp_path / "vendor").mkdir()
    (tmp_path / "vendor" / "lib.py").write_text("# vendored dependency")
    (tmp_path / "debug.log").write_text("log output")

    files = extract_files_from_directory(str(tmp_path))
    paths = {f["path"] for f in files}

    assert "src/app.py" in paths
    assert not any("vendor/" in p for p in paths)
    assert "debug.log" not in paths


def test_extract_files_from_directory_gitignore_negation_still_indexed(tmp_path):
    (tmp_path / ".gitignore").write_text("*.pyc\n")
    (tmp_path / "cache.pyc").write_text("# not real bytecode, just testing extension filtering")
    (tmp_path / "keep.py").write_text("x = 1")

    files = extract_files_from_directory(str(tmp_path))
    paths = {f["path"] for f in files}

    # .pyc isn't in SUPPORTED_EXTENSIONS anyway, so this mainly checks that
    # gitignore filtering doesn't error out or accidentally drop keep.py.
    assert "keep.py" in paths


# ---------- zip bomb protection ----------

def _open_zip(zip_bytes: bytes) -> zipfile.ZipFile:
    return zipfile.ZipFile(io.BytesIO(zip_bytes))


def test_validate_zip_safety_accepts_normal_archive():
    zip_bytes = _make_zip({"a.py": "print('hi')", "b.py": "print('bye')"})
    with _open_zip(zip_bytes) as zf:
        _validate_zip_safety(zf, max_uncompressed_mb=500, max_entries=20000, max_compression_ratio=100)
        # no exception -> passes


def test_validate_zip_safety_rejects_too_many_entries():
    zip_bytes = _make_zip({f"file_{i}.py": "x" for i in range(50)})
    with _open_zip(zip_bytes) as zf:
        with pytest.raises(UnsafeZipError, match="entries"):
            _validate_zip_safety(zf, max_uncompressed_mb=500, max_entries=10, max_compression_ratio=100)


def test_validate_zip_safety_rejects_excessive_total_uncompressed_size():
    zip_bytes = _make_zip({"big.py": "x" * (2 * 1024 * 1024)})  # 2MB uncompressed
    with _open_zip(zip_bytes) as zf:
        with pytest.raises(UnsafeZipError, match="uncompressed"):
            _validate_zip_safety(zf, max_uncompressed_mb=1, max_entries=20000, max_compression_ratio=100)


def test_validate_zip_safety_rejects_high_compression_ratio_zip_bomb():
    # Highly repetitive content compresses extremely well — a classic
    # zip-bomb shape: small on disk, huge once expanded.
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("bomb.txt", "0" * (5 * 1024 * 1024))  # 5MB of a single repeated byte
    zip_bytes = buf.getvalue()

    with _open_zip(zip_bytes) as zf:
        # total-size cap deliberately set high so only the ratio check can trigger
        with pytest.raises(UnsafeZipError, match="compression ratio"):
            _validate_zip_safety(zf, max_uncompressed_mb=500, max_entries=20000, max_compression_ratio=50)


def test_validate_zip_safety_ratio_check_exempts_small_entries():
    # A tiny file that happens to compress well shouldn't trip the ratio
    # check — the floor exists specifically to avoid this false positive.
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("tiny.txt", "0" * 500)  # well under the 4096-byte compressed-size floor
    zip_bytes = buf.getvalue()

    with _open_zip(zip_bytes) as zf:
        _validate_zip_safety(zf, max_uncompressed_mb=500, max_entries=20000, max_compression_ratio=10)
        # no exception -> passes despite a high ratio, because it's tiny


def test_extract_files_from_zip_rejects_zip_bomb_via_entry_count(monkeypatch):
    import utils.file_handler as fh
    monkeypatch.setattr(fh, "MAX_ZIP_ENTRIES", 5)

    zip_bytes = _make_zip({f"file_{i}.py": "x = 1" for i in range(10)})
    with pytest.raises(UnsafeZipError):
        extract_files_from_zip(zip_bytes)


def test_extract_files_from_zip_rejects_zip_bomb_via_uncompressed_size(monkeypatch):
    import utils.file_handler as fh
    monkeypatch.setattr(fh, "MAX_ZIP_UNCOMPRESSED_MB", 1)

    zip_bytes = _make_zip({"big.py": "x" * (2 * 1024 * 1024)})
    with pytest.raises(UnsafeZipError):
        extract_files_from_zip(zip_bytes)


def test_extract_files_from_zip_raises_unsafe_zip_error_for_corrupted_archive():
    with pytest.raises(UnsafeZipError, match="[Nn]ot a valid zip"):
        extract_files_from_zip(b"this is not a zip file at all")


def test_extract_files_from_zip_rejected_bomb_never_reaches_safe_extract(monkeypatch):
    # A rejected zip-bomb-shaped archive must be caught before extraction
    # ever runs — _safe_extract (which writes the (potentially huge)
    # content to disk) should never even be called.
    import utils.file_handler as fh
    monkeypatch.setattr(fh, "MAX_ZIP_UNCOMPRESSED_MB", 1)

    called = {"safe_extract": False}
    original_safe_extract = fh._safe_extract

    def _tracking_safe_extract(*args, **kwargs):
        called["safe_extract"] = True
        return original_safe_extract(*args, **kwargs)

    monkeypatch.setattr(fh, "_safe_extract", _tracking_safe_extract)

    zip_bytes = _make_zip({"big.py": "x" * (2 * 1024 * 1024)})
    with pytest.raises(UnsafeZipError):
        extract_files_from_zip(zip_bytes)

    assert called["safe_extract"] is False