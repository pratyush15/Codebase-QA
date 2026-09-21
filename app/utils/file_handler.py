
import os
import zipfile
import tempfile
import shutil
import logging
import chardet
from pathlib import Path
from typing import List, Dict, Optional

from config import (
    SUPPORTED_EXTENSIONS, SKIP_EXTENSIONS, SKIP_DIRS, RESPECT_GITIGNORE,
    MAX_ZIP_UNCOMPRESSED_MB, MAX_ZIP_ENTRIES, MAX_ZIP_COMPRESSION_RATIO,
)
from utils.gitignore_filter import build_gitignore_spec, is_path_ignored

logger = logging.getLogger(__name__)


class UnsafeZipError(Exception):
    """Raised when a zip archive is unsafe to extract: a path that would
    escape the extraction directory, a corrupted/non-zip file, or one that
    looks like a "zip bomb" (would expand to an unreasonable amount of
    data or file count relative to its own size)."""


def _validate_zip_safety(
    zf: zipfile.ZipFile,
    max_uncompressed_mb: int = MAX_ZIP_UNCOMPRESSED_MB,
    max_entries: int = MAX_ZIP_ENTRIES,
    max_compression_ratio: int = MAX_ZIP_COMPRESSION_RATIO,
) -> None:
    """
    Inspects a zip's central directory — before extracting anything — to
    reject archives that would blow up memory/disk/CPU once extracted
    ("zip bombs"). All three checks read only metadata (entry count,
    declared compressed/uncompressed sizes) already present in the zip's
    directory listing; none of them touch actual file content, so a
    malicious archive is rejected without ever writing a byte of its
    payload to disk.

    Three independent checks, any one of which is grounds for rejection:
      - total entry count (a "quantity bomb" — e.g. a million empty files
        — is a resource-exhaustion problem via metadata/hashing/chunking
        overhead alone, regardless of total byte size)
      - total uncompressed size across every entry
      - any single entry's compression ratio (uncompressed / compressed):
        a legitimate source file rarely compresses more than ~10-20x,
        while a deliberately crafted bomb can exceed 1000x — entries
        under a small floor size are exempt, since ratio is a noisy,
        meaningless signal for tiny files and the total-size check above
        already bounds the actual damage they could do.
    """
    infolist = zf.infolist()

    if len(infolist) > max_entries:
        raise UnsafeZipError(
            f"Zip contains {len(infolist)} entries, over the {max_entries}-entry "
            f"limit — refusing to extract this archive."
        )

    max_uncompressed_bytes = max_uncompressed_mb * 1024 * 1024
    ratio_floor_bytes = 4096  # entries smaller than this are exempt from the ratio check

    total_uncompressed = 0
    for member in infolist:
        total_uncompressed += member.file_size
        if total_uncompressed > max_uncompressed_bytes:
            raise UnsafeZipError(
                f"Zip would extract to over {max_uncompressed_mb}MB uncompressed — "
                f"refusing to extract this archive (possible zip bomb)."
            )

        if member.compress_size > ratio_floor_bytes:
            ratio = member.file_size / max(member.compress_size, 1)
            if ratio > max_compression_ratio:
                raise UnsafeZipError(
                    f"Zip entry '{member.filename}' has a suspicious "
                    f"{ratio:.0f}:1 compression ratio — refusing to extract "
                    f"this archive (possible zip bomb)."
                )


def _safe_extract(zf: zipfile.ZipFile, dest_dir: str) -> None:
    """
    Extract a zip archive while blocking zip-slip path traversal.

    Every member's resolved path must stay inside dest_dir. Members using
    '../', absolute paths, or symlink tricks to escape dest_dir are rejected
    outright rather than silently skipped, since a partially-extracted
    malicious archive is still a red flag worth surfacing to the caller.
    """
    dest_root = Path(dest_dir).resolve()

    for member in zf.infolist():
        member_path = Path(dest_dir) / member.filename
        resolved = member_path.resolve()

        if resolved != dest_root and dest_root not in resolved.parents:
            raise UnsafeZipError(
                f"Zip entry '{member.filename}' resolves outside the extraction "
                f"directory — refusing to extract this archive."
            )

    zf.extractall(dest_dir)


def detect_encoding(file_bytes: bytes) -> str:
    result = chardet.detect(file_bytes)
    return result.get("encoding") or "utf-8"


def is_readable_file(path: Path) -> bool:
    if path.suffix.lower() in SKIP_EXTENSIONS:
        return False
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        return False
    for skip_dir in SKIP_DIRS:
        if skip_dir in path.parts:
            return False
    return True


def read_file_content(file_path: Path) -> Optional[str]:
    try:
        raw = file_path.read_bytes()
        return raw.decode(detect_encoding(raw), errors="replace")
    except Exception:
        return None


def extract_files_from_zip(zip_bytes: bytes) -> List[Dict]:
    extracted = []
    tmp_dir = tempfile.mkdtemp()

    try:
        zip_path = os.path.join(tmp_dir, "upload.zip")
        with open(zip_path, "wb") as f:
            f.write(zip_bytes)

        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                _validate_zip_safety(zf, MAX_ZIP_UNCOMPRESSED_MB, MAX_ZIP_ENTRIES, MAX_ZIP_COMPRESSION_RATIO)
                _safe_extract(zf, tmp_dir)
        except zipfile.BadZipFile as e:
            raise UnsafeZipError(f"Not a valid zip file: {e}") from e

        gitignore_spec, gitignore_root = (
            build_gitignore_spec(Path(tmp_dir)) if RESPECT_GITIGNORE else (None, None)
        )
        gitignore_skipped = 0

        for root, dirs, files in os.walk(tmp_dir):
            dirs[:] = [
                d for d in dirs
                if d not in SKIP_DIRS
                and not is_path_ignored(gitignore_spec, gitignore_root, Path(root) / d, is_dir=True)
            ]
            for filename in files:
                full_path = Path(root) / filename
                if not is_readable_file(full_path):
                    continue
                if is_path_ignored(gitignore_spec, gitignore_root, full_path):
                    gitignore_skipped += 1
                    continue
                content = read_file_content(full_path)
                if not content or not content.strip():
                    continue
                rel_path = full_path.relative_to(tmp_dir)
                parts = rel_path.parts
                if len(parts) > 1 and parts[0].endswith(".zip"):
                    rel_path = Path(*parts[1:])
                extracted.append({
                    # as_posix(): keep "/"-separated paths on every OS, since
                    # this becomes metadata["source"] and must match the
                    # forward-slash paths hardcoded in eval/eval_set.py.
                    "path": rel_path.as_posix(),
                    "content": content,
                    "extension": full_path.suffix.lower(),
                    "size_kb": round(full_path.stat().st_size / 1024, 2),
                })

        if gitignore_skipped:
            logger.info("Skipped %d file(s) matching .gitignore rules", gitignore_skipped)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return extracted


def extract_files_from_uploads(uploaded_files: list) -> List[Dict]:
    extracted = []
    for uploaded_file in uploaded_files:
        file_bytes = uploaded_file.read()
        file_name = uploaded_file.name

        if file_name.endswith(".zip"):
            extracted.extend(extract_files_from_zip(file_bytes))
            continue

        path = Path(file_name)
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        content = file_bytes.decode(detect_encoding(file_bytes), errors="replace")
        if not content.strip():
            continue

        extracted.append({
            "path": path.as_posix(),
            "content": content,
            "extension": path.suffix.lower(),
            "size_kb": round(len(file_bytes) / 1024, 2),
        })

    return extracted


def extract_files_from_directory(root_dir: str) -> List[Dict]:
    """
    Walk a local directory on disk and extract readable files the same way
    extract_files_from_zip does for an uploaded archive — used by the eval
    harness (and any other tool that wants to index a repo already checked
    out locally, without zipping it first).
    """
    extracted = []
    root_path = Path(root_dir).resolve()

    gitignore_spec, gitignore_root = (
        build_gitignore_spec(root_path) if RESPECT_GITIGNORE else (None, None)
    )
    gitignore_skipped = 0

    for root, dirs, files in os.walk(root_path):
        dirs[:] = [
            d for d in dirs
            if d not in SKIP_DIRS
            and not is_path_ignored(gitignore_spec, gitignore_root, Path(root) / d, is_dir=True)
        ]
        for filename in files:
            full_path = Path(root) / filename
            if not is_readable_file(full_path):
                continue
            if is_path_ignored(gitignore_spec, gitignore_root, full_path):
                gitignore_skipped += 1
                continue
            content = read_file_content(full_path)
            if not content or not content.strip():
                continue
            rel_path = full_path.relative_to(root_path)
            extracted.append({
                # as_posix(): keep "/"-separated paths on every OS (Windows'
                # str(rel_path) uses "\\", which broke eval matching against
                # eval_set.py's hardcoded forward-slash expected_sources).
                "path": rel_path.as_posix(),
                "content": content,
                "extension": full_path.suffix.lower(),
                "size_kb": round(full_path.stat().st_size / 1024, 2),
            })

    if gitignore_skipped:
        logger.info("Skipped %d file(s) matching .gitignore rules", gitignore_skipped)

    return extracted


def get_summary(files: List[Dict]) -> Dict:
    total_size = sum(f["size_kb"] for f in files)
    ext_counts: Dict[str, int] = {}
    for f in files:
        ext = f["extension"] or "unknown"
        ext_counts[ext] = ext_counts.get(ext, 0) + 1
    return {
        "total_files": len(files),
        "total_size_kb": round(total_size, 2),
        "by_extension": dict(sorted(ext_counts.items(), key=lambda x: -x[1])),
    }