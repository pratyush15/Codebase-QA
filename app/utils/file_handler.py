import os
import zipfile
import tempfile
import shutil
import chardet
from pathlib import Path
from typing import List, Dict, Optional

from config import SUPPORTED_EXTENSIONS, SKIP_EXTENSIONS, SKIP_DIRS


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

        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(tmp_dir)

        for root, dirs, files in os.walk(tmp_dir):
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
            for filename in files:
                full_path = Path(root) / filename
                if not is_readable_file(full_path):
                    continue
                content = read_file_content(full_path)
                if not content or not content.strip():
                    continue
                rel_path = full_path.relative_to(tmp_dir)
                parts = rel_path.parts
                if len(parts) > 1 and parts[0].endswith(".zip"):
                    rel_path = Path(*parts[1:])
                extracted.append({
                    "path": str(rel_path),
                    "content": content,
                    "extension": full_path.suffix.lower(),
                    "size_kb": round(full_path.stat().st_size / 1024, 2),
                })
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
            "path": file_name,
            "content": content,
            "extension": path.suffix.lower(),
            "size_kb": round(len(file_bytes) / 1024, 2),
        })

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