
"""
Incremental re-indexing.

Re-uploading a project used to always re-embed and re-add every file, which
silently duplicated chunks for anything already indexed (flagged as a known
limitation in the README). This module diffs an upload against what's
already in the collection by content hash, so a re-index only pays the
embedding cost for files that are actually new or changed.

It also handles the mirror-image problem: a file that WAS indexed but is
absent from the current upload (deleted, renamed, moved out of the repo).
Left alone, its chunks stay in the collection forever and keep surfacing
in retrieval for a file that no longer exists. See `full_sync` below.
"""
import time
import logging
from typing import List, Dict

from core.ingestion import ingest_files, hash_file_content, get_ingestion_stats
from core.vectorstore import (
    get_collection_file_hashes,
    delete_documents_by_source,
    delete_documents_by_sources,
    add_documents_to_collection,
)

logger = logging.getLogger(__name__)


def sync_files_to_collection(collection_name: str, files: List[Dict], full_sync: bool = False) -> Dict:
    """
    Index only new or changed files into collection_name.

    Args:
        collection_name: target Chroma collection.
        files: list of file dicts from file_handler (path, content, ...).
        full_sync: whether `files` represents the *entire* current state of
            the project, so that anything indexed in the collection but
            missing from `files` is safely known to have been deleted (or
            renamed) and its chunks should be removed. Set this True for a
            full-codebase re-upload (e.g. a fresh zip of the whole repo).
            Leave it False for a partial upload (e.g. a handful of
            individual files) — there, a file's absence just means "not
            part of this batch", not "deleted", so nothing is removed.
            Defaults to False so callers must opt in explicitly rather
            than risk wiping out an untouched part of a project.

    Returns:
        Stats dict: new_files, updated_files, skipped_files, deleted_files
        (counts), their path lists, total_chunks added/updated, and timing.
    """
    t0 = time.perf_counter()

    existing_hashes = get_collection_file_hashes(collection_name)

    new_files: List[Dict] = []
    updated_files: List[Dict] = []
    skipped_paths: List[str] = []

    for f in files:
        current_hash = hash_file_content(f["content"])
        prior_hash = existing_hashes.get(f["path"])

        if prior_hash is None:
            new_files.append(f)
        elif prior_hash != current_hash:
            updated_files.append(f)
        else:
            skipped_paths.append(f["path"])

    # Changed files: clear their old chunks before re-adding, so we don't
    # accumulate stale chunks alongside the new ones for the same file.
    for f in updated_files:
        delete_documents_by_source(collection_name, f["path"])

    # Full sync only: anything previously indexed that isn't in this
    # upload at all was removed from the project — drop its chunks too,
    # or they'd keep answering questions about code that no longer exists.
    deleted_paths: List[str] = []
    if full_sync:
        uploaded_paths = {f["path"] for f in files}
        deleted_paths = sorted(set(existing_hashes) - uploaded_paths)
        if deleted_paths:
            delete_documents_by_sources(collection_name, deleted_paths)

    files_to_embed = new_files + updated_files
    documents = ingest_files(files_to_embed)

    total_chunk_count = 0
    if documents or deleted_paths:
        total_chunk_count = add_documents_to_collection(collection_name, documents)

    ingestion_stats = get_ingestion_stats(documents)
    elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)

    stats = {
        "new_files": len(new_files),
        "updated_files": len(updated_files),
        "skipped_files": len(skipped_paths),
        "deleted_files": len(deleted_paths),
        "new_file_paths": [f["path"] for f in new_files],
        "updated_file_paths": [f["path"] for f in updated_files],
        "skipped_file_paths": skipped_paths,
        "deleted_file_paths": deleted_paths,
        "chunks_embedded": len(documents),
        "collection_chunk_count": total_chunk_count,
        "chunk_methods": ingestion_stats["chunk_methods"],
        "elapsed_ms": elapsed_ms,
    }

    logger.info(
        "Sync indexed collection '%s': %d new, %d updated, %d unchanged, %d deleted "
        "(%d chunks embedded) full_sync=%s in %.1fms",
        collection_name, stats["new_files"], stats["updated_files"],
        stats["skipped_files"], stats["deleted_files"], stats["chunks_embedded"],
        full_sync, elapsed_ms,
    )

    return stats