
from unittest.mock import patch

from core.indexing import sync_files_to_collection
from core.ingestion import hash_file_content


def _file(path, content, size_kb=0.1):
    return {"path": path, "content": content, "size_kb": size_kb}


@patch("core.indexing.add_documents_to_collection", return_value=10)
@patch("core.indexing.delete_documents_by_source")
@patch("core.indexing.get_collection_file_hashes")
def test_sync_treats_all_files_as_new_when_collection_is_empty(
    mock_hashes, mock_delete, mock_add
):
    mock_hashes.return_value = {}
    files = [_file("a.py", "def a(): pass"), _file("b.py", "def b(): pass")]

    stats = sync_files_to_collection("proj", files)

    assert stats["new_files"] == 2
    assert stats["updated_files"] == 0
    assert stats["skipped_files"] == 0
    mock_delete.assert_not_called()
    mock_add.assert_called_once()


@patch("core.indexing.add_documents_to_collection", return_value=10)
@patch("core.indexing.delete_documents_by_source")
@patch("core.indexing.get_collection_file_hashes")
def test_sync_skips_unchanged_files(mock_hashes, mock_delete, mock_add):
    content = "def a(): pass"
    mock_hashes.return_value = {"a.py": hash_file_content(content)}
    files = [_file("a.py", content)]

    stats = sync_files_to_collection("proj", files)

    assert stats["skipped_files"] == 1
    assert stats["new_files"] == 0
    assert stats["updated_files"] == 0
    assert stats["chunks_embedded"] == 0
    mock_delete.assert_not_called()
    mock_add.assert_not_called()  # nothing to embed — should not touch the vectorstore


@patch("core.indexing.add_documents_to_collection", return_value=10)
@patch("core.indexing.delete_documents_by_source")
@patch("core.indexing.get_collection_file_hashes")
def test_sync_reembeds_changed_files_after_deleting_old_chunks(
    mock_hashes, mock_delete, mock_add
):
    mock_hashes.return_value = {"a.py": hash_file_content("def a(): return 1")}
    files = [_file("a.py", "def a(): return 2")]  # content changed

    stats = sync_files_to_collection("proj", files)

    assert stats["updated_files"] == 1
    assert stats["new_files"] == 0
    mock_delete.assert_called_once_with("proj", "a.py")
    mock_add.assert_called_once()


@patch("core.indexing.add_documents_to_collection", return_value=10)
@patch("core.indexing.delete_documents_by_source")
@patch("core.indexing.get_collection_file_hashes")
def test_sync_handles_mixed_new_updated_and_unchanged_files(
    mock_hashes, mock_delete, mock_add
):
    unchanged_content = "def c(): pass"
    mock_hashes.return_value = {
        "unchanged.py": hash_file_content(unchanged_content),
        "changed.py": hash_file_content("def old(): pass"),
    }
    files = [
        _file("new.py", "def new(): pass"),
        _file("changed.py", "def new_version(): pass"),
        _file("unchanged.py", unchanged_content),
    ]

    stats = sync_files_to_collection("proj", files)

    assert stats["new_files"] == 1
    assert stats["updated_files"] == 1
    assert stats["skipped_files"] == 1
    mock_delete.assert_called_once_with("proj", "changed.py")

# ---------- full_sync: stale-file (deleted) cleanup ----------

@patch("core.indexing.add_documents_to_collection", return_value=10)
@patch("core.indexing.delete_documents_by_sources")
@patch("core.indexing.delete_documents_by_source")
@patch("core.indexing.get_collection_file_hashes")
def test_sync_default_does_not_delete_stale_files(
    mock_hashes, mock_delete_one, mock_delete_many, mock_add
):
    # "gone.py" was indexed previously but isn't in this upload at all.
    mock_hashes.return_value = {"a.py": hash_file_content("def a(): pass"), "gone.py": "somehash"}
    files = [_file("a.py", "def a(): pass")]  # unchanged; gone.py simply absent

    stats = sync_files_to_collection("proj", files)  # full_sync defaults to False

    assert stats["deleted_files"] == 0
    assert stats["deleted_file_paths"] == []
    mock_delete_many.assert_not_called()


@patch("core.indexing.add_documents_to_collection", return_value=10)
@patch("core.indexing.delete_documents_by_sources")
@patch("core.indexing.delete_documents_by_source")
@patch("core.indexing.get_collection_file_hashes")
def test_sync_full_sync_deletes_files_missing_from_upload(
    mock_hashes, mock_delete_one, mock_delete_many, mock_add
):
    mock_hashes.return_value = {"a.py": hash_file_content("def a(): pass"), "gone.py": "somehash"}
    files = [_file("a.py", "def a(): pass")]  # unchanged; gone.py removed from the project

    stats = sync_files_to_collection("proj", files, full_sync=True)

    assert stats["deleted_files"] == 1
    assert stats["deleted_file_paths"] == ["gone.py"]
    mock_delete_many.assert_called_once_with("proj", ["gone.py"])


@patch("core.indexing.add_documents_to_collection", return_value=10)
@patch("core.indexing.delete_documents_by_sources")
@patch("core.indexing.delete_documents_by_source")
@patch("core.indexing.get_collection_file_hashes")
def test_sync_full_sync_deletes_nothing_when_everything_still_present(
    mock_hashes, mock_delete_one, mock_delete_many, mock_add
):
    mock_hashes.return_value = {"a.py": hash_file_content("def a(): pass")}
    files = [_file("a.py", "def a(): pass")]

    stats = sync_files_to_collection("proj", files, full_sync=True)

    assert stats["deleted_files"] == 0
    mock_delete_many.assert_not_called()


@patch("core.indexing.add_documents_to_collection", return_value=10)
@patch("core.indexing.delete_documents_by_sources")
@patch("core.indexing.delete_documents_by_source")
@patch("core.indexing.get_collection_file_hashes")
def test_sync_full_sync_refreshes_chunk_count_when_only_a_deletion_happened(
    mock_hashes, mock_delete_one, mock_delete_many, mock_add
):
    # Everything in the upload is unchanged (nothing to embed), but a file
    # was removed from the project - the collection count should still be
    # refreshed to reflect the deletion, even with zero chunks embedded.
    content = "def a(): pass"
    mock_hashes.return_value = {"a.py": hash_file_content(content), "gone.py": "somehash"}
    files = [_file("a.py", content)]

    stats = sync_files_to_collection("proj", files, full_sync=True)

    assert stats["chunks_embedded"] == 0
    assert stats["deleted_files"] == 1
    mock_add.assert_called_once_with("proj", [])
    assert stats["collection_chunk_count"] == 10