
import logging
import streamlit as st
from core.qa_chain import check_ollama_connection
from core.vectorstore import (
    list_collections,
    delete_collection,
    sanitize_collection_name,
    get_collection_sources,
    get_collection_symbols,
)
from core.indexing import sync_files_to_collection
from utils.file_handler import (
    extract_files_from_zip,
    extract_files_from_uploads,
    get_summary,
    UnsafeZipError,
)

logger = logging.getLogger(__name__)


def render_ollama_status():
    st.markdown("### 🤖 Model Status")
    result = check_ollama_connection()
    if result["status"] == "ok":
        st.success(f"Ollama connected — `{result['model']}`")
    else:
        st.error(f"Ollama unreachable — `{result['model']}`")
        st.caption(result.get("message", ""))
    st.divider()


def render_project_selector() -> str:
    st.markdown("### 📁 Project")
    collections = list_collections()

    if not collections:
        st.info("No projects yet. Upload a codebase below.")
        return None

    names = [c["name"] for c in collections]
    counts = {c["name"]: c["chunk_count"] for c in collections}

    default_index = 0
    if "selected_collection" in st.session_state and st.session_state.selected_collection in names:
        default_index = names.index(st.session_state.selected_collection)

    selected = st.selectbox(
        "Active project",
        options=names,
        index=default_index,
        format_func=lambda n: f"{n}  ({counts[n]} chunks)",
        key="active_collection",
    )

    st.session_state.selected_collection = selected

    if selected:
        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("🗑️ Delete", key="delete_collection"):
                if delete_collection(selected):
                    st.session_state.pop("selected_collection", None)
                    st.success(f"Deleted `{selected}`")
                    st.rerun()
                else:
                    st.error("Delete failed.")

    st.divider()
    return selected


def render_upload_section():
    st.markdown("### ⬆️ Upload Codebase")

    project_name = st.text_input(
        "Project name",
        placeholder="my-fastapi-app",
        key="project_name_input",
    )

    upload_mode = st.radio(
        "Upload type",
        options=["Zip folder", "Individual files"],
        horizontal=True,
        key="upload_mode",
    )

    if upload_mode == "Zip folder":
        uploaded = st.file_uploader(
            "Upload a .zip of your project",
            type=["zip"],
            key="zip_uploader",
        )
    else:
        uploaded = st.file_uploader(
            "Upload individual files",
            accept_multiple_files=True,
            key="files_uploader",
        )

    if st.button("📥 Index Codebase", key="index_button"):
        if not project_name.strip():
            st.warning("Enter a project name.")
            return

        if not uploaded:
            st.warning("Upload a zip or files first.")
            return

        collection_name = sanitize_collection_name(project_name.strip())

        with st.spinner("Reading files..."):
            try:
                if upload_mode == "Zip folder":
                    files = extract_files_from_zip(uploaded.read())
                else:
                    files = extract_files_from_uploads(uploaded)
            except UnsafeZipError as e:
                logger.warning("Rejected unsafe zip upload for project '%s': %s", project_name, e)
                st.error("This zip file looks unsafe (it contains paths that escape the target folder) and was rejected.")
                return

        if not files:
            st.error("No readable code files found.")
            return

        summary = get_summary(files)

        # A zip is a snapshot of the whole project, so it's safe to treat
        # missing files as deletions. A batch of individually-picked files
        # is a partial upload — we don't know if the rest of the project
        # is meant to be untouched, so we never delete on that path.
        full_sync = (upload_mode == "Zip folder")

        with st.spinner("Syncing changes (embedding new/changed files only)..."):
            sync_stats = sync_files_to_collection(collection_name, files, full_sync=full_sync)

        if sync_stats["chunks_embedded"] == 0 and sync_stats["deleted_files"] == 0 and sync_stats["skipped_files"] > 0:
            st.info(
                f"`{collection_name}` is already up to date — "
                f"{sync_stats['skipped_files']} file(s) unchanged, nothing re-embedded."
            )
        else:
            message = (
                f"Synced `{collection_name}` — "
                f"{sync_stats['new_files']} new, {sync_stats['updated_files']} updated, "
                f"{sync_stats['skipped_files']} unchanged"
            )
            if full_sync:
                message += f", {sync_stats['deleted_files']} removed"
            message += f" ({sync_stats['chunks_embedded']} chunks embedded)"
            st.success(message)

        with st.expander("Index details", expanded=False):
            st.markdown(f"**Total size scanned:** {summary['total_size_kb']} KB")
            st.markdown(f"**Time:** {sync_stats['elapsed_ms']} ms")
            if sync_stats.get("chunk_methods"):
                method_parts = [f"{count} via {method}" for method, count in sync_stats["chunk_methods"].items()]
                st.markdown(f"**Chunking:** {', '.join(method_parts)}")
            if sync_stats["new_file_paths"]:
                st.markdown("**New files:**")
                for p in sync_stats["new_file_paths"]:
                    st.markdown(f"- {p}")
            if sync_stats["updated_file_paths"]:
                st.markdown("**Updated files:**")
                for p in sync_stats["updated_file_paths"]:
                    st.markdown(f"- {p}")
            if sync_stats.get("deleted_file_paths"):
                st.markdown("**Removed (no longer in the project):**")
                for p in sync_stats["deleted_file_paths"]:
                    st.markdown(f"- {p}")

        st.session_state.selected_collection = collection_name
        st.session_state.messages = []
        st.rerun()

    st.divider()


def render_filter_options(collection_name: str) -> dict:
    if not collection_name:
        return {"language": None, "source": None, "symbol": None}

    st.markdown("### 🔍 Filter (optional)")

    sources = get_collection_sources(collection_name)
    source_options = ["All files"] + sources
    selected_source = st.selectbox("Scope to file", source_options, key="filter_source")

    symbols = get_collection_symbols(collection_name)
    symbol_options = ["All symbols"] + symbols
    selected_symbol = st.selectbox(
        "Scope to symbol",
        symbol_options,
        key="filter_symbol",
        help="Function, class, or method names extracted from the codebase — "
             "narrows retrieval to just that definition.",
    )

    filters = {
        "language": None,
        "source": None if selected_source == "All files" else selected_source,
        "symbol": None if selected_symbol == "All symbols" else selected_symbol,
    }

    st.divider()
    return filters


def render_sidebar() -> dict:
    with st.sidebar:
        st.title("🧠 Codebase Q&A")
        st.caption("Powered by Qwen + ChromaDB")
        st.divider()

        render_ollama_status()
        active_collection = render_project_selector()
        render_upload_section()

        filters = render_filter_options(active_collection)

        return {
            "collection": active_collection,
            "filters": filters,
        }