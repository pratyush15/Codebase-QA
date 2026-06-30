import streamlit as st
from core.qa_chain import check_ollama_connection
from core.vectorstore import (
    list_collections,
    delete_collection,
    sanitize_collection_name,
    get_collection_sources,
    add_documents_to_collection,
)
from core.ingestion import ingest_files, get_ingestion_stats
from utils.file_handler import (
    extract_files_from_zip,
    extract_files_from_uploads,
    get_summary,
)


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
            if upload_mode == "Zip folder":
                files = extract_files_from_zip(uploaded.read())
            else:
                files = extract_files_from_uploads(uploaded)

        if not files:
            st.error("No readable code files found.")
            return

        summary = get_summary(files)

        with st.spinner("Chunking and embedding..."):
            documents = ingest_files(files)
            add_documents_to_collection(collection_name, documents)
            stats = get_ingestion_stats(documents)

        st.success(f"Indexed `{collection_name}` — {stats['total_chunks']} chunks from {stats['unique_files']} files")

        with st.expander("Index details", expanded=False):
            st.markdown(f"**Total size:** {summary['total_size_kb']} KB")
            st.markdown("**Languages found:**")
            for lang, count in stats["languages"].items():
                st.markdown(f"- {lang}: {count} chunks")

        st.session_state.selected_collection = collection_name
        st.session_state.messages = []
        st.rerun()

    st.divider()


def render_filter_options(collection_name: str) -> dict:
    if not collection_name:
        return {"language": None, "source": None}

    st.markdown("### 🔍 Filter (optional)")

    sources = get_collection_sources(collection_name)
    source_options = ["All files"] + sources
    selected_source = st.selectbox("Scope to file", source_options, key="filter_source")

    filters = {
        "language": None,
        "source": None if selected_source == "All files" else selected_source,
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