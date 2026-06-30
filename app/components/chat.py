import streamlit as st
from typing import List, Dict
from core.qa_chain import run_qa_streaming


def init_chat_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "current_collection" not in st.session_state:
        st.session_state.current_collection = None


def render_sources(sources: List[Dict]):
    if not sources:
        return
    with st.expander(f"📎 Sources ({len(sources)} files)", expanded=False):
        for s in sources:
            role = f" — *{s['role']}*" if s.get("role") else ""
            lang = s.get("language", "")
            st.markdown(
                f"📄 `{s['source']}` &nbsp; <span style='color:gray;font-size:0.8em'>{lang}{role}</span>",
                unsafe_allow_html=True,
            )


def render_chat_history():
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                render_sources(msg["sources"])


def render_chat(collection_name: str, filters: dict):
    init_chat_state()

    if not collection_name:
        st.info("👈 Select or upload a project from the sidebar to start querying.")
        return

    if st.session_state.current_collection != collection_name:
        st.session_state.messages = []
        st.session_state.current_collection = collection_name

    col1, col2 = st.columns([5, 1])
    with col1:
        st.markdown(f"#### 💬 `{collection_name}`")
    with col2:
        if st.session_state.messages:
            if st.button("🗑️ Clear", key="clear_chat"):
                st.session_state.messages = []
                st.rerun()

    render_chat_history()

    question = st.chat_input("Ask anything about your codebase...")

    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            placeholder = st.empty()
            full_response = ""
            sources = []

            stream = run_qa_streaming(
                collection_name=collection_name,
                question=question,
                filter_language=filters.get("language"),
                filter_source=filters.get("source"),
            )

            for chunk in stream:
                if chunk["type"] == "token":
                    full_response += chunk["value"]
                    placeholder.markdown(full_response + "▌")
                elif chunk["type"] == "sources":
                    sources = chunk["value"]
                elif chunk["type"] == "error":
                    full_response = chunk["message"]
                    placeholder.warning(full_response)
                    break

            placeholder.markdown(full_response)
            render_sources(sources)

        st.session_state.messages.append({
            "role": "assistant",
            "content": full_response,
            "sources": sources,
        })