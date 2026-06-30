import streamlit as st

st.set_page_config(
    page_title="Codebase Q&A",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

from components.sidebar import render_sidebar
from components.chat import render_chat
from components.file_tree import render_file_tree


def apply_styles():
    st.markdown("""
    <style>
        /* Tighten chat bubbles */
        .stChatMessage { padding: 0.5rem 0.75rem; }

        /* File tree monospace */
        code { font-size: 0.82em; }

        /* Subtle divider */
        hr { margin: 0.5rem 0; border-color: #333; }

        /* Source expander smaller font */
        .streamlit-expanderContent p {
            font-size: 0.85em;
            margin: 0.15rem 0;
        }

        /* Hide streamlit branding */
        #MainMenu { visibility: hidden; }
        footer { visibility: hidden; }
    </style>
    """, unsafe_allow_html=True)


def main():
    apply_styles()

    # Sidebar returns active collection + filters
    sidebar_state = render_sidebar()
    collection = sidebar_state["collection"]
    filters = sidebar_state["filters"]

    # Main layout: chat on left, file tree on right
    chat_col, tree_col = st.columns([3, 1])

    with chat_col:
        render_chat(collection_name=collection, filters=filters)

    with tree_col:
        render_file_tree(collection_name=collection)


if __name__ == "__main__":
    main()