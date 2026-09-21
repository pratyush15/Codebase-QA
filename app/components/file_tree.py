import streamlit as st
from typing import Dict, List
from core.vectorstore import get_collection_sources
from utils.language_detect import detect_language


LANGUAGE_ICONS = {
    "Python": "🐍",
    "JavaScript": "🟨",
    "TypeScript": "🔷",
    "JavaScript (React)": "⚛️",
    "TypeScript (React)": "⚛️",
    "Java": "☕",
    "Go": "🐹",
    "Rust": "🦀",
    "C++": "⚙️",
    "C": "⚙️",
    "C#": "💜",
    "Ruby": "💎",
    "PHP": "🐘",
    "SQL": "🗄️",
    "HTML": "🌐",
    "CSS": "🎨",
    "SCSS": "🎨",
    "Shell": "🖥️",
    "Bash": "🖥️",
    "JSON": "📋",
    "YAML": "📋",
    "TOML": "📋",
    "Markdown": "📝",
    "Plain Text": "📄",
}


def get_icon(file_path: str) -> str:
    lang = detect_language(file_path)
    return LANGUAGE_ICONS.get(lang, "📄")


def build_tree(paths: List[str]) -> Dict:
    tree = {}
    for path in paths:
        parts = path.replace("\\", "/").split("/")
        node = tree
        for part in parts[:-1]:
            node = node.setdefault(f"📂 {part}", {})
        node[parts[-1]] = path
    return tree


def render_tree_node(node: Dict, depth: int = 0):
    for key, value in sorted(node.items()):
        if isinstance(value, dict):
            # It's a folder
            with st.expander(key, expanded=(depth == 0)):
                render_tree_node(value, depth + 1)
        else:
            # It's a file (value = full path)
            icon = get_icon(value)
            lang = detect_language(value)
            st.markdown(
                f"{'&nbsp;' * (depth * 2)}{icon} `{key}` <span style='color:gray;font-size:0.75em'>{lang}</span>",
                unsafe_allow_html=True,
            )


def render_file_tree(collection_name: str):
    if not collection_name:
        return

    st.markdown("### 🗂️ Indexed Files")

    sources = get_collection_sources(collection_name)

    if not sources:
        st.info("No files indexed yet.")
        return

    tree = build_tree(sources)
    render_tree_node(tree, depth=0)

    st.caption(f"{len(sources)} file(s) indexed")