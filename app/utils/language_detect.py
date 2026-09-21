from pathlib import Path
from typing import Optional


EXTENSION_LANGUAGE_MAP = {
    ".py": "Python",
    ".js": "JavaScript",
    ".ts": "TypeScript",
    ".jsx": "JavaScript (React)",
    ".tsx": "TypeScript (React)",
    ".java": "Java",
    ".cpp": "C++",
    ".cc": "C++",
    ".c": "C",
    ".h": "C/C++ Header",
    ".cs": "C#",
    ".go": "Go",
    ".rs": "Rust",
    ".rb": "Ruby",
    ".php": "PHP",
    ".swift": "Swift",
    ".kt": "Kotlin",
    ".scala": "Scala",
    ".html": "HTML",
    ".css": "CSS",
    ".scss": "SCSS",
    ".json": "JSON",
    ".yaml": "YAML",
    ".yml": "YAML",
    ".toml": "TOML",
    ".md": "Markdown",
    ".txt": "Plain Text",
    ".sh": "Shell",
    ".bash": "Bash",
    ".sql": "SQL",
    ".r": "R",
    ".m": "MATLAB/Objective-C",
    ".lua": "Lua",
}

# Map language -> markdown fence identifier
LANGUAGE_FENCE_MAP = {
    "Python": "python",
    "JavaScript": "javascript",
    "TypeScript": "typescript",
    "JavaScript (React)": "jsx",
    "TypeScript (React)": "tsx",
    "Java": "java",
    "C++": "cpp",
    "C": "c",
    "C/C++ Header": "c",
    "C#": "csharp",
    "Go": "go",
    "Rust": "rust",
    "Ruby": "ruby",
    "PHP": "php",
    "Swift": "swift",
    "Kotlin": "kotlin",
    "Scala": "scala",
    "HTML": "html",
    "CSS": "css",
    "SCSS": "scss",
    "JSON": "json",
    "YAML": "yaml",
    "TOML": "toml",
    "Shell": "bash",
    "Bash": "bash",
    "SQL": "sql",
    "R": "r",
    "Lua": "lua",
    "Markdown": "markdown",
    "Plain Text": "",
}


def detect_language(file_path: str) -> str:
    ext = Path(file_path).suffix.lower()
    return EXTENSION_LANGUAGE_MAP.get(ext, "Unknown")


def get_fence_identifier(language: str) -> str:
    return LANGUAGE_FENCE_MAP.get(language, "")


def get_language_context(file_path: str) -> dict:
    """
    Returns language metadata used to enrich chunk metadata and prompts.
    """
    language = detect_language(file_path)
    fence = get_fence_identifier(language)

    is_config = Path(file_path).suffix.lower() in {".json", ".yaml", ".yml", ".toml", ".env"}
    is_markup = Path(file_path).suffix.lower() in {".html", ".md", ".txt"}
    is_code = not is_config and not is_markup and language != "Unknown"

    return {
        "language": language,
        "fence": fence,
        "is_code": is_code,
        "is_config": is_config,
        "is_markup": is_markup,
    }


def group_files_by_language(files: list) -> dict:
    """
    Groups a list of file dicts by detected language.
    Used in sidebar for a language breakdown display.
    """
    groups = {}
    for f in files:
        lang = detect_language(f["path"])
        groups.setdefault(lang, []).append(f["path"])
    return dict(sorted(groups.items(), key=lambda x: -len(x[1])))


def get_file_role_hint(file_path: str) -> Optional[str]:
    """
    Returns a plain-English role hint for a file based on its name/path.
    Injected into chunk metadata to give the LLM extra context.
    """
    name = Path(file_path).name.lower()
    path_lower = file_path.lower()

    hints = {
        "main.py": "application entry point",
        "app.py": "application entry point",
        "index.js": "JavaScript entry point",
        "index.ts": "TypeScript entry point",
        "config.py": "configuration file",
        "settings.py": "configuration/settings file",
        "models.py": "data models",
        "schema.py": "data schema definitions",
        "routes.py": "API route definitions",
        "views.py": "view/controller logic",
        "utils.py": "utility/helper functions",
        "helpers.py": "helper functions",
        "requirements.txt": "Python dependencies",
        "package.json": "Node.js dependencies and scripts",
        "dockerfile": "Docker container definition",
        "docker-compose.yml": "Docker Compose configuration",
        "readme.md": "project documentation",
    }

    if name in hints:
        return hints[name]

    if "test" in path_lower or "spec" in path_lower:
        return "test file"
    if "migration" in path_lower:
        return "database migration"
    if "middleware" in path_lower:
        return "middleware"
    if "api" in path_lower:
        return "API layer"

    return None