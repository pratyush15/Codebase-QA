# Single image, two entrypoints: the Streamlit UI (main.py) and the FastAPI
# layer (api.py) both import from core/ the same way, so they share one
# image and differ only in the command docker-compose runs (see
# docker-compose.yml). No Ollama here — it runs as its own service/container;
# this image only ever talks to it over HTTP via OLLAMA_BASE_URL.

FROM python:3.13-slim

# build-essential: some deps (e.g. chromadb's hnswlib, tree-sitter grammars)
# fall back to building from source when no prebuilt wheel matches the
# platform. curl: used by the Docker HEALTHCHECK below.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Installed before the rest of the source so this layer is only rebuilt
# when dependencies actually change, not on every code edit.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Non-root: matches the /app/data bind mount used for the Chroma DB and
# query metrics, which docker-compose owns as the host user.
RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p data \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8501 8000

# Default to the Streamlit UI; docker-compose overrides `command:` for the
# api service to run uvicorn against the same image instead.
CMD ["streamlit", "run", "app/main.py", "--server.port=8501", "--server.address=0.0.0.0"]