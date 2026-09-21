
"""
Shared pytest fixtures/setup.

config.py fails fast if OLLAMA_BASE_URL / OLLAMA_MODEL / OLLAMA_EMBEDDING_MODEL /
CHROMA_DB_PATH aren't set (see config._require). That's the right behavior for
the running app, but it means tests would otherwise depend on a developer's
local .env file existing — which isn't guaranteed (it's gitignored) and
shouldn't be required just to run unit tests against pure functions.

We set harmless dummy values here, before any test module imports config
(directly or transitively), so the test suite is hermetic: it runs the same
on a fresh checkout with no .env and no Ollama/Chroma actually running. None
of the tests in this suite make real network calls to Ollama or persist to
Chroma — they exercise pure functions (chunking, hashing, filtering, file
handling) that don't need those values to be *real*, just *present*.
"""
import os

os.environ.setdefault("OLLAMA_BASE_URL", "http://localhost:11434")
os.environ.setdefault("OLLAMA_MODEL", "qwen2.5:3b")
os.environ.setdefault("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
os.environ.setdefault("CHROMA_DB_PATH", "./data/chroma_db")

# core.qa_chain.run_qa_streaming writes a query-telemetry record (see
# core/query_log.py) after every completed query by default. Most
# qa_chain/api tests don't mock that call — they're testing everything
# *around* it — so without this, every test run would silently append
# real records to the project's actual data/query_metrics.jsonl on disk.
# Off by default here, the same way the dummy Ollama values above keep
# the suite hermetic; tests/test_query_log.py exercises the logging
# functions directly against tmp_path, and any test that specifically
# wants to verify the qa_chain -> query_log wiring re-enables this
# locally via monkeypatch.
os.environ.setdefault("ENABLE_QUERY_METRICS_LOG", "false")