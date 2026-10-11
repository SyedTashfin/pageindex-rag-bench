"""Model and connection defaults, in one place.

These used to be duplicated: `app/server.py` defaulted to `qwen3:8b` while the
adapters defaulted to `qwen2.5:7b-instruct`, so the demo worked and the command
line tools failed with "model not found" on the same machine. One definition
removes that class of inconsistency.

Override any of them by environment variable, which is also how you point the
harness at a stronger model on another machine:

    OLLAMA_API_BASE=http://192.168.1.24:11434 \\
    CHAT_MODEL=ollama/qwen3:32b \\
    python -m app.ask "your question"
"""

from __future__ import annotations

import os

# Answers the questions. The single biggest quality lever in the whole harness.
CHAT_MODEL = os.environ.get("CHAT_MODEL", "ollama/qwen3:8b")

# Builds PageIndex's tree. Speed matters more than quality here, so this is the
# smaller model by default.
INDEX_MODEL = os.environ.get("INDEX_MODEL", "ollama/qwen3:1.7b")

# Embeddings for the vector baseline. Dimension is read from the response.
EMBED_MODEL = os.environ.get("EMBED_MODEL", "ollama/nomic-embed-text")

# Postgres with pgvector. Matches docker-compose.yml.
PGVECTOR_DSN = os.environ.get(
    "PGVECTOR_DSN", "postgresql://postgres:postgres@localhost:5433/ragbench"
)


def available_models(base: str = "http://127.0.0.1:11434") -> list[str]:
    """Ask the Ollama daemon what it actually has.

    Used to turn "model not found" into a list of the models that would work,
    because the bare litellm error does not say what is installed.
    """
    import json
    import urllib.error
    import urllib.request

    url = os.environ.get("OLLAMA_API_BASE", base).rstrip("/") + "/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            payload = json.load(response)
        return sorted(m["name"] for m in payload.get("models", []))
    except (urllib.error.URLError, OSError, ValueError, KeyError):
        return []
