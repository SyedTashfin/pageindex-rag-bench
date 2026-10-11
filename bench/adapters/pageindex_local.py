"""Adapter for PageIndex in local mode: vectorless, reasoning-based retrieval.

PageIndex builds a hierarchical tree index per document (Flash indexing by
default: layout-based extraction, a deterministic merge, then an LLM expansion
pass) and answers by reasoning over that tree. There is no vector index and no
chunking step.

Model names are LiteLLM's, so a local Ollama model works by prefix:
``ollama/qwen2.5:7b-instruct``. That keeps a full run at zero API cost.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Sequence

from .base import Answer, pages_from_citations

DEFAULT_CHAT_MODEL = "ollama/qwen2.5:7b-instruct"
DEFAULT_INDEX_MODEL = "ollama/qwen2.5:7b-instruct"


class PageIndexLocal:
    """Local-mode PageIndex, indexed once into a persistent store."""

    name = "pageindex_local"

    def __init__(
        self,
        chat_model: str | None = None,
        index_model: str | None = None,
        storage_path: str | None = None,
        mode: str | None = None,
        citations: bool = True,
    ) -> None:
        self.chat_model = chat_model or os.environ.get("PAGEINDEX_CHAT_MODEL", DEFAULT_CHAT_MODEL)
        self.index_model = index_model or os.environ.get(
            "PAGEINDEX_INDEX_MODEL", DEFAULT_INDEX_MODEL
        )
        self.storage_path = storage_path or os.environ.get(
            "PAGEINDEX_STORAGE", os.path.join("corpus", "pageindex_store")
        )
        # "flash" is the local default and the faster path; "standard" builds a
        # full LLM tree. Kept as a knob so the harness can compare them.
        self.mode = mode
        self.citations = citations
        self._client = None

    # -- client -----------------------------------------------------------

    @property
    def client(self):
        """Constructed lazily so importing this module needs no API key."""
        if self._client is None:
            from pageindex import PageIndexClient

            self._client = PageIndexClient(
                mode="local",
                chat_model=self.chat_model,
                index_model=self.index_model,
                storage_path=self.storage_path,
            )
        return self._client

    # -- ingestion --------------------------------------------------------

    def prepare(self, doc_ids: Sequence[str], documents_dir: str) -> None:
        """Index each document, caching doc ids so later runs skip the work.

        Local mode indexes synchronously inside ``submit_document``, which is
        the expensive step: a full 34-document run is measured in hours on a
        local model, so the cache is not an optimisation, it is the difference
        between a resumable benchmark and a restartable one.
        """
        os.makedirs(self.storage_path, exist_ok=True)
        cache_path = os.path.join(self.storage_path, "doc_ids.json")
        cache: dict[str, str] = {}
        if os.path.exists(cache_path):
            with open(cache_path) as fh:
                cache = json.load(fh)

        pending = [d for d in doc_ids if d not in cache]
        if pending:
            print(f"indexing {len(pending)} document(s) with {self.index_model} ...")

        for i, doc_id in enumerate(pending, 1):
            path = os.path.join(documents_dir, doc_id)
            if not os.path.exists(path):
                print(f"  [{i}/{len(pending)}] MISSING {doc_id}")
                continue
            started = time.time()
            result = self.client.submit_document(path, mode=self.mode)
            # submit_document returns the internal doc id; key the cache by the
            # corpus filename so a manifest change does not silently mis-map.
            cache[doc_id] = result["doc_id"]
            with open(cache_path, "w") as fh:
                json.dump(cache, fh, indent=2)
            elapsed = time.time() - started
            print(f"  [{i}/{len(pending)}] {doc_id} -> {cache[doc_id]} ({elapsed:.1f}s)")

        self._doc_ids = cache

    # -- answering --------------------------------------------------------

    def answer(self, doc_id: str, question: str) -> Answer:
        internal = getattr(self, "_doc_ids", {}).get(doc_id)
        if not internal:
            raise RuntimeError(f"{doc_id} has not been indexed; call prepare() first")

        started = time.time()
        result = self.client.chat(question, doc_id=internal, citations=self.citations)
        elapsed = time.time() - started

        if isinstance(result, str):
            text, raw = result, {}
        elif isinstance(result, dict):
            text = result.get("answer") or result.get("content") or ""
            raw = result
        else:  # a stream, if citations/stream settings ever change
            text, raw = str(result), {}

        pages: list[int] = []
        if self.citations and text:
            try:
                pages = pages_from_citations(self.client.get_citations(text, internal))
            except Exception as exc:  # noqa: BLE001 - provenance is best-effort
                raw = {**raw, "citation_error": str(exc)}

        return Answer(
            response=text,
            retrieved_pages=pages,
            latency_s=elapsed,
            context_chars=len(json.dumps(raw)),
            raw=raw,
        )
