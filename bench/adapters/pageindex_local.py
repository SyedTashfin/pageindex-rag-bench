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
from typing import Any

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

    @staticmethod
    def staged_name(index: int) -> str:
        """A short, stable filename for a corpus document.

        PageIndex's local agent identifies documents to its *tools* by internal
        ``pi-`` id, but its targeting block tells the model the document *name* —
        so the model has to bridge the two by reproducing the name inside a tool
        call. Two things break that with a small local model, both measured:

        1. A 36-character hex corpus filename ("a5879805d70c854ea4361e43a84e3bb
           2.pdf") is not reproduced reliably. Ingested under that name, every
           attempt with qwen3:8b returned ``Document not found``.
        2. **Zero-padding is worse than useless.** With ``doc-03.pdf`` the model
           asked for ``doc-3.pdf`` — it kept the word, the digit and the
           extension, and dropped only the leading zero. Rigid identifiers that
           differ by a character the model considers cosmetic are the worst case.

        So names are short, unpadded and punctuation-free: ``doc1.pdf`` …
        ``doc34.pdf``. Nothing here is cosmetic, so there is nothing to drop. The
        mapping is written to the store next to the doc ids, so a run stays
        reproducible. Evidence: docs/FINDINGS.md.
        """
        return f"doc{index}.pdf"

    def prepare(self, doc_ids: Sequence[str], documents_dir: str) -> None:
        """Index each document, caching doc ids so later runs skip the work.

        Local mode indexes synchronously inside ``submit_document``, which is
        the expensive step: a full 34-document run is measured in hours on a
        local model, so the cache is not an optimisation, it is the difference
        between a resumable benchmark and a restartable one.
        """
        import shutil

        os.makedirs(self.storage_path, exist_ok=True)
        staging = os.path.join(self.storage_path, "staging")
        os.makedirs(staging, exist_ok=True)

        cache_path = os.path.join(self.storage_path, "doc_ids.json")
        cache: dict[str, str] = {}
        if os.path.exists(cache_path):
            with open(cache_path) as fh:
                cache = json.load(fh)

        names = {doc_id: self.staged_name(i) for i, doc_id in enumerate(sorted(doc_ids), 1)}
        with open(os.path.join(self.storage_path, "staged_names.json"), "w") as fh:
            json.dump(names, fh, indent=2, sort_keys=True)

        pending = [d for d in doc_ids if d not in cache]
        if pending:
            print(f"indexing {len(pending)} document(s) with {self.index_model} ...")

        for i, doc_id in enumerate(pending, 1):
            source = os.path.join(documents_dir, doc_id)
            if not os.path.exists(source):
                print(f"  [{i}/{len(pending)}] MISSING {doc_id}")
                continue
            staged = os.path.join(staging, names[doc_id])
            if not os.path.exists(staged):
                shutil.copyfile(source, staged)
            started = time.time()
            result = self.client.submit_document(staged, mode=self.mode)
            # submit_document returns the internal doc id; key the cache by the
            # corpus filename so a manifest change does not silently mis-map.
            cache[doc_id] = result["doc_id"]
            with open(cache_path, "w") as fh:
                json.dump(cache, fh, indent=2)
            print(
                f"  [{i}/{len(pending)}] {doc_id} -> {cache[doc_id]} "
                f"as {names[doc_id]} ({time.time()-started:.1f}s)"
            )

        self._doc_ids = cache

    # -- answering --------------------------------------------------------

    @staticmethod
    def parse_chat(result: Any) -> tuple[str, dict, list[int]]:
        """Read PageIndex's chat envelope.

        Three shapes occur in practice, all serialised as a *string* rather than
        a dict:

            answer    {"status": "success", "content": "February 8, 2021"}
            failure   {"status": "error", "error": "...", "errorCode": "NOT_FOUND"}
            tool      {"page": 14, "text": "<page 14 contents>"}

        The third is the subtle one. Local mode runs an agent loop, and the
        envelope it returns can be the raw result of the last tool call rather
        than a synthesised answer — so the caller receives a page of the document
        where it expected a sentence. Treating that text as the answer would
        score a retrieval artifact as a response, which is both wrong and
        invisible. The page number it carries *is* genuine retrieval provenance,
        so it is returned as such and the shape is flagged in the payload.

        Failures raise, so they land in the run's error column instead of being
        recorded as a successful response.
        """
        payload: Any = result
        if isinstance(result, str):
            try:
                payload = json.loads(result)
            except json.JSONDecodeError:
                return result.strip(), {}, []  # already plain text

        if isinstance(payload, dict):
            code = payload.get("errorCode") or payload.get("error_code")
            if code or payload.get("error"):
                raise RuntimeError(
                    f"PageIndex {code or 'error'}: {payload.get('error') or payload.get('detail')}"
                )
            if "page" in payload and "text" in payload:
                try:
                    pages = [int(payload["page"])]
                except (TypeError, ValueError):
                    pages = []
                return str(payload["text"]).strip(), {**payload, "_shape": "page_content"}, pages

            text = payload.get("content") or payload.get("answer") or payload.get("text") or ""
            return str(text).strip(), payload, []

        return str(result).strip(), {}, []

    def answer(self, doc_id: str, question: str) -> Answer:
        internal = getattr(self, "_doc_ids", {}).get(doc_id)
        if not internal:
            raise RuntimeError(
                f"{doc_id} has not been indexed; call prepare() first "
                f"(known documents: {sorted(getattr(self, '_doc_ids', {}))[:3]})"
            )

        started = time.time()
        result = self.client.chat(question, doc_id=internal, citations=self.citations)
        elapsed = time.time() - started
        text, raw, pages = self.parse_chat(result)

        if self.citations and text:
            try:
                cited = pages_from_citations(self.client.get_citations(text, internal))
                pages = sorted(set(pages) | set(cited))
            except Exception as exc:  # noqa: BLE001 - provenance is best-effort
                raw = {**raw, "citation_error": str(exc)}

        return Answer(
            response=text,
            retrieved_pages=pages,
            latency_s=elapsed,
            context_chars=len(json.dumps(raw)),
            raw=raw,
        )
