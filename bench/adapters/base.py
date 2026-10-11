"""The contract every system in this benchmark implements."""

from __future__ import annotations

import os
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class Answer:
    """One system's response to one question, with its retrieval provenance."""

    response: str
    retrieved_pages: list[int] = field(default_factory=list)
    latency_s: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    context_chars: int = 0
    raw: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class System(Protocol):
    """A retriever-plus-generator under test."""

    name: str

    def prepare(self, doc_ids: Sequence[str], documents_dir: str) -> None:
        """Ingest documents. Must be idempotent: a run may be resumed."""
        ...

    def answer(self, doc_id: str, question: str) -> Answer:
        """Answer one question about one document."""
        ...


_PAGE_KEYS = ("page", "pages", "page_number", "page_num", "start_page")


def think_extra_body(think: bool) -> dict:
    """Provider options that disable a reasoning model's hidden reasoning.

    `qwen3` and similar hybrid models emit reasoning by default, and it is not
    free: measured on this corpus, one question cost 155 completion tokens with
    thinking on against 3 with it off, and it dominates the latency. A small
    output-token cap can also be spent entirely on reasoning, producing an
    **empty response** with no error — the failure mode most likely to corrupt a
    result silently.

    It defaults to **on**, because the upstream benchmark shows accuracy rising
    with reasoning effort, so turning it off trades real accuracy for speed. This
    returns the provider option rather than setting a policy, so the caller
    decides. See docs/FINDINGS.md.
    """
    return {} if think else {"think": False}


def pages_from_citations(citations: Sequence[Any]) -> list[int]:
    """Extract page numbers from a PageIndex citation list.

    PageIndex writes page provenance into citation tags of the form
    ``<cite doc= page= block=/>`` and resolves them through ``get_citations()``.
    The resolved shape is a list of dicts that carry the page either directly or
    nested under the cited ``block``. Both shapes are read here, and anything
    unrecognised is skipped rather than raising — a citation format change should
    degrade retrieval scoring, not abort a benchmark run.
    """
    found: set[int] = set()
    for cite in citations or []:
        if not isinstance(cite, dict):
            continue
        candidates = [cite]
        block = cite.get("block")
        if isinstance(block, dict):
            candidates.append(block)
        for candidate in candidates:
            for key in _PAGE_KEYS:
                if key not in candidate:
                    continue
                value = candidate[key]
                if isinstance(value, int):
                    found.add(value)
                elif isinstance(value, str):
                    found.update(int(m) for m in re.findall(r"\d+", value))
                elif isinstance(value, (list, tuple)):
                    found.update(parse_page_values(value))
    return sorted(found)


def env_flag(name: str, default: bool) -> bool:
    """Read a boolean from the environment, accepting the usual spellings.

    Kept explicit rather than using ``bool(os.environ.get(...))``, which reads
    "0" and "false" as True — a mistake that silently flips a measured setting.
    """
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off", "")


def parse_page_values(values: Sequence[Any]) -> set[int]:
    out: set[int] = set()
    for v in values:
        if isinstance(v, int):
            out.add(v)
        elif isinstance(v, str):
            out.update(int(m) for m in re.findall(r"\d+", v))
    return out
