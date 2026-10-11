"""Scoring primitives.

Deliberately dependency-light (numpy only) and pure, so the scoring logic can be
tested without a database, a model, or a network — see tests/test_metrics.py.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field

import numpy as np

ERROR = "error"


@dataclass
class QuestionResult:
    """One (system, question) observation."""

    system: str
    set_name: str
    doc_id: str
    question: str
    reference: str
    response: str | None = None
    correct: bool | None = None
    evidence_pages: list[int] = field(default_factory=list)
    retrieved_pages: list[int] = field(default_factory=list)
    latency_s: float | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    context_chars: int = 0
    error: str | None = None

    @property
    def scored(self) -> bool:
        """True when the question produced a judgeable answer."""
        return self.correct is not None and self.error is None

    def to_dict(self) -> dict:
        return asdict(self)


def parse_pages(value: object) -> list[int]:
    """Parse an evidence-page field.

    Upstream stores this inconsistently: as a string like ``"[19]"``,
    ``"[19, 20]"``, or occasionally as a real list. Anything unparseable yields
    an empty list rather than raising, because a missing page hint should
    degrade retrieval scoring, not abort a run.
    """
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        out = []
        for v in value:
            out.extend(parse_pages(v))
        return sorted(set(out))
    if isinstance(value, (int, float)):
        return [int(value)]
    return sorted({int(m) for m in re.findall(r"\d+", str(value))})


def _flags(results: Iterable[QuestionResult]) -> list[bool]:
    return [bool(r.correct) for r in results if r.scored]


def accuracy(results: Sequence[QuestionResult]) -> float | None:
    """Fraction of scored questions answered correctly, or None if none scored."""
    flags = _flags(results)
    return float(sum(flags) / len(flags)) if flags else None


def bootstrap_ci(
    results: Sequence[QuestionResult],
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float] | None:
    """Percentile bootstrap CI for accuracy.

    Reported alongside every accuracy figure. With 62 questions a point estimate
    is nearly meaningless on its own: the interval is roughly +/- 9 points, which
    is wider than most differences people claim from benchmarks this size.
    """
    flags = _flags(results)
    if len(flags) < 2:
        return None
    rng = np.random.default_rng(seed)
    arr = np.asarray(flags, dtype=float)
    idx = rng.integers(0, len(arr), size=(n_boot, len(arr)))
    means = arr[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def latency_summary(results: Sequence[QuestionResult]) -> dict:
    """p50/p95 latency in seconds over successfully timed questions."""
    vals = [r.latency_s for r in results if r.latency_s is not None and r.error is None]
    if not vals:
        return {"n": 0, "p50_s": None, "p95_s": None, "mean_s": None}
    arr = np.asarray(vals, dtype=float)
    return {
        "n": len(vals),
        "p50_s": float(np.percentile(arr, 50)),
        "p95_s": float(np.percentile(arr, 95)),
        "mean_s": float(arr.mean()),
    }


def retrieval_hit_rate(results: Sequence[QuestionResult]) -> float | None:
    """Fraction of questions where at least one evidence page was retrieved.

    This is the metric that isolates retrieval from generation: the chat model
    and the judge are identical across systems, so a gap here is a gap in the
    retrieval strategy. It is the primary claim this harness tests.
    """
    applicable = [r for r in results if r.evidence_pages and r.error is None]
    if not applicable:
        return None
    hits = 0
    for r in applicable:
        if set(r.evidence_pages) & set(r.retrieved_pages):
            hits += 1
    return float(hits / len(applicable))


def summarize(results: Sequence[QuestionResult]) -> dict:
    """Aggregate one run into the figures that go in the report."""
    errors = [r for r in results if r.error is not None]
    scored = [r for r in results if r.scored]
    costs = [r.cost_usd for r in results if r.error is None]
    return {
        "n_total": len(results),
        "n_scored": len(scored),
        "n_errors": len(errors),
        "accuracy": accuracy(results),
        "accuracy_ci95": bootstrap_ci(results),
        "retrieval_hit_rate": retrieval_hit_rate(results),
        "latency": latency_summary(results),
        "cost_usd": {
            "total": float(sum(costs)),
            "mean_per_question": float(np.mean(costs)) if costs else None,
        },
        "tokens": {
            "prompt": sum(r.prompt_tokens for r in results),
            "completion": sum(r.completion_tokens for r in results),
        },
        "context_chars_mean": (
            float(np.mean([r.context_chars for r in results])) if results else None
        ),
    }
