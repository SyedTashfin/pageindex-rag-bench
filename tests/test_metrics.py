"""Scoring logic. No database, no model, no network — these must always run in CI."""

from __future__ import annotations

import pytest

from bench import metrics
from bench.metrics import QuestionResult


def make(correct=None, evidence=None, retrieved=None, latency=None, error=None, cost=0.0):
    return QuestionResult(
        system="s",
        set_name="t",
        doc_id="d.pdf",
        question="q",
        reference="r",
        correct=correct,
        evidence_pages=evidence or [],
        retrieved_pages=retrieved or [],
        latency_s=latency,
        cost_usd=cost,
        error=error,
    )


class TestParsePages:
    @pytest.mark.parametrize(
        "value,expected",
        [
            ("[19]", [19]),
            ("[19, 20]", [19, 20]),
            ("19", [19]),
            ([1, 2], [1, 2]),
            ("['Table']", []),
            (None, []),
            ("", []),
            ("[3, 3, 1]", [1, 3]),
            (5, [5]),
        ],
    )
    def test_parses_upstream_shapes(self, value, expected):
        assert metrics.parse_pages(value) == expected


class TestAccuracy:
    def test_counts_only_scored_questions(self):
        results = [make(True), make(False), make(None), make(True, error="boom")]
        # Only the first two are scored: the third has no verdict and the fourth
        # errored, so its `correct=True` must not be counted as a success.
        assert metrics.accuracy(results) == pytest.approx(0.5)

    def test_an_error_never_counts_as_correct(self):
        results = [make(True, error="boom"), make(False)]
        assert metrics.accuracy(results) == 0.0

    def test_none_when_nothing_scored(self):
        assert metrics.accuracy([make(None), make(True, error="x")]) is None


class TestBootstrapCI:
    def test_brackets_the_point_estimate(self):
        results = [make(i % 2 == 0) for i in range(62)]
        lo, hi = metrics.bootstrap_ci(results)
        assert lo <= metrics.accuracy(results) <= hi

    def test_is_deterministic_for_a_fixed_seed(self):
        results = [make(i % 3 == 0) for i in range(40)]
        assert metrics.bootstrap_ci(results, seed=7) == metrics.bootstrap_ci(results, seed=7)

    def test_is_wide_at_this_sample_size(self):
        # A 62-question set carries roughly +/- 9 points. If this ever narrows
        # dramatically, the interval is broken, not the benchmark.
        results = [make(i % 2 == 0) for i in range(62)]
        lo, hi = metrics.bootstrap_ci(results)
        assert (hi - lo) > 0.15

    def test_none_below_two_samples(self):
        assert metrics.bootstrap_ci([make(True)]) is None


class TestLatency:
    def test_percentiles(self):
        results = [make(True, latency=v) for v in (1.0, 2.0, 3.0, 4.0, 100.0)]
        summary = metrics.latency_summary(results)
        assert summary["n"] == 5
        assert summary["p50_s"] == pytest.approx(3.0)
        assert summary["p95_s"] > 4.0

    def test_ignores_errored_runs(self):
        assert metrics.latency_summary([make(None, latency=1.0, error="x")])["n"] == 0


class TestRetrievalHitRate:
    def test_hit_when_any_evidence_page_retrieved(self):
        results = [make(True, evidence=[3, 4], retrieved=[4, 9])]
        assert metrics.retrieval_hit_rate(results) == 1.0

    def test_miss_when_no_evidence_page_retrieved(self):
        results = [make(False, evidence=[3], retrieved=[7, 8])]
        assert metrics.retrieval_hit_rate(results) == 0.0

    def test_skips_questions_without_evidence_annotations(self):
        results = [make(True, evidence=[], retrieved=[]), make(True, evidence=[1], retrieved=[1])]
        assert metrics.retrieval_hit_rate(results) == 1.0

    def test_none_when_not_applicable(self):
        assert metrics.retrieval_hit_rate([make(True)]) is None


class TestSummarize:
    def test_reports_all_headline_fields(self):
        results = [
            make(True, evidence=[1], retrieved=[1], latency=1.0, cost=0.001),
            make(False, evidence=[2], retrieved=[5], latency=3.0, cost=0.002),
        ]
        summary = metrics.summarize(results)
        assert summary["n_scored"] == 2
        assert summary["n_errors"] == 0
        assert summary["accuracy"] == pytest.approx(0.5)
        assert summary["retrieval_hit_rate"] == pytest.approx(0.5)
        assert summary["cost_usd"]["total"] == pytest.approx(0.003)
        assert summary["cost_usd"]["mean_per_question"] == pytest.approx(0.0015)
