"""Judge verdict parsing and adapter helpers. No network."""

from __future__ import annotations

import pytest

from bench.adapters.base import pages_from_citations
from bench.adapters.pgvector_baseline import chunk_pages
from bench.judge import Judge


class TestJudgeParseVerdict:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("YES", True),
            ("yes", True),
            ("Yes.", True),
            ("NO", False),
            ("no", False),
            ("NO.", False),
            ("  NO  ", False),
            ("", None),
            (None, None),
            ("   ", None),
            # A verdict followed by a short justification still counts: only the
            # opening token is read.
            ("YES - the figures match", True),
            ("NO, the dates differ", False),
            # A judge that restates the question before deciding has not committed
            # on its first token, and must not be silently accepted as a pass.
            ("The candidate response means the same thing, so YES", None),
        ],
    )
    def test_reads_only_the_opening_tokens(self, text, expected):
        assert Judge.parse_verdict(text) is expected

    def test_a_hedge_is_unscored_rather_than_a_pass(self):
        # The failure this guards: reading a hedge as a pass inflates accuracy.
        hedge = "It may be YES, but the context does not clearly confirm it."
        assert Judge.parse_verdict(hedge) is None

    def test_prompt_does_not_leak_which_system_answered(self):
        judge = Judge(model="x")
        messages = judge.build_messages("What is the revenue?", "1,358,000", "1358000")
        blob = " ".join(m["content"] for m in messages)
        assert "pageindex" not in blob.lower()
        assert "vector" not in blob.lower()
        assert "baseline" not in blob.lower()

    def test_prompt_carries_reference_and_response(self):
        judge = Judge(model="x")
        messages = judge.build_messages("Q", "REF", "CAND")
        blob = " ".join(m["content"] for m in messages)
        assert "REF" in blob and "CAND" in blob


class TestPagesFromCitations:
    def test_reads_page_directly(self):
        assert pages_from_citations([{"page": 19}]) == [19]

    def test_reads_page_nested_under_block(self):
        assert pages_from_citations([{"doc_id": "d", "block": {"page": 7}}]) == [7]

    def test_accepts_page_ranges_as_strings(self):
        assert pages_from_citations([{"page": "12-13"}]) == [12, 13]

    def test_merges_and_deduplicates(self):
        cites = [{"page": 3}, {"block": {"page": 3}}, {"page": 9}]
        assert pages_from_citations(cites) == [3, 9]

    def test_ignores_unrecognised_shapes_instead_of_raising(self):
        assert pages_from_citations([{"nonsense": 1}, "a string", None]) == []
        assert pages_from_citations([]) == []


class TestChunkPages:
    def test_pages_are_one_based_and_never_merge(self):
        chunks = chunk_pages(["a" * 50, "b" * 50], chunk_chars=30, overlap_chars=5)
        assert {c.page for c in chunks} == {1, 2}

    def test_overlap_produces_more_chunks_than_a_single_pass(self):
        text = "word " * 400
        no_overlap = chunk_pages([text], chunk_chars=600, overlap_chars=1)
        with_overlap = chunk_pages([text], chunk_chars=600, overlap_chars=300)
        assert len(with_overlap) > len(no_overlap)

    def test_blank_pages_are_skipped(self):
        assert chunk_pages(["", "   \n "]) == []

    def test_rejects_overlap_at_or_above_chunk_size(self):
        with pytest.raises(ValueError):
            chunk_pages(["text"], chunk_chars=100, overlap_chars=100)

    def test_terminates_and_covers_the_input(self):
        text = " ".join(f"w{i}" for i in range(500))
        chunks = chunk_pages([text], chunk_chars=200, overlap_chars=50)
        assert chunks
        assert "w0" in chunks[0].text
        assert "w499" in chunks[-1].text
