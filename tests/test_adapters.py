"""Judge verdict parsing and adapter helpers. No network."""

from __future__ import annotations

import pytest

from bench.adapters.base import env_flag, pages_from_citations, think_extra_body
from bench.adapters.pageindex_local import PageIndexLocal
from bench.adapters.pgvector_baseline import chunk_pages, create_table_sql
from bench.judge import Judge


class TestThinkKnob:
    """Reasoning models emit hidden reasoning by default. The knob exists so the
    two systems can be held identical on it, and so the accuracy/speed trade-off
    is explicit rather than accidental."""

    def test_default_leaves_thinking_on(self):
        assert think_extra_body(True) == {}

    def test_disabled_emits_the_provider_option(self):
        assert think_extra_body(False) == {"think": False}

    def test_callers_pass_empty_as_none(self):
        # litellm should not receive an empty dict as extra_body.
        assert (think_extra_body(True) or None) is None
        assert (think_extra_body(False) or None) == {"think": False}

    def test_adapter_defaults_to_thinking_on(self, monkeypatch):
        monkeypatch.delenv("PAGEINDEX_THINK", raising=False)
        assert PageIndexLocal().think is True

    def test_adapter_reads_the_env_knob(self, monkeypatch):
        monkeypatch.setenv("PAGEINDEX_THINK", "false")
        assert PageIndexLocal().think is False

    def test_explicit_argument_beats_the_environment(self, monkeypatch):
        monkeypatch.setenv("PAGEINDEX_THINK", "false")
        assert PageIndexLocal(think=True).think is True


class TestEnvFlag:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("0", False), ("false", False), ("False", False), ("FALSE", False),
            ("no", False), ("off", False), ("", False),
            ("1", True), ("true", True), ("True", True), ("yes", True), ("on", True),
        ],
    )
    def test_parses_the_usual_spellings(self, raw, expected, monkeypatch):
        monkeypatch.setenv("X_FLAG", raw)
        assert env_flag("X_FLAG", True) is expected

    def test_returns_the_default_when_unset(self, monkeypatch):
        monkeypatch.delenv("X_FLAG", raising=False)
        assert env_flag("X_FLAG", True) is True
        assert env_flag("X_FLAG", False) is False

    def test_the_naive_reading_is_the_bug_this_avoids(self, monkeypatch):
        import os

        monkeypatch.setenv("X_FLAG", "0")
        assert bool(os.environ["X_FLAG"]) is True   # what bool(env) would give
        assert env_flag("X_FLAG", True) is False    # what it should give


class TestParseChatEnvelope:
    """PageIndex returns a JSON envelope serialised as a string. Treating that
    string as the answer records a NOT_FOUND payload as a successful response —
    which is what the first version of this adapter did."""

    def test_reads_content_from_a_success_envelope(self):
        text, raw, pages = PageIndexLocal.parse_chat(
            '{"status": "success", "content": "February 8, 2021"}'
        )
        assert text == "February 8, 2021"
        assert raw["status"] == "success"
        assert pages == []

    def test_raises_on_a_not_found_envelope(self):
        payload = (
            '{"error": "Document not found.", "errorCode": "NOT_FOUND",'
            ' "doc_name": "x.pdf"}'
        )
        with pytest.raises(RuntimeError, match="NOT_FOUND"):
            PageIndexLocal.parse_chat(payload)

    def test_raises_on_a_bare_error_field(self):
        with pytest.raises(RuntimeError, match="boom"):
            PageIndexLocal.parse_chat('{"error": "boom"}')

    def test_passes_through_plain_text(self):
        # If the SDK ever returns prose instead of an envelope, that is the answer.
        text, _, pages = PageIndexLocal.parse_chat("The answer is 42.")
        assert text == "The answer is 42."
        assert pages == []

    def test_reads_a_real_dict_too(self):
        text, _, _ = PageIndexLocal.parse_chat({"status": "success", "content": "ok"})
        assert text == "ok"

    def test_never_returns_a_raw_json_string_as_the_answer(self):
        text, _, _ = PageIndexLocal.parse_chat('{"status": "success", "content": "fine"}')
        assert not text.startswith("{")

    def test_page_content_is_flagged_and_yields_provenance(self):
        """The agent loop sometimes returns the last tool result — a page of the
        document — where a synthesised answer was expected. The page number is
        real retrieval provenance and must be recorded as such, not silently
        scored as an answer."""
        payload = '{"page": 14, "text": "Montr\\u00e9al, February 8, 2021"}'
        text, raw, pages = PageIndexLocal.parse_chat(payload)
        assert raw["_shape"] == "page_content"
        assert pages == [14]
        assert "February 8, 2021" in text


class TestStagedName:
    """PageIndex's local agent tells the model the document *name* but resolves
    its tools by internal id, so the model has to reproduce the name in a tool
    call. Both a long hex filename and zero-padding break that on small local
    models — the second is the subtler trap."""

    def test_is_short_and_unpadded(self):
        assert PageIndexLocal.staged_name(1) == "doc1.pdf"
        assert PageIndexLocal.staged_name(34) == "doc34.pdf"

    def test_no_zero_padding_to_be_dropped(self):
        # Measured: with 'doc-03.pdf' the model asked for 'doc-3.pdf'. A name
        # that differs from its own unpadded form only by a leading zero invites
        # the model to drop it.
        for i in range(1, 10):
            assert not PageIndexLocal.staged_name(i).startswith("doc0")

    def test_no_punctuation_inside_the_name(self):
        # Hyphens and underscores are also candidates for being dropped or
        # mis-typed; the stem should be alphanumeric only.
        stem = PageIndexLocal.staged_name(12).removesuffix(".pdf")
        assert stem.isalnum()

    def test_is_much_shorter_than_a_corpus_filename(self):
        corpus_name = "a5879805d70c854ea4361e43a84e3bb2.pdf"
        assert len(PageIndexLocal.staged_name(1)) < len(corpus_name) / 3

    def test_is_deterministic(self):
        assert PageIndexLocal.staged_name(7) == PageIndexLocal.staged_name(7)

    def test_differs_per_document(self):
        names = {PageIndexLocal.staged_name(i) for i in range(1, 35)}
        assert len(names) == 34


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


class TestCreateTableSql:
    """Regression guard for a bug that only appeared when actually run against
    Postgres: a bind parameter is not allowed as a type modifier."""

    def test_dimension_is_a_literal_not_a_placeholder(self):
        sql = create_table_sql("chunks", 768)
        assert "vector(768)" in sql
        assert "%s" not in sql
        assert "$1" not in sql

    def test_dimension_is_coerced_to_an_integer(self):
        # The value is interpolated into DDL, so it must never be a string.
        assert "vector(1024)" in create_table_sql("chunks", "1024")

    def test_table_name_is_used_and_primary_key_present(self):
        sql = create_table_sql("my_chunks", 3)
        assert sql.startswith("CREATE TABLE IF NOT EXISTS my_chunks")
        assert "PRIMARY KEY (doc_id, page, chunk_index)" in sql

    def test_primary_key_includes_the_page(self):
        """chunk_pages numbers chunks within each page, so chunk_index restarts
        at 0 on every page. Keying without `page` collides on page 2 of every
        document - this actually happened against Postgres."""
        assert "page" in create_table_sql("chunks", 768).split("PRIMARY KEY")[1]


class TestChunkIndexRestartsPerPage:
    def test_chunk_index_is_per_page_not_global(self):
        # Documents the behaviour the primary key above has to accommodate.
        chunks = chunk_pages(["x " * 400, "y " * 400], chunk_chars=300, overlap_chars=50)
        by_page = {}
        for c in chunks:
            by_page.setdefault(c.page, []).append(c.index)
        assert set(by_page) == {1, 2}
        assert by_page[1][0] == 0 and by_page[2][0] == 0

    def test_no_duplicate_doc_page_index_tuples(self):
        chunks = chunk_pages(
            ["x " * 500, "y " * 500, "z " * 500], chunk_chars=250, overlap_chars=40
        )
        keys = [(c.page, c.index) for c in chunks]
        assert len(keys) == len(set(keys)), "same (page, index) emitted twice"


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
