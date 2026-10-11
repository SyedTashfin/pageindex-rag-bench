"""Baseline: chunk-and-embed vector RAG over pgvector.

This is the ordinary implementation: split each document into overlapping
character chunks, embed them, cosine top-k, put the winning chunks in the
context, ask the chat model.

**This baseline is deliberately the naïve version, and that is a limitation, not
a claim.** A real deployment would typically add BM25 hybrid search, a
cross-encoder reranker, and query rewriting. Beating this baseline does not show
that tree reasoning beats *good* vector RAG — only that it beats the version
most people actually ship. `docs/DECISIONS.md` records this and hybrid search is
tracked as a separate system so the comparison can be made honestly later.

A second asymmetry is worth stating plainly: PageIndex answers agentically
(multi-turn tool use over the tree), while this baseline answers single-shot from
a fixed context window. That favours PageIndex. It is disclosed here rather than
buried, and an agentic variant of this baseline is the fair follow-up.
"""

from __future__ import annotations

import os
import time
from collections.abc import Sequence
from dataclasses import dataclass

from pgvector import Vector

from .base import Answer, env_flag, think_extra_body

DEFAULT_CHAT_MODEL = "ollama/qwen2.5:7b-instruct"
DEFAULT_EMBED_MODEL = "ollama/nomic-embed-text"
DEFAULT_DSN = "postgresql://postgres:postgres@localhost:5433/ragbench"
TABLE = "chunks"

GROUNDED_PROMPT = """Answer the question using only the context below.

The context is an excerpt from a longer document. Some answers are a single \
value, a name, or a short phrase. If the context does not contain the answer, \
say so plainly rather than guessing.

Give the answer only — no preamble, no restating the question.

Context:
{context}

Question: {question}

Answer:"""


@dataclass
class Chunk:
    page: int
    index: int
    text: str


def extract_pages(path: str) -> list[str]:
    """Per-page plain text. Page numbers are 1-based to match the annotations."""
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(path)
    try:
        out = []
        for page in pdf:
            textpage = page.get_textpage()
            out.append(textpage.get_text_range())
        return out
    finally:
        pdf.close()


def chunk_pages(
    pages: Sequence[str], chunk_chars: int = 1200, overlap_chars: int = 200
) -> list[Chunk]:
    """Fixed-size overlapping chunks, cut on whitespace where possible.

    Chunks never span pages, so every chunk has an unambiguous page number —
    which is what makes retrieval hit rate computable for this baseline at all.
    """
    if overlap_chars >= chunk_chars:
        raise ValueError("overlap_chars must be smaller than chunk_chars")

    chunks: list[Chunk] = []
    for page_number, text in enumerate(pages, start=1):
        clean = " ".join(text.split())
        if not clean:
            continue
        start, index = 0, 0
        while start < len(clean):
            end = min(start + chunk_chars, len(clean))
            window = clean[start:end]
            if end < len(clean):
                cut = window.rfind(" ")
                if cut > chunk_chars // 2:
                    window = window[:cut]
            chunks.append(Chunk(page=page_number, index=index, text=window.strip()))
            index += 1
            if end >= len(clean):
                break
            start = max(end - overlap_chars, start + 1)
    return chunks


def create_table_sql(table: str, dim: int) -> str:
    """DDL for the chunk table.

    Postgres does not accept a bind parameter as a type modifier: ``vector($1)``
    fails with "type modifiers must be simple constants or identifiers". The
    dimension therefore has to be interpolated as a literal. That is safe here —
    it is an int derived from the embedding response, never user input — and it
    is kept as a pure function so a test can assert that no placeholder ever
    reaches the dimension slot again.

    The key includes ``page`` because ``chunk_pages`` numbers chunks *within* a
    page, so ``chunk_index`` restarts at 0 on every page. Keying on
    ``(doc_id, chunk_index)`` alone collides on the second page of every
    document.
    """
    return (
        f"CREATE TABLE IF NOT EXISTS {table} ("
        " doc_id text NOT NULL,"
        " page int NOT NULL,"
        " chunk_index int NOT NULL,"
        " chunk_chars int NOT NULL,"
        f" embedding vector({int(dim)}) NOT NULL,"
        " text text NOT NULL,"
        " PRIMARY KEY (doc_id, page, chunk_index)"
        ")"
    )


class PgvectorBaseline:
    """Chunk + embed + cosine top-k, backed by pgvector."""

    name = "pgvector_baseline"

    def __init__(
        self,
        chat_model: str | None = None,
        embed_model: str | None = None,
        dsn: str | None = None,
        chunk_chars: int = 1200,
        overlap_chars: int = 200,
        top_k: int = 5,
        embed_batch: int = 32,
        think: bool | None = None,
    ) -> None:
        self.chat_model = chat_model or os.environ.get("BASELINE_CHAT_MODEL", DEFAULT_CHAT_MODEL)
        self.embed_model = embed_model or os.environ.get(
            "BASELINE_EMBED_MODEL", DEFAULT_EMBED_MODEL
        )
        self.dsn = dsn or os.environ.get("PGVECTOR_DSN", DEFAULT_DSN)
        self.chunk_chars = chunk_chars
        self.overlap_chars = overlap_chars
        self.top_k = top_k
        self.embed_batch = embed_batch
        # Same reasoning-model knob as the PageIndex adapter, so the two systems
        # can be held identical on this too. See bench/adapters/base.py.
        self.think = env_flag("BASELINE_THINK", True) if think is None else think

    # -- storage ----------------------------------------------------------

    def _connect(self, create: bool = True):
        import psycopg
        from pgvector.psycopg import register_vector

        conn = psycopg.connect(self.dsn, autocommit=True)
        register_vector(conn)
        if create:
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            dim = self._embedding_dim()
            conn.execute(create_table_sql(TABLE, dim))
            conn.execute(
                f"CREATE INDEX IF NOT EXISTS {TABLE}_embedding_idx ON {TABLE} "
                "USING hnsw (embedding vector_cosine_ops)"
            )
        return conn

    def _embedding_dim(self) -> int:
        if not hasattr(self, "_dim"):
            probe = self._embed(["dimension probe"])
            first = probe[0]
            # pgvector's Vector exposes its values via to_list(); it has no len().
            self._dim = len(first.to_list() if hasattr(first, "to_list") else first)
        return self._dim

    # -- embedding --------------------------------------------------------

    def _embed(self, texts: Sequence[str]) -> list[Vector]:
        import litellm

        vectors: list[Vector] = []
        for start in range(0, len(texts), self.embed_batch):
            batch = list(texts[start : start + self.embed_batch])
            response = litellm.embedding(model=self.embed_model, input=batch, timeout=120)
            vectors.extend(Vector(item["embedding"]) for item in response.data)
        return vectors

    # -- ingestion --------------------------------------------------------

    def prepare(self, doc_ids: Sequence[str], documents_dir: str) -> None:
        """Chunk, embed and store every document not already indexed."""
        conn = self._connect()
        try:
            for i, doc_id in enumerate(doc_ids, 1):
                existing = conn.execute(
                    f"SELECT count(*) FROM {TABLE} WHERE doc_id = %s", (doc_id,)
                ).fetchone()[0]
                if existing:
                    print(f"  [{i}/{len(doc_ids)}] {doc_id}: {existing} chunks cached")
                    continue

                path = os.path.join(documents_dir, doc_id)
                if not os.path.exists(path):
                    print(f"  [{i}/{len(doc_ids)}] MISSING {doc_id}")
                    continue

                started = time.time()
                chunks = chunk_pages(extract_pages(path), self.chunk_chars, self.overlap_chars)
                if not chunks:
                    print(f"  [{i}/{len(doc_ids)}] {doc_id}: no extractable text, skipped")
                    continue
                vectors = self._embed([c.text for c in chunks])
                with conn.cursor() as cur:
                    cur.executemany(
                        f"INSERT INTO {TABLE}"
                        " (doc_id, page, chunk_index, chunk_chars, embedding, text)"
                        " VALUES (%s, %s, %s, %s, %s, %s)",
                        [
                            (doc_id, c.page, c.index, len(c.text), vec, c.text)
                            for c, vec in zip(chunks, vectors, strict=True)
                        ],
                    )
                print(
                    f"  [{i}/{len(doc_ids)}] {doc_id}: {len(chunks)} chunks "
                    f"in {time.time()-started:.1f}s"
                )
        finally:
            conn.close()

    # -- answering --------------------------------------------------------

    def answer(self, doc_id: str, question: str) -> Answer:
        import litellm

        conn = self._connect(create=False)
        try:
            started = time.time()
            query_vector = self._embed([question])[0]
            rows = conn.execute(
                f"SELECT page, chunk_index, text, 1 - (embedding <=> %s) AS similarity"
                f" FROM {TABLE} WHERE doc_id = %s"
                " ORDER BY embedding <=> %s LIMIT %s",
                (query_vector, doc_id, query_vector, self.top_k),
            ).fetchall()
            retrieval_s = time.time() - started

            if not rows:
                return Answer(
                    response="", retrieved_pages=[], latency_s=retrieval_s,
                    raw={"error": "no chunks indexed for this document"},
                )

            pages = sorted({int(r[0]) for r in rows})
            context = "\n\n".join(f"[page {r[0]}]\n{r[2]}" for r in rows)
            completion = litellm.completion(
                model=self.chat_model,
                messages=[
                    {
                        "role": "user",
                        "content": GROUNDED_PROMPT.format(context=context, question=question),
                    }
                ],
                temperature=0.0,
                timeout=180,
                extra_body=think_extra_body(self.think) or None,
            )
            text = completion.choices[0].message.content or ""
            usage = getattr(completion, "usage", None)
            try:
                cost = float(litellm.completion_cost(completion) or 0.0)
            except Exception:  # noqa: BLE001 - unknown local models have no price table
                cost = 0.0

            return Answer(
                response=text.strip(),
                retrieved_pages=pages,
                latency_s=time.time() - started,
                prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
                completion_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
                cost_usd=cost,
                context_chars=len(context),
                raw={"retrieval_s": retrieval_s, "top_k": self.top_k},
            )
        finally:
            conn.close()
