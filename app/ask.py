"""Ask a question across every document you have ingested.

    python -m app.ask "what does the contract say about notice periods?"
    python -m app.ask "summarise the refund policy" --think off

Uses the vector retrieval path, which is the system that reliably answers on a
local model. Retrieval is deliberately unrestricted across the whole index, so
this is not the benchmark comparison — it says nothing about which retrieval
strategy is better, only what your documents contain.

Every answer reports which documents and pages it drew on, and the similarity of
each retrieved chunk, so you can check the claim rather than trust it.
"""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="app.ask", description=__doc__)
    ap.add_argument("question", help="the question to ask")
    ap.add_argument("--top-k", type=int, default=None, help="chunks to retrieve (default 5)")
    ap.add_argument("--think", default=None, help="on|off — reasoning costs ~24x latency here")
    ap.add_argument("--show-chunks", action="store_true", help="print the retrieved text")
    args = ap.parse_args(argv)

    from bench.preload import preload_litellm

    preload_litellm()
    from bench.adapters.pgvector_baseline import PgvectorBaseline

    from . import config

    kwargs: dict[str, object] = {
        "chat_model": config.CHAT_MODEL,
        "embed_model": config.EMBED_MODEL,
    }
    if args.top_k:
        kwargs["top_k"] = args.top_k
    if args.think is not None:
        kwargs["think"] = args.think.strip().lower() not in ("0", "false", "no", "off")

    baseline = PgvectorBaseline(**kwargs)

    available = config.available_models()
    model_name = config.CHAT_MODEL.removeprefix("ollama/")
    if available and not any(m.startswith(model_name) for m in available):
        print(f"model '{model_name}' is not installed.", file=sys.stderr)
        print("installed: " + ", ".join(available), file=sys.stderr)
        print("pick one with:  CHAT_MODEL=ollama/<name> python -m app.ask ...", file=sys.stderr)
        return 1

    indexed = baseline.list_documents()
    if not indexed:
        print("nothing indexed yet. Ingest some documents first:", file=sys.stderr)
        print("  python -m app.ingest ~/Documents/some-folder", file=sys.stderr)
        return 1

    answer = baseline.answer_any(args.question)
    if answer.raw.get("error"):
        print(f"error: {answer.raw['error']}", file=sys.stderr)
        return 1

    print(answer.response or "(the model returned nothing)")
    print()
    print(f"--- {len(indexed)} document(s) indexed, top {baseline.top_k} chunks retrieved "
          f"in {answer.raw.get('retrieval_s', 0):.2f}s, answered in {answer.latency_s:.2f}s total")
    print(f"--- reasoning: {'on' if baseline.think else 'off'}  |  "
          f"{answer.completion_tokens} completion tokens")

    sources = answer.raw.get("sources", [])
    if sources:
        print("\n--- sources consulted")
        for s in sources:
            print(f"  {s['doc_id']}  page {s['page']}  similarity {s['similarity']:.4f}")
            if args.show_chunks:
                print(f"      {s['preview']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
