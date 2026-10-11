"""Ingest your own documents so you can ask questions across them.

This is the document-assistant path, not the benchmark. It uses the vector
retrieval system, which is the one that reliably produces answers on a local
model — see docs/FINDINGS.md for why PageIndex's local agent does not.

    python -m app.ingest ~/Documents/some-folder
    python -m app.ingest ~/Documents/a.pdf ~/Documents/b.pdf --top-k 8
    python -m app.ingest --list

Then ask questions across everything you ingested:

    python -m app.ask "what does the contract say about notice periods?"

Nothing here is public. Documents you ingest stay in your local Postgres and are
never committed, never uploaded, and never sent anywhere except to your own local
model. That is why your own documents belong here and not in the benchmark, whose
whole point is a corpus other people can reproduce.
"""

from __future__ import annotations

import argparse
import os
import sys


def collect(paths: list[str]) -> list[str]:
    """Expand directories into PDF paths, deterministically ordered."""
    files: list[str] = []
    for path in paths:
        expanded = os.path.expanduser(path)
        if os.path.isdir(expanded):
            for root, _, names in os.walk(expanded):
                for name in sorted(names):
                    if name.lower().endswith(".pdf"):
                        files.append(os.path.join(root, name))
        elif os.path.isfile(expanded):
            files.append(expanded)
        else:
            print(f"skipping (not found): {path}", file=sys.stderr)
    return sorted(set(files))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="app.ingest", description=__doc__)
    ap.add_argument("paths", nargs="*", help="folders and/or PDF files")
    ap.add_argument("--list", action="store_true", help="show what is already indexed")
    ap.add_argument("--top-k", type=int, default=None, help="chunks to retrieve per question")
    ap.add_argument("--chunk-chars", type=int, default=None)
    ap.add_argument("--think", default=None, help="on|off — reasoning costs ~24x latency here")
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
    if args.chunk_chars:
        kwargs["chunk_chars"] = args.chunk_chars
    if args.think is not None:
        kwargs["think"] = args.think.strip().lower() not in ("0", "false", "no", "off")

    baseline = PgvectorBaseline(**kwargs)

    if args.list:
        docs = baseline.list_documents()
        if not docs:
            print("nothing indexed yet")
            return 0
        total = 0
        for d in docs:
            total += d["chunks"]
            pages = f"pages {d['first_page']}-{d['last_page']}"
            print(f"  {d['doc_id'][:52]:<52} {d['chunks']:>5} chunks  {pages}")
        print(f"\n{len(docs)} document(s), {total} chunks")
        return 0

    if not args.paths:
        ap.print_help()
        return 1

    files = collect(args.paths)
    if not files:
        print("no PDFs found", file=sys.stderr)
        return 1

    print(f"ingesting {len(files)} PDF(s) with {baseline.embed_model} ...")
    ingested = baseline.ingest_files(files)
    print(f"\n{len(ingested)} document(s) available. Ask with:")
    print('  python -m app.ask "your question"')
    return 0 if ingested else 1


if __name__ == "__main__":
    raise SystemExit(main())
