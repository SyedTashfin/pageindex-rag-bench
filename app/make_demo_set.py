"""Pick a small, fast slice of the corpus for the interactive demo.

The full benchmark indexes 34 documents through an LLM in local mode, which is
measured in hours. That is the right scope for a published result and the wrong
scope for looking at the thing. This picks the smallest documents that still
carry questions in the plain-text set, so the same code path is exercised on a
slice that indexes in minutes.

    python -m app.make_demo_set --documents 3

Writes app/demo_set.json. Deterministic: same corpus, same output.
"""

from __future__ import annotations

import argparse
import json
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(REPO_ROOT, "corpus", "manifest.json")
QUESTIONS = os.path.join(REPO_ROOT, "corpus", "annotations", "comparable_90.json")
OUT = os.path.join(REPO_ROOT, "app", "demo_set.json")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="app.make_demo_set", description=__doc__)
    ap.add_argument("--documents", type=int, default=3, help="how many documents to include")
    args = ap.parse_args(argv)

    manifest = json.load(open(MANIFEST))
    pages = {d["doc_id"]: d.get("pages") or 0 for d in manifest["documents"]}
    sizes = {d["doc_id"]: d["bytes"] for d in manifest["documents"]}

    if not os.path.exists(QUESTIONS):
        raise SystemExit(f"{QUESTIONS} not found. Run: python -m corpus.questions")

    with open(QUESTIONS) as fh:
        all_q = json.load(fh)

    # Prefer documents that are small in pages, then small in bytes: page count
    # drives indexing time, byte count drives disk and parse time.
    counts: dict[str, int] = {}
    for q in all_q:
        counts[q["doc_id"]] = counts.get(q["doc_id"], 0) + 1

    chosen = sorted(counts, key=lambda d: (pages[d], sizes[d]))[: args.documents]

    questions = [
        {
            "id": i,
            "doc_id": q["doc_id"],
            "question": q["question"],
            "reference": str(q.get("answer", "")),
            "evidence_pages": q.get("evidence_pages"),
            "evidence_sources": q.get("evidence_sources"),
            "doc_type": q.get("doc_type"),
        }
        for i, q in enumerate(all_q)
        if q["doc_id"] in chosen
    ]

    payload = {
        "note": (
            "Small slice for interactive use. Regenerate with "
            "python -m app.make_demo_set. The full benchmark runs the same code "
            "over all 34 documents (corpus/annotations/comparable_90.json)."
        ),
        "selection_rule": (
            f"{args.documents} documents with the fewest pages that carry questions "
            "in the plain-text set"
        ),
        "documents": [
            {
                "doc_id": d,
                "pages": pages[d],
                "bytes": sizes[d],
                "questions": counts[d],
            }
            for d in chosen
        ],
        "total_pages": sum(pages[d] for d in chosen),
        "questions": questions,
    }

    json.dump(payload, open(OUT, "w"), indent=2)
    open(OUT, "a").write("\n")

    print(f"wrote {OUT}")
    for d in payload["documents"]:
        print(
            f"  {d['doc_id'][:46]:48} {d['pages']:>3}p  "
            f"{d['bytes']/1e6:>5.1f}MB  {d['questions']} questions"
        )
    print(f"  TOTAL: {payload['total_pages']} pages, {len(questions)} questions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
