"""Derive the question sets from Apache-2.0 annotations.

The obvious thing to do — copy the curated 62-question file from
PageIndex-OSS-Benchmark — is not legally available: that repository grants no
licence (its /license endpoint returns 404).

Instead the questions are re-derived from `MMLongBench-Doc-V2`, which is
Apache-2.0 and is itself a corrected release of the same upstream dataset. The
derivation was verified to reproduce all 62 curated questions verbatim
(62/62 exact match on doc_id + question text) while additionally surfacing 28
further questions from the same documents that the curated set drops.

Two sets are produced:

  comparable_90  the 34 documents filtered to questions whose evidence is
                 purely plain text (`evidence_sources == ['Pure-text (Plain-text)']`).
                 Contains the curated 62 as a strict subset — so results on the
                 62-question subset are directly comparable to published ones.
  full_284       every question attached to the 34 documents, across all
                 evidence sources (tables, figures, charts, layout).

Usage:
    python -m corpus.questions            # fetch annotations and write both sets
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from . import fetch, sources

PURE_TEXT = "['Pure-text (Plain-text)']"


def load_annotations() -> list[dict]:
    path = os.path.join(sources.ANNOTATIONS_DIR, "samples.json")
    if not os.path.exists(path):
        os.makedirs(sources.ANNOTATIONS_DIR, exist_ok=True)
        print(f"fetching annotations from {sources.annotations_url()} ...")
        fetch.download(sources.annotations_url(), path)
    with open(path) as fh:
        return json.load(fh)


def normalize(value: object) -> str:
    return " ".join(str(value or "").lower().split())


def derive(annotations: list[dict], doc_ids: set[str]) -> dict[str, list[dict]]:
    """Split the annotations into the two sets. Pure function, no I/O."""
    pool = [a for a in annotations if a.get("doc_id") in doc_ids]
    pure_text = [a for a in pool if a.get("evidence_sources") == PURE_TEXT]
    return {"comparable_90": pure_text, "full_284": pool}


def write_sets(manifest: dict, sets: dict[str, list[dict]]) -> dict:
    os.makedirs(sources.ANNOTATIONS_DIR, exist_ok=True)
    counts = {}
    for name, rows in sets.items():
        out = os.path.join(sources.ANNOTATIONS_DIR, f"{name}.json")
        with open(out, "w") as fh:
            json.dump(rows, fh, indent=2)
            fh.write("\n")
        counts[name] = len(rows)
        docs = len({r["doc_id"] for r in rows})
        print(f"  {name:16} {len(rows):>4} questions over {docs} documents -> {out}")
    json.dump(counts, open(os.path.join(sources.ANNOTATIONS_DIR, "counts.json"), "w"), indent=2)
    return counts


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="corpus.questions", description=__doc__)
    ap.add_argument("--check", action="store_true", help="verify counts against the manifest")
    args = ap.parse_args(argv)

    manifest = fetch.load_manifest()
    ids = set(fetch.doc_ids(manifest))

    if args.check:
        expected = manifest["question_sets"]
        ok = True
        for name, meta in expected.items():
            path = os.path.join(sources.ANNOTATIONS_DIR, f"{name}.json")
            # A set with no licence grant is never redistributed or fetched here,
            # so its absence on disk is correct rather than a failure.
            if meta.get("redistributed") is False:
                print(f"{name}: not redistributed by design (see THIRD_PARTY.md)")
                continue
            if not os.path.exists(path):
                print(f"missing: {path}", file=sys.stderr)
                ok = False
                continue
            with open(path) as fh:
                rows = json.load(fh)
            if len(rows) != meta["count"]:
                print(f"{name}: expected {meta['count']}, found {len(rows)}", file=sys.stderr)
                ok = False
            else:
                print(f"{name}: {len(rows)} questions ok")
        return 0 if ok else 1

    annotations = load_annotations()
    print(f"annotations: {len(annotations)} questions total")
    sets = derive(annotations, ids)

    # Equivalence against the curated set is verified against a *local* copy that
    # the user may have fetched for comparison. That copy is gitignored and is
    # never redistributed: the comparison repository grants no licence.
    ref_path = os.path.join(sources.ANNOTATIONS_DIR, "curated_reference.json")
    if os.path.exists(ref_path):
        with open(ref_path) as fh:
            reference = json.load(fh)
        derived_keys = {
            (normalize(r["doc_id"]), normalize(r["question"])) for r in sets["comparable_90"]
        }
        ref_keys = {(normalize(r["doc_id"]), normalize(r["question"])) for r in reference}
        overlap = len(derived_keys & ref_keys)
        print(
            f"  curated {len(ref_keys)} reproduced from Apache-2.0 data: "
            f"{overlap}/{len(ref_keys)}"
        )
        if overlap != len(ref_keys):
            print("  WARNING: derivation no longer reproduces the curated set", file=sys.stderr)
    else:
        meta = manifest["question_sets"]["curated_62"]
        print(
            f"  curated {meta['count']} not checked locally (no {os.path.basename(ref_path)}). "
            f"Recorded equivalence at derivation time: {meta['verified_overlap']}."
        )

    counts = write_sets(manifest, sets)
    manifest["question_sets"]["comparable_90"]["count"] = counts["comparable_90"]
    manifest["question_sets"]["full_284"]["count"] = counts["full_284"]
    with open(sources.MANIFEST_PATH, "w") as fh:
        json.dump(manifest, fh, indent=2)
        fh.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
