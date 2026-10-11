"""Run one system over one question set.

    python -m bench.run --system pgvector_baseline --set comparable_90
    python -m bench.run --system pageindex_local --set comparable_90 --limit 10

Writes `results/raw/<system>__<set>.json`, which contains the run configuration
alongside every per-question observation. Runs are resumable: questions already
answered in the output file are skipped, so a long local-model run can be
interrupted and restarted without losing work.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import traceback

from . import metrics
from .adapters.base import System
from .judge import Judge, default_judge
from .preload import preload_litellm

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANNOTATIONS_DIR = os.path.join(REPO_ROOT, "corpus", "annotations")
DOCUMENTS_DIR = os.path.join(REPO_ROOT, "corpus", "documents")
RAW_DIR = os.path.join(REPO_ROOT, "results", "raw")

# A question set is either derived from Apache-2.0 annotations (committed as a
# fetch step) or the reference set, which is never redistributed and must be
# supplied locally.
DERIVED_SETS = ("comparable_90", "full_284")
REFERENCE_SETS = ("curated_62",)


def build_system(name: str) -> System:
    if name == "pageindex_local":
        from .adapters.pageindex_local import PageIndexLocal

        return PageIndexLocal()
    if name == "pgvector_baseline":
        from .adapters.pgvector_baseline import PgvectorBaseline

        return PgvectorBaseline()
    raise SystemExit(f"unknown system: {name}")


def load_questions(set_name: str) -> list[dict]:
    if set_name in DERIVED_SETS:
        path = os.path.join(ANNOTATIONS_DIR, f"{set_name}.json")
    elif set_name in REFERENCE_SETS:
        path = os.path.join(ANNOTATIONS_DIR, "curated_reference.json")
    else:
        raise SystemExit(f"unknown set: {set_name}")

    if not os.path.exists(path):
        raise SystemExit(
            f"{path} not found.\n"
            "For derived sets run:  python -m corpus.questions\n"
            "For curated_62, place a local copy at corpus/annotations/curated_reference.json. "
            "That set is deliberately not redistributed — its source grants no licence "
            "(see THIRD_PARTY.md)."
        )
    with open(path) as fh:
        return json.load(fh)


def load_progress(path: str) -> tuple[dict, dict]:
    if not os.path.exists(path):
        return {}, {}
    with open(path) as fh:
        payload = json.load(fh)
    done = {r["question"]: r for r in payload.get("results", [])}
    return payload.get("run", {}), done


def save(path: str, run: dict, results: list[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump({"run": run, "results": results}, fh, indent=2)
        fh.write("\n")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="bench.run", description=__doc__)
    ap.add_argument("--system", required=True)
    ap.add_argument("--set", required=True, dest="set_name")
    ap.add_argument("--limit", type=int, default=None, help="only the first N questions")
    ap.add_argument(
        "--judge-model",
        default=None,
        help="overrides JUDGE_MODEL; must be the same across systems being compared",
    )
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-judge", action="store_true", help="collect responses without grading")
    ap.add_argument(
        "--prepare-only",
        action="store_true",
        help="ingest the corpus and exit, without answering questions",
    )
    args = ap.parse_args(argv)

    # Before any adapter is constructed: PageIndex preloads litellm on its own
    # thread, and a concurrent first-import of litellm wedges the process.
    preload_litellm()

    system = build_system(args.system)
    questions = load_questions(args.set_name)
    doc_ids = sorted({q["doc_id"] for q in questions})
    if args.limit:
        questions = questions[: args.limit]
        doc_ids = sorted({q["doc_id"] for q in questions})

    out_path = args.out or os.path.join(RAW_DIR, f"{system.name}__{args.set_name}.json")
    prior_run, done = load_progress(out_path)

    run = {
        "system": system.name,
        "config": {
            k: v
            for k, v in vars(system).items()
            if isinstance(v, (str, int, float, bool, type(None)))
        },
        "set": args.set_name,
        "n_questions": len(questions),
        "n_documents": len(doc_ids),
        "judge_model": None if args.no_judge else (args.judge_model or default_judge().model),
        "started_at": prior_run.get("started_at")
        or datetime.datetime.now().isoformat(timespec="seconds"),
        "updated_at": None,
        "harness_version": "0.1.0",
    }

    print(f"{system.name} on {args.set_name}: {len(questions)} questions, {len(doc_ids)} documents")
    if done:
        print(f"resuming: {len(done)} question(s) already answered")

    system.prepare(doc_ids, DOCUMENTS_DIR)
    if args.prepare_only:
        print("prepare-only: corpus ingested, exiting")
        return 0

    judge: Judge | None = None
    if not args.no_judge:
        judge = Judge(model=run["judge_model"])
        run["judge_model"] = judge.model

    results = list(done.values())
    for i, item in enumerate(questions, 1):
        question = item["question"]
        if question in done:
            continue

        record = metrics.QuestionResult(
            system=system.name,
            set_name=args.set_name,
            doc_id=item["doc_id"],
            question=question,
            reference=str(item.get("answer", "")),
            evidence_pages=metrics.parse_pages(item.get("evidence_pages")),
        )
        try:
            answer = system.answer(item["doc_id"], question)
            record.response = answer.response
            record.retrieved_pages = answer.retrieved_pages
            record.latency_s = answer.latency_s
            record.prompt_tokens = answer.prompt_tokens
            record.completion_tokens = answer.completion_tokens
            record.cost_usd = answer.cost_usd
            record.context_chars = answer.context_chars
            if judge is not None and answer.response:
                verdict, usage = judge.grade(question, record.reference, answer.response)
                record.correct = verdict
                record.prompt_tokens += usage["prompt_tokens"]
                record.completion_tokens += usage["completion_tokens"]
        except Exception as exc:  # noqa: BLE001 - one failure must not kill the run
            record.error = f"{type(exc).__name__}: {exc}"
            traceback.print_exc(limit=3)

        results.append(record.to_dict())
        run["updated_at"] = datetime.datetime.now().isoformat(timespec="seconds")
        save(out_path, run, results)

        flag = "ERR" if record.error else ("ok " if record.correct else "miss")
        print(
            f"[{i:>3}/{len(questions)}] {flag} {record.latency_s or 0:6.1f}s "
            f"pages={record.retrieved_pages[:4]} {question[:58]}"
        )

    summary = metrics.summarize([metrics.QuestionResult(**r) for r in results])
    run["summary"] = summary
    save(out_path, run, results)

    print(f"\nwrote {out_path}")
    acc = summary["accuracy"]
    ci = summary["accuracy_ci95"]
    print(
        f"  accuracy {acc:.3f}" if acc is not None else "  accuracy n/a",
        f"(95% CI {ci[0]:.3f}-{ci[1]:.3f})" if ci else "",
        f"| hit rate {summary['retrieval_hit_rate']}"
        if summary["retrieval_hit_rate"] is not None
        else "",
        f"| errors {summary['n_errors']}",
    )
    return 0 if summary["n_errors"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
