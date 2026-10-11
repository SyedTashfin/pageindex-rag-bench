"""Interactive demo: both retrieval systems, side by side, in a browser.

The benchmark answers 90 questions and writes a results file. That is the right
artefact for a result and a poor one for understanding the thing. This server
runs one question at a time through both systems and shows what each one
actually retrieved, so the difference between reasoning over a document tree and
cosine-searching embedded chunks is visible rather than asserted.

    python -m app.server            # then open http://127.0.0.1:8000

Scope is deliberately a slice (app/demo_set.json), because indexing the full 34
documents through a local model is measured in hours. The systems and the code
path are identical to the benchmark; only the corpus is smaller.
"""

from __future__ import annotations

import json
import os
import threading
import time
import traceback

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from bench.preload import preload_litellm

from . import config

# Import litellm in the main thread, before any worker thread exists. Both
# PageIndex (its own preload thread) and the baseline import it lazily, and two
# concurrent first-imports race inside importlib and wedge the process. See
# bench/preload.py.
preload_litellm()

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEMO_SET = os.path.join(REPO_ROOT, "app", "demo_set.json")
DOCUMENTS_DIR = os.path.join(REPO_ROOT, "corpus", "documents")
INDEX_HTML = os.path.join(REPO_ROOT, "app", "index.html")

# From app/config.py — one definition shared with the command line tools, so the
# demo and the CLIs cannot disagree about which model they are using.
CHAT_MODEL = config.CHAT_MODEL
INDEX_MODEL = config.INDEX_MODEL
EMBED_MODEL = config.EMBED_MODEL


def load_demo() -> dict:
    with open(DEMO_SET) as fh:
        return json.load(fh)


app = FastAPI(title="pageindex-rag-bench demo")

# One instance of each system, built once. Indexing is the expensive part, so it
# must not happen per request.
STATE: dict = {
    "systems": {},
    "ready": False,
    "indexing": False,
    "log": [],
    "error": None,
    "started_at": None,
}


def log(message: str) -> None:
    stamp = time.strftime("%H:%M:%S")
    STATE["log"].append(f"[{stamp}] {message}")
    STATE["log"] = STATE["log"][-200:]
    print(f"[{stamp}] {message}", flush=True)


def build_systems() -> dict:
    if STATE["systems"]:
        return STATE["systems"]
    from bench.adapters.pageindex_local import PageIndexLocal
    from bench.adapters.pgvector_baseline import PgvectorBaseline

    STATE["systems"] = {
        "pageindex_local": PageIndexLocal(chat_model=CHAT_MODEL, index_model=INDEX_MODEL),
        "pgvector_baseline": PgvectorBaseline(chat_model=CHAT_MODEL, embed_model=EMBED_MODEL),
    }
    return STATE["systems"]


def index_everything(doc_ids: list[str]) -> None:
    """Index both systems. Runs in a thread; the page polls /api/status."""
    try:
        systems = build_systems()
        for name, system in systems.items():
            log(f"indexing {len(doc_ids)} document(s) with {name} ...")
            system.prepare(doc_ids, DOCUMENTS_DIR)
            log(f"{name} ready")
        STATE["ready"] = True
        log("both systems ready")
    except Exception as exc:  # noqa: BLE001 - surfaced to the page
        STATE["error"] = f"{type(exc).__name__}: {exc}"
        log(f"FAILED: {STATE['error']}")
        traceback.print_exc()
    finally:
        STATE["indexing"] = False


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    with open(INDEX_HTML) as fh:
        return fh.read()


@app.get("/api/status")
def status() -> dict:
    demo = load_demo()
    return {
        "ready": STATE["ready"],
        "indexing": STATE["indexing"],
        "error": STATE["error"],
        "log": STATE["log"],
        "models": {"chat": CHAT_MODEL, "index": INDEX_MODEL, "embed": EMBED_MODEL},
        "corpus": {
            "documents": demo["documents"],
            "total_pages": demo["total_pages"],
            "questions": len(demo["questions"]),
            "full_benchmark": "34 documents, 90 questions, 1,945 pages (same code path)",
        },
    }


@app.post("/api/prepare")
def prepare() -> JSONResponse:
    if STATE["indexing"]:
        return JSONResponse({"status": "already running"}, status_code=409)
    if STATE["ready"]:
        return JSONResponse({"status": "already indexed"})

    demo = load_demo()
    doc_ids = [d["doc_id"] for d in demo["documents"]]

    # Fail early with a readable message rather than mid-index if the corpus is
    # not on disk or pgvector is down.
    missing = [d for d in doc_ids if not os.path.exists(os.path.join(DOCUMENTS_DIR, d))]
    if missing:
        return JSONResponse(
            {
                "error": "corpus not fetched",
                "detail": f"{len(missing)} document(s) missing, e.g. {missing[0]}",
                "fix": "python -m corpus.fetch --docs",
            },
            status_code=400,
        )

    STATE["indexing"] = True
    STATE["error"] = None
    STATE["log"] = []
    STATE["started_at"] = time.time()
    threading.Thread(target=index_everything, args=(doc_ids,), daemon=True).start()
    return JSONResponse({"status": "started", "documents": len(doc_ids)})


class Ask(BaseModel):
    question_id: int


@app.post("/api/ask")
def ask(payload: Ask) -> JSONResponse:
    if not STATE["ready"]:
        return JSONResponse({"error": "not indexed yet"}, status_code=409)

    demo = load_demo()
    match = [q for q in demo["questions"] if q["id"] == payload.question_id]
    if not match:
        return JSONResponse(
            {"error": f"unknown question id {payload.question_id}"}, status_code=404
        )
    item = match[0]

    results = []
    for name, system in build_systems().items():
        started = time.time()
        try:
            answer = system.answer(item["doc_id"], item["question"])
            results.append(
                {
                    "system": name,
                    "ok": True,
                    "response": answer.response,
                    "retrieved_pages": answer.retrieved_pages,
                    "latency_s": round(answer.latency_s, 2),
                    "prompt_tokens": answer.prompt_tokens,
                    "completion_tokens": answer.completion_tokens,
                    "context_chars": answer.context_chars,
                }
            )
        except Exception as exc:  # noqa: BLE001 - one system failing must not hide the other
            results.append(
                {
                    "system": name,
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "latency_s": round(time.time() - started, 2),
                }
            )

    def pages(value):
        import re

        return sorted({int(m) for m in re.findall(r"\d+", str(value or ""))})

    evidence = pages(item.get("evidence_pages"))
    for r in results:
        if r.get("ok") and evidence:
            r["hit"] = bool(set(r["retrieved_pages"]) & set(evidence))
        else:
            r["hit"] = None

    return JSONResponse(
        {
            "question": item["question"],
            "doc_id": item["doc_id"],
            "reference": item["reference"],
            "evidence_pages": evidence,
            "results": results,
        }
    )


@app.get("/api/questions")
def questions() -> dict:
    demo = load_demo()
    return {
        "documents": demo["documents"],
        "questions": [
            {"id": q["id"], "doc_id": q["doc_id"], "question": q["question"]}
            for q in demo["questions"]
        ],
    }


class FreeAsk(BaseModel):
    question: str


@app.post("/api/ask-free")
def ask_free(payload: FreeAsk) -> JSONResponse:
    """Ask anything across every indexed document.

    This is the document-assistant path, not the benchmark. It uses the vector
    baseline because that is the system which reliably produces answers; PageIndex
    is deliberately not offered here, because its local agent returns tool results
    instead of answers on this model. See docs/FINDINGS.md.
    """
    if not STATE["ready"]:
        return JSONResponse({"error": "not indexed yet"}, status_code=409)

    question = (payload.question or "").strip()
    if not question:
        return JSONResponse({"error": "empty question"}, status_code=400)

    baseline = build_systems()["pgvector_baseline"]
    try:
        answer = baseline.answer_any(question)
    except Exception as exc:  # noqa: BLE001 - surfaced to the page
        return JSONResponse({"error": f"{type(exc).__name__}: {exc}"}, status_code=500)

    return JSONResponse(
        {
            "question": question,
            "system": baseline.name,
            "answer": answer.response,
            "documents_used": answer.raw.get("documents_used", []),
            "sources": answer.raw.get("sources", []),
            "latency_s": round(answer.latency_s, 2),
            "top_k": baseline.top_k,
            "think": baseline.think,
        }
    )


@app.get("/api/index")
def indexed() -> dict:
    """What is actually in the vector index, across every ingested document."""
    baseline = build_systems()["pgvector_baseline"]
    try:
        docs = baseline.list_documents()
    except Exception as exc:  # noqa: BLE001 - the page should still load
        return {"error": f"{type(exc).__name__}: {exc}", "documents": []}
    return {"documents": docs, "total": len(docs), "top_k": baseline.top_k}
