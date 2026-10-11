# pageindex-rag-bench

An evaluation harness that measures whether **vectorless, reasoning-based retrieval** actually
beats ordinary **vector RAG** on long professional documents — using the same documents, the same
chat model, and the same judge for both.

`PageIndex` replaces the vector index with a hierarchical tree index and lets an LLM reason its way
down it. The claim is that similarity is not relevance. This harness tests that claim rather than
assuming it.

## Run it yourself

The benchmark is a CLI; the demo is a local web page that runs both systems on
one question at a time and shows what each of them actually retrieved.

```bash
# 1. dependencies
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install -e ".[dev,demo]"

# 2. corpus (34 documents, 108 MB, hash-verified against the manifest)
python -m corpus.fetch --docs
python -m corpus.questions          # derive the question sets

# 3. models — local, so a run costs nothing
ollama serve
ollama pull qwen3:8b                # answers the questions
ollama pull qwen3:1.7b              # builds the tree index (speed matters more here)
ollama pull nomic-embed-text        # embeddings for the vector baseline

# 4. the vector baseline needs a database
docker compose up -d                # pgvector on :5433

# 5a. interactive: pick a question, see both systems side by side
python -m app.make_demo_set --documents 3
python -m uvicorn app.server:app --port 8000     # open http://127.0.0.1:8000

# 5b. or run the benchmark and generate the results table
python -m bench.run --system pageindex_local  --set comparable_90
python -m bench.run --system pgvector_baseline --set comparable_90
python -m bench.report
```

The demo runs a **slice** (three of the 34 documents by default), because
indexing all 34 through a local model is measured in hours. The code path is
identical; only the corpus is smaller. Both `bench.run` and `app.server` are
resumable — indexing is cached, so an interrupted run continues where it stopped.

### What running it actually looks like

First end-to-end run on the demo slice, question: *"Format the date mentioned on
page 14 as YYYY-MM-DD."* — reference answer `2021-02-08`, evidence on page 14:

| | PageIndex (local) | pgvector baseline |
| --- | --- | --- |
| Found page 14 (the evidence page) | yes | yes |
| Response | the text of page 14 | `2021-02-08` |
| Latency | 58.2 s | 16.3 s |

Both retrieved the right page. Only the baseline produced the formatted answer —
PageIndex returned the page content itself, because its local agent loop can end
on a retrieval tool call instead of synthesising a final answer. That is recorded
honestly (`_shape: page_content` in the run output) rather than scored as a
correct response. One question is not a result; it is a demonstration that the
harness runs and that the comparison is worth making properly.

## The question this answers

The obvious baseline for a vectorless retrieval engine is vector RAG — chunk the document, embed
the chunks, cosine top-k, stuff the context. It is the thing almost everyone ships.

The upstream benchmark for PageIndex OSS measures PageIndex **against itself**: it varies the chat
model and reasoning effort, and reports accuracy and cost per model. That is a useful model-selection
study, but it does not tell you whether tree reasoning beats a pgvector index. Nothing in the
published numbers is a vector-RAG baseline.

This harness adds exactly that baseline, and holds everything else constant:

| | PageIndex (local) | pgvector baseline |
| --- | --- | --- |
| Retrieval | LLM reasoning over a document tree | chunk + embed + cosine top-k |
| Chat model | same | same |
| Judge | same | same |
| Documents | same 34 | same 34 |
| Questions | same set | same set |

Because only the retrieval strategy differs, a difference in accuracy is attributable to retrieval.
The harness also reports **retrieval hit rate** — whether an evidence page was actually retrieved —
which isolates retrieval quality from generation quality.

## What is mine, and what is not

This is not a fork of PageIndex, and it does not vendor anything. Attribution is in [`NOTICE`](NOTICE).

**Not mine:** the PageIndex engine (Vectify AI, MIT) is consumed as a dependency. The 34 documents
and the question annotations come from two Apache-2.0 upstream projects and are downloaded at setup
time. See [`THIRD_PARTY.md`](THIRD_PARTY.md).

**Mine:** the baseline, the harness, the metrics, the corpus pinning, and the findings.

## Question sets

Three sets are derived from Apache-2.0 annotations, so all of them are reproducible by anyone:

| Set | n | Rule | Comparable to published results? |
| --- | --- | --- | --- |
| `comparable_90` | 90 | questions over the 34 documents whose evidence is purely plain text | Yes — contains the curated 62 as a strict subset |
| `full_284` | 284 | every question attached to the 34 documents, all evidence sources | No — upstream excludes tables, figures and charts |
| `curated_62` | 62 | reference point only, **not redistributed** (its source grants no licence) | This is the published set |

The curated 62 was verified at derivation time to be **62 of 62 verbatim present** in the
Apache-2.0 derivation. Results on those 62 are therefore directly comparable to the published
figures, and the 222 further questions in `full_284` are scored too — which the comparison point
does not do.

## Results

_Not yet run._ The results table is generated by `bench.report` from `results/raw/`, so nothing here
is hand-entered. Until a run is committed, this section stays empty — the harness deliberately does
not ship placeholder numbers.

To produce them:

```bash
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install -e ".[dev]"

python -m corpus.fetch --docs          # 34 documents, 108 MB, hash-verified
python -m corpus.questions             # derive both question sets

python -m bench.run --system pageindex_local --set comparable_90
python -m bench.run --system pgvector_baseline --set comparable_90
python -m bench.report                 # writes results/SUMMARY.md
```

## Scope and limitations

Stated up front, because a benchmark that hides its boundaries is not worth reading:

- **34 documents, 90 questions in the comparable set.** With n=90 the 95% confidence interval on
  accuracy is roughly ±7 points. The harness reports a bootstrap CI next to every accuracy figure
  for that reason, and differences inside the interval are not claimed as wins.
- **Text-evidence questions only, for the comparable set.** `full_284` includes table, figure and
  chart evidence, which exercises the multimodal path; results there are reported separately and
  are not comparable to published numbers.
- **Documents are pre-verified to index successfully**, following the comparison point's practice.
  Documents that fail to ingest are excluded and reported as coverage, not scored as failures.
  `docs/FINDINGS.md` records which ones and why.
- **Cost figures for local models are zero by construction.** Cost only becomes informative when
  the same harness is pointed at a paid model; the price table is configurable.

## Layout

```
corpus/     fetch + verify documents; derive question sets from Apache-2.0 annotations
bench/      metrics, judge, adapters (pageindex_local, pgvector_baseline), runner, report
docs/       FINDINGS.md (results and failure modes), DECISIONS.md (why it is built this way)
tests/      scoring logic, tested without a database, a model, or a network
results/    aggregated summaries (raw per-question output is gitignored)
```

## Licence

MIT — see [`LICENSE`](LICENSE). Third-party attributions in [`NOTICE`](NOTICE) and
[`THIRD_PARTY.md`](THIRD_PARTY.md).
