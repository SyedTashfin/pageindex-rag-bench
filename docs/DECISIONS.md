# Design decisions

Why this harness is built the way it is. Each entry states the decision, the
alternative that was rejected, and the reason.

## 1. The baseline is the naïve implementation, and that is a weakness

**Decision.** The vector baseline chunks at 1200 characters with 200 characters
of overlap, embeds, takes cosine top-5, and answers from those chunks in one
shot.

**Rejected alternative.** Hybrid retrieval (BM25 + vector, fused) with a
cross-encoder reranker.

**Why.** The honest object of study is "what most people actually ship", and that
is naive vector RAG. A tuned hybrid pipeline would be a stronger opponent but
also a different experiment.

**Consequence, stated plainly.** A win over this baseline does **not** show that
tree reasoning beats good vector RAG. It shows it beats naive vector RAG. Hybrid
retrieval is tracked as a separate system so the stronger comparison can be made
later without re-running everything.

## 2. The baseline answers single-shot; PageIndex answers agentically

**Decision.** Disclose the asymmetry rather than hide it.

**Why.** PageIndex runs a multi-turn tool loop over the document tree. The
baseline gets one fixed context window and one completion. That structurally
favours PageIndex, and a reader who knows how these systems work will spot it
immediately. Publishing a favourable comparison without naming the asymmetry
would be the dishonest part; naming it lets the reader discount appropriately.

**Follow-up.** An agentic baseline — give the vector retriever the same
multi-turn loop and the same turn budget — is the fair comparison. It is not
implemented yet, and results should not be described as "vectorless beats vector
RAG" until it is.

## 3. The questions are re-derived rather than copied

**Decision.** Derive all question sets from `MMLongBench-Doc-V2`'s Apache-2.0
`samples.json` by filtering to the 34 documents used by the comparison point.

**Rejected alternative.** Copy `questions.json` and the 34 PDFs from
`VectifyAI/PageIndex-OSS-Benchmark`, which is the obvious thing to do and is what
the first draft of this harness did.

**Why.** That repository grants **no licence** — its `/license` endpoint returns
404 and it ships no LICENSE file. No licence means all rights reserved, so its
files cannot be redistributed. Copying them into a public repository would be a
copyright violation in the second file a reviewer opens, which is a worse
outcome than any benchmark result.

**Verified outcome.** The derivation reproduces all 62 curated questions
verbatim (62/62 exact match on `doc_id` + question text) and additionally
surfaces 28 more questions from the same documents. So the harness is legally
distributable *and* strictly more informative than the source it compares
against.

## 4. Three question sets, not one

**Decision.** Report `comparable_90`, `full_284`, and, where a local reference
copy exists, `curated_62`.

**Why.** One set cannot serve both purposes. `comparable_90` is scoped to
plain-text evidence so it is comparable to published figures and so a wrong
answer means a retrieval failure rather than a chart-reading failure.
`full_284` covers tables, figures, charts and layout — where the two systems may
differ most — but is not comparable to anything published. Reporting only the
first understates the system; reporting only the second makes the comparison
meaningless. Both are reported, with the distinction maintained.

## 5. Semantic judging, not string matching

**Decision.** An LLM judge decides whether the response means the reference
answer.

**Why.** String comparison is broken in a way that is easy to demonstrate:
`Operating Activities` loses to `Operations activities`; `1,358,000` loses to
`1358000`. Both are the same answer. MMLongBench-Doc-V2 replaced its string
metric with a judge for this reason, and its documentation warns that V2 numbers
are therefore not comparable with V1 numbers.

**Guard against bias.** The judge prompt contains no reference to which system
produced the response, and a test asserts that the prompt never contains the
strings `pageindex`, `vector` or `baseline`. The judge model is pinned and shared
across systems, and `bench.report` states whether the judge was in fact identical
across the runs being compared.

**Known cost.** A judge is a model, so it has its own error rate, and a judge
that fails to commit (returns neither YES nor NO in its opening tokens) is
recorded as unscored rather than as a pass. `parse_verdict` reads only the
opening tokens for exactly this reason.

## 6. Retrieval hit rate is reported alongside accuracy

**Decision.** Report the fraction of questions where at least one annotated
evidence page was actually retrieved.

**Why.** Accuracy conflates two failures: not finding the evidence, and finding
it but answering badly. Because the chat model and judge are identical across
systems, a gap in hit rate is a gap in the retrieval strategy — which is the
actual claim under test. Without it, a difference in accuracy would be
unattributable.

**Caveat.** The metric depends on page-level provenance being reported by each
system. PageIndex exposes it through citation tags; the baseline gets it free
because its chunks never span pages. If a system reports no provenance, its hit
rate is `None`, not zero — an absence of evidence is not scored as a failure.

## 7. The corpus is fetched, never committed

**Decision.** `corpus/documents/` and `corpus/annotations/` are gitignored. The
repository ships `corpus/manifest.json` with a SHA-256 per document and refuses
to run on a mismatch.

**Why.** Two reasons at once: redistribution of the documents is avoided, and
reproducibility is improved, because a run can prove which bytes it scored.
Verification was confirmed by the manifest total (113,507,445 bytes across 34
documents), which matches the upstream API's reported total exactly.

## 8. Uncertainty is reported, not buried

**Decision.** Every accuracy figure carries a bootstrap 95% confidence interval,
and a test asserts the interval stays wide at this sample size.

**Why.** At n≈90 the interval is roughly ±7 points, which is wider than most
differences people claim from benchmarks this size. Reporting a bare point
estimate invites exactly the overclaiming this harness exists to avoid. The test
is a guard: if the interval ever narrows dramatically, the interval is broken.
