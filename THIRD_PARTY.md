# Third-party components and licence hygiene

Every third-party component is listed here with its licence and the *mode* in
which it is used. Nothing is vendored.

## Rules this repository follows

1. **No third-party bytes are committed.** Documents and annotation files are
   downloaded at setup time into `corpus/documents/` and `corpus/annotations/`,
   both `.gitignore`d. The repository ships a fetch script and a manifest of
   SHA-256 hashes, not the data.
2. **Every download is pinned and verified.** `corpus/manifest.json` records
   the expected size and SHA-256 of each document. `corpus/fetch.py` refuses to
   continue on a mismatch.
3. **No file is taken from an unlicensed source.** See the note on
   PageIndex-OSS-Benchmark below.
4. **Attribution is retained for MIT dependencies.** See `NOTICE`.

## Components

| Component | Licence | Mode | Committed? |
| --- | --- | --- | --- |
| [VectifyAI/PageIndex](https://github.com/VectifyAI/PageIndex) | MIT | Python dependency | No |
| [mayubo2333/MMLongBench-Doc](https://github.com/mayubo2333/MMLongBench-Doc) | Apache-2.0 | Documents, fetched at setup | No |
| [VectifyAI/MMLongBench-Doc-V2](https://github.com/VectifyAI/MMLongBench-Doc-V2) | Apache-2.0 | Question annotations, fetched at setup | No |
| [VectifyAI/PageIndex-OSS-Benchmark](https://github.com/VectifyAI/PageIndex-OSS-Benchmark) | **None granted** | Cited as a comparison point only | No |
| pgvector/pgvector | PostgreSQL Licence | Docker image | No |
| litellm | MIT | Python dependency | No |

## The PageIndex-OSS-Benchmark problem

`VectifyAI/PageIndex-OSS-Benchmark` publishes a curated 62-question benchmark
over 34 PDFs, and it is the natural comparison point for this harness. It also
has **no LICENSE file** — verified via the GitHub API, which returns 404 for
`/license`. No licence means all rights reserved: its contents may not be
redistributed.

The earlier version of this harness would therefore have copied
`questions.json` and the 34 PDFs straight into the repository. It does not.

Instead, both the documents and the questions are re-derived from
Apache-2.0 sources, and the derivation is reproducible:

- **Documents** come from `mayubo2333/MMLongBench-Doc`, the Apache-2.0 upstream
  that MMLongBench-Doc-V2 is itself derived from. All 34 of the benchmark's
  `doc_id`s are present there.
- **Questions** are re-derived from `MMLongBench-Doc-V2`'s `samples.json`
  (Apache-2.0) by filtering to those same 34 `doc_id`s. This was verified to
  reproduce **62 of 62** of the curated questions verbatim, plus 222 further
  questions from the same documents that the curated set drops.
- The curated benchmark is then used only as an **external comparison point**:
  its published accuracy and cost numbers appear in `docs/FINDINGS.md` as a
  reference, and no artefact of it is copied into this repository.

This makes the harness legally distributable *and* strictly more informative
than the source it compares against, because the 222 additional questions are
scored too.

## Licence compatibility

This repository is MIT. It depends on MIT and Apache-2.0 components only.
Apache-2.0 code is consumed as a dependency or fetched as data; nothing is
combined into a single derivative work in a way that would require licence
propagation. No AGPL, ELv2, BUSL, or unlicensed component is embedded.
