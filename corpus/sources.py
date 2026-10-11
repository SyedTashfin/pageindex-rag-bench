"""Pinned upstream sources.

Every artefact this harness uses is fetched from one of these. All are
Apache-2.0. Nothing is taken from an unlicensed source; see THIRD_PARTY.md.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCUMENTS_DIR = os.path.join(REPO_ROOT, "corpus", "documents")
ANNOTATIONS_DIR = os.path.join(REPO_ROOT, "corpus", "annotations")
MANIFEST_PATH = os.path.join(REPO_ROOT, "corpus", "manifest.json")


@dataclass(frozen=True)
class Source:
    """An upstream artefact: where it lives, how to fetch it, and its licence."""

    repo: str
    ref: str
    license: str
    note: str

    def raw_url(self, path: str) -> str:
        return f"https://raw.githubusercontent.com/{self.repo}/{self.ref}/{path}"


# Canonical source of the 135 evaluation documents. Apache-2.0.
# Chosen over the curated benchmark's copy because the curated copy carries no
# licence grant at all (its /license endpoint returns 404).
DOCUMENTS = Source(
    repo="mayubo2333/MMLongBench-Doc",
    ref="main",
    license="Apache-2.0",
    note="Upstream of MMLongBench-Doc; canonical, licensed source of the PDFs.",
)

# Canonical source of the question annotations. Apache-2.0.
# Derived from DOCUMENTS' upstream, and it documents every correction it made.
ANNOTATIONS = Source(
    repo="VectifyAI/MMLongBench-Doc-V2",
    ref="main",
    license="Apache-2.0",
    note="Corrected-annotation release of MMLongBench-Doc; provides samples.json.",
)

ANNOTATIONS_PATH = "data/samples.json"
DOCUMENTS_PATH = "data/documents"

# The comparison point. NO LICENCE IS GRANTED by this repository, so no file is
# taken from it. Its published accuracy/cost numbers are cited in
# docs/FINDINGS.md as an external reference only.
COMPARISON_REPO = "VectifyAI/PageIndex-OSS-Benchmark"
COMPARISON_LICENSE = None


def document_url(doc_id: str) -> str:
    return DOCUMENTS.raw_url(f"{DOCUMENTS_PATH}/{doc_id}")


def annotations_url() -> str:
    return ANNOTATIONS.raw_url(ANNOTATIONS_PATH)
