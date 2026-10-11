"""System adapters.

Each adapter answers the same question against the same document, and reports
which pages it retrieved. The interface is deliberately small: `prepare` ingests
a corpus, `answer` produces one response plus its provenance.
"""

from .base import Answer, System, pages_from_citations

__all__ = ["Answer", "System", "pages_from_citations", "pageindex_local", "pgvector_baseline"]
