"""Benchmark harness for reasoning-based (vectorless) retrieval vs vector RAG.

Two systems, one corpus, one judge: only the retrieval strategy differs, so any
difference in accuracy is attributable to retrieval rather than to the model.
"""

__all__ = ["metrics", "judge", "adapters"]
