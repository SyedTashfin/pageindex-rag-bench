"""Corpus acquisition and question-set derivation.

Nothing in this package ships third-party bytes. It downloads the evaluation
documents and annotations from their Apache-2.0 canonical sources, verifies
them against the pinned manifest, and derives the question sets.

See THIRD_PARTY.md at the repository root for the licence reasoning.
"""

__all__ = ["sources", "fetch", "questions"]
