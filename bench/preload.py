"""Work around a concurrent first-import of litellm.

PageIndex starts a background thread to preload litellm, because importing it
takes seconds and the SDK does not want to block on it. Any *other* thread that
imports litellm at the same time races inside importlib's machinery and dies
with::

    File ".../importlib/_bootstrap_external.py", line 1355, in _get_parent_path
    KeyError: 'litellm'

That failure is not a normal exception — it happens while the import lock and
`sys.modules` are mid-mutation, so it can wedge every thread, not just the one
that hit it. Observed in practice: the demo server stopped answering any request
at all, because the indexing thread was importing litellm for the pgvector
baseline while PageIndex's preload thread was doing the same.

Importing litellm once here, in the main thread, before any worker thread is
started, removes the race: both later imports become lookups in `sys.modules`.

This is deliberately not in `bench/__init__.py`. CI installs only pytest, numpy
and ruff, on purpose, so the scoring tests need no model stack; a module-level
litellm import there would break that.
"""

from __future__ import annotations


def preload_litellm() -> None:
    """Import litellm in the calling thread. Idempotent."""
    import litellm  # noqa: F401
