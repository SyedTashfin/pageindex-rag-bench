"""Download and verify the evaluation documents.

Usage:
    python -m corpus.fetch --docs              # download the 34 documents
    python -m corpus.fetch --verify            # verify against the manifest
    python -m corpus.fetch --write-manifest    # hash on-disk files into manifest.json

The manifest records, per document, its byte size and SHA-256. Verification
fails loudly on any mismatch so a run can never silently score a different
corpus than the one that was published.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import urllib.error
import urllib.request

from . import sources

CHUNK = 1 << 16
USER_AGENT = "pageindex-rag-bench/0.1 (+https://github.com/SyedTashfin/pageindex-rag-bench)"


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def load_manifest() -> dict:
    if not os.path.exists(sources.MANIFEST_PATH):
        raise SystemExit(
            f"manifest not found at {sources.MANIFEST_PATH}. "
            "This file is committed; restore it from git rather than regenerating it."
        )
    with open(sources.MANIFEST_PATH) as fh:
        return json.load(fh)


def doc_ids(manifest: dict) -> list[str]:
    return [d["doc_id"] for d in manifest["documents"]]


def download(url: str, dest: str) -> int:
    """Stream a URL to dest. Returns bytes written."""
    tmp = dest + ".part"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    written = 0
    with urllib.request.urlopen(req, timeout=120) as resp, open(tmp, "wb") as out:
        while True:
            block = resp.read(CHUNK)
            if not block:
                break
            out.write(block)
            written += len(block)
    os.replace(tmp, dest)
    return written


def cmd_docs(manifest: dict, limit: int | None, force: bool) -> int:
    os.makedirs(sources.DOCUMENTS_DIR, exist_ok=True)
    entries = manifest["documents"][: limit or len(manifest["documents"])]
    failures = []
    for i, entry in enumerate(entries, 1):
        doc_id = entry["doc_id"]
        dest = os.path.join(sources.DOCUMENTS_DIR, doc_id)
        if os.path.exists(dest) and not force:
            actual = sha256_file(dest)
            if actual == entry["sha256"]:
                print(f"[{i:>2}/{len(entries)}] ok (cached)  {doc_id}")
                continue
            print(f"[{i:>2}/{len(entries)}] re-fetching (hash mismatch)  {doc_id}")
        try:
            n = download(sources.document_url(doc_id), dest)
        except urllib.error.URLError as exc:
            print(f"[{i:>2}/{len(entries)}] FAILED {doc_id}: {exc}", file=sys.stderr)
            failures.append(doc_id)
            continue
        actual = sha256_file(dest)
        status = "ok" if actual == entry["sha256"] else "HASH MISMATCH"
        print(f"[{i:>2}/{len(entries)}] {status}  {doc_id}  ({n / 1024 / 1024:.1f} MB)")
        if actual != entry["sha256"]:
            failures.append(doc_id)
    if failures:
        print(f"\n{len(failures)} document(s) failed: {failures}", file=sys.stderr)
        return 1
    print(f"\n{len(entries)} documents verified against manifest.")
    return 0


def cmd_verify(manifest: dict) -> int:
    bad = []
    for entry in manifest["documents"]:
        path = os.path.join(sources.DOCUMENTS_DIR, entry["doc_id"])
        if not os.path.exists(path):
            bad.append((entry["doc_id"], "missing"))
            continue
        if os.path.getsize(path) != entry["bytes"]:
            bad.append((entry["doc_id"], "size"))
            continue
        if sha256_file(path) != entry["sha256"]:
            bad.append((entry["doc_id"], "sha256"))
    if bad:
        for doc_id, why in bad:
            print(f"MISMATCH ({why}): {doc_id}", file=sys.stderr)
        return 1
    print(f"all {len(manifest['documents'])} documents verified.")
    return 0


def cmd_write_manifest(manifest: dict) -> int:
    """Re-hash files on disk. Only for the maintainer pinning a new corpus."""
    changed = 0
    for entry in manifest["documents"]:
        path = os.path.join(sources.DOCUMENTS_DIR, entry["doc_id"])
        if not os.path.exists(path):
            print(f"missing, cannot pin: {entry['doc_id']}", file=sys.stderr)
            continue
        size = os.path.getsize(path)
        digest = sha256_file(path)
        if entry.get("bytes") != size or entry.get("sha256") != digest:
            entry["bytes"], entry["sha256"] = size, digest
            changed += 1
    with open(sources.MANIFEST_PATH, "w") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=False)
        fh.write("\n")
    print(f"manifest updated: {changed} entr(ies) changed")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="corpus.fetch", description=__doc__)
    ap.add_argument("--docs", action="store_true", help="download the documents")
    ap.add_argument("--verify", action="store_true", help="verify documents against the manifest")
    ap.add_argument(
        "--write-manifest", action="store_true", help="re-hash files into the manifest"
    )
    ap.add_argument(
        "--limit", type=int, default=None, help="only process the first N (for testing)"
    )
    ap.add_argument("--force", action="store_true", help="re-download even if the hash matches")
    args = ap.parse_args(argv)

    manifest = load_manifest()
    if args.write_manifest:
        return cmd_write_manifest(manifest)
    if args.verify:
        return cmd_verify(manifest)
    if args.docs:
        return cmd_docs(manifest, args.limit, args.force)
    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
