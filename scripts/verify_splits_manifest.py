"""Verify frozen-split JSONLs match a reference splits_manifest.json.

The committed `data/frozen/splits_manifest.json` is the source of truth for the
sweep: the QLoRA, TF-IDF, and 3-shot baselines all evaluate on the same
`test_held_out.jsonl` bytes. This script re-hashes every split JSONL on disk
and compares against the reference manifest's `sha256` fields.

Exit codes
----------
* 0 — every split listed in the reference manifest is present on disk and
      hashes to the recorded value (extra files in the dir are ignored).
* 2 — any mismatch, missing split, or unreadable file. Prints a structured
      diff on stderr.

Usage
-----
    python scripts/verify_splits_manifest.py \\
        --frozen-dir data/frozen \\
        --reference-manifest data/frozen/splits_manifest.json

In the Tillicum slurm launcher the reference manifest is snapshotted before
freeze_splits.py runs, so this script can detect a regenerated split that
drifted from the committed bytes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frozen-dir", required=True,
                    help="Directory containing the JSONL splits to verify.")
    ap.add_argument("--reference-manifest", required=True,
                    help="Path to the splits_manifest.json that defines expected hashes.")
    args = ap.parse_args()

    frozen_dir = Path(args.frozen_dir)
    ref_path = Path(args.reference_manifest)

    if not frozen_dir.is_dir():
        print(f"[verify] frozen-dir does not exist: {frozen_dir}", file=sys.stderr)
        return 2
    if not ref_path.is_file():
        print(f"[verify] reference manifest does not exist: {ref_path}", file=sys.stderr)
        return 2

    ref = json.loads(ref_path.read_text())

    # Flatten both schemas into a list of (name, path, sha256) triples.
    #   v1: ref["splits"][<name>] = {"path": ..., "sha256": ...}
    #   v2: ref["sizes"][<size>][<role>] = {"path": ..., "sha256": ...}
    triples: list[tuple[str, str, str | None]] = []
    if ref.get("splits"):
        for name, info in ref["splits"].items():
            triples.append((name, info.get("path"), info.get("sha256")))
    elif ref.get("sizes"):
        for size_name, size_block in ref["sizes"].items():
            for role, info in size_block.items():
                triples.append((f"{size_name}/{role}", info.get("path"), info.get("sha256")))
    else:
        print(
            f"[verify] reference manifest has no 'splits' (v1) or 'sizes' (v2) entries: {ref_path}",
            file=sys.stderr,
        )
        return 2

    mismatches: list[str] = []
    missing: list[str] = []
    ok: list[str] = []

    for name, rel, expected in triples:
        if not rel or not expected:
            mismatches.append(f"{name}: manifest entry missing 'path' or 'sha256'")
            continue
        path = frozen_dir / rel
        if not path.is_file():
            missing.append(f"{name}: expected file {path} not found")
            continue
        actual = sha256_of_file(path)
        if actual != expected:
            mismatches.append(
                f"{name}: sha256 mismatch\n"
                f"    file:     {path}\n"
                f"    expected: {expected}\n"
                f"    actual:   {actual}"
            )
        else:
            ok.append(f"{name}: {path.name} ✓")

    if not mismatches and not missing:
        for line in ok:
            print(f"[verify] {line}")
        print(f"[verify] OK — all {len(ok)} splits match {ref_path}")
        return 0

    print(f"[verify] FAILED against {ref_path}", file=sys.stderr)
    for line in ok:
        print(f"[verify]   ok: {line}", file=sys.stderr)
    for line in missing:
        print(f"[verify]   MISSING: {line}", file=sys.stderr)
    for line in mismatches:
        print(f"[verify]   MISMATCH: {line}", file=sys.stderr)
    print(
        "[verify] The on-disk splits drifted from the committed manifest. "
        "Either regenerate via scripts/freeze_splits.py with the same args, "
        "or update the committed manifest if the drift is intentional.",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    sys.exit(main())
