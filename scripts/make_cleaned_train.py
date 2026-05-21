"""Build cleaned BEADs training datasets for the cleaning + retrain experiment.

Background
----------
The cross-eval matrix (2026-05-19) + hand-label scoring (2026-05-20) gave us
two locked decisions:

1. **Flag rule**: a train row is flagged when all three non-BEADs adapters
   (BABE, cajcodes, WNC) unanimously vote against BEADs gold
   (``non_beads_vote != "split" AND non_beads_vote != gold_label``).
   Measured train flag count: 10,371 of 27,263 rows (38.0%).

2. **Cleaning actions to test**: remove flagged rows ("remove") and relabel
   them to the ensemble consensus ("flip"). For each action we also produce
   a class-balanced variant (undersample majority class to 50/50) — so the
   sweep can disentangle "cleaning helped" from "balanced class distribution
   helped."

What this script produces
-------------------------
Four new frozen datasets, each mirroring ``data/frozen/beads/``'s layout
(nested ``sizes/{100,500,1k,5k,full}/{train,val,test}.jsonl`` + a
``splits_manifest.json`` with SHA256s):

  data/frozen/beads_cleaned_remove/             (~16,892 train rows; ~70/30 biased)
  data/frozen/beads_cleaned_remove_balanced/    (~10,362 train rows; 50/50)
  data/frozen/beads_cleaned_flip/               (27,263 train rows; ~74/26 biased)
  data/frozen/beads_cleaned_flip_balanced/      (~14,204 train rows; 50/50)

val.jsonl and test.jsonl in each size dir are byte-identical copies of the
original BEADs val and test (we're cleaning training data, not changing the
evaluation target).

Inputs
------
- ``data/frozen/beads/sizes/full/train.jsonl`` — original train pool (27,263 rows)
- ``data/frozen/beads/sizes/full/val.jsonl``   — copied as-is into every cleaned size dir
- ``data/frozen/beads/sizes/full/test.jsonl``  — copied as-is into every cleaned size dir
- ``outputs/cross_eval/qlora_{babe,cajcodes,wnc}_full__on__beads_train/predictions.jsonl``
  — 3 non-BEADs adapter predictions on the train rows (produced 2026-05-19
  by ``scripts/predict_beads_train_val.slurm``)

Reproducibility
---------------
- Seed 42 (matches the original sweep).
- Stratified nesting via ``carve_nested()`` reused from ``freeze_splits.py``.
- SHA256s emitted per JSONL; same inputs + seed -> byte-identical outputs.

Usage
-----
    python scripts/make_cleaned_train.py
    python scripts/make_cleaned_train.py --dry-run    # print plan, write nothing
    python scripts/make_cleaned_train.py --out-dir /tmp/_cleaned_smoke   # smoke test
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd

# Re-use the originals so we don't rewrite (and accidentally change) their behavior.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.freeze_splits import (  # noqa: E402
    carve_nested,
    class_balance,
    name_for_size,
    set_seed,
    to_records,
    verify_nesting,
    write_jsonl,
)


CROSS_ADAPTERS = ["babe", "cajcodes", "wnc"]
SEED = 42
SWEEP_SIZES = [100, 500, 1000, 5000]  # 'full' is implicit (everything that remains)


def load_train_jsonl(path: Path) -> list[dict]:
    """Load the original BEADs train JSONL preserving row index (== beads_row_idx)."""
    rows: list[dict] = []
    with path.open() as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            rows.append({
                "beads_row_idx": i,
                "text": r["text"],
                "label_int": int(r["label_int"]),
                "label_str": r["label_str"],
            })
    return rows


def load_pred_int_by_idx(cell_dir: Path, n_expected: int) -> dict[int, int]:
    """Load predictions.jsonl from a cross-eval cell; return {row_idx: pred_int}."""
    path = cell_dir / "predictions.jsonl"
    if not path.is_file():
        sys.exit(f"[clean] predictions.jsonl missing at {path} — run "
                 f"scripts/predict_beads_train_val.slurm first.")
    out: dict[int, int] = {}
    with path.open() as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            out[i] = int(json.loads(line)["pred_int"])
    if len(out) != n_expected:
        sys.exit(f"[clean] {path} has {len(out)} rows, expected {n_expected}")
    return out


def compute_flags(rows: list[dict],
                  preds_by_adapter: dict[str, dict[int, int]]) -> list[dict]:
    """For each train row, compute non_beads_vote and the flag bit.

    Returns the same rows with two extra fields:
      non_beads_vote_int : 0 | 1 | -1 (split)
      flagged            : bool

    The flag fires when the three non-BEADs adapters unanimously agree AND
    their consensus disagrees with BEADs gold — exactly the cleaning rule
    locked in the build log.
    """
    out: list[dict] = []
    for r in rows:
        idx = r["beads_row_idx"]
        votes = {preds_by_adapter[ds][idx] for ds in CROSS_ADAPTERS}
        if len(votes) == 1:
            nbv = next(iter(votes))
            flagged = (nbv != r["label_int"])
        else:
            nbv = -1
            flagged = False  # split votes never flag
        out.append({**r, "non_beads_vote_int": nbv, "flagged": flagged})
    return out


def make_remove_pool(rows_with_flag: list[dict]) -> list[dict]:
    """Drop flagged rows, keep labels as-is."""
    return [{k: r[k] for k in ("beads_row_idx", "text", "label_int", "label_str")}
            for r in rows_with_flag if not r["flagged"]]


def make_flip_pool(rows_with_flag: list[dict], label0_text: str, label1_text: str) -> list[dict]:
    """Keep all rows; flagged rows have their label flipped to the ensemble's vote."""
    label_strs = {0: label0_text, 1: label1_text}
    out: list[dict] = []
    for r in rows_with_flag:
        if r["flagged"]:
            new_label = r["non_beads_vote_int"]
            out.append({
                "beads_row_idx": r["beads_row_idx"],
                "text": r["text"],
                "label_int": new_label,
                "label_str": label_strs[new_label],
            })
        else:
            out.append({k: r[k] for k in ("beads_row_idx", "text", "label_int", "label_str")})
    return out


def undersample_to_balanced(rows: list[dict], seed: int) -> list[dict]:
    """Undersample the majority class to match the minority size. Stratified.

    Returns a 50/50 class-balanced pool drawn from ``rows``. Uses a
    deterministic shuffle (seeded RNG) so the same input always yields the
    same output.
    """
    by_class: dict[int, list[dict]] = {0: [], 1: []}
    for r in rows:
        by_class[r["label_int"]].append(r)
    n_min = min(len(by_class[0]), len(by_class[1]))
    if n_min == 0:
        return []
    # Stable seeded shuffle per class to make the undersample deterministic.
    import random
    rng = random.Random(seed)
    for k in (0, 1):
        rng.shuffle(by_class[k])
    return by_class[0][:n_min] + by_class[1][:n_min]


def write_cleaned_dataset(name: str,
                          full_pool: list[dict],
                          original_val: list[dict],
                          original_test: list[dict],
                          out_root: Path,
                          source_label: str,
                          label0_text: str,
                          label1_text: str,
                          seed: int) -> dict:
    """Produce a freezed dataset under ``<out_root>/<name>/sizes/<sz>/{train,val,test}.jsonl``.

    Sizes are 100 / 500 / 1k / 5k carved out as nested stratified subsets of
    ``full_pool``, plus the full pool itself. val + test are written byte-
    identically into every size dir.

    Returns the splits_manifest dict (which is also written to disk).
    """
    print(f"\n[clean] Building dataset '{name}'  (full pool: {len(full_pool)} rows)")
    if not full_pool:
        sys.exit(f"[clean] '{name}' full pool is empty — refusing to write.")

    pool_df = pd.DataFrame(full_pool)
    # The training pool needs at least max(SWEEP_SIZES) + minority-class headroom
    # for stratify to be valid. If even the smallest pool falls short, fail loud.
    if len(pool_df) < max(SWEEP_SIZES):
        sys.exit(f"[clean] '{name}' has {len(pool_df)} rows; can't carve {max(SWEEP_SIZES)}-size subset")

    nested_dfs, nested_order = carve_nested(pool_df, SWEEP_SIZES, seed)

    train_records_by_size: dict[str, list[dict]] = {}
    for size_name, df in nested_dfs.items():
        train_records_by_size[size_name] = to_records(df, label0_text, label1_text)

    verify_nesting(train_records_by_size, nested_order)
    print(f"[clean]   nesting verified: {' -> '.join('train_' + n for n in nested_order)}")

    out_dataset = out_root / name
    sizes_block: dict[str, dict] = {}
    for size_name, train_recs in train_records_by_size.items():
        size_dir = out_dataset / "sizes" / size_name
        train_sha = write_jsonl(train_recs, size_dir / "train.jsonl")
        val_sha = write_jsonl(original_val, size_dir / "val.jsonl")
        test_sha = write_jsonl(original_test, size_dir / "test.jsonl")
        sizes_block[size_name] = {
            "train": {"path": f"sizes/{size_name}/train.jsonl", "sha256": train_sha, **class_balance(train_recs)},
            "val":   {"path": f"sizes/{size_name}/val.jsonl",   "sha256": val_sha,   **class_balance(original_val)},
            "test":  {"path": f"sizes/{size_name}/test.jsonl",  "sha256": test_sha,  **class_balance(original_test)},
        }
        bal = sizes_block[size_name]["train"]
        print(f"[clean]   {size_name}: n={bal['n']:>6}  biased={bal['label_1']:>6}  "
              f"non-biased={bal['label_0']:>6}  ({bal['pct_label_1']:.1%} biased)")

    manifest = {
        "schema_version": 2,
        "dataset": name,
        "seed": seed,
        "source": source_label,
        "label_map": {"0": label0_text, "1": label1_text},
        "split_strategy": "clean_then_nested_stratified_subsets",
        "nesting": " ⊂ ".join("train_" + n for n in nested_order),
        "sizes": sizes_block,
    }
    manifest_path = out_dataset / "splits_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"[clean]   wrote manifest -> {manifest_path}")
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--beads-root", default=str(REPO_ROOT / "data" / "frozen" / "beads"))
    ap.add_argument("--cross-eval-dir", default=str(REPO_ROOT / "outputs" / "cross_eval"))
    ap.add_argument("--out-dir", default=str(REPO_ROOT / "data" / "frozen"))
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--label0-text", default="non-biased")
    ap.add_argument("--label1-text", default="biased")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the planned datasets + counts and exit without writing.")
    args = ap.parse_args()

    set_seed(args.seed)

    beads_root = Path(args.beads_root)
    train_path = beads_root / "sizes" / "full" / "train.jsonl"
    val_path = beads_root / "sizes" / "full" / "val.jsonl"
    test_path = beads_root / "sizes" / "full" / "test.jsonl"
    for p in (train_path, val_path, test_path):
        if not p.is_file():
            sys.exit(f"[clean] missing required input: {p}")

    print(f"[clean] Loading original BEADs train pool from {train_path}")
    rows = load_train_jsonl(train_path)
    print(f"[clean]   {len(rows)} rows loaded")

    # Load val and test as record-lists for byte-identical copy-out.
    def _load_records(path: Path) -> list[dict]:
        out: list[dict] = []
        with path.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
        return out
    val_records = _load_records(val_path)
    test_records = _load_records(test_path)
    print(f"[clean]   val: {len(val_records)} rows, test: {len(test_records)} rows")

    print(f"[clean] Loading non-BEADs adapter predictions on train ...")
    preds_by_adapter: dict[str, dict[int, int]] = {}
    for ds in CROSS_ADAPTERS:
        cell = Path(args.cross_eval_dir) / f"qlora_{ds}_full__on__beads_train"
        preds_by_adapter[ds] = load_pred_int_by_idx(cell, len(rows))
        print(f"[clean]   {ds}: {len(preds_by_adapter[ds])} predictions loaded")

    print("[clean] Computing flag per row ...")
    rows_with_flag = compute_flags(rows, preds_by_adapter)
    n_flagged = sum(1 for r in rows_with_flag if r["flagged"])
    n_unanimous = sum(1 for r in rows_with_flag if r["non_beads_vote_int"] != -1)
    n_flagged_gold0 = sum(1 for r in rows_with_flag
                          if r["flagged"] and r["label_int"] == 0)
    n_flagged_gold1 = sum(1 for r in rows_with_flag
                          if r["flagged"] and r["label_int"] == 1)
    print(f"[clean]   non-BEADs unanimous (any direction): {n_unanimous:>6} / {len(rows)}")
    print(f"[clean]   flagged (unanimous != gold):         {n_flagged:>6} / {len(rows)}  "
          f"({n_flagged/len(rows):.1%})")
    print(f"[clean]     gold=non-biased flagged (BEADs missed bias):  {n_flagged_gold0:>5}")
    print(f"[clean]     gold=biased     flagged (BEADs over-called):  {n_flagged_gold1:>5}")

    print("[clean] Building four cleaned full pools ...")
    pool_remove = make_remove_pool(rows_with_flag)
    pool_flip = make_flip_pool(rows_with_flag, args.label0_text, args.label1_text)
    pool_remove_balanced = undersample_to_balanced(pool_remove, args.seed)
    pool_flip_balanced = undersample_to_balanced(pool_flip, args.seed)

    pool_summary = [
        ("beads_cleaned_remove",          pool_remove,          "cleaning_rule=cross_unanimous_disagree;action=remove"),
        ("beads_cleaned_remove_balanced", pool_remove_balanced, "cleaning_rule=cross_unanimous_disagree;action=remove;then_undersample_to_50_50"),
        ("beads_cleaned_flip",            pool_flip,            "cleaning_rule=cross_unanimous_disagree;action=flip_to_non_beads_vote"),
        ("beads_cleaned_flip_balanced",   pool_flip_balanced,   "cleaning_rule=cross_unanimous_disagree;action=flip_to_non_beads_vote;then_undersample_to_50_50"),
    ]
    print("[clean]   pool sizes (before nesting):")
    for name, pool, _ in pool_summary:
        n_b = sum(1 for r in pool if r["label_int"] == 1)
        n_n = sum(1 for r in pool if r["label_int"] == 0)
        n = len(pool)
        print(f"[clean]     {name:>32s}: {n:>6}  biased={n_b:>6}  non-biased={n_n:>6}  "
              f"({n_b/n:.1%} biased)" if n else f"[clean]     {name:>32s}: empty")

    if args.dry_run:
        print("[clean] --dry-run set; not writing any frozen dirs.")
        return 0

    out_root = Path(args.out_dir)

    for name, pool, source_label in pool_summary:
        # Make sure we're not writing on top of unintended data.
        existing = out_root / name
        if existing.exists():
            print(f"[clean]   removing existing {existing} before rewriting")
            shutil.rmtree(existing)
        write_cleaned_dataset(
            name=name,
            full_pool=pool,
            original_val=val_records,
            original_test=test_records,
            out_root=out_root,
            source_label=source_label,
            label0_text=args.label0_text,
            label1_text=args.label1_text,
            seed=args.seed,
        )

    print("\n[clean] Done. Submit retrain sweep with "
          "scripts/launch_cleaned_retrain_sweep.sh once committed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
