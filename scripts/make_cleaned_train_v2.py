"""Build round-2 cleaned BEADs training datasets using a 4-voter rule.

Background
----------
Round 1 (2026-05-20) cleaned BEADs with a 3-voter rule: a row is flagged
when all three non-BEADs adapters (BABE, cajcodes, WNC) unanimously
disagree with BEADs gold. The flip-correctness rate against the
hand-label consensus was 85% — meaning ~15% of round-1 relabels were
wrong. That ~15% noise floor caused the cleaned_flip_balanced sweep to
peak at 500 training rows and degrade at larger sizes.

Round 2 adds the round-1 winning cleaned model
(``qlora_beads_cleaned_flip_balanced_500``) as a *4th voter*. The
cleaned model has true accuracy 0.77 vs human consensus on BEADs (far
better than any single non-BEADs adapter), so its agreement strengthens
the flag and its disagreement is informative ("the cross-dataset
ensemble wanted to flip this row, but the cleaned model — which knows
BEADs cleanly — says BEADs gold was actually right").

Cleaning variants produced
--------------------------
Both are flip + balanced (the round-1 winning combination):

* ``beads_cleaned_v2_strict``: a row is flagged iff **all 4 voters
  unanimously predict the same label AND that label != BEADs gold**.
  Most conservative. Fewer rows flagged, much cleaner relabels —
  expected wrong-relabel rate ~3-5%.

* ``beads_cleaned_v2_majority``: a row is flagged iff **at least 3
  of 4 voters agree on the same label AND that label != BEADs gold**.
  Wider net, similar to round-1 flag rate (~38%) but with quality boost
  from the 4th voter's confirmation.

Each variant gets undersampled to 50/50 class balance, then nested into
the standard {100, 500, 1k, 5k, full} sizes.

Inputs
------
- ``data/frozen/beads/sizes/full/train.jsonl`` (27,263 rows)
- ``data/frozen/beads/sizes/full/val.jsonl`` (8,520 rows, copied as-is)
- ``data/frozen/beads/sizes/full/test.jsonl`` (6,816 rows, copied as-is)
- ``outputs/cross_eval/qlora_{babe,cajcodes,wnc}_full__on__beads_train/predictions.jsonl``
  (the round-1 3-voter ensemble, produced 2026-05-19)
- ``outputs/cross_eval/qlora_beads_cleaned_flip_balanced_500__on__beads_train/predictions.jsonl``
  (the round-1 winner as 4th voter — produced by
  scripts/predict_round2_voter.slurm)

Outputs
-------
- ``data/frozen/beads_cleaned_v2_strict/`` with nested sizes + manifest
- ``data/frozen/beads_cleaned_v2_majority/`` with nested sizes + manifest

Usage
-----
    python scripts/make_cleaned_train_v2.py
    python scripts/make_cleaned_train_v2.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.freeze_splits import (  # noqa: E402
    carve_nested,
    class_balance,
    set_seed,
    to_records,
    verify_nesting,
    write_jsonl,
)
from scripts.make_cleaned_train import (  # noqa: E402
    SWEEP_SIZES,
    load_pred_int_by_idx,
    load_train_jsonl,
    undersample_to_balanced,
)

ROUND1_VOTERS = ["babe", "cajcodes", "wnc"]
ROUND2_FOURTH_VOTER_RUN = "qlora_beads_cleaned_flip_balanced_500"
SEED = 42


def compute_v2_flags(rows: list[dict],
                     preds_by_voter: dict[str, dict[int, int]]) -> list[dict]:
    """For each row, determine whether the strict / majority rules flag it.

    The voter set is {babe, cajcodes, wnc, cleaned_500}. A flag fires when:

      strict   -> all 4 voters predict the SAME label AND that label != gold
      majority -> at least 3 of 4 voters predict the SAME label AND that
                  label != gold

    When flagged, the new label is the consensus of the agreeing voters.

    Returns the same rows with extra fields:
      strict_flag (bool), strict_label (int | None)
      majority_flag (bool), majority_label (int | None)
    """
    voter_names = list(preds_by_voter.keys())
    out: list[dict] = []
    for r in rows:
        idx = r["beads_row_idx"]
        votes = [preds_by_voter[v][idx] for v in voter_names]
        vote_counter = Counter(votes)
        top_label, top_count = vote_counter.most_common(1)[0]
        gold = r["label_int"]

        # Strict: all 4 unanimous AND consensus != gold
        strict_flag = (top_count == 4 and top_label != gold)
        strict_label = top_label if strict_flag else None

        # Majority: >=3 of 4 same label AND consensus != gold
        majority_flag = (top_count >= 3 and top_label != gold)
        majority_label = top_label if majority_flag else None

        out.append({**r,
                    "strict_flag": strict_flag, "strict_label": strict_label,
                    "majority_flag": majority_flag, "majority_label": majority_label})
    return out


def make_flip_pool_v2(rows_with_flag: list[dict],
                      flag_key: str, label_key: str,
                      label0_text: str, label1_text: str) -> list[dict]:
    """Build a flip pool using the chosen flag/label fields from compute_v2_flags."""
    label_strs = {0: label0_text, 1: label1_text}
    out: list[dict] = []
    for r in rows_with_flag:
        if r[flag_key]:
            new_label = r[label_key]
            out.append({
                "beads_row_idx": r["beads_row_idx"],
                "text": r["text"],
                "label_int": new_label,
                "label_str": label_strs[new_label],
            })
        else:
            out.append({k: r[k] for k in ("beads_row_idx", "text", "label_int", "label_str")})
    return out


def write_cleaned_v2_dataset(name: str,
                             full_pool: list[dict],
                             original_val: list[dict],
                             original_test: list[dict],
                             out_root: Path,
                             source_label: str,
                             label0_text: str,
                             label1_text: str,
                             seed: int) -> dict:
    """Same shape as make_cleaned_train.write_cleaned_dataset, with the v2 schema label."""
    print(f"\n[clean-v2] Building dataset '{name}'  (full pool: {len(full_pool)} rows)")
    if not full_pool:
        sys.exit(f"[clean-v2] '{name}' full pool is empty — refusing to write.")
    pool_df = pd.DataFrame(full_pool)
    if len(pool_df) < max(SWEEP_SIZES):
        sys.exit(f"[clean-v2] '{name}' has {len(pool_df)} rows; can't carve "
                 f"{max(SWEEP_SIZES)}-size subset")

    nested_dfs, nested_order = carve_nested(pool_df, SWEEP_SIZES, seed)
    train_records_by_size: dict[str, list[dict]] = {}
    for size_name, df in nested_dfs.items():
        train_records_by_size[size_name] = to_records(df, label0_text, label1_text)

    verify_nesting(train_records_by_size, nested_order)
    print(f"[clean-v2]   nesting verified: {' -> '.join('train_' + n for n in nested_order)}")

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
        print(f"[clean-v2]   {size_name}: n={bal['n']:>6}  biased={bal['label_1']:>6}  "
              f"non-biased={bal['label_0']:>6}  ({bal['pct_label_1']:.1%} biased)")

    manifest = {
        "schema_version": 2,
        "dataset": name,
        "seed": seed,
        "source": source_label,
        "label_map": {"0": label0_text, "1": label1_text},
        "split_strategy": "round2_4voter_clean_then_undersample_then_nested_stratified",
        "nesting": " ⊂ ".join("train_" + n for n in nested_order),
        "sizes": sizes_block,
    }
    manifest_path = out_dataset / "splits_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"[clean-v2]   wrote manifest -> {manifest_path}")
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
                    help="Print the planned counts + class balances and exit without writing.")
    args = ap.parse_args()

    set_seed(args.seed)

    beads_root = Path(args.beads_root)
    train_path = beads_root / "sizes" / "full" / "train.jsonl"
    val_path   = beads_root / "sizes" / "full" / "val.jsonl"
    test_path  = beads_root / "sizes" / "full" / "test.jsonl"
    for p in (train_path, val_path, test_path):
        if not p.is_file():
            sys.exit(f"[clean-v2] missing required input: {p}")

    print(f"[clean-v2] Loading original BEADs train pool from {train_path}")
    rows = load_train_jsonl(train_path)
    print(f"[clean-v2]   {len(rows)} rows loaded")

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
    print(f"[clean-v2]   val: {len(val_records)} rows, test: {len(test_records)} rows")

    print("[clean-v2] Loading 4 voters' predictions on BEADs train ...")
    preds_by_voter: dict[str, dict[int, int]] = {}
    for ds in ROUND1_VOTERS:
        cell = Path(args.cross_eval_dir) / f"qlora_{ds}_full__on__beads_train"
        preds_by_voter[ds] = load_pred_int_by_idx(cell, len(rows))
        print(f"[clean-v2]   {ds}: {len(preds_by_voter[ds])} predictions")
    cleaned_cell = Path(args.cross_eval_dir) / f"{ROUND2_FOURTH_VOTER_RUN}__on__beads_train"
    preds_by_voter["cleaned_500"] = load_pred_int_by_idx(cleaned_cell, len(rows))
    print(f"[clean-v2]   cleaned_500 ({ROUND2_FOURTH_VOTER_RUN}): "
          f"{len(preds_by_voter['cleaned_500'])} predictions")

    print("[clean-v2] Computing strict + majority flags ...")
    rows_with_flag = compute_v2_flags(rows, preds_by_voter)
    n_strict = sum(1 for r in rows_with_flag if r["strict_flag"])
    n_majority = sum(1 for r in rows_with_flag if r["majority_flag"])
    n_round1 = sum(  # for comparison: round-1 flag (3-voter unanimous against gold)
        1 for r in rows_with_flag
        if len({preds_by_voter[v][r["beads_row_idx"]] for v in ROUND1_VOTERS}) == 1
        and next(iter({preds_by_voter[v][r["beads_row_idx"]] for v in ROUND1_VOTERS})) != r["label_int"]
    )
    print(f"[clean-v2]   round-1 flag (3-voter unanimous != gold): {n_round1:>6} / {len(rows)}  "
          f"({n_round1/len(rows):.1%})  ← for comparison")
    print(f"[clean-v2]   v2 strict   (all 4 unanimous != gold):    {n_strict:>6} / {len(rows)}  "
          f"({n_strict/len(rows):.1%})")
    print(f"[clean-v2]   v2 majority (>=3 of 4 same != gold):       {n_majority:>6} / {len(rows)}  "
          f"({n_majority/len(rows):.1%})")

    # The 4th voter's vetoes: rows the round-1 rule would have flagged but
    # the cleaned model "vetoes" (because it agrees with gold instead).
    n_vetoed = sum(
        1 for r in rows_with_flag
        if len({preds_by_voter[v][r["beads_row_idx"]] for v in ROUND1_VOTERS}) == 1
        and next(iter({preds_by_voter[v][r["beads_row_idx"]] for v in ROUND1_VOTERS})) != r["label_int"]
        and preds_by_voter["cleaned_500"][r["beads_row_idx"]] == r["label_int"]
    )
    print(f"[clean-v2]   4th voter vetoes (round-1 flag → cleaned says gold OK): "
          f"{n_vetoed:>6} rows  ({n_vetoed/max(1,n_round1):.1%} of round-1 flags)")

    print("[clean-v2] Building cleaned full pools (flip then balanced) ...")
    pool_strict_flip = make_flip_pool_v2(rows_with_flag, "strict_flag", "strict_label",
                                          args.label0_text, args.label1_text)
    pool_majority_flip = make_flip_pool_v2(rows_with_flag, "majority_flag", "majority_label",
                                            args.label0_text, args.label1_text)
    pool_strict_balanced = undersample_to_balanced(pool_strict_flip, args.seed)
    pool_majority_balanced = undersample_to_balanced(pool_majority_flip, args.seed)

    pool_summary = [
        ("beads_cleaned_v2_strict",   pool_strict_balanced,
         "round2_cleaning_rule=4_voter_unanimous;action=flip;then_balance"),
        ("beads_cleaned_v2_majority", pool_majority_balanced,
         "round2_cleaning_rule=4_voter_majority_3_of_4;action=flip;then_balance"),
    ]
    print("[clean-v2]   pool sizes (after flip + balance):")
    for name, pool, _ in pool_summary:
        if not pool:
            print(f"[clean-v2]     {name:>30s}: EMPTY")
            continue
        n_b = sum(1 for r in pool if r["label_int"] == 1)
        n_n = sum(1 for r in pool if r["label_int"] == 0)
        print(f"[clean-v2]     {name:>30s}: {len(pool):>6}  "
              f"biased={n_b:>6}  non-biased={n_n:>6}  ({n_b/len(pool):.1%} biased)")

    if args.dry_run:
        print("[clean-v2] --dry-run set; not writing any frozen dirs.")
        return 0

    out_root = Path(args.out_dir)
    for name, pool, source_label in pool_summary:
        existing = out_root / name
        if existing.exists():
            print(f"[clean-v2]   removing existing {existing} before rewriting")
            shutil.rmtree(existing)
        write_cleaned_v2_dataset(
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

    print("\n[clean-v2] Done. Submit retrain sweep with "
          "scripts/launch_cleaned_retrain_v2_sweep.sh once committed and push-data'd.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
