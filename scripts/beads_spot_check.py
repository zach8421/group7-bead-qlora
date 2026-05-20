"""Build a share-ready BEADs label-audit CSV from the cross-eval adapter
predictions.

Use case
--------
Hypothesis: BEADs is noisy. If the four QLoRA adapters trained on different
bias datasets (BEADs itself, BABE, cajcodes, WNC) collectively disagree
with BEADs's gold label on a row, that ensemble disagreement is a
candidate "BEADs gold looks wrong" signal. The strongest cases are rows
where *all four* adapters agree against the gold — including BEADs's own
QLoRA model, which trained on that gold and at test time refuses to
reproduce it.

Joins the four
``outputs/cross_eval/qlora_<ds>_full__on__beads/predictions.jsonl``
files row-by-row and emits one share-ready CSV plus a brief stdout
summary.

Output columns (in this order)
------------------------------
* row_idx                       — position in BEADs test split (stable ID).
* verdict                       — one of:
    - mislabel_likely_missed_bias       (BEADs said clean, all 4 said biased)
    - mislabel_likely_over_called_bias  (BEADs said biased, all 4 said clean)
    - agree_biased                      (BEADs + all 4 say biased)
    - agree_clean                       (BEADs + all 4 say clean)
    - mixed                             (partial disagreement; check
                                         models_disagreeing_with_gold to
                                         see how partial)
* confidence                    — mean model decisiveness on this row.
                                  Higher = the models were collectively
                                  more confident in their votes (regardless
                                  of which way they voted). Use this to
                                  sort within a verdict bucket.
* text                          — the BEADs sentence.
* gold_label                    — "biased" / "non-biased" — what BEADs says.
* pred_beads, pred_babe,        — "biased" / "non-biased" per adapter.
  pred_cajcodes, pred_wnc
* non_beads_vote                — the consensus of just BABE + cajcodes + WNC
                                  (the BEADs adapter is excluded so this is an
                                  independent vote):
    - "biased"     — all three non-BEADs adapters said biased
    - "non-biased" — all three non-BEADs adapters said non-biased
    - "split"      — they disagreed with each other
* models_disagreeing_with_gold  — 0–4. Higher = more models reject gold.

Sort
----
Strongest mislabel candidates first, then high-confidence rows within
each bucket:
  1. mislabel_likely_missed_bias
  2. mislabel_likely_over_called_bias
  3. mixed (sorted by models_disagreeing_with_gold descending — so
     3-of-4 partial disagreements rise to the top of this bucket)
  4. agree_biased
  5. agree_clean

Within each bucket: confidence descending.

Usage
-----
    python scripts/beads_spot_check.py
    python scripts/beads_spot_check.py --output beads_label_audit.csv

The output path defaults to ``beads_label_audit.csv`` in the repo root.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CELL_DIR = REPO_ROOT / "outputs" / "cross_eval"
ADAPTERS = ["beads", "babe", "cajcodes", "wnc"]
CROSS_ADAPTERS = ["babe", "cajcodes", "wnc"]
LABEL_STR = {0: "non-biased", 1: "biased"}

# Sort priority for the verdict buckets (lower number = appears earlier in CSV).
VERDICT_PRIORITY = {
    "mislabel_likely_missed_bias":      0,
    "mislabel_likely_over_called_bias": 1,
    "mixed":                            2,
    "agree_biased":                     3,
    "agree_clean":                      4,
}


def load_predictions(cell_dir: Path) -> list[dict]:
    path = cell_dir / "predictions.jsonl"
    if not path.is_file():
        sys.exit(f"[spot-check] Missing {path}. Run scripts/cross_eval.py first.")
    out: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def non_beads_consensus(preds: dict[str, int]) -> str:
    """Vote of {babe, cajcodes, wnc} only. BEADs adapter is excluded so the
    result is independent of BEADs's own training signal.

    Returns "biased" / "non-biased" when all three non-BEADs adapters agree,
    "split" when at least one disagrees with the others.
    """
    votes = {preds[ds] for ds in CROSS_ADAPTERS}
    if len(votes) == 1:
        return LABEL_STR[next(iter(votes))]
    return "split"


def classify(gold: int, preds: dict[str, int]) -> str:
    """Bucket a row into one of the five verdicts."""
    n_disagree = sum(1 for ds in ADAPTERS if preds[ds] != gold)
    if n_disagree == 0:
        return "agree_biased" if gold == 1 else "agree_clean"
    if n_disagree == 4:
        return (
            "mislabel_likely_missed_bias" if gold == 0
            else "mislabel_likely_over_called_bias"
        )
    return "mixed"


def per_adapter_margin(row: dict) -> float:
    """How decisive this adapter was on this row, in log-odds units.

    Always positive; equals log p(predicted) - log p(other). Larger = more
    confident pick.
    """
    if row["pred_int"] == 1:
        return float(row["score_biased"] - row["score_non_biased"])
    return float(row["score_non_biased"] - row["score_biased"])


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--cross-eval-dir", default=str(DEFAULT_CELL_DIR),
                    help="Root with qlora_<ds>_full__on__beads subdirs.")
    ap.add_argument("--output", default="beads_label_audit.csv",
                    help="CSV path (relative to repo root if not absolute).")
    args = ap.parse_args()

    root = Path(args.cross_eval_dir)
    preds_by_adapter = {ds: load_predictions(root / f"qlora_{ds}_full__on__beads")
                        for ds in ADAPTERS}

    # All four cells must have aligned row counts and order.
    lengths = {ds: len(p) for ds, p in preds_by_adapter.items()}
    if len(set(lengths.values())) != 1:
        sys.exit(f"[spot-check] Adapter prediction files disagree on row count: {lengths}")
    n = next(iter(lengths.values()))

    rows_out: list[dict] = []
    for i in range(n):
        per_adapter = {ds: preds_by_adapter[ds][i] for ds in ADAPTERS}

        # Sanity: text + gold must align across all four files.
        texts = {ds: per_adapter[ds]["text"] for ds in ADAPTERS}
        if len(set(texts.values())) != 1:
            sys.exit(f"[spot-check] row {i}: text mismatch across adapter files.")
        golds = {ds: per_adapter[ds]["gold_int"] for ds in ADAPTERS}
        if len(set(golds.values())) != 1:
            sys.exit(f"[spot-check] row {i}: gold_int mismatch — {golds}")

        text = next(iter(texts.values()))
        gold = next(iter(golds.values()))
        preds_int = {ds: per_adapter[ds]["pred_int"] for ds in ADAPTERS}
        margins = {ds: per_adapter_margin(per_adapter[ds]) for ds in ADAPTERS}

        verdict = classify(gold, preds_int)
        n_disagree = sum(1 for ds in ADAPTERS if preds_int[ds] != gold)
        mean_conf = sum(margins.values()) / len(margins)

        rows_out.append({
            "row_idx": i,
            "verdict": verdict,
            "confidence": round(mean_conf, 3),
            "text": text,
            "gold_label": LABEL_STR[gold],
            "pred_beads":    LABEL_STR[preds_int["beads"]],
            "pred_babe":     LABEL_STR[preds_int["babe"]],
            "pred_cajcodes": LABEL_STR[preds_int["cajcodes"]],
            "pred_wnc":      LABEL_STR[preds_int["wnc"]],
            "non_beads_vote": non_beads_consensus(preds_int),
            "models_disagreeing_with_gold": n_disagree,
        })

    # Sort:
    #   primary  — verdict priority (most interesting buckets first)
    #   secondary— within mixed bucket, push 3-of-4 partial-disagreements above
    #              1-of-4 noise. For other buckets this is a no-op.
    #   tertiary — confidence descending (high-confidence rows first)
    def sort_key(r):
        return (
            VERDICT_PRIORITY[r["verdict"]],
            -r["models_disagreeing_with_gold"] if r["verdict"] == "mixed" else 0,
            -r["confidence"],
        )
    rows_out.sort(key=sort_key)

    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = REPO_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "row_idx", "verdict", "confidence", "text", "gold_label",
        "pred_beads", "pred_babe", "pred_cajcodes", "pred_wnc",
        "non_beads_vote", "models_disagreeing_with_gold",
    ]
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)

    # Summary.
    bucket_counts = Counter(r["verdict"] for r in rows_out)
    total = len(rows_out)
    print(f"[spot-check] wrote {out_path}  ({total} rows)\n")
    print("Verdict breakdown:")
    for v in sorted(bucket_counts, key=lambda k: VERDICT_PRIORITY[k]):
        count = bucket_counts[v]
        pct = count / total
        print(f"  {v:38s} {count:>5}  ({pct:.1%})")

    # Quick partial-disagreement breakdown inside 'mixed'.
    mixed = [r for r in rows_out if r["verdict"] == "mixed"]
    if mixed:
        print("\nWithin 'mixed', by how many models disagreed with gold:")
        partial = Counter(r["models_disagreeing_with_gold"] for r in mixed)
        for k in sorted(partial, reverse=True):
            print(f"  {k} of 4 disagree: {partial[k]:>5} rows")

    # Non-BEADs-only ensemble: how often the 3 non-BEADs adapters agree, and
    # how their consensus lines up with BEADs gold.
    print("\nnon_beads_vote (vote of just babe + cajcodes + wnc):")
    nb_counts = Counter(r["non_beads_vote"] for r in rows_out)
    for v in ("biased", "non-biased", "split"):
        c = nb_counts.get(v, 0)
        print(f"  {v:>11}: {c:>5}  ({c/total:.1%})")

    print("\nnon_beads_vote vs gold_label (cells where they disagree are candidate mislabels):")
    nb_vs_gold = Counter(
        (r["non_beads_vote"], r["gold_label"]) for r in rows_out
    )
    print(f"  {'non_beads_vote':>14}  vs  {'gold':<11}  count")
    for nb in ("biased", "non-biased", "split"):
        for g in ("biased", "non-biased"):
            count = nb_vs_gold.get((nb, g), 0)
            marker = "  <-- non-BEADs disagree with gold" if (nb != "split" and nb != g) else ""
            print(f"  {nb:>14}  vs  {g:<11}  {count:>5}{marker}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
