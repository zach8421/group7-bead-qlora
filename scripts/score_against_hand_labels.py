"""Score one or more adapters against the team's hand-label consensus.

After the cleaning + retrain sweep finishes, every adapter writes a
``predictions.jsonl`` against the *noisy* BEADs test set
(``outputs/<run>/predictions.jsonl`` from the on-the-fly slurm eval). The
500 hand-labeled rows are a random subset of that test set, so we can
join by position (the same ``beads_row_idx`` indexing
``score_hand_labels.py`` already uses).

This script consumes the hand-labels + any number of ``predictions.jsonl``
files and emits a small comparison table showing each adapter's accuracy
and F1_macro vs the consensus, on the 500-row hand-labeled subset.

Usage
-----
    # Single adapter
    python scripts/score_against_hand_labels.py \
        --predictions outputs/qlora_beads_cleaned_remove_full/predictions.jsonl

    # All cleaned-sweep adapters (glob)
    python scripts/score_against_hand_labels.py \
        --predictions outputs/qlora_beads_cleaned_*_full/predictions.jsonl \
                      outputs/qlora_beads_cleaned_*_5k/predictions.jsonl

    # Override the hand-label join (defaults to the labeling/ + label-map zach used)
    python scripts/score_against_hand_labels.py \
        --predictions outputs/qlora_beads_full/predictions.jsonl \
        --label-map a=abrevaa b=ash c=zach

Output
------
- stdout: a ranked table (one row per predictions.jsonl) with n,
  accuracy, F1_macro, precision_pos, recall_pos.
- ``hand_label_eval.csv`` (in repo root, configurable) with the same
  table for easy import into spreadsheets / plots.

This script does NOT re-run any model inference. It assumes the
``predictions.jsonl`` files already exist (the slurm runner produces
them as a side-effect of the on-the-fly eval).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


REPO_ROOT = Path(__file__).resolve().parent.parent
ABSTAIN = "abstain"
LABEL_VOCAB = {"biased", "non-biased"}


# --- Hand-label loading (mirrors score_hand_labels.py logic) -----------


def parse_label_map(values: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for v in values:
        if "=" not in v:
            sys.exit(f"[score-vs-hl] --label-map entry '{v}' must be letter=name")
        letter, name = v.split("=", 1)
        out[letter.strip()] = name.strip()
    return out


def load_mapping(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open() as f:
        for r in csv.DictReader(f):
            rows.append({
                "labeler": r["labeler"],
                "labeler_row_id": int(r["labeler_row_id"]),
                "is_iaa": r["is_iaa"] == "1",
                "beads_row_idx": int(r["beads_row_idx"]),
                "gold_label": r["gold_label"],
            })
    return rows


def load_labeled_csv(path: Path) -> dict[int, str]:
    out: dict[int, str] = {}
    with path.open() as f:
        for r in csv.DictReader(f):
            r.pop(None, None)
            try:
                row_id = int(r["id"])
            except (KeyError, ValueError):
                sys.exit(f"[score-vs-hl] {path}: bad id field: {r!r}")
            label = (r.get("label") or "").strip().lower()
            if label and label not in LABEL_VOCAB:
                label = ""
            out[row_id] = label
    return out


def consensus_per_beads_idx(label_map: dict[str, str],
                            mapping: list[dict],
                            labeled_csvs: dict[str, dict[int, str]]) -> dict[int, str]:
    """Same consensus rule as score_hand_labels.py: majority of non-blank
    labels; single-labeler row uses that labeler's label; all-blank -> abstain."""
    by_idx: dict[int, list[str]] = {}
    for r in mapping:
        idx = r["beads_row_idx"]
        letter = r["labeler"]
        labeler_row_id = r["labeler_row_id"]
        label = labeled_csvs[letter].get(labeler_row_id, "")
        by_idx.setdefault(idx, []).append(label)

    out: dict[int, str] = {}
    for idx, labels in by_idx.items():
        votes = Counter(lab for lab in labels if lab)
        if not votes:
            out[idx] = ABSTAIN
        else:
            out[idx] = votes.most_common(1)[0][0]
    return out


# --- Predictions loading -----------------------------------------------


def load_predictions(path: Path) -> dict[int, str]:
    """Read predictions.jsonl; return {row_idx: 'biased'|'non-biased'}.

    Row index is the *position* in the file, which by the eval_adapter.py
    contract == BEADs test row_idx (the script iterates test.jsonl in
    order).
    """
    if not path.is_file():
        sys.exit(f"[score-vs-hl] predictions.jsonl missing: {path}")
    out: dict[int, str] = {}
    with path.open() as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            out[i] = "biased" if int(r["pred_int"]) == 1 else "non-biased"
    return out


# --- Adapter naming for display ----------------------------------------


def adapter_label_from_path(path: Path) -> str:
    """`outputs/qlora_<...>/predictions.jsonl` -> `qlora_<...>`.

    Falls back to the parent directory name otherwise.
    """
    p = Path(path).resolve()
    return p.parent.name


# --- Main --------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--predictions", required=True, nargs="+",
                    help="One or more predictions.jsonl paths (or glob patterns).")
    ap.add_argument("--mapping", default=str(REPO_ROOT / "labeling" / "_mapping.csv"))
    ap.add_argument("--labeling-dir", default=str(REPO_ROOT / "labeling"))
    ap.add_argument("--label-map", default=["a=abrevaa", "b=ash", "c=zach"], nargs="+",
                    metavar="LETTER=NAME",
                    help="Letter->name mapping for the labeled CSVs. "
                         "Default matches the locked experiment.")
    ap.add_argument("--output", default="hand_label_eval.csv",
                    help="CSV output (relative to repo root if not absolute).")
    args = ap.parse_args()

    # 1. Load the consensus once.
    label_map = parse_label_map(args.label_map)
    mapping = load_mapping(Path(args.mapping))
    labeled = {
        letter: load_labeled_csv(Path(args.labeling_dir) / f"labeler_{name}_labeled.csv")
        for letter, name in label_map.items()
    }
    consensus = consensus_per_beads_idx(label_map, mapping, labeled)
    n_total = len(consensus)
    n_non_abstain = sum(1 for v in consensus.values() if v != ABSTAIN)
    print(f"[score-vs-hl] hand-labeled rows: {n_total}  ({n_non_abstain} non-abstain)")

    # 2. Resolve adapter predictions paths.
    pred_paths: list[Path] = []
    for raw in args.predictions:
        if "*" in raw or "?" in raw:
            matches = sorted(REPO_ROOT.glob(raw)) if not Path(raw).is_absolute() else sorted(Path("/").glob(raw[1:]))
            if not matches:
                print(f"[score-vs-hl] WARNING: glob '{raw}' matched nothing", file=sys.stderr)
            pred_paths.extend(matches)
        else:
            pred_paths.append(Path(raw))
    if not pred_paths:
        sys.exit("[score-vs-hl] no predictions.jsonl paths to score.")
    print(f"[score-vs-hl] scoring {len(pred_paths)} adapter(s) against the consensus.")

    # 3. For each adapter, compute the headline numbers on the hand-labeled subset.
    rows_out: list[dict] = []
    for path in pred_paths:
        label = adapter_label_from_path(path)
        preds = load_predictions(path)
        y_true: list[int] = []
        y_pred: list[int] = []
        for idx, cons in consensus.items():
            if cons == ABSTAIN:
                continue
            if idx not in preds:
                print(f"[score-vs-hl] WARNING: {label} has no prediction for beads_row_idx {idx} "
                      f"— this adapter's predictions.jsonl may be incomplete.", file=sys.stderr)
                continue
            y_true.append(1 if cons == "biased" else 0)
            y_pred.append(1 if preds[idx] == "biased" else 0)
        if not y_true:
            print(f"[score-vs-hl] {label}: no overlap with hand-labels; skipping.", file=sys.stderr)
            continue
        rows_out.append({
            "adapter": label,
            "n": len(y_true),
            "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
            "f1_macro": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
            "f1_pos": round(float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)), 4),
            "precision_pos": round(float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)), 4),
            "recall_pos": round(float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)), 4),
            "predictions_path": str(path),
        })

    if not rows_out:
        sys.exit("[score-vs-hl] no rows scored — exiting non-zero so callers notice.")

    # 4. Sort by accuracy descending for an at-a-glance ranking.
    rows_out.sort(key=lambda r: (-r["accuracy"], r["adapter"]))

    # 5. Write CSV.
    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = REPO_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["adapter", "n", "accuracy", "f1_macro", "f1_pos",
                  "precision_pos", "recall_pos", "predictions_path"]
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)
    print(f"[score-vs-hl] wrote {out_path}")

    # 6. Stdout table.
    print()
    print(f"  {'adapter':<46s} {'n':>5s}  {'acc':>6s}  {'f1m':>6s}  {'f1p':>6s}  {'P':>5s}  {'R':>5s}")
    print(f"  {'-'*46} {'-'*5}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*5}  {'-'*5}")
    for r in rows_out:
        print(f"  {r['adapter']:<46s} {r['n']:>5d}  {r['accuracy']:>6.4f}  "
              f"{r['f1_macro']:>6.4f}  {r['f1_pos']:>6.4f}  "
              f"{r['precision_pos']:>5.3f}  {r['recall_pos']:>5.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
