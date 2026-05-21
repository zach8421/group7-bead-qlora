"""Score the BEADs hand-labeling results.

Inputs
------
- ``labeling/_mapping.csv`` — private join file produced by
  ``scripts/make_labeling_csvs.py``. Keyed by (labeler letter, labeler_row_id),
  resolves to (beads_row_idx, gold_label, non_beads_vote, is_iaa).
- ``labeling/labeler_<name>_labeled.csv`` × 3 — the team's hand-labels.
  Columns: id, text, label (label is one of "biased" / "non-biased" / blank).
- ``outputs/cross_eval/qlora_<ds>_full__on__beads/predictions.jsonl`` × 4 —
  per-adapter predictions on the BEADs test split (position-indexed by
  BEADs test row_idx).
- ``data/frozen/beads/sizes/full/test.jsonl`` — for the canonical text per
  row in the per-row CSV output.

Outputs
-------
- ``hand_label_scoring_per_row.csv`` (repo root) — 500 rows of per-row
  diagnostics (one row per BEADs row in the hand-labeled sample, with the
  50 IAA rows deduped to one row each).
- ``hand_label_scoring_summary.json`` (repo root) — IAA pairwise agreement
  + Cohen's kappa, BEADs mislabel rate (with directional breakdown +
  contingency table), qlora_beads_full true accuracy, ensemble
  flip-correctness on flagged rows, and the two gate decisions.
- stdout: a clean headline-numbers summary.

Decision branches encoded
-------------------------
- ``gate_iaa_pass = min(pairwise raw %) >= 70`` (per the locked plan).
- ``gate_flip_branch`` is one of:
    - ``remove_and_flip``  (flip-correctness >= 70%)
    - ``remove_only``      (55-70%)
    - ``skip``             (< 55%)

The script does NOT take any action on the gate decisions — it just reports
them. The user reads the summary and decides whether to launch the retrain
slurm.

Usage
-----
    python scripts/score_hand_labels.py --label-map a=abrevaa b=ash c=zach
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

from sklearn.metrics import (accuracy_score, cohen_kappa_score, f1_score,
                             precision_score, recall_score)


REPO_ROOT = Path(__file__).resolve().parent.parent
ADAPTERS = ["beads", "babe", "cajcodes", "wnc"]
LABEL_VOCAB = {"biased", "non-biased"}
ABSTAIN = "abstain"


def normalize_whitespace(text: str) -> str:
    """Same helper as scripts/make_labeling_csvs.py — collapse all whitespace
    to single spaces so the per-row CSV reads cleanly in spreadsheet tools."""
    return " ".join(text.split())


def parse_label_map(values: list[str]) -> dict[str, str]:
    """Parse a list of ``letter=name`` strings into {letter: name}."""
    out: dict[str, str] = {}
    for v in values:
        if "=" not in v:
            sys.exit(f"[score] --label-map entry '{v}' must be of the form letter=name")
        letter, name = v.split("=", 1)
        letter = letter.strip()
        name = name.strip()
        if not letter or not name:
            sys.exit(f"[score] --label-map entry '{v}' has an empty letter or name")
        if letter in out:
            sys.exit(f"[score] --label-map repeats letter '{letter}'")
        out[letter] = name
    return out


def load_mapping(path: Path) -> list[dict]:
    """Read labeling/_mapping.csv into a list of dicts."""
    if not path.is_file():
        sys.exit(f"[score] mapping file not found: {path}")
    rows: list[dict] = []
    with path.open() as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                "labeler": r["labeler"],
                "labeler_row_id": int(r["labeler_row_id"]),
                "is_iaa": r["is_iaa"] == "1",
                "beads_row_idx": int(r["beads_row_idx"]),
                "gold_label": r["gold_label"],
                "non_beads_vote": r["non_beads_vote"],
            })
    return rows


def load_labeled_csv(path: Path) -> dict[int, str]:
    """Read a single labeler's labeled CSV. Returns {labeler_row_id: label_str}.

    Labels are normalized to lowercase. Blank ('' or whitespace only) maps to
    the empty string. ``csv.DictReader`` can produce a ``None`` field name
    when a CSV has a trailing comma in the header (ash's CSV has this) —
    we just ignore that phantom column.
    """
    if not path.is_file():
        sys.exit(f"[score] labeled CSV not found: {path}")
    out: dict[int, str] = {}
    with path.open() as f:
        reader = csv.DictReader(f)
        for r in reader:
            # Defensive: pop None-keyed phantom column from trailing-comma headers.
            r.pop(None, None)
            try:
                row_id = int(r["id"])
            except (KeyError, ValueError):
                sys.exit(f"[score] {path}: row missing/invalid 'id' field: {r!r}")
            label = (r.get("label") or "").strip().lower()
            if label and label not in LABEL_VOCAB:
                print(f"[score] WARNING: {path.name} row {row_id} has unexpected label "
                      f"{label!r}; treating as blank.", file=sys.stderr)
                label = ""
            out[row_id] = label
    return out


def load_predictions(cell_dir: Path) -> list[dict]:
    """Read predictions.jsonl from an outputs/cross_eval/<cell>/ dir.

    Same pattern as scripts/beads_spot_check.py:load_predictions. The list is
    position-indexed: index i == BEADs test row_idx i.
    """
    path = cell_dir / "predictions.jsonl"
    if not path.is_file():
        sys.exit(f"[score] predictions.jsonl missing at {path}")
    out: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def load_beads_test_text(path: Path) -> dict[int, str]:
    """Load BEADs test JSONL and return {row_idx: normalized_text}."""
    if not path.is_file():
        sys.exit(f"[score] BEADs test JSONL not found: {path}")
    out: dict[int, str] = {}
    with path.open() as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            out[i] = normalize_whitespace(r["text"])
    return out


def consensus_for_row(labels_by_labeler: dict[str, str]) -> str:
    """Compute consensus from the non-blank labels of (up to) 3 labelers.

    - 2 or 3 non-blank labels: majority of non-blank. With 3 binary raters a
      genuine 2-vs-1 tie is impossible (and we never end up with 2 vs 2).
    - 1 non-blank label: use it (same signal quality as a unique-row label).
    - 0 non-blank labels: ``ABSTAIN``. Excluded from downstream metrics.
    """
    votes = Counter(lab for lab in labels_by_labeler.values() if lab)
    if not votes:
        return ABSTAIN
    top_label, _ = votes.most_common(1)[0]
    return top_label


def pairwise_pct_agreement(a: list[str], b: list[str]) -> tuple[float, int]:
    """Return (pct agreement, n_pairs used) — pairs are only counted when
    both labelers gave a non-blank label."""
    pairs = [(x, y) for x, y in zip(a, b) if x and y]
    if not pairs:
        return float("nan"), 0
    agree = sum(1 for x, y in pairs if x == y)
    return agree / len(pairs), len(pairs)


def pairwise_kappa(a: list[str], b: list[str]) -> float:
    """Cohen's kappa over rows where both rated. Returns float('nan') if no
    common rows."""
    pairs = [(x, y) for x, y in zip(a, b) if x and y]
    if not pairs:
        return float("nan")
    xs = [x for x, _ in pairs]
    ys = [y for _, y in pairs]
    # sklearn returns 0 (and warns) when there's only one class observed by
    # one rater; that's an honest result for a degenerate case.
    return float(cohen_kappa_score(xs, ys, labels=sorted(LABEL_VOCAB)))


def three_way_unanimous_pct(rows: list[dict], letters: list[str]) -> tuple[float, int]:
    """% of IAA rows where all 3 labelers gave the same non-blank label.

    Denominator is rows where all 3 labelers gave a non-blank label.
    """
    denom = 0
    agree = 0
    for r in rows:
        labs = [r["labels"][L] for L in letters]
        if all(labs):
            denom += 1
            if len(set(labs)) == 1:
                agree += 1
    if denom == 0:
        return float("nan"), 0
    return agree / denom, denom


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--label-map", required=True, nargs="+",
                    metavar="LETTER=NAME",
                    help='Required. Map labeler letter (a/b/c in _mapping.csv) '
                         'to filename suffix in labeling/labeler_<name>_labeled.csv. '
                         'Example: --label-map a=abrevaa b=ash c=zach')
    ap.add_argument("--mapping", default=str(REPO_ROOT / "labeling" / "_mapping.csv"))
    ap.add_argument("--labeling-dir", default=str(REPO_ROOT / "labeling"))
    ap.add_argument("--cross-eval-dir", default=str(REPO_ROOT / "outputs" / "cross_eval"))
    ap.add_argument("--beads-test", default=str(REPO_ROOT / "data" / "frozen" / "beads"
                                                / "sizes" / "full" / "test.jsonl"))
    ap.add_argument("--output-prefix", default="hand_label_scoring",
                    help='Outputs land at <prefix>_per_row.csv and <prefix>_summary.json '
                         'in the repo root (matches beads_label_audit.csv convention).')
    args = ap.parse_args()

    label_map = parse_label_map(args.label_map)  # letter -> name
    letters = sorted(label_map.keys())

    # 1. Load mapping.
    mapping_rows = load_mapping(Path(args.mapping))

    # Sanity: every labeler letter present in the mapping must also appear in --label-map.
    mapping_letters = sorted({r["labeler"] for r in mapping_rows})
    missing = [L for L in mapping_letters if L not in label_map]
    if missing:
        sys.exit(f"[score] --label-map does not cover labeler letters present in mapping: {missing}. "
                 f"Mapping letters: {mapping_letters}, --label-map letters: {letters}")
    extras = [L for L in letters if L not in mapping_letters]
    if extras:
        print(f"[score] WARNING: --label-map has extra letters not in mapping: {extras}",
              file=sys.stderr)

    # 2. Load labeled CSVs (one per letter).
    labeled: dict[str, dict[int, str]] = {}
    for letter, name in label_map.items():
        path = Path(args.labeling_dir) / f"labeler_{name}_labeled.csv"
        labeled[letter] = load_labeled_csv(path)
        n_blank = sum(1 for v in labeled[letter].values() if not v)
        print(f"[score] loaded labeler {letter} ({name}): {len(labeled[letter])} rows, "
              f"{n_blank} blank/abstain")

    # 3. Build per-BEADs-row records (de-dup IAA rows).
    by_beads_idx: dict[int, dict] = {}
    for r in mapping_rows:
        idx = r["beads_row_idx"]
        letter = r["labeler"]
        labeler_row_id = r["labeler_row_id"]
        label = labeled[letter].get(labeler_row_id, "")
        if idx not in by_beads_idx:
            by_beads_idx[idx] = {
                "beads_row_idx": idx,
                "is_iaa": r["is_iaa"],
                "gold_label": r["gold_label"],
                "non_beads_vote": r["non_beads_vote"],
                "labels": {L: "" for L in letters},  # populated below
            }
        by_beads_idx[idx]["labels"][letter] = label

    n_unique = len(by_beads_idx)
    n_iaa = sum(1 for r in by_beads_idx.values() if r["is_iaa"])
    n_nonblank_per_iaa_row = [
        sum(1 for v in r["labels"].values() if v)
        for r in by_beads_idx.values() if r["is_iaa"]
    ]
    print(f"[score] unique BEADs rows in sample: {n_unique}")
    print(f"[score]   IAA-shared rows: {n_iaa}")
    print(f"[score]   non-IAA rows:    {n_unique - n_iaa}")
    if n_iaa > 0:
        print(f"[score]   non-blank labels per IAA row: "
              f"mean={statistics.mean(n_nonblank_per_iaa_row):.2f}, "
              f"min={min(n_nonblank_per_iaa_row)}, max={max(n_nonblank_per_iaa_row)}")

    # 4. Compute consensus per row.
    for r in by_beads_idx.values():
        r["consensus_label"] = consensus_for_row(r["labels"])

    n_abstain = sum(1 for r in by_beads_idx.values() if r["consensus_label"] == ABSTAIN)
    n_non_abstain = n_unique - n_abstain
    print(f"[score]   consensus: {n_non_abstain} non-abstain, {n_abstain} abstain")

    # 5. IAA on the 50 IAA-shared rows.
    iaa_rows = [r for r in by_beads_idx.values() if r["is_iaa"]]
    pairwise_agree: dict[str, dict] = {}
    pairwise_kappa_results: dict[str, float] = {}
    for i, L1 in enumerate(letters):
        for L2 in letters[i + 1:]:
            a = [r["labels"][L1] for r in iaa_rows]
            b = [r["labels"][L2] for r in iaa_rows]
            pct, n = pairwise_pct_agreement(a, b)
            pairwise_agree[f"{L1}-{L2}"] = {"pct": pct, "n_pairs": n}
            pairwise_kappa_results[f"{L1}-{L2}"] = pairwise_kappa(a, b)

    unanimous_pct, unanimous_n = three_way_unanimous_pct(iaa_rows, letters)

    pairwise_pcts = [pa["pct"] for pa in pairwise_agree.values()
                     if not (pa["pct"] != pa["pct"])]  # filter NaN
    gate_iaa_pass = bool(pairwise_pcts) and min(pairwise_pcts) >= 0.70

    # 6. BEADs mislabel rate against consensus (non-abstain rows only).
    contingency: dict[str, int] = Counter()
    missed_bias = 0    # gold=non-biased, consensus=biased
    over_called = 0    # gold=biased,     consensus=non-biased
    mislabel_count = 0
    for r in by_beads_idx.values():
        if r["consensus_label"] == ABSTAIN:
            continue
        g = r["gold_label"]
        c = r["consensus_label"]
        contingency[f"gold={g},consensus={c}"] += 1
        if g != c:
            mislabel_count += 1
            if g == "non-biased" and c == "biased":
                missed_bias += 1
            elif g == "biased" and c == "non-biased":
                over_called += 1
    beads_mislabel_rate = mislabel_count / n_non_abstain if n_non_abstain else float("nan")

    # 7. qlora_beads_full true accuracy on hand-labels (non-abstain only).
    beads_preds_list = load_predictions(Path(args.cross_eval_dir) / "qlora_beads_full__on__beads")
    # Position index of predictions.jsonl == beads_row_idx in test.
    beads_pred_by_idx: dict[int, str] = {}
    for i, p in enumerate(beads_preds_list):
        beads_pred_by_idx[i] = "biased" if p["pred_int"] == 1 else "non-biased"

    # Also load the other adapters' preds for the per-row CSV (and just-in-case).
    pred_by_adapter_by_idx: dict[str, dict[int, str]] = {}
    for ds in ADAPTERS:
        cells = load_predictions(Path(args.cross_eval_dir) / f"qlora_{ds}_full__on__beads")
        pred_by_adapter_by_idx[ds] = {
            i: ("biased" if p["pred_int"] == 1 else "non-biased") for i, p in enumerate(cells)
        }

    # Score qlora_beads_full vs consensus, on non-abstain rows.
    y_true_int: list[int] = []
    y_pred_int: list[int] = []
    for r in by_beads_idx.values():
        if r["consensus_label"] == ABSTAIN:
            continue
        y_true_int.append(1 if r["consensus_label"] == "biased" else 0)
        y_pred_int.append(1 if pred_by_adapter_by_idx["beads"][r["beads_row_idx"]] == "biased" else 0)
    qlora_beads_full_accuracy = float(accuracy_score(y_true_int, y_pred_int))
    qlora_beads_full_precision_pos = float(precision_score(y_true_int, y_pred_int, pos_label=1, zero_division=0))
    qlora_beads_full_recall_pos = float(recall_score(y_true_int, y_pred_int, pos_label=1, zero_division=0))
    qlora_beads_full_f1_pos = float(f1_score(y_true_int, y_pred_int, pos_label=1, zero_division=0))
    qlora_beads_full_f1_macro = float(f1_score(y_true_int, y_pred_int, average="macro", zero_division=0))

    # 8. Ensemble flip-correctness on flagged rows.
    # Flagged = (non_beads_vote != "split") AND (non_beads_vote != gold_label).
    # On flagged AND non-abstain rows: how often does non_beads_vote == consensus?
    flagged_idxs: list[int] = []
    flip_match = 0
    flip_total = 0
    for r in by_beads_idx.values():
        nbv = r["non_beads_vote"]
        if nbv == "split" or nbv == r["gold_label"]:
            continue
        flagged_idxs.append(r["beads_row_idx"])
        if r["consensus_label"] == ABSTAIN:
            continue
        flip_total += 1
        if nbv == r["consensus_label"]:
            flip_match += 1
    flip_correctness_rate = flip_match / flip_total if flip_total else float("nan")

    if flip_correctness_rate != flip_correctness_rate:  # NaN
        gate_flip_branch = "skip"
    elif flip_correctness_rate >= 0.70:
        gate_flip_branch = "remove_and_flip"
    elif flip_correctness_rate >= 0.55:
        gate_flip_branch = "remove_only"
    else:
        gate_flip_branch = "skip"

    # 9. Write per-row CSV.
    text_by_idx = load_beads_test_text(Path(args.beads_test))
    per_row_path = REPO_ROOT / f"{args.output_prefix}_per_row.csv"
    fieldnames = [
        "beads_row_idx", "text", "is_iaa", "gold_label",
        *[f"labeler_{letter}_label" for letter in letters],
        "consensus_label", "gold_matches_consensus",
        "pred_beads", "pred_babe", "pred_cajcodes", "pred_wnc",
        "non_beads_vote", "flagged_for_cleaning", "flip_matches_consensus",
    ]
    # Sort by beads_row_idx so the file is deterministic.
    rows_for_csv: list[dict] = []
    for r in sorted(by_beads_idx.values(), key=lambda x: x["beads_row_idx"]):
        idx = r["beads_row_idx"]
        consensus = r["consensus_label"]
        nbv = r["non_beads_vote"]
        flagged = (nbv != "split") and (nbv != r["gold_label"])
        if flagged and consensus != ABSTAIN:
            flip_match_this = (nbv == consensus)
        else:
            flip_match_this = ""  # not applicable
        out_row = {
            "beads_row_idx": idx,
            "text": text_by_idx.get(idx, ""),
            "is_iaa": int(r["is_iaa"]),
            "gold_label": r["gold_label"],
            **{f"labeler_{L}_label": r["labels"][L] for L in letters},
            "consensus_label": consensus,
            "gold_matches_consensus": ("" if consensus == ABSTAIN
                                       else int(r["gold_label"] == consensus)),
            **{f"pred_{ds}": pred_by_adapter_by_idx[ds][idx] for ds in ADAPTERS},
            "non_beads_vote": nbv,
            "flagged_for_cleaning": int(flagged),
            "flip_matches_consensus": flip_match_this if flip_match_this == "" else int(flip_match_this),
        }
        rows_for_csv.append(out_row)

    with per_row_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_for_csv)
    print(f"[score] wrote {per_row_path}  ({len(rows_for_csv)} rows)")

    # 9b. Quick-look CSV: just the 50 IAA rows, all three labels side-by-side,
    # using labeler NAMES (not letters) and sorted with outlier rows at the top.
    # Intended for the recalibration discussion.
    def outlier_category(r: dict) -> tuple[int, str]:
        """Return (sort_priority, label) for the outlier-pattern of this IAA row.

        Lower priority sorts first. Priorities:
          0: one labeler disagrees with the other two who both labeled
             (the calibration-discussion gold) — label names the outlier
          1: someone abstained (we want these visible too — they may have
             been the hard cases)
          2: all three agreed (least useful for the discussion, sort last)
        """
        labels_by_letter = r["labels"]  # {letter: label_str}
        non_blank = {L: v for L, v in labels_by_letter.items() if v}
        if len(non_blank) < len(letters):
            # At least one abstained.
            return (1, "with_abstain")
        # All three labeled. Either all agree or one is the odd one out.
        votes = Counter(non_blank.values())
        if len(votes) == 1:
            only = next(iter(votes))
            return (2, f"all_agree_{only}")
        # Some letter is the minority of 1.
        for L, lab in non_blank.items():
            if votes[lab] == 1:
                return (0, f"{label_map[L]}_outlier")
        return (2, "all_agree_unknown")  # unreachable for binary labels

    iaa_quicklook: list[dict] = []
    for r in iaa_rows:
        idx = r["beads_row_idx"]
        prio, category = outlier_category(r)
        iaa_quicklook.append({
            "sort_priority": prio,
            "beads_row_idx": idx,
            "text": text_by_idx.get(idx, ""),
            "gold_label": r["gold_label"],
            "non_beads_vote": r["non_beads_vote"],
            **{label_map[L]: r["labels"][L] for L in letters},
            "outlier": category,
            "consensus_label": r["consensus_label"],
        })
    iaa_quicklook.sort(key=lambda x: (x["sort_priority"], x["beads_row_idx"]))

    quicklook_fields = (
        ["beads_row_idx", "text", "gold_label", "non_beads_vote"]
        + [label_map[L] for L in letters]
        + ["outlier", "consensus_label"]
    )
    quicklook_path = REPO_ROOT / f"{args.output_prefix}_iaa_quicklook.csv"
    with quicklook_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=quicklook_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(iaa_quicklook)
    print(f"[score] wrote {quicklook_path}  ({len(iaa_quicklook)} IAA rows, "
          f"sorted: outliers -> abstains -> all-agree)")
    iaa_breakdown = Counter(r["outlier"] for r in iaa_quicklook)
    for cat, count in sorted(iaa_breakdown.items()):
        print(f"[score]   {cat:>30s}: {count:>2}")

    # 10. Write summary JSON.
    summary = {
        "n_total": n_unique,
        "n_non_abstain": n_non_abstain,
        "n_abstain": n_abstain,
        "n_iaa_rows": n_iaa,
        "iaa_pairwise_pct_agreement": {
            pair: round(info["pct"], 4) if info["pct"] == info["pct"] else None
            for pair, info in pairwise_agree.items()
        },
        "iaa_pairwise_n_pairs": {pair: info["n_pairs"] for pair, info in pairwise_agree.items()},
        "iaa_pairwise_cohens_kappa": {
            pair: round(k, 4) if k == k else None
            for pair, k in pairwise_kappa_results.items()
        },
        "iaa_three_way_unanimous_pct": round(unanimous_pct, 4) if unanimous_pct == unanimous_pct else None,
        "iaa_three_way_n_rows_all_labeled": unanimous_n,
        "gate_iaa_pass": gate_iaa_pass,
        "beads_mislabel_count": mislabel_count,
        "beads_mislabel_rate": round(beads_mislabel_rate, 4)
                               if beads_mislabel_rate == beads_mislabel_rate else None,
        "beads_contingency_table": dict(contingency),
        "beads_directional": {
            "missed_bias": missed_bias,    # gold=non-biased, consensus=biased
            "over_called": over_called,    # gold=biased,     consensus=non-biased
        },
        "qlora_beads_full_accuracy": round(qlora_beads_full_accuracy, 4),
        "qlora_beads_full_precision_pos": round(qlora_beads_full_precision_pos, 4),
        "qlora_beads_full_recall_pos": round(qlora_beads_full_recall_pos, 4),
        "qlora_beads_full_f1_pos": round(qlora_beads_full_f1_pos, 4),
        "qlora_beads_full_f1_macro": round(qlora_beads_full_f1_macro, 4),
        "flagged_in_sample": len(flagged_idxs),
        "flagged_non_abstain_in_sample": flip_total,
        "flip_correctness_rate": round(flip_correctness_rate, 4)
                                 if flip_correctness_rate == flip_correctness_rate else None,
        "gate_flip_branch": gate_flip_branch,
    }
    summary_path = REPO_ROOT / f"{args.output_prefix}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"[score] wrote {summary_path}")

    # 11. Stdout headline summary.
    print()
    print("=" * 72)
    print("  IAA (50 shared rows)")
    print("=" * 72)
    for pair, info in pairwise_agree.items():
        pct = info["pct"]
        k = pairwise_kappa_results[pair]
        pct_str = f"{pct:.1%}" if pct == pct else "n/a"
        k_str = f"κ={k:+.2f}" if k == k else "κ=n/a"
        print(f"  pair {pair}:  {pct_str:>8}  ({info['n_pairs']:>2} pairs, {k_str})")
    if unanimous_pct == unanimous_pct:
        print(f"  three-way unanimous:  {unanimous_pct:.1%}  ({unanimous_n} rows where all 3 labeled)")
    print()
    if gate_iaa_pass:
        print("  ✓ IAA GATE PASSED (min pairwise >= 70%)")
    else:
        print("  ✗ IAA GATE FAILED (min pairwise < 70%) — recalibrate before retraining.")
    print()
    print("=" * 72)
    print(f"  BEADs vs hand-labels  (n={n_non_abstain} non-abstain rows)")
    print("=" * 72)
    if beads_mislabel_rate == beads_mislabel_rate:
        print(f"  mislabel rate:       {beads_mislabel_rate:.1%}  ({mislabel_count} of {n_non_abstain})")
        if over_called > 0:
            ratio = missed_bias / over_called
            print(f"  directional:         {missed_bias} missed bias / {over_called} over-called "
                  f"(asymmetry {ratio:.1f}:1)")
        else:
            print(f"  directional:         {missed_bias} missed bias / {over_called} over-called")
        print(f"  contingency table:")
        for k, v in sorted(contingency.items()):
            print(f"    {k:<40} {v:>4}")
    print()
    print("=" * 72)
    print(f"  qlora_beads_full vs hand-labels  (n={n_non_abstain})")
    print("=" * 72)
    print(f"  accuracy:            {qlora_beads_full_accuracy:.4f}")
    print(f"  precision (biased):  {qlora_beads_full_precision_pos:.4f}")
    print(f"  recall    (biased):  {qlora_beads_full_recall_pos:.4f}")
    print(f"  f1        (biased):  {qlora_beads_full_f1_pos:.4f}")
    print(f"  f1_macro:            {qlora_beads_full_f1_macro:.4f}")
    print()
    print("=" * 72)
    print(f"  Ensemble flip-correctness on flagged rows")
    print("=" * 72)
    print(f"  flagged rows in sample:        {len(flagged_idxs)}  "
          f"(~{len(flagged_idxs)/n_unique:.1%} of {n_unique})")
    print(f"  flagged AND non-abstain:       {flip_total}")
    if flip_correctness_rate == flip_correctness_rate:
        print(f"  flip-correctness rate:         {flip_correctness_rate:.1%}  "
              f"({flip_match} of {flip_total})")
    else:
        print(f"  flip-correctness rate:         n/a (no flagged non-abstain rows)")
    print()
    branch_decision = {
        "remove_and_flip": "  -> run BOTH remove-train and flip-train sweeps (3-way comparison)",
        "remove_only":     "  -> run ONLY remove-train (flip is too unreliable to trust the relabel)",
        "skip":            "  -> SKIP retrain; the noise-detection result stands alone.",
    }
    print(f"  gate decision: {gate_flip_branch}")
    print(branch_decision[gate_flip_branch])
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
