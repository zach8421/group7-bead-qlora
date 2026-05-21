"""Generate the cross-dataset transfer figures for the cleaning writeup.

Two figures:

1. ``docs/figures/transfer_curve.png`` — 4-panel learning curve, one panel
   per target dataset (BEADs hand-labels, BABE, cajcodes, WNC). Each
   panel plots accuracy vs train_size for the 5 sweep families (original
   + 4 cleaned variants), mirroring the cleaning_curve.png style.

2. ``docs/figures/transfer_before_after.png`` — focused bar chart for
   the headline result: original qlora_beads_full vs the best-cleaned
   adapter on each target dataset. Shows the lift directly.

Data sources
------------
- BEADs hand-label panel: reads ``hand_label_eval.csv`` (produced by
  ``scripts/score_against_hand_labels.py``).
- BABE / cajcodes / WNC panels: reads ``eval_metrics.json`` from each
  cell under ``outputs/cross_eval/qlora_beads*__on__<ds>/``. The
  ``accuracy`` field uses each dataset's published gold (which is what
  the slurm runner's on-the-fly eval produces — no hand-labels for
  the non-BEADs datasets).
- Original adapter on non-BEADs targets: a single point at full size
  (the 2026-05-19 cross-eval cell). On the BABE/cajcodes/WNC panels
  this is drawn as a horizontal reference line instead of a 5-point
  curve since the original sweep only ran cross-eval on the full-size
  adapter.

Usage
-----
    python scripts/make_transfer_figures.py
    python scripts/make_transfer_figures.py --metric f1_macro
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


REPO_ROOT = Path(__file__).resolve().parent.parent

# Mirror make_hand_label_curve.py exactly so the figures are visually consistent.
FAMILY_FULL_SIZE = {
    "original": 27263,
    "remove": 16892,
    "remove_balanced": 10404,
    "flip": 27263,
    "flip_balanced": 14246,
}
SIZE_TO_INT = {"100": 100, "500": 500, "1k": 1000, "5k": 5000}
SIZES_ORDERED = ["100", "500", "1k", "5k", "full"]

FAMILY_STYLE = [
    ("original",         "Original (noisy gold)",       "#404040", "-",  "o"),
    ("remove",           "Cleaned: remove",             "#1f77b4", "-",  "o"),
    ("remove_balanced",  "Cleaned: remove + balanced",  "#1f77b4", "--", "s"),
    ("flip",             "Cleaned: flip",               "#d62728", "-",  "o"),
    ("flip_balanced",    "Cleaned: flip + balanced",    "#d62728", "--", "s"),
]

TARGETS = [
    ("BEADs (hand-labels, n=492)", "beads_hl"),
    ("BABE (n=413)",               "babe"),
    ("cajcodes (n=66)",            "cajcodes"),
    ("WNC (n=11,041)",             "wnc"),
]


# --- Data loading ------------------------------------------------------


def load_cell_metrics(cell_name: str, metric: str) -> float | None:
    p = REPO_ROOT / "outputs" / "cross_eval" / cell_name / "eval_metrics.json"
    if not p.is_file():
        return None
    try:
        return float(json.loads(p.read_text())[metric])
    except (KeyError, ValueError, TypeError):
        return None


def parse_handlabel_adapter(name: str) -> tuple[str, str] | None:
    """``qlora_beads_[cleaned_]<family>_<size>`` → (family, size).

    Same logic as make_hand_label_curve.py. Cross-eval cells like
    ``qlora_beads_full__on__beads`` are skipped (they're a duplicate
    of the original at full size, which we already pick up from the
    sweep dir).
    """
    if not name.startswith("qlora_beads_"):
        return None
    rest = name[len("qlora_beads_"):]
    if "__on__" in rest:
        return None
    if rest in {"100", "500", "1k", "5k", "full"}:
        return ("original", rest)
    if rest.startswith("cleaned_"):
        rest = rest[len("cleaned_"):]
        family, _, size = rest.rpartition("_")
        if family and size in {"100", "500", "1k", "5k", "full"}:
            return (family, size)
    return None


def load_beads_handlabel_points(csv_path: Path, metric: str) -> dict[str, list[tuple[int, float]]]:
    """Return {family: sorted [(x_train_rows, metric_value), ...]}."""
    out: dict[str, list[tuple[int, float]]] = {}
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            try:
                v = float(row[metric])
            except (KeyError, TypeError, ValueError):
                continue
            parsed = parse_handlabel_adapter(row["adapter"])
            if not parsed:
                continue
            family, size = parsed
            if family not in FAMILY_FULL_SIZE:
                continue
            x = FAMILY_FULL_SIZE[family] if size == "full" else SIZE_TO_INT[size]
            out.setdefault(family, []).append((x, v))
    for k in out:
        out[k].sort()
    return out


def load_target_points(target_ds: str, metric: str) -> dict[str, list[tuple[int, float]]]:
    """Cross-eval cell metrics for cleaned variants on a non-BEADs target.

    Returns {family: [(x_train_rows, metric_value), ...]} for the 4 cleaned
    families. The 'original' family ends up with a single point at
    full size (the 2026-05-19 cross-eval cell).
    """
    out: dict[str, list[tuple[int, float]]] = {}

    # Original full
    v = load_cell_metrics(f"qlora_beads_full__on__{target_ds}", metric)
    if v is not None:
        out.setdefault("original", []).append((FAMILY_FULL_SIZE["original"], v))

    # Cleaned variants at each size
    for family in ("remove", "remove_balanced", "flip", "flip_balanced"):
        for size in SIZES_ORDERED:
            cell = f"qlora_beads_cleaned_{family}_{size}__on__{target_ds}"
            v = load_cell_metrics(cell, metric)
            if v is None:
                continue
            x = FAMILY_FULL_SIZE[family] if size == "full" else SIZE_TO_INT[size]
            out.setdefault(family, []).append((x, v))
    for k in out:
        out[k].sort()
    return out


# --- Figure 1: 4-panel transfer learning curves -----------------------


def plot_transfer_curve(metric: str, out_path: Path,
                        handlabel_csv: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), dpi=150, sharex=False)
    axes = axes.flatten()

    legend_handles = []  # collect once, render once below the panels

    for i, (panel_title, target_id) in enumerate(TARGETS):
        ax = axes[i]
        if target_id == "beads_hl":
            points = load_beads_handlabel_points(handlabel_csv, metric)
        else:
            points = load_target_points(target_id, metric)

        for family, label, color, ls, marker in FAMILY_STYLE:
            if family not in points:
                continue
            pts = points[family]
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            if family == "original" and len(pts) == 1:
                # Single-point original on babe/cajcodes/wnc → horizontal
                # reference line spanning the cleaned-variant x-range.
                ax.axhline(ys[0], color=color, linestyle=ls, linewidth=2,
                           label=f"{label} (full)")
                ax.scatter(xs, ys, color=color, marker=marker, s=60,
                           edgecolor="white", linewidth=1.5, zorder=4)
                ax.annotate(f"{ys[0]:.3f}", xy=(xs[0], ys[0]),
                            xytext=(7, -2), textcoords="offset points",
                            fontsize=9, va="top", color=color, fontweight="bold")
            else:
                line, = ax.plot(xs, ys, color=color, linestyle=ls, marker=marker,
                                linewidth=2, markersize=6, label=label, zorder=3)
                # Annotate rightmost (full-size) point.
                if pts:
                    x_last, y_last = pts[-1]
                    ax.annotate(f"{y_last:.3f}", xy=(x_last, y_last),
                                xytext=(7, 0), textcoords="offset points",
                                fontsize=8.5, va="center", color=color,
                                fontweight="bold")
            if i == 0:
                # Collect legend handles from the first panel only.
                legend_handles.append(mpatches.Patch(color=color, label=label))

        ax.set_xscale("log")
        all_xs = sorted({x for pts in points.values() for x, _ in pts})
        ax.set_xticks(all_xs)
        ax.set_xticklabels(
            [f"{x:,}" if x < 1000 else f"{x/1000:.1f}k" for x in all_xs],
            fontsize=7.5, rotation=30, ha="right"
        )
        ax.minorticks_off()
        ax.set_title(panel_title, fontsize=11, fontweight="bold")
        ax.grid(True, axis="y", linestyle=":", alpha=0.4, zorder=1)
        ax.set_axisbelow(True)
        # Per-panel y range (each target has a different ceiling)
        all_ys = [y for pts in points.values() for _, y in pts]
        if all_ys:
            ax.set_ylim(max(0, min(all_ys) - 0.05), min(1.0, max(all_ys) + 0.07))

        if i % 2 == 0:
            metric_label = {"accuracy": "Accuracy", "f1_macro": "Macro F1",
                            "f1_pos": "F1 (biased)"}.get(metric, metric)
            ax.set_ylabel(metric_label, fontsize=10)
        if i >= 2:
            ax.set_xlabel("Training examples (log scale)", fontsize=10)

    # Single shared legend below the panels.
    fig.legend(handles=[mpatches.Patch(color=c, label=l)
                        for _, l, c, _, _ in FAMILY_STYLE if l != "Original (noisy gold)"]
                       + [mpatches.Patch(color="#404040", label="Original (noisy gold or hand-label)")],
               loc="lower center", ncol=3, fontsize=10, bbox_to_anchor=(0.5, -0.02),
               framealpha=0.95)

    metric_label = {"accuracy": "Accuracy", "f1_macro": "Macro F1",
                    "f1_pos": "F1 (biased)"}.get(metric, metric)
    fig.suptitle(
        f"BEAD QLoRA cleaning experiment — cross-dataset transfer ({metric_label})",
        fontsize=13, fontweight="bold", y=0.995
    )

    plt.tight_layout(rect=[0, 0.05, 1, 0.98])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    print(f"[fig] wrote {out_path}  ({out_path.stat().st_size / 1024:.1f} KB)")


# --- Figure 2: Before-after bar chart ----------------------------------


def plot_before_after(metric: str, out_path: Path,
                      handlabel_csv: Path) -> None:
    """For each target, plot Original (full) vs Best Cleaned, side by side.

    Best Cleaned is selected per (target, metric): the highest-scoring
    cleaned adapter across all families and sizes.
    """
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=150)

    target_labels = []
    original_vals = []
    cleaned_vals = []
    cleaned_label_texts = []

    for panel_title, target_id in TARGETS:
        if target_id == "beads_hl":
            points = load_beads_handlabel_points(handlabel_csv, metric)
        else:
            points = load_target_points(target_id, metric)

        # Original: take the full-size value for that target (or the
        # only original point on the non-BEADs targets).
        orig_val = None
        for x, y in points.get("original", []):
            orig_val = y  # last/largest
        if orig_val is None:
            continue

        # Best cleaned: max over all 4 cleaned families and all sizes.
        best_val = None
        best_label = None
        for family in ("remove", "remove_balanced", "flip", "flip_balanced"):
            for x, y in points.get(family, []):
                if best_val is None or y > best_val:
                    best_val = y
                    # Convert x back to a size label for the annotation
                    if x == FAMILY_FULL_SIZE[family]:
                        size_label = "full"
                    else:
                        # Inverse SIZE_TO_INT
                        size_label = {v: k for k, v in SIZE_TO_INT.items()}.get(x, str(x))
                    best_label = f"{family}_{size_label}"
        if best_val is None:
            continue

        target_labels.append(panel_title)
        original_vals.append(orig_val)
        cleaned_vals.append(best_val)
        cleaned_label_texts.append(best_label)

    n = len(target_labels)
    x_pos = list(range(n))
    width = 0.38

    bars_o = ax.bar([p - width/2 for p in x_pos], original_vals, width,
                    color="#9090a0", edgecolor="black", linewidth=0.8,
                    label="Original (qlora_beads_full)")
    bars_c = ax.bar([p + width/2 for p in x_pos], cleaned_vals, width,
                    color="#d62728", edgecolor="black", linewidth=0.8,
                    label="Best cleaned")

    # Annotate each bar with its value
    for bar, v in zip(bars_o, original_vals):
        ax.annotate(f"{v:.3f}", xy=(bar.get_x() + bar.get_width()/2, v),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", fontsize=9)
    for bar, v, lab in zip(bars_c, cleaned_vals, cleaned_label_texts):
        ax.annotate(f"{v:.3f}\n({lab})", xy=(bar.get_x() + bar.get_width()/2, v),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", fontsize=8, color="#404040")

    # Lift annotation (between bars, slightly above the lower bar)
    for i, (o, c) in enumerate(zip(original_vals, cleaned_vals)):
        lift = c - o
        ax.annotate(f"Δ {lift:+.1%}",
                    xy=(i, max(o, c) + 0.05),
                    ha="center", fontsize=10, fontweight="bold",
                    color="#0a8000" if lift > 0 else "#a00000")

    ax.set_xticks(x_pos)
    ax.set_xticklabels(target_labels, fontsize=10)
    metric_label = {"accuracy": "Accuracy", "f1_macro": "Macro F1",
                    "f1_pos": "F1 (biased)"}.get(metric, metric)
    ax.set_ylabel(metric_label, fontsize=11)
    ax.set_ylim(0, min(1.0, max(cleaned_vals) + 0.2))
    ax.set_title(f"Before vs after cleaning: {metric_label} on each target dataset\n"
                 f"(BEADs panel uses 492 hand-labels; others use each dataset's published gold)",
                 fontsize=12, fontweight="bold")
    ax.grid(True, axis="y", linestyle=":", alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", fontsize=10, framealpha=0.95)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    print(f"[fig] wrote {out_path}  ({out_path.stat().st_size / 1024:.1f} KB)")


# --- Main --------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--metric", default="accuracy",
                    choices=["accuracy", "f1_macro", "f1_pos"])
    ap.add_argument("--handlabel-csv", default=str(REPO_ROOT / "hand_label_eval.csv"))
    ap.add_argument("--out-curve", default=str(REPO_ROOT / "docs" / "figures" / "transfer_curve.png"))
    ap.add_argument("--out-bars",  default=str(REPO_ROOT / "docs" / "figures" / "transfer_before_after.png"))
    args = ap.parse_args()

    handlabel_csv = Path(args.handlabel_csv)
    plot_transfer_curve(args.metric, Path(args.out_curve), handlabel_csv)
    plot_before_after(args.metric, Path(args.out_bars), handlabel_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
