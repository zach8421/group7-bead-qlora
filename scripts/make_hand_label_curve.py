"""Generate the BEAD cleaning-experiment learning-curve figure.

Five sweep families plotted against the 492-row hand-label consensus
(``score_against_hand_labels.py`` output). The headline plot for the
cleaning experiment writeup.

Reads from ``hand_label_eval.csv`` in the repo root and emits the figure
to ``docs/figures/cleaning_curve.png`` by default.

Sweep families
--------------
- **Original** (``qlora_beads_<sz>``): the noisy baseline from the
  2026-05-16 sweep. Flat-to-declining against hand-labels.
- **cleaned_remove** (``qlora_beads_cleaned_remove_<sz>``): flagged rows
  dropped, natural class balance (~70/30 biased).
- **cleaned_remove_balanced**: drop + undersample to 50/50.
- **cleaned_flip** (``qlora_beads_cleaned_flip_<sz>``): flagged rows
  relabeled to the cross-dataset ensemble's vote, natural balance (~74/26).
- **cleaned_flip_balanced**: relabel + undersample to 50/50. This is the
  winning family.

Colour convention
-----------------
- Black = original baseline (visual reference; everything else should beat it)
- Blue  = ``remove`` family
- Red   = ``flip`` family
- Solid = natural class balance
- Dashed = 50/50 balanced

So the winner (``cleaned_flip_balanced``) appears as a dashed red line.

Usage
-----
    python scripts/make_hand_label_curve.py
    python scripts/make_hand_label_curve.py --metric f1_macro
    python scripts/make_hand_label_curve.py --out docs/figures/curve_f1.png
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parent.parent

# Actual full-pool training row counts per family (the smaller sizes are
# exactly 100/500/1000/5000 across all families since they're stratified
# subsamples of the nominal target).
FAMILY_FULL_SIZE = {
    "original": 27263,
    "remove": 16892,
    "remove_balanced": 10404,
    "flip": 27263,
    "flip_balanced": 14246,
}

# Nominal sweep sizes, mapped to actual row counts.
SIZE_TO_INT = {"100": 100, "500": 500, "1k": 1000, "5k": 5000}


# Display name + matplotlib style per family.
# Order in this list determines z-order (later = on top).
FAMILY_STYLE = [
    ("original",         "Original (noisy gold)",       "#404040", "-",  "o"),
    ("remove",           "Cleaned: remove",             "#1f77b4", "-",  "o"),
    ("remove_balanced",  "Cleaned: remove + balanced",  "#1f77b4", "--", "s"),
    ("flip",             "Cleaned: flip",               "#d62728", "-",  "o"),
    ("flip_balanced",    "Cleaned: flip + balanced",    "#d62728", "--", "s"),
]


def parse_adapter(name: str) -> tuple[str, str] | None:
    """Parse 'qlora_beads_[cleaned_]<family>_<size>' into (family, size).

    Returns None for any adapter the plot shouldn't include (e.g. the
    cross-eval cell name ``qlora_beads_full__on__beads``).
    """
    if not name.startswith("qlora_beads_"):
        return None
    rest = name[len("qlora_beads_"):]
    if "__on__" in rest:
        return None  # cross-eval cells, not a sweep point
    if rest in {"100", "500", "1k", "5k", "full"}:
        return ("original", rest)
    if rest.startswith("cleaned_"):
        rest = rest[len("cleaned_"):]
        # rsplit so 'flip_balanced_500' -> ('flip_balanced', '_', '500')
        family, _, size = rest.rpartition("_")
        if family and size in {"100", "500", "1k", "5k", "full"}:
            return (family, size)
    return None


def adapter_to_xy(adapter: str, value: float) -> tuple[str, int, float] | None:
    """Return (family, train_row_count, value) for a known adapter, else None."""
    parsed = parse_adapter(adapter)
    if not parsed:
        return None
    family, size = parsed
    if family not in FAMILY_FULL_SIZE:
        return None
    if size == "full":
        x = FAMILY_FULL_SIZE[family]
    else:
        x = SIZE_TO_INT[size]
    return (family, x, value)


def load_points(csv_path: Path, metric: str) -> dict[str, list[tuple[int, float]]]:
    """Read hand_label_eval.csv, group by family. Returns {family: [(x, y), ...]}."""
    out: dict[str, list[tuple[int, float]]] = {}
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            try:
                v = float(row[metric])
            except (KeyError, TypeError, ValueError):
                continue
            r = adapter_to_xy(row["adapter"], v)
            if r is None:
                continue
            family, x, y = r
            out.setdefault(family, []).append((x, y))
    for family in out:
        out[family].sort()
    return out


def plot(points_by_family: dict[str, list[tuple[int, float]]],
         metric: str,
         out_path: Path,
         original_noisy_gold: float | None = None) -> None:
    if not points_by_family:
        raise SystemExit("[plot] no points found in hand_label_eval.csv")

    fig, ax = plt.subplots(figsize=(9.5, 6), dpi=150)

    # Plot each family in declared order.
    for family, label, color, ls, marker in FAMILY_STYLE:
        if family not in points_by_family:
            continue
        pts = points_by_family[family]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, color=color, linestyle=ls, marker=marker,
                linewidth=2, markersize=7, label=label, zorder=3)
        # Annotate the rightmost (full-size) point so the winner is unambiguous.
        if pts:
            x_last, y_last = pts[-1]
            ax.annotate(f"{y_last:.3f}", xy=(x_last, y_last),
                        xytext=(7, 0), textcoords="offset points",
                        fontsize=9, va="center", color=color, fontweight="bold")

    # Optional: horizontal reference line for the original's accuracy against
    # *its own noisy gold* — shows how much of the "0.80" benchmark was
    # actually noise memorization.
    if original_noisy_gold is not None:
        ax.axhline(original_noisy_gold, color="#a0a0a0", linestyle=":", linewidth=1.5,
                   label=f"Original vs noisy gold ({original_noisy_gold:.3f})", zorder=1)

    ax.set_xscale("log")
    # Tick at every unique x that appears across families.
    all_xs = sorted({x for pts in points_by_family.values() for x, _ in pts})
    ax.set_xticks(all_xs)
    ax.set_xticklabels([f"{x:,}" if x < 1000 else f"{x/1000:.1f}k" for x in all_xs],
                       fontsize=8, rotation=30, ha="right")
    ax.minorticks_off()

    metric_label = {"accuracy": "Accuracy",
                    "f1_macro": "Macro F1",
                    "f1_pos": "F1 (biased)"}.get(metric, metric)
    ax.set_xlabel("Training examples (log scale)", fontsize=11)
    ax.set_ylabel(f"{metric_label} on 492 hand-labeled rows", fontsize=11)
    ax.set_title(
        "BEAD QLoRA cleaning experiment\n"
        f"{metric_label} vs. team consensus (IAA 81-88%)",
        fontsize=12, fontweight="bold"
    )
    ax.grid(True, axis="y", linestyle=":", alpha=0.4, zorder=1)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.95, ncol=1)

    # Y-limits: 0 to a bit above max for breathing room.
    all_ys = [y for pts in points_by_family.values() for _, y in pts]
    y_max = max(all_ys + ([original_noisy_gold] if original_noisy_gold else []))
    y_min = min(all_ys)
    ax.set_ylim(max(0, y_min - 0.05), min(1.0, y_max + 0.07))

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    print(f"[plot] wrote {out_path}  ({out_path.stat().st_size / 1024:.1f} KB)")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--input", default=str(REPO_ROOT / "hand_label_eval.csv"))
    ap.add_argument("--metric", default="accuracy",
                    choices=["accuracy", "f1_macro", "f1_pos"])
    ap.add_argument("--out", default=str(REPO_ROOT / "docs" / "figures" / "cleaning_curve.png"))
    ap.add_argument("--original-noisy-gold", type=float, default=0.7987,
                    help="Original qlora_beads_full's accuracy against the *noisy* BEADs "
                         "test gold (the original sweep result). Shown as a faint dotted "
                         "horizontal reference. Set to a negative number to hide.")
    args = ap.parse_args()

    points = load_points(Path(args.input), args.metric)

    print(f"[plot] families found: {len(points)}")
    for family, pts in sorted(points.items()):
        print(f"  {family:<20s} {len(pts)} pts")
        for x, y in pts:
            print(f"    n={x:>6,d}  {args.metric}={y:.4f}")

    plot(
        points,
        metric=args.metric,
        out_path=Path(args.out),
        original_noisy_gold=args.original_noisy_gold if args.original_noisy_gold > 0 else None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
