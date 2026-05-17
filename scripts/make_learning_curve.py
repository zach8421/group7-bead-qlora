"""Generate the BEAD QLoRA learning-curve figure for sharing with the team.

Reads from the committed sweep manifest (`outputs/manifest.csv`) plus any
baseline `eval_metrics.json` files under `baselines/<name>/`, then plots
accuracy vs. number of training examples on a log-x axis with each
baseline overlaid as a horizontal reference. Saved as
`docs/figures/learning_curve.png` by default.

The script is **reproducible**: no hand-coded numbers, no matplotlib
state, no JSON edits required when a new run lands. Re-run after any
sweep or baseline addition and the figure regenerates from disk.

Usage
-----
    python scripts/make_learning_curve.py
    python scripts/make_learning_curve.py --metric f1_macro
    python scripts/make_learning_curve.py --out docs/figures/curve_f1.png
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parent.parent


def load_qlora_points(manifest_csv: Path, metric: str) -> list[tuple[int, float, str]]:
    """Return (train_size, metric_value, run_name) tuples sorted by train_size."""
    points: list[tuple[int, float, str]] = []
    with manifest_csv.open() as f:
        for row in csv.DictReader(f):
            try:
                n = int(row["train_size"])
                v = float(row[metric])
            except (TypeError, ValueError):
                continue
            points.append((n, v, row["run_name"]))
    return sorted(points)


def load_baselines(baselines_dir: Path, metric: str) -> list[tuple[str, float]]:
    """Return (display_name, metric_value) for each baseline that has an
    eval_metrics.json with the requested metric."""
    out: list[tuple[str, float]] = []
    for em in sorted(baselines_dir.glob("*/*eval_metrics*.json")):
        try:
            d = json.loads(em.read_text())
            v = float(d[metric])
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            continue
        # Display name: dir name + run_name if distinguishable
        display = d.get("run_name") or em.parent.name
        out.append((display, v))
    return out


def plot(points: list[tuple[int, float, str]],
         baselines: list[tuple[str, float]],
         metric: str,
         out_path: Path) -> None:
    if not points:
        raise SystemExit("No QLoRA points found — is outputs/manifest.csv populated?")

    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=150)

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    labels = [p[2] for p in points]

    # QLoRA learning curve
    ax.plot(xs, ys, marker="o", linewidth=2, markersize=8,
            color="#1f4e79", label="QLoRA (Llama-3.1-8B-Instruct)", zorder=3)

    # Per-point value annotations.
    for x, y, lab in zip(xs, ys, labels):
        ax.annotate(f"{lab}\n{y:.3f}", xy=(x, y),
                    xytext=(8, -4), textcoords="offset points",
                    fontsize=8.5, va="top", color="#1f4e79", zorder=4)

    # Baselines as horizontal reference lines.
    baseline_styles = [
        ("#c00000", "--"),  # TF-IDF (first baseline) — red dashed
        ("#7030a0", ":"),   # 3-shot (when it lands) — purple dotted
        ("#404040", "-."),  # any future baseline — grey dash-dot
    ]
    for i, (name, val) in enumerate(baselines):
        color, linestyle = baseline_styles[i % len(baseline_styles)]
        ax.axhline(val, color=color, linestyle=linestyle, linewidth=1.5,
                   label=f"{name} ({val:.3f})", zorder=2)

    ax.set_xscale("log")
    # Major ticks at the actual sweep sizes so the x-axis reads as data, not powers of 10.
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{x:,}" for x in xs])
    ax.minorticks_off()

    metric_label = {"accuracy": "Accuracy",
                    "f1_macro": "Macro F1",
                    "f1_pos": "F1 (biased)"}.get(metric, metric)
    ax.set_xlabel("Training examples (log scale)", fontsize=11)
    ax.set_ylabel(f"{metric_label} on held-out test (n=6,816)", fontsize=11)
    ax.set_title(
        "BEAD bias classification — learning curve\n"
        "Llama-3.1-8B-Instruct + QLoRA, vs. TF-IDF + logreg baseline",
        fontsize=12, fontweight="bold"
    )
    ax.grid(True, axis="y", linestyle=":", alpha=0.4, zorder=1)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.95)
    ax.set_ylim(min(min(ys), *(v for _, v in baselines), 0.5) - 0.05,
                max(max(ys), *(v for _, v in baselines)) + 0.05)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    print(f"Wrote {out_path}  ({out_path.stat().st_size / 1024:.1f} KB)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default=str(REPO_ROOT / "outputs/manifest.csv"))
    ap.add_argument("--baselines-dir", default=str(REPO_ROOT / "baselines"))
    ap.add_argument("--metric", default="accuracy",
                    choices=["accuracy", "f1_macro", "f1_pos", "precision_pos", "recall_pos"])
    ap.add_argument("--out", default=str(REPO_ROOT / "docs/figures/learning_curve.png"))
    args = ap.parse_args()

    points = load_qlora_points(Path(args.manifest), args.metric)
    baselines = load_baselines(Path(args.baselines_dir), args.metric)

    print(f"QLoRA points ({len(points)}):")
    for n, v, lab in points:
        print(f"  {lab:12s}  n={n:>6,d}  {args.metric}={v:.4f}")
    print(f"Baselines ({len(baselines)}):")
    for name, v in baselines:
        print(f"  {name:24s}  {args.metric}={v:.4f}")

    plot(points, baselines, args.metric, Path(args.out))


if __name__ == "__main__":
    main()
