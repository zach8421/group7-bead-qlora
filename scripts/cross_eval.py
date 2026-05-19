"""Run the (adapter × eval-dataset) cross-evaluation matrix.

For every (trained adapter, dataset's test split) pair the user specifies, this
script invokes ``scripts/eval_adapter.py`` with appropriate paths and upserts
the resulting metrics into ``outputs/manifest.csv`` via
``scripts/update_manifest.py``.

Each cross-eval cell is written to its own directory so the per-cell
``eval_metrics.json`` and ``predictions.jsonl`` don't overwrite the original
same-dataset eval results inside ``outputs/qlora_*/``:

    outputs/cross_eval/{adapter_run_name}__on__{eval_dataset}/
        predictions.jsonl
        eval_metrics.json

The manifest's ``run_name`` for cross-eval cells follows the same convention.

Examples
--------
Evaluate every ``outputs/qlora_*_full/`` adapter on every dataset:
    python scripts/cross_eval.py \\
        --adapters outputs/qlora_beads_full outputs/qlora_babe_full \\
                   outputs/qlora_cajcodes_full outputs/qlora_wnc_full \\
        --eval-datasets beads babe cajcodes wnc

Evaluate just the BEADs sweep on every dataset (32 cells):
    python scripts/cross_eval.py \\
        --adapters outputs/qlora_beads_* \\
        --eval-datasets beads babe cajcodes wnc

Dry-run (print the planned cells, run nothing):
    python scripts/cross_eval.py --adapters outputs/qlora_*_full --eval-datasets beads babe --dry-run
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent


def test_jsonl_for(dataset: str) -> Path:
    """Resolve the canonical test JSONL for a dataset, regardless of layout shape.

    BEADs uses ``data/frozen/beads/sizes/full/test.jsonl``; the simple datasets
    use ``data/frozen/<name>/full/test.jsonl``.
    """
    candidates = [
        REPO_ROOT / "data" / "frozen" / dataset / "sizes" / "full" / "test.jsonl",
        REPO_ROOT / "data" / "frozen" / dataset / "full" / "test.jsonl",
    ]
    for p in candidates:
        if p.is_file():
            return p
    raise FileNotFoundError(
        f"No test JSONL found for dataset '{dataset}'. Looked at:\n  "
        + "\n  ".join(str(p) for p in candidates)
        + "\nRun: python scripts/freeze_splits.py --dataset " + dataset
    )


def adapter_dir_for(run_dir: Path) -> Path:
    adapter_path = run_dir / "adapter"
    if not adapter_path.is_dir():
        raise FileNotFoundError(f"No adapter/ inside {run_dir}")
    return adapter_path


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--adapters", required=True, nargs="+",
                    help="One or more outputs/qlora_*/ directories. Each must "
                         "contain an adapter/ subdir.")
    ap.add_argument("--eval-datasets", required=True, nargs="+",
                    choices=["beads", "babe", "cajcodes", "wnc"],
                    help="Datasets whose test split each adapter is scored on.")
    ap.add_argument("--out-root", default="outputs/cross_eval",
                    help="Cells land at <out-root>/<adapter_run_name>__on__<eval_dataset>/.")
    ap.add_argument("--manifest", default="outputs/manifest.csv",
                    help="Manifest to upsert rows into. Sibling .json is also written.")
    ap.add_argument("--eval-batch-size", type=int, default=16)
    ap.add_argument("--max-test-rows", type=int, default=0,
                    help="If >0, evaluate only the first N rows (smoke).")
    ap.add_argument("--skip-existing", action="store_true",
                    help="Don't re-run a cell if its eval_metrics.json already exists.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the planned cells; don't actually run eval_adapter.")
    args = ap.parse_args()

    # Resolve adapters → (run_dir, adapter_dir, run_name)
    cells: list[tuple[Path, Path, str, str, Path]] = []
    for raw in args.adapters:
        run_dir = Path(raw).resolve()
        if not run_dir.is_dir():
            print(f"[cross_eval] skip: {run_dir} is not a directory", file=sys.stderr)
            continue
        try:
            adapter_path = adapter_dir_for(run_dir)
        except FileNotFoundError as e:
            print(f"[cross_eval] skip: {e}", file=sys.stderr)
            continue
        run_name = run_dir.name  # e.g. qlora_beads_full
        for ds in args.eval_datasets:
            try:
                test_path = test_jsonl_for(ds)
            except FileNotFoundError as e:
                print(f"[cross_eval] skip ({run_name}, {ds}): {e}", file=sys.stderr)
                continue
            cells.append((run_dir, adapter_path, run_name, ds, test_path))

    if not cells:
        print("[cross_eval] no runnable cells", file=sys.stderr)
        return 2

    print(f"[cross_eval] planning {len(cells)} cells:")
    for _, _, run_name, ds, test_path in cells:
        cell_name = f"{run_name}__on__{ds}"
        print(f"  {cell_name:60s}  test={test_path}")

    if args.dry_run:
        return 0

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    n_run = 0
    n_skipped = 0
    n_failed = 0
    for run_dir, adapter_path, run_name, ds, test_path in cells:
        cell_name = f"{run_name}__on__{ds}"
        cell_dir = out_root / cell_name
        cell_dir.mkdir(parents=True, exist_ok=True)

        metrics_path = cell_dir / "eval_metrics.json"
        if args.skip_existing and metrics_path.exists():
            print(f"[cross_eval] skip (exists): {cell_name}")
            n_skipped += 1
            continue

        eval_cmd = [
            sys.executable, "scripts/eval_adapter.py",
            "--adapter-path", str(adapter_path),
            "--test-jsonl", str(test_path),
            "--eval-dataset", ds,
            "--output-dir", str(cell_dir),
            "--run-name", cell_name,
            "--eval-batch-size", str(args.eval_batch_size),
        ]
        if args.max_test_rows > 0:
            eval_cmd += ["--max-test-rows", str(args.max_test_rows)]

        print(f"\n[cross_eval] ▶ {cell_name}")
        try:
            subprocess.run(eval_cmd, check=True, cwd=REPO_ROOT)
        except subprocess.CalledProcessError as e:
            print(f"[cross_eval] FAILED: {cell_name} (returncode={e.returncode})", file=sys.stderr)
            n_failed += 1
            continue

        # Copy the training-side run_meta/train_metrics into the cell dir so
        # update_manifest picks up their fields (train_dataset, learning_rate,
        # throughput, etc.). Override the embedded run_name with the cell's
        # name so the manifest row keys off the cell, not the original adapter.
        import json as _json
        for fname in ("run_meta.json", "train_metrics.json"):
            src = run_dir / fname
            if not src.is_file():
                continue
            data = _json.loads(src.read_text())
            if isinstance(data, dict) and "run_name" in data:
                data["run_name"] = cell_name
            (cell_dir / fname).write_text(_json.dumps(data, indent=2))

        try:
            subprocess.run(
                [sys.executable, "scripts/update_manifest.py",
                 "--run-dir", str(cell_dir),
                 "--manifest", args.manifest],
                check=True, cwd=REPO_ROOT,
            )
        except subprocess.CalledProcessError as e:
            print(f"[cross_eval] manifest update failed: {cell_name} ({e})", file=sys.stderr)
            n_failed += 1
            continue

        n_run += 1

    print(f"\n[cross_eval] done. ran={n_run} skipped={n_skipped} failed={n_failed}")
    return 0 if n_failed == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
