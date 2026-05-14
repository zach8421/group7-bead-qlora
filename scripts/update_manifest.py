"""Upsert one row into the sweep manifest from a completed run directory.

Reads from `<run-dir>/run_meta.json`, `<run-dir>/train_metrics.json`, and
`<run-dir>/eval_metrics.json` (whichever exist), keyed by `run_name`. Writes
two formats side-by-side:

  outputs/manifest.csv   — pandas-friendly summary table (one row per run)
  outputs/manifest.json  — same content, dict keyed by run_name

If an existing manifest already has this run_name, the row is replaced.

Usage
-----
    python scripts/update_manifest.py --run-dir outputs/qlora_5k
    python scripts/update_manifest.py --run-dir outputs/qlora_5k --manifest outputs/manifest.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


COLUMNS = [
    "run_name",
    "smoke_test",
    "model_name",
    "train_size",
    "num_epochs",
    "effective_batch_size",
    "max_seq_length",
    "learning_rate",
    "wall_clock_min",
    "peak_cuda_memory_gb",
    "throughput_examples_per_sec",
    "trainable_params",
    "train_loss_final",
    "accuracy",
    "f1_pos",
    "f1_macro",
    "precision_pos",
    "recall_pos",
    "eval_n_examples",
    "eval_seconds",
    "adapter_path",
    "started_at",
    "finished_at",
    "git_head",
    "hostname",
    "slurm_job_id",
    "train_jsonl_sha256",
    "splits_manifest_sha256",
    "label_str_0",
    "label_str_1",
]


def _read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def build_row(run_dir: Path) -> dict:
    # Smoke runs use *_smoke.json so they don't get mistaken for real results,
    # but the manifest happily reports them when the real files aren't present.
    meta = _read_json(run_dir / "run_meta.json") or _read_json(run_dir / "run_meta_smoke.json") or {}
    train = _read_json(run_dir / "train_metrics.json") or _read_json(run_dir / "train_metrics_smoke.json") or {}
    ev = _read_json(run_dir / "eval_metrics.json") or _read_json(run_dir / "eval_metrics_smoke.json") or {}

    run_name = (
        meta.get("run_name")
        or train.get("run_name")
        or ev.get("run_name")
        or run_dir.name
    )

    host = meta.get("host") or {}
    inputs = meta.get("inputs") or {}
    label_map = inputs.get("splits_label_map") or {}

    row = {
        "run_name": run_name,
        "smoke_test": train.get("smoke_test", meta.get("smoke_test")),
        "model_name": train.get("model_name") or (meta.get("model") or {}).get("model_name"),
        "train_size": train.get("examples"),
        "num_epochs": train.get("num_epochs"),
        "effective_batch_size": train.get("effective_batch_size"),
        "max_seq_length": train.get("max_seq_length"),
        "learning_rate": train.get("learning_rate"),
        "wall_clock_min": train.get("wall_clock_min"),
        "peak_cuda_memory_gb": train.get("peak_cuda_memory_gb"),
        "throughput_examples_per_sec": train.get("throughput_examples_per_sec"),
        "trainable_params": train.get("trainable_params"),
        "train_loss_final": train.get("train_loss_final"),
        "accuracy": ev.get("accuracy"),
        "f1_pos": ev.get("f1_pos"),
        "f1_macro": ev.get("f1_macro"),
        "precision_pos": ev.get("precision_pos"),
        "recall_pos": ev.get("recall_pos"),
        "eval_n_examples": ev.get("n_examples"),
        "eval_seconds": ev.get("eval_seconds"),
        "adapter_path": train.get("adapter_path") or ev.get("adapter_path"),
        "started_at": meta.get("started_at"),
        "finished_at": meta.get("finished_at"),
        "git_head": meta.get("git_head"),
        "hostname": host.get("hostname"),
        "slurm_job_id": host.get("slurm_job_id"),
        "train_jsonl_sha256": inputs.get("train_jsonl_sha256"),
        "splits_manifest_sha256": inputs.get("splits_manifest_sha256"),
        "label_str_0": label_map.get("0"),
        "label_str_1": label_map.get("1"),
    }
    return row


def load_existing_csv(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    out = {}
    with path.open() as f:
        for r in csv.DictReader(f):
            out[r["run_name"]] = r
    return out


def write_csv(path: Path, rows: dict[str, dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(rows.values(), key=lambda r: (r.get("train_size") or 0, r.get("run_name") or ""))
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        w.writeheader()
        for row in ordered:
            w.writerow({k: ("" if row.get(k) is None else row.get(k)) for k in COLUMNS})


def write_json(path: Path, rows: dict[str, dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2, default=str))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True, help="Path to one completed run directory.")
    ap.add_argument("--manifest", default="outputs/manifest.csv",
                    help="CSV path. A sibling .json is written alongside.")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        sys.exit(f"[manifest] {run_dir} does not exist.")

    row = build_row(run_dir)
    if not row["run_name"]:
        sys.exit(f"[manifest] Could not determine run_name from {run_dir}.")

    csv_path = Path(args.manifest)
    json_path = csv_path.with_suffix(".json")

    rows = load_existing_csv(csv_path)
    rows[row["run_name"]] = row

    write_csv(csv_path, rows)
    write_json(json_path, rows)

    print(f"[manifest] Upserted {row['run_name']} into {csv_path}")
    print(json.dumps(row, indent=2, default=str))


if __name__ == "__main__":
    main()
