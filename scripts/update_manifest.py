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
import contextlib
import csv
import json
import os
import sys
import time
from pathlib import Path

try:
    import fcntl  # POSIX advisory lock; not available on Windows
    _HAVE_FCNTL = True
except ImportError:
    _HAVE_FCNTL = False


@contextlib.contextmanager
def manifest_lock(csv_path: Path, timeout_s: float = 60.0):
    """Advisory exclusive lock around a manifest's read-modify-write.

    Concurrent QLoRA jobs (and the 16-cell cross-eval matrix) all upsert into
    the same `outputs/manifest.csv`. Without a lock, a slow reader on one job
    can race with a faster writer on another and clobber rows — see the
    2026-05-16 entry in docs/build_log.md for the incident this guards against.

    Uses an on-disk `.lock` sibling so the lock survives short-lived Python
    process boundaries. On Windows (no fcntl) this degrades to a no-op; that
    matches the prior behavior — we never claimed Windows support — and the
    Tillicum/macOS dev paths both have fcntl.
    """
    if not _HAVE_FCNTL:
        yield
        return
    lock_path = csv_path.with_suffix(csv_path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o644)
    deadline = time.monotonic() + timeout_s
    try:
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise SystemExit(
                        f"[manifest] timed out after {timeout_s:.0f}s waiting for {lock_path}. "
                        f"Stale lock? Remove the file and retry."
                    )
                time.sleep(0.25)
        yield
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


COLUMNS = [
    "run_name",
    "train_dataset",
    "eval_dataset",
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

    # train_dataset / eval_dataset are tracked separately so cross-eval rows
    # can be pivoted from this table.
    KNOWN_DATASETS = {"beads", "babe", "cajcodes", "wnc"}

    def _infer_dataset_from_path(jsonl_path: str | None) -> str | None:
        if not jsonl_path:
            return None
        parts = Path(jsonl_path).parts
        if "frozen" not in parts:
            return None
        i = parts.index("frozen")
        if i + 1 >= len(parts):
            return None
        candidate = parts[i + 1]
        # Reject flat-layout legacy paths (data/frozen/train_full.jsonl) — the
        # part after "frozen" there is a filename, not a dataset dir. Those
        # runs predate the multi-dataset restructure and were all BEADs.
        if candidate.endswith(".jsonl"):
            return "beads"
        return candidate if candidate in KNOWN_DATASETS else None

    train_dataset = inputs.get("train_dataset") or _infer_dataset_from_path(inputs.get("train_jsonl"))
    eval_dataset = ev.get("eval_dataset") or _infer_dataset_from_path(ev.get("test_jsonl"))

    row = {
        "run_name": run_name,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
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


def _train_size_int(r: dict) -> int:
    # CSV round-trip stringifies everything, freshly-built rows are ints — coerce
    # so the sort key doesn't TypeError when both shapes coexist mid-rebuild.
    ts = r.get("train_size")
    if ts is None or ts == "":
        return 0
    try:
        return int(ts)
    except (ValueError, TypeError):
        return 0


def write_csv(path: Path, rows: dict[str, dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(rows.values(), key=lambda r: (_train_size_int(r), r.get("run_name") or ""))
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

    with manifest_lock(csv_path):
        rows = load_existing_csv(csv_path)
        rows[row["run_name"]] = row

        write_csv(csv_path, rows)
        write_json(json_path, rows)

    print(f"[manifest] Upserted {row['run_name']} into {csv_path}")
    print(json.dumps(row, indent=2, default=str))


if __name__ == "__main__":
    main()
