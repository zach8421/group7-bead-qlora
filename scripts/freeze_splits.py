"""Freeze train/val/test JSONL splits for the cross-eval datasets.

Dispatches on ``--dataset {beads,babe,cajcodes,wnc}``:

* **beads** — preserves BEADs' native train/valid boundary, carves a 20%
  stratified test off the original train, then builds *nested* stratified
  subsets so smaller training pools are byte-subsets of larger ones:

      train_100 ⊂ train_500 ⊂ train_1k ⊂ train_5k ⊂ train_full

  Output layout:
      data/frozen/beads/sizes/{100,500,1k,5k,full}/{train,val,test}.jsonl
      data/frozen/beads/splits_manifest.json

  val.jsonl and test.jsonl are copied into every size dir so each one is
  self-contained — that matches the new-dataset layout and keeps the
  cross-eval orchestrator's path discovery simple.

* **babe / cajcodes / wnc** — stratified 80/10/10 train/val/test split off the
  full dataset (the loaders union all native splits). Output layout:

      data/frozen/{name}/full/{train,val,test}.jsonl
      data/frozen/{name}/splits_manifest.json

Each output line:
    {"text": "<sentence>", "label_int": 0|1, "label_str": "biased"|"non-biased"}

The ``id`` field intentionally does **not** round-trip into the JSONL. The
trainer/evaluator never use IDs, and keeping records id-less means re-running
``freeze_splits.py --dataset beads`` reproduces the v2 sweep's SHA256s
byte-for-byte (subject to library determinism).

Smoke test
----------
    python scripts/freeze_splits.py --dataset beads --mock --out-dir data/frozen_mock
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Make sibling-package imports work whether this is run as `python scripts/freeze_splits.py`
# or as a module.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.dataset_loaders import load_babe, load_beads, load_cajcodes, load_wnc  # noqa: E402


SEED = 42
SWEEP_SIZES = [100, 500, 1000, 5000]  # 'full' is everything that remains
DEFAULT_TEST_FRAC = 0.2       # BEADs: 20% of original train is held out
DEFAULT_VAL_FRAC = 0.1        # simple datasets: 10% val, 10% test
DEFAULT_SIMPLE_TEST_FRAC = 0.1


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


# ---------------------------------------------------------------------------
# Pure helpers (reused across dataset paths)
# ---------------------------------------------------------------------------


def to_records(df: pd.DataFrame, label0_text: str, label1_text: str) -> list[dict]:
    label_map = {0: label0_text, 1: label1_text}
    out = []
    for _, row in df.iterrows():
        label_int = int(row["label_int"])
        if label_int not in label_map:
            raise SystemExit(f"Unexpected label value: {label_int}. Expected 0/1.")
        text = row["text"]
        if not isinstance(text, str) or not text.strip():
            continue
        out.append({"text": text, "label_int": label_int, "label_str": label_map[label_int]})
    return out


def write_jsonl(records: list[dict], path: Path) -> str:
    """Write JSONL deterministically and return its SHA256 hex digest."""
    path.parent.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha256()
    with path.open("w") as f:
        for r in records:
            line = json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n"
            f.write(line)
            h.update(line.encode("utf-8"))
    return h.hexdigest()


def class_balance(records: list[dict]) -> dict:
    if not records:
        return {"n": 0, "label_0": 0, "label_1": 0, "pct_label_1": 0.0}
    n = len(records)
    n1 = sum(1 for r in records if r["label_int"] == 1)
    return {"n": n, "label_0": n - n1, "label_1": n1, "pct_label_1": round(n1 / n, 4)}


def name_for_size(s: int) -> str:
    """train_100 / train_500 / train_1k / train_5k style — used for the size dir name."""
    if s % 1000 == 0 and s >= 1000:
        return f"{s // 1000}k"
    return str(s)


def carve_nested(train_pool_df: pd.DataFrame, sizes: list[int], seed: int):
    """Build nested stratified subsets so smaller subsets are subsets of larger ones."""
    sizes_asc = sorted(sizes)
    sizes_desc = list(reversed(sizes_asc))

    if max(sizes_asc) > len(train_pool_df):
        raise SystemExit(
            f"max subset size {max(sizes_asc)} exceeds train_pool size {len(train_pool_df)}"
        )

    out: dict[str, pd.DataFrame] = {"full": train_pool_df.copy()}
    parent = train_pool_df
    for s in sizes_desc:
        _rest, smaller = train_test_split(
            parent,
            test_size=s,
            stratify=parent["label_int"],
            random_state=seed,
        )
        out[name_for_size(s)] = smaller
        parent = smaller

    ordered_names = [name_for_size(s) for s in sizes_asc] + ["full"]
    return out, ordered_names


def verify_nesting(records_by_name: dict[str, list[dict]], order: list[str]) -> None:
    sets = {name: set(json.dumps(r, sort_keys=True) for r in records_by_name[name]) for name in order}
    for i in range(len(order) - 1):
        smaller, larger = order[i], order[i + 1]
        if not sets[smaller].issubset(sets[larger]):
            missing = len(sets[smaller] - sets[larger])
            raise SystemExit(
                f"Nesting violated: {missing} rows in {smaller} not present in {larger}"
            )


# ---------------------------------------------------------------------------
# Mock data (smoke testing without HF / disk)
# ---------------------------------------------------------------------------


def build_mock(n_train: int = 6000, n_val: int = 600):
    rng = np.random.default_rng(SEED)
    biased = [
        "Those people always cause problems.",
        "Everyone knows that group can't be trusted.",
        "It is obvious which side is correct on this issue.",
    ]
    neutral = [
        "The committee released its annual report this morning.",
        "Researchers measured the temperature at three locations.",
        "The library will close at 6pm on Sunday.",
    ]

    def gen(n):
        rows = []
        for i in range(n):
            label = int(rng.integers(0, 2))
            tpl = biased if label == 1 else neutral
            rows.append({
                "id": f"mock_{i}",
                "text": tpl[i % len(tpl)] + f" (item {i})",
                "label_int": label,
            })
        return pd.DataFrame(rows)

    return gen(n_train), gen(n_val)


# ---------------------------------------------------------------------------
# BEADs (multi-size nested sweep)
# ---------------------------------------------------------------------------


def _prepare_beads(args) -> dict:
    if args.mock:
        print("[freeze] beads: using synthetic mock dataset.")
        train_df, val_df = build_mock()
        source = "MOCK"
    elif args.from_hf:
        print(f"[freeze] beads: loading from HF ({load_beads.HF_NAME}:{load_beads.HF_CONFIG}) ...")
        train_df, val_df = load_beads.load_split_hf()
        source = f"hf:{load_beads.HF_NAME}:{load_beads.HF_CONFIG}"
    else:
        csv_dir = Path(args.csv_dir)
        print(f"[freeze] beads: loading CSVs from {csv_dir} ...")
        train_df, val_df = load_beads.load_split_csv(csv_dir)
        source = f"csv:{csv_dir}"

    print(f"[freeze] beads: original train rows={len(train_df)}, validation rows={len(val_df)}")

    train_pool_df, test_df = train_test_split(
        train_df,
        test_size=args.test_frac,
        stratify=train_df["label_int"],
        random_state=args.seed,
    )
    print(f"[freeze] beads: after test carve-off: train_pool={len(train_pool_df)}, test={len(test_df)}")

    nested_dfs, nested_order = carve_nested(train_pool_df, args.sizes, args.seed)

    # Convert each split to records once.
    train_records_by_size: dict[str, list[dict]] = {}
    for size_name, df in nested_dfs.items():
        train_records_by_size[size_name] = to_records(df, args.label0_text, args.label1_text)
    val_records = to_records(val_df, args.label0_text, args.label1_text)
    test_records = to_records(test_df, args.label0_text, args.label1_text)

    verify_nesting(train_records_by_size, nested_order)
    print(f"[freeze] beads: nesting verified: {' ⊂ '.join('train_' + n for n in nested_order)}")

    out_root = Path(args.out_dir) / "beads"
    sizes_block: dict[str, dict] = {}
    for size_name, train_recs in train_records_by_size.items():
        size_dir = out_root / "sizes" / size_name
        train_sha = write_jsonl(train_recs, size_dir / "train.jsonl")
        val_sha = write_jsonl(val_records, size_dir / "val.jsonl")
        test_sha = write_jsonl(test_records, size_dir / "test.jsonl")
        sizes_block[size_name] = {
            "train": {"path": f"sizes/{size_name}/train.jsonl", "sha256": train_sha, **class_balance(train_recs)},
            "val":   {"path": f"sizes/{size_name}/val.jsonl",   "sha256": val_sha,   **class_balance(val_records)},
            "test":  {"path": f"sizes/{size_name}/test.jsonl",  "sha256": test_sha,  **class_balance(test_records)},
        }

    manifest = {
        "schema_version": 2,
        "dataset": "beads",
        "seed": args.seed,
        "source": source,
        "label_map": {"0": args.label0_text, "1": args.label1_text},
        "split_strategy": "preserve_native_val_then_carve_test_then_nested_subsets",
        "test_frac_of_original_train": args.test_frac,
        "nesting": " ⊂ ".join("train_" + n for n in nested_order),
        "sizes": sizes_block,
    }
    return _emit_manifest(manifest, out_root)


# ---------------------------------------------------------------------------
# Simple datasets (BABE, cajcodes, WNC) — single 80/10/10 split
# ---------------------------------------------------------------------------


SIMPLE_LOADERS = {
    "babe": load_babe,
    "cajcodes": load_cajcodes,
    "wnc": load_wnc,
}


def _prepare_simple(args) -> dict:
    name = args.dataset
    loader = SIMPLE_LOADERS[name]
    print(f"[freeze] {name}: loading via {loader.__name__} ...")
    df = loader.load()
    print(f"[freeze] {name}: total rows={len(df)}, label dist={df['label_int'].value_counts().to_dict()}")

    # 80/10/10 stratified — two splits because train_test_split is binary.
    train_df, rest_df = train_test_split(
        df,
        test_size=args.val_frac + args.test_frac_simple,
        stratify=df["label_int"],
        random_state=args.seed,
    )
    val_share = args.val_frac / (args.val_frac + args.test_frac_simple)
    val_df, test_df = train_test_split(
        rest_df,
        test_size=1 - val_share,
        stratify=rest_df["label_int"],
        random_state=args.seed,
    )
    print(f"[freeze] {name}: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    train_records = to_records(train_df, args.label0_text, args.label1_text)
    val_records = to_records(val_df, args.label0_text, args.label1_text)
    test_records = to_records(test_df, args.label0_text, args.label1_text)

    out_root = Path(args.out_dir) / name
    size_dir = out_root / "full"
    train_sha = write_jsonl(train_records, size_dir / "train.jsonl")
    val_sha = write_jsonl(val_records, size_dir / "val.jsonl")
    test_sha = write_jsonl(test_records, size_dir / "test.jsonl")

    manifest = {
        "schema_version": 2,
        "dataset": name,
        "seed": args.seed,
        "source": f"hf:{getattr(loader, 'HF_NAME', '?')}" if name != "wnc" else "local:data/wnc/bias_data/WNC",
        "label_map": {"0": args.label0_text, "1": args.label1_text},
        "split_strategy": "stratified_80_10_10",
        "val_frac": args.val_frac,
        "test_frac": args.test_frac_simple,
        "sizes": {
            "full": {
                "train": {"path": "full/train.jsonl", "sha256": train_sha, **class_balance(train_records)},
                "val":   {"path": "full/val.jsonl",   "sha256": val_sha,   **class_balance(val_records)},
                "test":  {"path": "full/test.jsonl",  "sha256": test_sha,  **class_balance(test_records)},
            }
        },
    }
    return _emit_manifest(manifest, out_root)


def _emit_manifest(manifest: dict, out_root: Path) -> dict:
    manifest_path = out_root / "splits_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"[freeze] wrote manifest → {manifest_path}")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return manifest


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--dataset", required=True, choices=["beads", "babe", "cajcodes", "wnc"])
    ap.add_argument("--out-dir", default="data/frozen",
                    help="Root output dir. Per-dataset subdir is appended.")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--label0-text", default="non-biased")
    ap.add_argument("--label1-text", default="biased")

    # BEADs-only options
    ap.add_argument("--csv-dir", default="data/bead", help="(beads) BEADs CSV directory.")
    ap.add_argument("--from-hf", action="store_true", help="(beads) Load from HuggingFace instead of CSV.")
    ap.add_argument("--mock", action="store_true", help="(beads) Use synthetic mock data.")
    ap.add_argument("--test-frac", type=float, default=DEFAULT_TEST_FRAC,
                    help="(beads) Fraction of native train to hold out as test.")
    ap.add_argument("--sizes", default=",".join(str(s) for s in SWEEP_SIZES),
                    help="(beads) Comma-separated subset sizes (excluding 'full').")

    # Simple-dataset options
    ap.add_argument("--val-frac", type=float, default=DEFAULT_VAL_FRAC,
                    help="(simple datasets) Fraction of full data assigned to validation.")
    ap.add_argument("--test-frac-simple", type=float, default=DEFAULT_SIMPLE_TEST_FRAC,
                    help="(simple datasets) Fraction of full data assigned to test.")

    args = ap.parse_args()
    set_seed(args.seed)
    args.sizes = [int(s) for s in args.sizes.split(",") if s.strip()]

    if args.dataset == "beads":
        _prepare_beads(args)
    else:
        _prepare_simple(args)


if __name__ == "__main__":
    main()
