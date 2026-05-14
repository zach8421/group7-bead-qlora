"""Prepare BEAD bias-classification splits for the QLoRA calibration run.

What this does
--------------
1. Loads BEAD from the pre-downloaded CSVs in ./data/bead/ by default
   (the group7 layout). Switch to HuggingFace download with --from-hf, point
   at a different CSV dir with --csv-dir, or use synthetic data with --mock.
2. Carves a held-out test split off the original *train* split (default: 20%,
   stratified on label).
3. Keeps the dataset's provided validation split as-is (validation is for
   tuning during the full sweep; the calibration run does not need it but we
   save it so the directory matches the v2 plan).
4. From what remains of the original train split, draws a stratified
   subset of size --train-size (default 1000) for the calibration run.
5. Writes everything as JSONL (one example per line) into ./data/processed/
   plus a splits_summary.json with counts and class balance.

Each output line:
  {"text": "<sentence>", "label_int": 0|1, "label_str": "biased"|"non-biased"}

Label mapping
-------------
Per data/bead/README.md: 1 = biased, 0 = non-biased. Override with
--label0-text / --label1-text if a future drop flips this.

Smoke test (no internet, no HF gating)
--------------------------------------
  python scripts/prepare_bead_splits.py --mock --train-size 64 --out-dir data/processed_mock
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


SEED = 42


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def load_bead_csv(csv_dir: Path, text_col: str, label_col: str):
    """Load BEAD from the pre-rsynced CSVs. Returns (train_df, val_df)."""
    train_path = csv_dir / "bias-train.csv"
    val_path = csv_dir / "bias-valid.csv"
    for p in (train_path, val_path):
        if not p.exists():
            raise SystemExit(
                f"Missing {p}. Expected the BEAD CSV layout under {csv_dir}. "
                f"Pass --from-hf to download from HuggingFace, or --csv-dir to point elsewhere."
            )
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    for df, name in [(train_df, "train"), (val_df, "validation")]:
        for col in (text_col, label_col):
            if col not in df.columns:
                raise SystemExit(
                    f"Column '{col}' not in {name} CSV. Available: {list(df.columns)}. "
                    f"Override with --text-col / --label-col."
                )
    return (
        train_df[[text_col, label_col]].rename(columns={text_col: "text", label_col: "label_int"}),
        val_df[[text_col, label_col]].rename(columns={text_col: "text", label_col: "label_int"}),
    )


def load_bead_hf(dataset_name: str, dataset_config: str, text_col: str, label_col: str):
    """Load BEAD via HuggingFace datasets. Returns (train_df, val_df)."""
    from datasets import load_dataset

    ds = load_dataset(dataset_name, dataset_config)
    if "train" not in ds:
        raise SystemExit(f"Dataset {dataset_name}:{dataset_config} has no 'train' split. Got: {list(ds.keys())}")
    train_df = ds["train"].to_pandas()
    if "validation" in ds:
        val_df = ds["validation"].to_pandas()
    elif "valid" in ds:
        val_df = ds["valid"].to_pandas()
    elif "val" in ds:
        val_df = ds["val"].to_pandas()
    else:
        raise SystemExit(f"Dataset has no validation split. Got: {list(ds.keys())}")
    for df, name in [(train_df, "train"), (val_df, "validation")]:
        for col in (text_col, label_col):
            if col not in df.columns:
                raise SystemExit(
                    f"Column '{col}' not in {name} split. Available: {list(df.columns)}. "
                    f"Override with --text-col / --label-col."
                )
    return train_df[[text_col, label_col]].rename(columns={text_col: "text", label_col: "label_int"}), \
           val_df[[text_col, label_col]].rename(columns={text_col: "text", label_col: "label_int"})


def build_mock(n_train: int = 6000, n_val: int = 600) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Synthetic BEAD-shaped data for offline smoke tests. ~50/50 balance."""
    rng = np.random.default_rng(SEED)
    biased_templates = [
        "Those people always cause problems.",
        "Everyone knows that group can't be trusted.",
        "It is obvious which side is correct on this issue.",
    ]
    neutral_templates = [
        "The committee released its annual report this morning.",
        "Researchers measured the temperature at three locations.",
        "The library will close at 6pm on Sunday.",
    ]

    def gen(n):
        rows = []
        for i in range(n):
            label = int(rng.integers(0, 2))
            tpl = biased_templates if label == 1 else neutral_templates
            rows.append({"text": tpl[i % len(tpl)] + f" (item {i})", "label_int": label})
        return pd.DataFrame(rows)

    return gen(n_train), gen(n_val)


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


def write_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def class_balance(records: list[dict]) -> dict:
    if not records:
        return {"n": 0, "label_0": 0, "label_1": 0, "pct_label_1": 0.0}
    n = len(records)
    n1 = sum(1 for r in records if r["label_int"] == 1)
    return {"n": n, "label_0": n - n1, "label_1": n1, "pct_label_1": round(n1 / n, 4)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv-dir", default="data/bead",
                    help="Directory containing bias-train.csv + bias-valid.csv. Default: data/bead")
    ap.add_argument("--from-hf", action="store_true",
                    help="Download from HuggingFace instead of reading local CSVs")
    ap.add_argument("--dataset", default="shainar/BEAD",
                    help="HuggingFace dataset path (used only with --from-hf)")
    ap.add_argument("--config", default="Bias_classification",
                    help="HuggingFace dataset config (used only with --from-hf)")
    ap.add_argument("--text-col", default="text", help="Text column name in the dataset")
    ap.add_argument("--label-col", default="label", help="Integer label column name")
    ap.add_argument("--label0-text", default="non-biased", help="String label for class 0")
    ap.add_argument("--label1-text", default="biased", help="String label for class 1")
    ap.add_argument("--train-size", type=int, default=1000,
                    help="Stratified subset size for the 1k calibration run")
    ap.add_argument("--test-frac", type=float, default=0.2,
                    help="Fraction of original train carved off as held-out test")
    ap.add_argument("--out-dir", default="data/processed", help="Output directory")
    ap.add_argument("--mock", action="store_true",
                    help="Use synthetic data instead of downloading BEAD (offline smoke test)")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    set_seed(args.seed)

    if args.mock:
        print("[prepare] Using synthetic mock dataset (offline mode).")
        train_df, val_df = build_mock()
    elif args.from_hf:
        print(f"[prepare] Loading {args.dataset}:{args.config} from HuggingFace ...")
        train_df, val_df = load_bead_hf(args.dataset, args.config, args.text_col, args.label_col)
    else:
        csv_dir = Path(args.csv_dir)
        print(f"[prepare] Loading BEAD CSVs from {csv_dir} ...")
        train_df, val_df = load_bead_csv(csv_dir, args.text_col, args.label_col)

    print(f"[prepare] Original train rows: {len(train_df)}; validation rows: {len(val_df)}")

    # Stratified 80/20 split off original train: 80% becomes train_pool, 20% becomes test.
    train_pool_df, test_df = train_test_split(
        train_df,
        test_size=args.test_frac,
        stratify=train_df["label_int"],
        random_state=args.seed,
    )
    print(f"[prepare] After test carve-off: train_pool={len(train_pool_df)}, test={len(test_df)}")

    if args.train_size > len(train_pool_df):
        raise SystemExit(
            f"--train-size {args.train_size} exceeds available train_pool size {len(train_pool_df)}"
        )

    train_pool_df, train_1k_df = train_test_split(
        train_pool_df,
        test_size=args.train_size,
        stratify=train_pool_df["label_int"],
        random_state=args.seed,
    )
    # train_1k_df is the stratified subset; train_pool_df is the leftover (saved as train_full for v2).
    print(f"[prepare] Calibration train subset: {len(train_1k_df)}; remaining train_full: {len(train_pool_df)}")

    out_dir = Path(args.out_dir)
    train_1k_records = to_records(train_1k_df, args.label0_text, args.label1_text)
    train_full_records = to_records(train_pool_df, args.label0_text, args.label1_text)
    val_records = to_records(val_df, args.label0_text, args.label1_text)
    test_records = to_records(test_df, args.label0_text, args.label1_text)

    write_jsonl(train_1k_records, out_dir / "train_1k.jsonl")
    write_jsonl(train_full_records, out_dir / "train_full.jsonl")
    write_jsonl(val_records, out_dir / "val.jsonl")
    write_jsonl(test_records, out_dir / "test_held_out.jsonl")

    if args.mock:
        source = "MOCK"
    elif args.from_hf:
        source = f"hf:{args.dataset}:{args.config}"
    else:
        source = f"csv:{args.csv_dir}"
    summary = {
        "seed": args.seed,
        "source": source,
        "label_map": {"0": args.label0_text, "1": args.label1_text},
        "splits": {
            "train_1k": class_balance(train_1k_records),
            "train_full": class_balance(train_full_records),
            "val": class_balance(val_records),
            "test_held_out": class_balance(test_records),
        },
        "test_frac_of_original_train": args.test_frac,
    }
    summary_path = out_dir / "splits_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print(f"[prepare] Wrote splits to {out_dir}/")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
