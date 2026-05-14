"""Freeze the v2 BEAD splits for the Week-7 sweep.

What this does
--------------
1. Loads BEAD from `data/bead/` (or HuggingFace with --from-hf, mock with --mock).
2. Carves a 20% stratified test split off the original train (test_held_out).
3. Keeps the dataset's validation split as-is (val).
4. From the remaining train_pool, builds a chain of *nested* stratified subsets
   so the smaller splits are byte-subsets of the larger ones:

       train_100 ⊂ train_500 ⊂ train_1k ⊂ train_5k ⊂ train_full

5. Writes JSONL for every split into `data/frozen/`.
6. Computes a SHA256 over the (newline-joined) JSONL bytes of each split and
   stores them, alongside per-split counts and class balance, in
   `data/frozen/splits_manifest.json`. This manifest is the source of truth
   the teammates' TF-IDF / 3-shot baselines and the QLoRA sweep all reference.

Why nest instead of independently re-stratifying each size
----------------------------------------------------------
A learning-curve study should answer "what does the model gain from seeing
*more* data?" — keeping smaller subsets as prefixes of larger ones isolates
that question from sampling noise across subset draws.

Note vs the v1 calibration splits
---------------------------------
`data/processed/` (v1) drew train_1k directly from train_pool with a
different stratified call. The v1 splits are kept as-is for reproducibility
of the v1 calibration result; the v2 sweep uses `data/frozen/` from here on.

Each output line:
    {"text": "<sentence>", "label_int": 0|1, "label_str": "biased"|"non-biased"}

Smoke test
----------
    python scripts/freeze_splits.py --mock --out-dir data/frozen_mock
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


SEED = 42
SWEEP_SIZES = [100, 500, 1000, 5000]  # 'full' is everything that remains


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def load_bead_csv(csv_dir: Path, text_col: str, label_col: str):
    train_path = csv_dir / "bias-train.csv"
    val_path = csv_dir / "bias-valid.csv"
    for p in (train_path, val_path):
        if not p.exists():
            raise SystemExit(
                f"Missing {p}. Pass --from-hf to download from HuggingFace, "
                f"or --csv-dir to point elsewhere."
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
        train_df[[text_col, label_col]].rename(
            columns={text_col: "text", label_col: "label_int"}
        ),
        val_df[[text_col, label_col]].rename(
            columns={text_col: "text", label_col: "label_int"}
        ),
    )


def load_bead_hf(dataset_name: str, dataset_config: str, text_col: str, label_col: str):
    from datasets import load_dataset

    ds = load_dataset(dataset_name, dataset_config)
    if "train" not in ds:
        raise SystemExit(f"Dataset {dataset_name}:{dataset_config} has no 'train' split.")
    train_df = ds["train"].to_pandas()
    val_key = next((k for k in ("validation", "valid", "val") if k in ds), None)
    if val_key is None:
        raise SystemExit(f"Dataset has no validation split. Got: {list(ds.keys())}")
    val_df = ds[val_key].to_pandas()
    return (
        train_df[[text_col, label_col]].rename(columns={text_col: "text", label_col: "label_int"}),
        val_df[[text_col, label_col]].rename(columns={text_col: "text", label_col: "label_int"}),
    )


def build_mock(n_train: int = 6000, n_val: int = 600):
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
    """train_100 / train_500 / train_1k / train_5k style. Falls back to train_<n> for arbitrary sizes."""
    if s % 1000 == 0 and s >= 1000:
        return f"train_{s // 1000}k"
    return f"train_{s}"


def carve_nested(train_pool_df: pd.DataFrame, sizes: list[int], seed: int) -> tuple[dict[str, pd.DataFrame], list[str]]:
    """Build nested stratified subsets: smallest ⊂ next ⊂ ... ⊂ train_pool.

    Strategy: sort sizes ascending. Walk descending: at each step, stratify-split
    the current parent into (rest, smaller). The "smaller" becomes the next parent.

    Returns (dict of name -> df including 'train_full', ordered list of names ascending).
    """
    sizes_asc = sorted(sizes)
    sizes_desc = list(reversed(sizes_asc))

    if max(sizes_asc) > len(train_pool_df):
        raise SystemExit(
            f"max subset size {max(sizes_asc)} exceeds train_pool size {len(train_pool_df)}"
        )

    out: dict[str, pd.DataFrame] = {"train_full": train_pool_df.copy()}
    parent = train_pool_df
    for s in sizes_desc:
        _rest, smaller = train_test_split(
            parent,
            test_size=s,
            stratify=parent["label_int"],
            random_state=seed,
        )
        out[name_for_size(s)] = smaller
        parent = smaller  # next-smaller subset comes from this one (guarantees nesting)

    ordered_names = [name_for_size(s) for s in sizes_asc] + ["train_full"]
    return out, ordered_names


def verify_nesting(splits: dict[str, list[dict]], order: list[str]) -> None:
    """Assert smallest ⊂ ... ⊂ train_full for the given chain of split names."""
    sets = {name: set(json.dumps(r, sort_keys=True) for r in splits[name]) for name in order}
    for i in range(len(order) - 1):
        smaller, larger = order[i], order[i + 1]
        if not sets[smaller].issubset(sets[larger]):
            missing = len(sets[smaller] - sets[larger])
            raise SystemExit(
                f"Nesting violated: {missing} rows in {smaller} not present in {larger}"
            )


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--csv-dir", default="data/bead")
    ap.add_argument("--from-hf", action="store_true")
    ap.add_argument("--dataset", default="shainar/BEAD")
    ap.add_argument("--config", default="Bias_classification")
    ap.add_argument("--text-col", default="text")
    ap.add_argument("--label-col", default="label")
    ap.add_argument("--label0-text", default="non-biased")
    ap.add_argument("--label1-text", default="biased")
    ap.add_argument(
        "--sizes",
        default=",".join(str(s) for s in SWEEP_SIZES),
        help="Comma-separated subset sizes (excluding 'full'). Default 100,500,1000,5000",
    )
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--out-dir", default="data/frozen")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    set_seed(args.seed)
    sizes = [int(s) for s in args.sizes.split(",") if s.strip()]

    if args.mock:
        print("[freeze] Using synthetic mock dataset.")
        train_df, val_df = build_mock()
        source = "MOCK"
    elif args.from_hf:
        print(f"[freeze] Loading {args.dataset}:{args.config} from HuggingFace ...")
        train_df, val_df = load_bead_hf(args.dataset, args.config, args.text_col, args.label_col)
        source = f"hf:{args.dataset}:{args.config}"
    else:
        csv_dir = Path(args.csv_dir)
        print(f"[freeze] Loading BEAD CSVs from {csv_dir} ...")
        train_df, val_df = load_bead_csv(csv_dir, args.text_col, args.label_col)
        source = f"csv:{args.csv_dir}"

    print(f"[freeze] Original train rows: {len(train_df)}; validation rows: {len(val_df)}")

    train_pool_df, test_df = train_test_split(
        train_df,
        test_size=args.test_frac,
        stratify=train_df["label_int"],
        random_state=args.seed,
    )
    print(
        f"[freeze] After test carve-off: train_pool={len(train_pool_df)}, test={len(test_df)}"
    )

    nested_dfs, nested_order = carve_nested(train_pool_df, sizes, args.seed)

    out_dir = Path(args.out_dir)
    splits_records: dict[str, list[dict]] = {}
    for name, df in nested_dfs.items():
        splits_records[name] = to_records(df, args.label0_text, args.label1_text)
    splits_records["val"] = to_records(val_df, args.label0_text, args.label1_text)
    splits_records["test_held_out"] = to_records(test_df, args.label0_text, args.label1_text)

    # Sanity: prove the nesting invariant before we hand the files to teammates.
    verify_nesting(splits_records, nested_order)
    print(f"[freeze] Nesting verified: {' ⊂ '.join(nested_order)}")

    hashes: dict[str, str] = {}
    for name, records in splits_records.items():
        path = out_dir / f"{name}.jsonl"
        hashes[name] = write_jsonl(records, path)

    manifest = {
        "schema_version": 1,
        "seed": args.seed,
        "source": source,
        "label_map": {"0": args.label0_text, "1": args.label1_text},
        "test_frac_of_original_train": args.test_frac,
        "nesting": " ⊂ ".join(nested_order),
        "splits": {
            name: {
                "path": f"{name}.jsonl",
                "sha256": hashes[name],
                **class_balance(splits_records[name]),
            }
            for name in splits_records
        },
    }
    manifest_path = out_dir / "splits_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

    print(f"[freeze] Wrote splits + manifest to {out_dir}/")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
