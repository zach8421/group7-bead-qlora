"""BABE loader. Source: HuggingFace ``mediabiasgroup/BABE``.

BABE ships ``train`` (~3.1k) and ``test`` (~1k) splits with columns ``uuid``,
``text``, ``label`` already in 0/1 binary form (0=non-biased, 1=biased). We
union both splits — the downstream stratified 80/10/10 in ``freeze_splits.py``
produces the train/val/test boundary for the cross-eval (we're not using
BABE's published benchmark numbers, just its annotated data).
"""
from __future__ import annotations

import pandas as pd

HF_NAME = "mediabiasgroup/BABE"


def load() -> pd.DataFrame:
    from datasets import load_dataset

    ds = load_dataset(HF_NAME)
    frames = []
    for split_name in ("train", "test"):
        if split_name not in ds:
            continue
        frames.append(ds[split_name].to_pandas())
    if not frames:
        raise RuntimeError(f"BABE has no usable splits. Got: {list(ds.keys())}")
    df = pd.concat(frames, ignore_index=True)
    if "uuid" not in df.columns or "text" not in df.columns or "label" not in df.columns:
        raise RuntimeError(f"Unexpected BABE columns: {list(df.columns)}")
    out = df[["uuid", "text", "label"]].rename(
        columns={"uuid": "id", "label": "label_int"}
    )
    out["id"] = "babe_" + out["id"].astype(str)
    out["label_int"] = out["label_int"].astype(int)
    return out[["id", "text", "label_int"]]
