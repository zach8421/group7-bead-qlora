"""BEADs loader. Source: HuggingFace ``shainar/BEAD`` (``Bias_classification`` config).

Local cache lives at ``data/bead/{bias-train,bias-valid}.csv`` (CC BY-NC 4.0;
gitignored). BEADs is the only one of the four cross-eval datasets that arrives
*pre-split* into train + validation — keep that boundary; the test split is
later carved off the train side inside ``freeze_splits.py`` to match the
existing v2 sweep behavior.

Returns either ``load()`` (single DataFrame, for parity with the other loaders)
or ``load_split()`` (the pre-existing train + valid pair, for ``freeze_splits``
to consume).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

DEFAULT_CSV_DIR = Path("data/bead")
HF_NAME = "shainar/BEAD"
HF_CONFIG = "Bias_classification"
TEXT_COL = "text"
LABEL_COL = "label"


def _normalize(df: pd.DataFrame, id_prefix: str) -> pd.DataFrame:
    out = df[[TEXT_COL, LABEL_COL]].rename(columns={TEXT_COL: "text", LABEL_COL: "label_int"})
    out = out.reset_index(drop=True)
    out["id"] = id_prefix + out.index.astype(str)
    return out[["id", "text", "label_int"]]


def load_split_csv(csv_dir: Path = DEFAULT_CSV_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_path = csv_dir / "bias-train.csv"
    val_path = csv_dir / "bias-valid.csv"
    for p in (train_path, val_path):
        if not p.exists():
            raise FileNotFoundError(
                f"Missing {p}. Either copy the BEADs CSVs there or call load_split_hf()."
            )
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    return _normalize(train_df, "beads_train_"), _normalize(val_df, "beads_val_")


def load_split_hf() -> tuple[pd.DataFrame, pd.DataFrame]:
    from datasets import load_dataset

    ds = load_dataset(HF_NAME, HF_CONFIG)
    val_key = next((k for k in ("validation", "valid", "val") if k in ds), None)
    if val_key is None:
        raise RuntimeError(f"BEADs has no validation split. Got: {list(ds.keys())}")
    return (
        _normalize(ds["train"].to_pandas(), "beads_train_"),
        _normalize(ds[val_key].to_pandas(), "beads_val_"),
    )


def load() -> pd.DataFrame:
    """Concatenated train+valid for parity with the single-DataFrame loaders.

    ``freeze_splits.py`` uses ``load_split_*`` directly for BEADs to preserve
    the original train/valid boundary. This entry point exists so memory-light
    smoke checks (Phase 7) can iterate uniformly.
    """
    try:
        train_df, val_df = load_split_csv()
    except FileNotFoundError:
        train_df, val_df = load_split_hf()
    return pd.concat([train_df, val_df], ignore_index=True)
