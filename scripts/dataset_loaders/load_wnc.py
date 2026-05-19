"""WNC loader. Source: ``data/wnc/bias_data/WNC/biased.word.{train,dev,test}`` TSVs.

Acquired by unzipping the corpus from https://www.dropbox.com/s/qol3rmn0rq0dfhn/bias_data.zip
(see the WNC README for the canonical link). All three WNC files are unioned
here — the original train/dev/test boundary is dropped so the downstream
stratified 80/10/10 in ``freeze_splits.py`` produces a test set large enough
for cross-eval (~11k rows vs WNC's native 1k test).

Each TSV row is a (biased, neutral) sentence pair edited by a Wikipedia NPOV
revision. We expand each pair into two rows:

  src_raw → label_int=1 (biased,  id=wnc_b_{pair_id}_{row})
  tgt_raw → label_int=0 (neutral, id=wnc_n_{pair_id}_{row})

The pair_id is not unique across rows (multiple edits can share a diff id), so
we suffix with the row index to keep IDs unique.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

DEFAULT_DATA_DIR = Path("data/wnc/bias_data/WNC")
TSV_COLS = ["pair_id", "src_tok", "tgt_tok", "src_raw", "tgt_raw", "src_pos", "tgt_parse"]
TSV_FILES = ["biased.word.train", "biased.word.dev", "biased.word.test"]


def _read_one(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        sep="\t",
        names=TSV_COLS,
        header=None,
        dtype=str,
        keep_default_na=False,
        on_bad_lines="skip",
        engine="python",
    )


def load(data_dir: Path = DEFAULT_DATA_DIR) -> pd.DataFrame:
    parts = []
    for fname in TSV_FILES:
        p = data_dir / fname
        if not p.exists():
            raise FileNotFoundError(
                f"WNC TSV not found at {p}. Download bias_data.zip and unzip into data/wnc/."
            )
        parts.append(_read_one(p))
    df = pd.concat(parts, ignore_index=True)
    # Drop rows where either side is empty — rare but possible after a bad-line skip.
    df = df[(df["src_raw"].str.len() > 0) & (df["tgt_raw"].str.len() > 0)].reset_index(drop=True)

    biased = pd.DataFrame({
        "id": "wnc_b_" + df["pair_id"].astype(str) + "_" + df.index.astype(str),
        "text": df["src_raw"],
        "label_int": 1,
    })
    neutral = pd.DataFrame({
        "id": "wnc_n_" + df["pair_id"].astype(str) + "_" + df.index.astype(str),
        "text": df["tgt_raw"],
        "label_int": 0,
    })
    out = pd.concat([biased, neutral], ignore_index=True)
    return out[["id", "text", "label_int"]]
