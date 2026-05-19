"""cajcodes/political-bias loader. Source: HuggingFace ``cajcodes/political-bias``.

The native ``label`` is a 0-4 ordinal on a left-right spectrum where 2 is
center. We collapse to BEADs' biased/non-biased framing:

  label_int = 0  iff  native label == 2  (center)
  label_int = 1  otherwise                (any kind of slant)

Synthetic statements, sentence-length. No native ID column → synthesized.
"""
from __future__ import annotations

import pandas as pd

HF_NAME = "cajcodes/political-bias"
CENTER_LABEL = 2


def load() -> pd.DataFrame:
    from datasets import load_dataset

    ds = load_dataset(HF_NAME, split="train").to_pandas()
    if "text" not in ds.columns or "label" not in ds.columns:
        raise RuntimeError(f"Unexpected cajcodes columns: {list(ds.columns)}")
    out = pd.DataFrame({
        "id": "caj_" + ds.index.astype(str),
        "text": ds["text"],
        "label_int": (ds["label"].astype(int) != CENTER_LABEL).astype(int),
    })
    return out[["id", "text", "label_int"]]
