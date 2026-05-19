"""Per-dataset loaders for the multi-dataset cross-eval pipeline.

This package is intentionally **not** named ``datasets`` — Python's import
resolution puts ``scripts/`` on ``sys.path`` when a script in that dir is
invoked, which would cause ``scripts/datasets/`` to shadow the HuggingFace
``datasets`` library inside the loaders.

Each submodule exposes ``load() -> pandas.DataFrame`` with columns
``id`` (str, unique), ``text`` (str, non-empty), ``label_int`` (int in {0, 1}).
``label_int == 1`` means biased; ``0`` means non-biased / neutral.

The dataset-specific quirks live here. Stratified train/val/test splitting,
JSONL emission, SHA256 manifesting, and label_str rendering all live in the
unified ``scripts/freeze_splits.py`` which dispatches on ``--dataset``.
"""
from __future__ import annotations

from . import load_babe, load_beads, load_cajcodes, load_wnc

LOADERS = {
    "beads": load_beads,
    "babe": load_babe,
    "cajcodes": load_cajcodes,
    "wnc": load_wnc,
}

__all__ = ["LOADERS", "load_babe", "load_beads", "load_cajcodes", "load_wnc"]
