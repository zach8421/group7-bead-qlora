# Build log — BEAD QLoRA sweep (group7)

Reverse-chronological log of architecturally meaningful changes, decisions,
and measurements. Intended as source material for the final technical
writeup. One entry per change worth remembering — not every commit.

For per-commit history, use `git log`. For per-run reproducibility metadata,
use the `run_meta.json` files in `outputs/<run>/`.

---

## 2026-05-16 — Batched eval scoring

**What changed**

[scripts/eval_adapter.py](../scripts/eval_adapter.py) rewritten so the
likelihood-scoring pass over the held-out test set runs in batches instead
of one (row, label) at a time:

- `prepare_scoring_items()` tokenizes each prompt once, then tokenizes
  `prompt + label` once per (row, label). The full prompt+label string is
  still tokenized per item so BPE boundary effects at the
  prompt-end/label-start boundary are byte-identical to the previous
  single-example path.
- `score_items_batched()` right-pads sequences within each batch, builds
  the attention mask, runs one `torch.inference_mode()` forward per batch,
  gathers per-token log-probs only at completion positions via a
  `(B, T-1)` boolean mask, and accumulates per-row scores in a GPU tensor.
- Results stay on the GPU until the entire eval is done — one CPU sync at
  the end instead of one per row.
- `--eval-batch-size N` (default 16) added; threaded through as
  `EVAL_BATCH_SIZE` in [scripts/run_qlora.slurm](../scripts/run_qlora.slurm).
- Eval-side base-model load now uses `device_map={"": 0}` (single GPU)
  instead of `"auto"`.

**Why**

The v1 eval path was `~7.9 rows/sec` on H200 — 864 s for the 6,816-row
test set in qlora_100. Inspection showed: every row did **2** forward
passes at batch size 1, **4** tokenizer calls, and forced a CUDA sync per
`.item()` to read out each scalar score. The H200 was idling. With the
test set unchanged across the 5 sweep sizes, fixing this once was worth
~13 min per sweep run, ~1 hour cumulative across the sweep.

**Verification**

Local smoke flow (fp32 tiny-Llama on MPS, 20 rows): predictions and
per-class scores **bit-identical** between `--eval-batch-size 1` and
`--eval-batch-size 16`. On Tillicum at bf16 the batched path will diverge
from bs=1 by reduction-order noise only — orders of magnitude below the
typical class-score gap, so argmax decisions stay stable.

**Expected impact**

~8–14× wall-clock speedup at batch 16. To be confirmed by the first
re-run of qlora_100 against the new code.

**Open follow-ups**

- The slurm wrapper launches `train_qlora.py` and `eval_adapter.py` as
  two separate Python processes, so the 8B base model loads twice per
  job (~30–90 s each on GPFS). A single-process orchestrator that loads
  once, trains, swaps the adapter into eval mode, and scores would
  recover this. Bigger refactor than this entry; deferred.

---

## 2026-05-14 — qlora_100 sanity run (slurm 115121)

**What ran**

First full v2 sweep run, smallest training size:

- Train: `train_100.jsonl` (100 rows), 3 epochs, effective batch 16,
  21 global steps, 29.2 s wall-clock, 13.0 GB peak VRAM.
- Eval: held-out `test_held_out.jsonl` (6,816 rows), 863.7 s.
- Adapter: `outputs/qlora_100/adapter/` (gitignored), metrics committed
  in [outputs/qlora_100/](../outputs/qlora_100/).

**Result**

| Metric | qlora_100 |
| --- | --- |
| accuracy | 0.4990 |
| f1_pos | 0.6655 |
| f1_macro | 0.3339 |
| precision_pos | 0.4991 |
| recall_pos | 0.9982 |
| recall_neg | 0.0012 |
| train_loss_final | 1.997 |

The adapter collapsed to "always predict biased" — recall_pos 0.998,
recall_neg 0.001. The training loss never came down meaningfully (started
at 2.6 mid-epoch-1, ended at ~1.4 mid-epoch-3, reported final 1.997
because TRL averages over the run).

**Why it matters**

This was the explicit sanity-check leg called for in Risk #2 of the v2
proposal — "The 100-example run serves as an explicit sanity check before
launching larger runs." The collapse is consistent with the v1
calibration's noted asymmetry (over-prediction of biased after 1 epoch
on 1k) and is **not** evidence of a broken pipeline:

- The mechanical pipeline ran cleanly end-to-end: split verification
  against the committed `splits_manifest.json` sha256s, training,
  adapter save, eval, manifest upsert.
- The pathology is consistent with under-data, not a code bug.
- The proposal's framing — learning-curve sweep across five sizes — treats
  a degenerate left endpoint as a valid (if uninformative) data point.

**Decision**

Continue with 500 / 1k / 5k / full rather than stopping to retune. The
500-row run is the first one where the model has enough supervision to
meaningfully move off the prior.

**Performance bottleneck flagged**

Eval time (863 s) dominated training (29 s) by ~30×. Single-example
likelihood scoring identified as the bottleneck — see 2026-05-16 entry.

---

## 2026-05-13 — v2 sweep scaffolding committed

**What landed**

Initial public commit (`2e0f0db`) plus three follow-ups: 

- `scripts/freeze_splits.py` — builds nested stratified subsets
  (`train_100 ⊂ 500 ⊂ 1k ⊂ 5k ⊂ full`) and a sha256 manifest.
- `scripts/train_qlora.py` — QLoRA SFT (Llama-3.1-8B-Instruct, 4-bit
  nf4, LoRA r=16 on q/k/v/o/gate/up/down). Loss is masked to the
  assistant turn via `trl.SFTConfig(completion_only_loss=True)` (asserted
  at startup; the script exits if the trl release doesn't support that
  param).
- `scripts/eval_adapter.py` — likelihood-scored binary eval, picks the
  higher-summed-log-prob completion between `"biased"` and `"non-biased"`
  under the same chat template the model was trained on.
- `scripts/update_manifest.py` — upserts one row per run into
  `outputs/manifest.csv` and `outputs/manifest.json` from the
  per-run `run_meta.json` + metrics files.
- `scripts/run_qlora.slurm` — single launcher; `SIZE=100|500|1k|5k|full`.
- `scripts/legacy/` — preserves the v1 calibration scripts.

**Audit guards (commit `b2a291c`)**

The commit message refers to C1/C2/C3 and I4/I6. In the codebase those
became:

- **C1 — split integrity.** `verify_splits_manifest.py` hashes every
  on-disk JSONL against the committed `splits_manifest.json` before
  training; the slurm wrapper snapshots the manifest into `/tmp` first
  to detect mid-job modification.
- **C2 — input drift at training time.** `train_qlora.py` re-hashes
  `--train-jsonl` and compares to the manifest entry; mismatch is a
  hard exit.
- **C3 — TRL contract.** `train_qlora.py` inspects `SFTConfig`'s
  signature and aborts if `completion_only_loss` isn't accepted by the
  installed trl release.
- **I4 / I6 — reproducibility metadata.** Every run writes
  `run_meta.json` with: argparse values, started/finished UTC
  timestamps, slurm job id + node, CUDA visible devices, library
  versions (torch / transformers / peft / trl / bitsandbytes / datasets
  / accelerate / CUDA / GPU name), git head, input sha256s, and label
  map. `update_manifest.py` joins these into the sweep manifest.

**Tillicum sync (commit `755e44a`)**

`scripts/tillicum_sync.sh` wraps the common rsync transfers:
`push-code`, `push-data`, `pull-results`, `pull-logs`, `pull-all`,
`status`, with a `--dry-run` mode. The team shares a single clone at
`/gpfs/projects/imt526a/group7` — no per-user copies — so the sync
script also documents the coordination protocol in its help text.

---

## 2026-05-09 — v1 1k-example timing calibration

**What ran**

A 1,000-example × 1-epoch QLoRA fine-tune on H200 via Tillicum, run
**solely** to size the compute budget in the v2 proposal. Not a results
run.

**Measured**

- 1.58 min wall-clock
- 14.6 GB peak VRAM
- 10.6 examples/sec sustained throughput

**Why it shaped the project**

These three numbers extrapolated linearly across the sweep gave the
H200-hours estimates in proposal §4 — `qlora_100: 0.30 H200-hrs`,
`qlora_500: 0.34`, `qlora_1k: 0.39`, `qlora_5k: 0.85`, `qlora_full: 3.04`
— plus a 25% contingency buffer of 4 hours. The full sweep budget was
~10 H200-hrs.

A secondary observation from this run informed Risk #1: after 1 epoch
on 1k examples the model produced 54 false positives against 16 false
negatives. That asymmetry is what the qlora_100 sanity run was looking
for, and is what it found in a more extreme form (recall_pos 0.998).

**Where to read more**

- [docs/v1_calibration_writeup.md](v1_calibration_writeup.md) — full
  v1 narrative.
- [outputs/tillicum_1k_calibration/](../outputs/tillicum_1k_calibration/)
  — committed metrics; adapter weights gitignored but regenerable from
  `scripts/legacy/run_tillicum_1k.slurm`.

---

## Standing snapshot (as of 2026-05-16)

**Sweep progress**

| Run | Train rows | Status | f1_macro | accuracy | recall_pos | recall_neg |
| --- | --- | --- | --- | --- | --- | --- |
| qlora_100 | 100 | done (collapsed to majority-positive) | 0.334 | 0.499 | 0.998 | 0.001 |
| qlora_500 | 500 | not yet launched | — | — | — | — |
| qlora_1k | 1,000 | not yet launched | — | — | — | — |
| qlora_5k | 5,000 | not yet launched | — | — | — | — |
| qlora_full | 27,263 | not yet launched | — | — | — | — |

**Baselines**

| Baseline | Owner | Status |
| --- | --- | --- |
| TF-IDF + logistic regression | Abrevaa | Completed 2026-05-15, added to repo |
| 3-shot Llama-3.1-8B prompting | Ash | Paused — example selection raised label-quality concerns; awaiting instructor guidance |

**Open questions**

- Label noise: dataset inspection surfaced apparently mislabeled rows.
  3-shot example selection feels unavoidably cherry-picked given the
  noise. Awaiting Prof. Harker's guidance before proceeding.
- Model-load cost on Tillicum: ~30–90 s of GPFS read + bnb nf4 quant
  per slurm job, paid **twice** because train and eval are separate
  Python processes. Worth merging if eval-side optimizations don't get
  the per-job time low enough.
