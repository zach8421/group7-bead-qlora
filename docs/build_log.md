# Build log — BEAD QLoRA sweep (group7)

Reverse-chronological log of architecturally meaningful changes, decisions,
and measurements. Intended as source material for the final technical
writeup. One entry per change worth remembering — not every commit.

For per-commit history, use `git log`. For per-run reproducibility metadata,
use the `run_meta.json` files in `outputs/<run>/`.

---

## 2026-05-16 — Full sweep complete (qlora_100 → qlora_full)

**What ran**

All five v2 sweep runs launched in parallel on Tillicum node g006 (8× H200,
each job got its own GPU). Jobs `117767`–`117771`. EVAL_BATCH_SIZE=16
(default) for all five.

**Headline result — the learning curve**

| Train rows | Accuracy | F1_macro | F1_pos | Prec_pos | Recall_pos | Train wall | Slurm |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 100 | 0.4991 | 0.3337 | 0.6657 | 0.4992 | 0.9988 | 0.5 min | 117767 |
| 500 | 0.6844 | 0.6817 | 0.7111 | 0.6549 | 0.7778 | 2.1 min | 117768 |
| 1,000 | 0.6981 | 0.6975 | 0.7108 | 0.6811 | 0.7432 | 4.1 min | 117769 |
| 5,000 | 0.7584 | 0.7584 | 0.7568 | 0.7605 | 0.7532 | 18.6 min | 117770 |
| 27,263 | **0.8036** | **0.8035** | 0.8024 | 0.8059 | 0.7990 | 99.3 min | 117771 |

Per-run metrics live in `outputs/qlora_<size>/eval_metrics.json` and
`train_metrics.json`. The consolidated manifest is at
[outputs/manifest.csv](../outputs/manifest.csv) /
[outputs/manifest.json](../outputs/manifest.json).

**What this means**

- **`qlora_full` reaches 0.804 accuracy / 0.804 F1_macro** on the 6,816-row
  held-out test set. That clears the BEAD paper's reported Llama2-7B
  baseline of 0.77 accuracy (Raza et al., 2024) by ~3 points. Headline
  number for the writeup.
- **Returns are still climbing at the right edge of the curve.**
  100→500 gives +0.185 accuracy, 500→1k gives +0.014, 1k→5k +0.060,
  5k→full +0.045. The 1k→5k and 5k→full deltas are similar in size, so
  more data would plausibly still help. Not yet in the diminishing-returns
  regime by the time the BEAD training split is exhausted.
- **The over-predict-biased asymmetry is a low-data phenomenon, not a
  permanent feature.** Precision/recall asymmetry resolves cleanly with
  scale: at 100 we have P 0.50 / R 1.00 (degenerate collapse), at 500
  P 0.65 / R 0.78, at 5k P 0.76 / R 0.75, at full P 0.81 / R 0.80
  (balanced). The proposal's Risk #1 ("model over-predicting biased after
  1 epoch on 1k") is now contextualized — it's a data-quantity artifact,
  not a stable decision boundary.
- **The 100-row collapse is reproducible.** Across the calibration sweep
  and the canonical run, all qlora_100 instances landed at ~0.499 accuracy
  with recall_pos > 0.998. Useful left-anchor for the curve.

**Calibration math sanity**

Sustained training throughput converged to ~13.7 ex/s on H200 with
gradient checkpointing — better than the v1 calibration's 10.6 ex/s
extrapolation. Training wall scaled almost exactly linearly with example
count (100→0.5 min, full→99.3 min ≈ 100×). Peak VRAM was 14.6 GB across
all runs ≥ 500, matching the calibration prediction. The proposal's
H200-hours budget was generous by ~1.8× on the train side.

**Eval wall held steady at ~180 s per run** at bs=16 (vs the original
864 s before the refactor) — the calibration result transfers cleanly
across sweep sizes since they all evaluate the same test set.

**Open follow-ups**

- TF-IDF baseline (Abrevaa) is in `TF-IDF baseline/` locally but not yet
  committed; needs a directory rename before commit (space in name).
- 3-shot Llama-3.1-8B prompting baseline still paused pending Prof. Harker's
  guidance on label-quality concerns surfaced during inspection.
- Learning-curve plot + qualitative inspection on 30–50 disagreement examples
  are Week 8 deliverables.
- The bigger structural eval-perf win — length-sorted batching — is
  unblocked but not implemented. See [[2026-05-16-calibration]]; the
  measured ceiling for random-order batches is ~180 s and we don't need
  faster than that for the remaining work.

---

## 2026-05-16 — Eval-batch-size calibration + completion-only scoring (OOM fix)

**Calibration sweep on qlora_100**

After the batched-scoring refactor (entry below), ran a four-point
sweep over `EVAL_BATCH_SIZE` against the qlora_100 adapter to find the
optimal batch size before launching the bigger sweeps.

| Slurm | bs | Forward (s) | rows/s | per-item (ms) | Result |
| ---: | ---: | ---: | ---: | ---: | --- |
| 117680 | 16 | 181.9 | 37.5 | 13.3 | baseline (post-refactor) |
| 117738 | 32 | 199.2 | 34.2 | 14.6 | slower than bs=16 |
| 117756 | 64 | — | — | — | **OOM (87 GB allocation)** |
| 117761 | 32 | 196.1 | 34.8 | 14.4 | post-OOM-patch, still slower |
| 117762 | 64 | 226.2 | 30.1 | 16.6 | slower again |

**Finding: per-item cost grows monotonically with batch size.** The H200
is fully compute-saturated at bs=16. Going bigger just invites more
padding waste — BEAD texts vary widely in length, so larger batches
right-pad to longer max-in-batch values and waste more compute on pad
tokens. bs=16 is the optimal point for the **current random-order
batching** code path.

**OOM fix — completion-only scoring**

The bs=64 run (slurm 117756) crashed with `torch.OutOfMemoryError`
trying to allocate 87.12 GB at this line in `score_items_batched`:

```python
log_probs = torch.log_softmax(logits[:, :-1, :].float(), dim=-1)  # (B, T-1, V)
```

Root cause: the original refactor materialized a full `(B, T-1, V)`
fp32 log-softmax tensor over the entire sequence even though we only
needed log-probs at the 1–3 completion-token positions per row. At bs=64,
max_seq_length=512, V=128,256: that tensor alone is 16.8 GB, and the
upstream `.float()` cast plus the gather staging push the allocation
chain to 87 GB.

**Fix (commit `d1aa8e4`)**: slice logits to the completion positions
**per row** before computing log-probs. Each slice is `(n_comp, V)` with
n_comp typically 1–3, so the fp32 cast + logsumexp math costs effectively
nothing. Peak eval memory is now bounded by the model's own `(B, T, V)`
bf16 logits output (~17 GB at bs=128, fine on a 141 GB H200).

Math is identical:
`log p(target | context) = logit[target] - logsumexp(logits)`. We compute
this on the small completion slice; the full-vocab reduction still
happens (it's how you normalize a distribution) but only at completion
positions.

**Verification**: local fp32 smoke flow on tiny-Llama (20 rows) showed
20/20 prediction matches between bs=1 and bs=16 with score deltas at
3.8e-06 — float32 reduction noise from a different kernel path
(`logsumexp + gather` vs `log_softmax`), nowhere near anything that
flips a prediction. On Tillicum at bf16 the metrics drift across the
three qlora_100 runs (0.4990 / 0.4994 / 0.4990) is entirely from
training non-determinism, not the eval path.

**Lesson worth remembering**

My first refactor (batching) optimized the per-row scoring loop but left
the per-batch math memory-naive. A peer agent flagged the issue
("score only the completion tokens instead of computing log-probs for
every prompt token") which matched the right diagnosis. Always slice
before reducing when only a small subset of positions matter.

---

## 2026-05-16 — Manifest race + sort-key bug

**What broke**

All five sweep jobs ran in parallel on g006 and tried to upsert into the
shared `outputs/manifest.csv` simultaneously. Only the last writer (the
qlora_100 row, since it finished first and re-wrote last after others
started reading the empty file) survived. The other four runs' rows were
lost from the consolidated manifest, though per-run `eval_metrics.json`,
`train_metrics.json`, and `run_meta.json` were unaffected (they live in
separate run directories).

When I tried to rebuild the manifest by re-running
`scripts/update_manifest.py` sequentially against each run directory,
the first call succeeded but the second through fifth raised
`TypeError: '<' not supported between instances of 'int' and 'str'`.

Root cause in [scripts/update_manifest.py:131](../scripts/update_manifest.py#L131):
the sort key was `(r.get("train_size") or 0, r.get("run_name") or "")`.
Freshly-built rows have `train_size` as an `int` (from `train_metrics.json`),
but rows loaded from an existing CSV have `train_size` as a `str` (CSV
stringifies everything). Once both shapes coexist mid-rebuild, Python 3
can't compare int with str → TypeError.

**Fix**

Added `_train_size_int()` helper that coerces the value to int via
try/except. The sort key uses the helper, so it works whether the row
came from a fresh JSON read or a CSV round-trip.

**Follow-ups**

- The race condition itself isn't fixed. If five concurrent jobs all
  upsert into the same CSV, the result is non-deterministic. A proper
  fix would either:
  1. Add a file lock (`fcntl.flock`) around the read-modify-write in
     `update_manifest.py`, or
  2. Have each job write to a per-run shard (e.g.
     `outputs/qlora_<size>/manifest_row.json`) and run a separate
     consolidation step after the sweep.
  The simplest near-term mitigation is to launch sweep jobs with a small
  stagger (e.g. `--dependency=afterany:<prev_id>` so they queue serially)
  or to just re-run `update_manifest.py` for each run dir after the
  sweep finishes.
- Worth noting: the per-run files are the ground truth. The manifest is
  a derived view that can always be rebuilt from them. Losing the
  manifest is annoying but not data loss.

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

**Sweep progress** — all five runs complete (2026-05-16)

| Run | Train rows | Accuracy | F1_macro | Prec_pos | Recall_pos | Slurm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| qlora_100 | 100 | 0.4991 | 0.3337 | 0.4992 | 0.9988 | 117767 |
| qlora_500 | 500 | 0.6844 | 0.6817 | 0.6549 | 0.7778 | 117768 |
| qlora_1k | 1,000 | 0.6981 | 0.6975 | 0.6811 | 0.7432 | 117769 |
| qlora_5k | 5,000 | 0.7584 | 0.7584 | 0.7605 | 0.7532 | 117770 |
| qlora_full | 27,263 | **0.8036** | **0.8035** | 0.8059 | 0.7990 | 117771 |

**Baselines**

| Baseline | Owner | Status |
| --- | --- | --- |
| TF-IDF + logistic regression | Abrevaa | Completed 2026-05-15, in repo at `TF-IDF baseline/` (not yet committed — needs dir rename, space in name) |
| 3-shot Llama-3.1-8B prompting | Ash | Paused — example selection raised label-quality concerns; awaiting Prof. Harker's guidance |

**Open questions / follow-ups**

- 3-shot prompting baseline blocked on label-noise question with the instructor.
- TF-IDF directory rename + commit (path with space is shell-hostile).
- Week 8 deliverables: learning-curve plot, qualitative inspection of
  30–50 disagreement examples (per proposal §5).
- **Length-sorted batching** would unlock another 2–3× eval speedup
  (estimated 60–90 s/run vs. the current 180 s) by eliminating
  padding waste in random-order batches. Not blocking — current eval
  cost is small relative to training, especially for qlora_full.
- **Manifest race condition**: `update_manifest.py` is not concurrency-safe.
  Either add a file lock or have each job write a per-run shard and
  consolidate post-sweep. Current workaround is to rerun the manifest
  builder sequentially after the sweep finishes (see 2026-05-16 entry).
- **Model-load cost on Tillicum**: ~9 s on warm GPFS cache (measured
  this round), paid twice because train and eval are separate Python
  processes. Combining them into a single orchestrator would recover
  ~9 s × 5 runs ≈ 45 s. Marginal; not worth doing now that eval is
  fast.
