# BEAD QLoRA - 1k-example Tillicum timing calibration (group7)

> **v1 / archival.** This is the writeup for the May 9 calibration run that
> sized the compute budget in Project Proposal v2. The v2 Week-7 sweep uses
> the scripts in `scripts/` (not the v1 scripts under `scripts/legacy/`).
> See the top-level [README.md](../README.md) for the current workflow.

A self-contained package for the v2 proposal compute calibration: fine-tune
**meta-llama/Llama-3.1-8B-Instruct** with **QLoRA** on a stratified
**1,000-example** subset of BEAD bias classification, **1 epoch**, log
wall-clock and peak GPU memory, then extrapolate to the full sweep.

This is **not** a results run - it's a *cost-shaped run*. We're measuring
seconds per example so the compute budget table in section 4 of the proposal
is defensible.

## Layout (this folder maps 1:1 to /gpfs/projects/imt526a/group7 on Tillicum)

```text
group7/
+-- scripts/
|   +-- prepare_bead_splits.py     # build stratified train/val/test JSONL + summary
|   +-- train_qlora_1k.py          # 4-bit Llama + LoRA, prompt-completion SFT
|   +-- eval_adapter.py            # likelihood-scored binary eval, F1/precision/recall/acc
|   +-- run_tillicum_1k.slurm      # Slurm batch script
+-- data/
|   +-- bead/                      # already rsynced: bias-train.csv, bias-valid.csv, README.md
|   +-- processed/                 # built by prepare_bead_splits.py
+-- outputs/tillicum_1k_calibration/  # adapter + metrics from the H200 run
+-- checkpoints/                   # placeholder for any longer runs you add later
+-- logs/                          # Slurm stdout/stderr
+-- README_1k_calibration.md       # this file
```

## Approach in one paragraph

Each BEAD example is rendered as a Llama-3 chat conversation: **system** =
labeling instructions, **user** = `Statement: <text>`, **assistant** =
`biased` or `non-biased`. Training is plain next-token cross-entropy with
loss masked to the assistant turn only (`completion_only_loss=True` in
TRL's `SFTTrainer`). The base is loaded in 4-bit nf4 via bitsandbytes with
double-quantization; LoRA (`r=16, alpha=32`) is attached to all attention +
MLP projections via PEFT. **No linear classification head** - the
prompt-completion framing is what addresses the V1 architecture feedback.
Eval scores both candidate completions under the prompt and picks the
higher-likelihood one (deterministic, no sampling).

## 1. Local smoke test (no GPU, no Llama download)

These run on `darwin` / `mps` against synthetic data and a ~5 MB tiny
Llama-shaped model. They validate the data formatting + training loop
end-to-end so we don't burn H200-hours debugging on Tillicum.

```bash
conda activate llm                       # the local Tillicum-aligned env
cd "Final project/group7"

# a. Prepare splits from synthetic data (no internet, no HF gating).
python scripts/legacy/prepare_bead_splits.py --mock --train-size 64 \
  --out-dir data/processed_mock

# b. Train smoke (~5 MB tiny Llama, 4 steps, CPU/MPS friendly).
python scripts/legacy/train_qlora_1k.py --smoke-test \
  --train-jsonl data/processed_mock/train_1k.jsonl \
  --output-dir outputs/smoke \
  --per-device-batch-size 2 --grad-accum 1 --max-seq-length 128

# c. Eval smoke (likelihood scoring path).
python scripts/eval_adapter.py --smoke-test \
  --adapter-path outputs/smoke/adapter_smoke \
  --test-jsonl data/processed_mock/test_held_out.jsonl \
  --max-test-rows 16
```

Numbers from a tiny random model are meaningless - the point is that
tokenization, chat template, completion-only masking, adapter save/load, and
likelihood inference all work.

## 1b. Optional local check against the real BEAD CSVs

Same code path the Slurm job will run, just on CPU - useful sanity check
before pushing to Tillicum:

```bash
python scripts/legacy/prepare_bead_splits.py \
  --csv-dir data/bead --train-size 1000 --out-dir data/processed
cat data/processed/splits_summary.json
```

You should see ~50/50 class balance in each split (matches the values in
`data/bead/README.md`: 50.07% non-biased / 49.93% biased).

## 2. Tillicum interactive sanity check (recommended before submitting batch)

The class onboarding doc strongly recommends running interactively first.
Mirror the flow from the quick-start:

```bash
ssh tillicum
cd /gpfs/projects/imt526a/group7

# Request a short interactive session.
salloc -A group7 --qos=interactive --gpus=1 --time=00:30:00
hostname
nvidia-smi

module purge
module load conda
conda activate /gpfs/projects/imt526a/conda/envs/imt526a-jupyter-torch
which python
python --version

# Confirm the heavy deps are available in the env. If any of these import,
# you're set; if not, switch to your personal llm env (see Slurm script).
python -c "import torch, bitsandbytes, peft, trl, transformers, datasets; \
print('torch', torch.__version__); print('peft', peft.__version__); print('trl', trl.__version__)"

# Quick prep + a tiny smoke training run on the real CSVs to confirm
# everything wires up before committing to a 1-hour batch job.
export HF_TOKEN=...   # or run `huggingface-cli login` once
python scripts/legacy/prepare_bead_splits.py --csv-dir data/bead --train-size 1000 --out-dir data/processed
python scripts/legacy/train_qlora_1k.py --smoke-test \
  --train-jsonl data/processed/train_1k.jsonl --output-dir outputs/_interactive_smoke

exit
```

If `bitsandbytes`, `peft`, or `trl` is missing in the shared env, switch to
your personal env:

```bash
conda activate /gpfs/projects/imt526a/$USER/conda/envs/llm
```

## 3. Tillicum batch run (the actual calibration)

```bash
cd /gpfs/projects/imt526a/group7

# Confirm the slurm header matches your group:
#   --account=group7   (edit if different)
#   conda activate path
# Then submit:
sbatch scripts/legacy/run_tillicum_1k.slurm
squeue -u $USER

# When it finishes, view the log and metrics:
cat logs/slurm-<jobid>.out
cat outputs/tillicum_1k_calibration/calibration_metrics.json
cat outputs/tillicum_1k_calibration/eval_metrics.json
```

To rsync results back to your laptop:

```bash
rsync -av zach8420@tillicum.uw.edu:/gpfs/projects/imt526a/group7/outputs/ \
  "Final project/group7/outputs/"
rsync -av zach8420@tillicum.uw.edu:/gpfs/projects/imt526a/group7/logs/ \
  "Final project/group7/logs/"
```

## 4. What numbers to copy into the proposal compute section

From `outputs/tillicum_1k_calibration/calibration_metrics.json`:

| Field                         | Use it for                                                         |
| ----------------------------- | ------------------------------------------------------------------ |
| `wall_clock_min`              | Wall-clock for "1k x 1 epoch" row in the compute table             |
| `peak_cuda_memory_gb`         | Storage / VRAM column; informs whether bigger batch sizes are safe |
| `effective_batch_size`        | Document the recipe so the extrapolation is reproducible           |
| `max_seq_length`              | Same                                                               |
| `throughput_examples_per_sec` | Used directly in section 4 sizing math (see below)                 |
| `trainable_params`            | "QLoRA only updates X% of the parameters" justification line       |

From `outputs/tillicum_1k_calibration/eval_metrics.json`:

| Field      | Use it for                                                       |
| ---------- | ---------------------------------------------------------------- |
| `f1_pos`   | Sanity-check signal that 1 epoch on 1k actually moves the metric |
| `accuracy` | Same                                                             |

Eval numbers from the calibration run are noisy (1k examples is deliberately
under-sized) - present them as a **floor**, not a target.

## 5. Extrapolating to the full sweep

The full proposal sweep is "learning curve at multiple training-set sizes".
Use the throughput from the calibration run as your unit:

```text
seconds_per_example_per_epoch = wall_clock_sec / (1000 * num_epochs)
```

Then for each planned run:

```text
predicted_seconds = seconds_per_example_per_epoch * N_examples * N_epochs
predicted_h200_hours = predicted_seconds / 3600
```

Multiply by **1.25** to bake in the 25% contingency George flagged.

Worked example with placeholder factors (replace with your measurements):

| Run               | N examples | Epochs | Predicted H200-hrs (x1.25) |
| ----------------- | ---------- | ------ | -------------------------- |
| Baseline eval     | 1,000      | -      | from `eval_seconds`        |
| QLoRA 5k          | 5,000      | 3      | `5*3 = 15x` calibration    |
| QLoRA 20k         | 20,000     | 3      | `20*3 = 60x` calibration   |
| QLoRA full (~34k) | 34,079     | 3      | `34*3 ~ 100x` calibration  |

**Caveats to mention in the section 4 paragraph:**

* Throughput is roughly linear in N at fixed sequence length and batch size,
  but only if the activation memory headroom on the H200 holds - if the
  larger sweeps need a smaller per-device batch, throughput drops.
* If `peak_cuda_memory_gb` is well below H200 capacity (~141 GB), you have
  headroom to raise per-device batch size or sequence length, which can
  reverse-engineer extra throughput. Note this in the slack paragraph.
* Eval cost is small but non-zero - count it.

## 6. Assumptions / placeholders to verify

* **Slurm account `group7`** in the Slurm header. Verify with
  `sacctmgr show user $USER` on Tillicum and edit if different.
* **Shared env at `/gpfs/projects/imt526a/conda/envs/imt526a-jupyter-torch`**
  is the default. If it doesn't have `bitsandbytes`, `peft`, or `trl`,
  switch to your personal env line in the Slurm script.
* **HF_TOKEN** must be present for the gated Llama download. Set in
  `~/.bashrc` on Tillicum or pass via `sbatch --export=ALL,HF_TOKEN=...`.
* **Sequence length 512** is generous for BEAD (most sentences << 100
  tokens). Drop to 256 for a tighter throughput number.
* **Effective batch size 16** (`per_device_batch_size=4 x grad_accum=4`)
  is a conservative starting point. Bump `per_device_batch_size` once you
  see `peak_cuda_memory_gb`.
* **Label mapping**: `data/bead/README.md` confirms 1=biased, 0=non-biased.
  Override with `--label0-text` / `--label1-text` if a future drop flips it.

## 7. Reproducibility

* Seed = 42 throughout (`prepare`, `train`, `eval`).
* Greedy / likelihood-scored decoding only - no sampling.
* `splits_summary.json` records exact counts and class balance per split.
* `calibration_metrics.json` records full hyperparameter + timing snapshot.
