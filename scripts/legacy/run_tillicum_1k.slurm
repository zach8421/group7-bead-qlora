#!/bin/bash
#SBATCH --job-name=bead-qlora-1k
#SBATCH --account=group7         # EDIT if your slurm account differs from `group7`
#SBATCH --qos=normal             # use --qos=interactive only when running via salloc
#SBATCH --gpus=1                 # 1x H200
#SBATCH --time=01:00:00          # 1k examples * 1 epoch finishes well under this on H200
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

set -euo pipefail
mkdir -p logs

# ---- Environment ----
# Default: the class-shared env from the onboarding repo. If it's missing
# bitsandbytes / peft / trl, switch the `conda activate` line to your personal
# Tillicum-aligned env at /gpfs/projects/imt526a/$USER/conda/envs/llm.
module purge
module load conda
# Default: shared class env. Switch to the personal `llm` env line below if
# the shared env is missing bitsandbytes/peft/trl.
# conda activate /gpfs/projects/imt526a/conda/envs/imt526a-jupyter-torch
conda activate /gpfs/projects/imt526a/$USER/conda/envs/llm

# Force conda's libstdc++ to win the link race over /lib64's older copy.
# Without this, pip-installed scipy/sklearn raise ImportError on CXXABI_1.3.15
# on Tillicum's compute nodes.
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

# Sanity-check the GPU and confirm we have the right Python.
which python
python --version
nvidia-smi || true

# HF auth: prefer the on-disk token written by `hf auth login` (~/.cache/huggingface/token).
# Read it explicitly into HF_TOKEN as a belt-and-suspenders so any non-default HF_HOME
# elsewhere in the env doesn't hide it.
if [[ -z "${HF_TOKEN:-}" && -z "${HUGGING_FACE_HUB_TOKEN:-}" ]]; then
  if [[ -r "$HOME/.cache/huggingface/token" ]]; then
    export HF_TOKEN="$(cat "$HOME/.cache/huggingface/token")"
    echo "[slurm] Loaded HF_TOKEN from ~/.cache/huggingface/token"
  else
    echo "WARN: HF_TOKEN unset and ~/.cache/huggingface/token missing - gated Llama will 401." >&2
  fi
fi

# Cache only the *model hub* in the project dir so we don't fill $HOME quota.
# Do NOT set HF_HOME here - that would also relocate the token file and break auth.
export HF_HUB_CACHE="${HF_HUB_CACHE:-$PWD/.hf_cache/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$PWD/.hf_cache/datasets}"
mkdir -p "$HF_HUB_CACHE" "$HF_DATASETS_CACHE"

echo "============================================================"
echo "Job:    $SLURM_JOB_ID  Node: $SLURMD_NODENAME  GPU: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "PWD:    $PWD"
echo "============================================================"

# 1. Build splits from the local BEAD CSVs (idempotent; set SKIP_PREPARE=1 to
#    reuse an existing data/processed/ from a prior run).
if [[ -z "${SKIP_PREPARE:-}" ]]; then
  echo "[slurm] Preparing BEAD splits ..."
  python scripts/prepare_bead_splits.py \
    --csv-dir data/bead \
    --train-size 1000 \
    --test-frac 0.2 \
    --out-dir data/processed
fi

# 2. Run the calibration training.
echo "[slurm] Starting QLoRA 1k calibration ..."
python scripts/train_qlora_1k.py \
  --train-jsonl data/processed/train_1k.jsonl \
  --output-dir outputs/tillicum_1k_calibration \
  --per-device-batch-size 4 \
  --grad-accum 4 \
  --max-seq-length 512 \
  --num-epochs 1.0

# 3. Quick sanity eval on 200 held-out test examples. Comment out if you only
#    want pure timing numbers from this job.
echo "[slurm] Running quick eval on a 200-example slice ..."
python scripts/eval_adapter.py \
  --adapter-path outputs/tillicum_1k_calibration/adapter \
  --test-jsonl data/processed/test_held_out.jsonl \
  --max-test-rows 200

echo "[slurm] Done."
