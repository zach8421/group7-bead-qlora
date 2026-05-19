#!/bin/bash
# Cross-evaluation matrix launcher.
#
# Runs scripts/cross_eval.py to score every adapter listed in ADAPTERS on every
# dataset listed in EVAL_DATASETS. Each (adapter, eval_dataset) cell lands at
#   outputs/cross_eval/{adapter_run_name}__on__{eval_dataset}/
# and gets upserted into outputs/manifest.csv.
#
# Typical submission (after all 4 train jobs have produced outputs/qlora_*_full/adapter/):
#   sbatch --export=ALL --time=01:30:00 scripts/run_cross_eval.slurm
#
# Override the inputs:
#   ADAPTERS        Space-separated run dirs (default: all four full-size adapters).
#   EVAL_DATASETS   Space-separated dataset names (default: beads babe cajcodes wnc).
#   EVAL_BATCH_SIZE Sequences per forward (default: 16). Each test row produces 2.
#   SKIP_EXISTING   "1" to skip cells whose eval_metrics.json already exists.
#
# This script is intentionally a single Python process: cross_eval.py iterates
# cells sequentially, so manifest writes don't race with each other (and the
# fcntl lock guards against any race with concurrent train jobs in the same
# squeue).

#SBATCH --job-name=bead-cross-eval
#SBATCH --account=group7
#SBATCH --qos=normal
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

set -euo pipefail
mkdir -p logs

ADAPTERS="${ADAPTERS:-outputs/qlora_beads_full outputs/qlora_babe_full outputs/qlora_cajcodes_full outputs/qlora_wnc_full}"
EVAL_DATASETS="${EVAL_DATASETS:-beads babe cajcodes wnc}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"

# ---- Environment ----
module purge
module load conda
conda activate /gpfs/projects/imt526a/$USER/conda/envs/llm
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

if [[ -z "${HF_TOKEN:-}" && -z "${HUGGING_FACE_HUB_TOKEN:-}" ]]; then
  if [[ -r "$HOME/.cache/huggingface/token" ]]; then
    export HF_TOKEN="$(cat "$HOME/.cache/huggingface/token")"
    echo "[slurm] Loaded HF_TOKEN from ~/.cache/huggingface/token"
  else
    echo "WARN: HF_TOKEN unset — gated Llama base model load will 401." >&2
  fi
fi
export HF_HUB_CACHE="${HF_HUB_CACHE:-$PWD/.hf_cache/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$PWD/.hf_cache/datasets}"
mkdir -p "$HF_HUB_CACHE" "$HF_DATASETS_CACHE"

echo "============================================================"
echo "Job:           $SLURM_JOB_ID   Node: ${SLURMD_NODENAME:-?}  GPU: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "PWD:           $PWD"
echo "Adapters:      $ADAPTERS"
echo "Eval datasets: $EVAL_DATASETS"
echo "Batch size:    $EVAL_BATCH_SIZE"
echo "Skip existing: $SKIP_EXISTING"
echo "============================================================"

which python
python --version
nvidia-smi || true

# Sanity: every adapter dir must have an adapter/ subdir.
for a in $ADAPTERS; do
  if [[ ! -d "$a/adapter" ]]; then
    echo "[slurm] ERROR: $a/adapter not found. Did the train job finish?" >&2
    exit 2
  fi
done

# Sanity: each eval dataset has a test JSONL at the expected layout.
for ds in $EVAL_DATASETS; do
  if [[ ! -f "data/frozen/$ds/full/test.jsonl" && ! -f "data/frozen/$ds/sizes/full/test.jsonl" ]]; then
    echo "[slurm] ERROR: no test.jsonl for dataset '$ds'. Run scripts/freeze_splits.py --dataset $ds." >&2
    exit 2
  fi
done

CROSS_EVAL_ARGS=( --adapters $ADAPTERS \
                  --eval-datasets $EVAL_DATASETS \
                  --eval-batch-size "$EVAL_BATCH_SIZE" )
if [[ "$SKIP_EXISTING" == "1" ]]; then
  CROSS_EVAL_ARGS+=( --skip-existing )
fi

echo "[slurm] Running cross-eval matrix ..."
python scripts/cross_eval.py "${CROSS_EVAL_ARGS[@]}"

echo "[slurm] Done."
