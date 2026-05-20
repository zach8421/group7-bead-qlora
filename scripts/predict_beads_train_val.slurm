#!/bin/bash
# Predict the 4 cross-eval adapters on BEADs train + val splits.
#
# Why
# ---
# The cleaning experiment needs per-row predictions on BEADs *train*
# rows so the flag rule (`cross_unanimous_disagree == 1`) can identify
# which of the 27,263 train rows to remove or flip in the retrain step.
# The 16-cell cross-eval matrix only covered BEADs test (6,816 rows).
# This job extends coverage to train + val without touching the existing
# test cells.
#
# Output: 8 new cells (4 adapters x 2 splits) at
# outputs/cross_eval/qlora_<ds>_full__on__beads_{train,val}/
#
# Each cell gets:
#   predictions.jsonl  — one row per (test row, label) pair
#   eval_metrics.json  — accuracy/F1 against BEADs gold (noisy but
#                         useful as a sanity check that the prediction
#                         pass actually ran)
#
# Wall estimate
# -------------
# Per the 2026-05-16 calibration: ~37.5 rows/s at eval_batch_size=16.
# Train (27,263) x 4 adapters = 109,052 row-evals = ~48 min
# Val   ( 8,520) x 4 adapters =  34,080 row-evals = ~15 min
# Plus ~9 s base-model load x 8 invocations = ~72 s overhead
# Total: ~65 min. Walltime request of 2h gives a safe margin.
#
# Cost estimate
# -------------
# 1 H200 hr x ~1.1 hrs = ~$1 at $0.90/hr.
#
# Submission
# ----------
#   sbatch --time=02:00:00 scripts/predict_beads_train_val.slurm
#
# This script does not update outputs/manifest.csv — these cells are
# internal artifacts for the cleaning step, not headline eval results.
# If you want them in the manifest later, run update_manifest.py per cell.

#SBATCH --job-name=beads-train-val-predict
#SBATCH --account=group7
#SBATCH --qos=normal
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

set -euo pipefail
mkdir -p logs

ADAPTERS=(beads babe cajcodes wnc)
SPLITS=(train val)
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"

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
echo "Job:         $SLURM_JOB_ID   Node: ${SLURMD_NODENAME:-?}  GPU: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "PWD:         $PWD"
echo "Adapters:    ${ADAPTERS[*]}"
echo "Splits:      ${SPLITS[*]}"
echo "Batch size:  $EVAL_BATCH_SIZE"
echo "============================================================"

which python
python --version
nvidia-smi || true

# Sanity: every adapter dir + JSONL must exist before we burn 65 min on the GPU.
for adapter in "${ADAPTERS[@]}"; do
  adapter_dir="outputs/qlora_${adapter}_full/adapter"
  if [[ ! -d "$adapter_dir" ]]; then
    echo "[slurm] ERROR: $adapter_dir not found." >&2
    exit 2
  fi
done
for split in "${SPLITS[@]}"; do
  jsonl="data/frozen/beads/sizes/full/${split}.jsonl"
  if [[ ! -f "$jsonl" ]]; then
    echo "[slurm] ERROR: $jsonl not found. Run scripts/freeze_splits.py --dataset beads." >&2
    exit 2
  fi
done

# ---- Predict ----
n_cells=$((${#ADAPTERS[@]} * ${#SPLITS[@]}))
i=0
for adapter in "${ADAPTERS[@]}"; do
  for split in "${SPLITS[@]}"; do
    i=$((i + 1))
    cell_name="qlora_${adapter}_full__on__beads_${split}"
    cell_dir="outputs/cross_eval/${cell_name}"
    test_jsonl="data/frozen/beads/sizes/full/${split}.jsonl"

    echo
    echo "[slurm] ($i/$n_cells)  $cell_name"
    echo "[slurm]   adapter: outputs/qlora_${adapter}_full/adapter"
    echo "[slurm]   test:    $test_jsonl"
    echo "[slurm]   output:  $cell_dir"

    mkdir -p "$cell_dir"
    python scripts/eval_adapter.py \
      --adapter-path "outputs/qlora_${adapter}_full/adapter" \
      --test-jsonl   "$test_jsonl" \
      --eval-dataset beads \
      --output-dir   "$cell_dir" \
      --run-name     "$cell_name" \
      --eval-batch-size "$EVAL_BATCH_SIZE"
  done
done

echo
echo "[slurm] All $n_cells cells complete."
ls -la outputs/cross_eval/qlora_*_full__on__beads_{train,val}/predictions.jsonl 2>/dev/null || true
echo "[slurm] Done."
