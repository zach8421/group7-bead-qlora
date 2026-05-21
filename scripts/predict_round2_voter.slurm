#!/bin/bash
# Predict the round-1 BEADs-hand-label winner (qlora_beads_cleaned_flip_balanced_500)
# on the 27,263-row BEADs train split and the 8,520-row val split.
#
# This is the *4th voter* for the round-2 cleaning experiment. The 3 non-BEADs
# adapters (babe, cajcodes, wnc) were the round-1 ensemble; the cleaned model
# adds a 4th independent vote. Round-2 flag rules:
#
#   strict   = all 4 voters unanimously agree against BEADs gold
#   majority = >= 3 of 4 voters agree against BEADs gold
#
# Output: 2 new cells at
#   outputs/cross_eval/qlora_beads_cleaned_flip_balanced_500__on__beads_train/
#   outputs/cross_eval/qlora_beads_cleaned_flip_balanced_500__on__beads_val/
#
# Submission:
#   sbatch scripts/predict_round2_voter.slurm
#
# Wall: ~50-60 min (35,783 rows * 2 sequences / 37.5 rows/s ~ 32 min forward
# + 2 model loads at ~9s + tokenization overhead).
# Cost: ~$1.

#SBATCH --job-name=bead-r2-voter-predict
#SBATCH --account=group7
#SBATCH --qos=normal
#SBATCH --gpus=1
#SBATCH --time=01:30:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

set -euo pipefail
mkdir -p logs

ADAPTER_RUN="qlora_beads_cleaned_flip_balanced_500"
SPLITS=(train val)
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"

module purge
module load conda
conda activate /gpfs/projects/imt526a/$USER/conda/envs/llm
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

if [[ -z "${HF_TOKEN:-}" && -z "${HUGGING_FACE_HUB_TOKEN:-}" ]]; then
  if [[ -r "$HOME/.cache/huggingface/token" ]]; then
    export HF_TOKEN="$(cat "$HOME/.cache/huggingface/token")"
  else
    echo "WARN: HF_TOKEN unset — gated Llama base model load will 401." >&2
  fi
fi
export HF_HUB_CACHE="${HF_HUB_CACHE:-$PWD/.hf_cache/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$PWD/.hf_cache/datasets}"
mkdir -p "$HF_HUB_CACHE" "$HF_DATASETS_CACHE"

echo "============================================================"
echo "Job:        $SLURM_JOB_ID   Node: ${SLURMD_NODENAME:-?}  GPU: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "Adapter:    $ADAPTER_RUN"
echo "Splits:     ${SPLITS[*]}"
echo "============================================================"
which python && python --version
nvidia-smi || true

ADAPTER_DIR="outputs/${ADAPTER_RUN}/adapter"
if [[ ! -d "$ADAPTER_DIR" ]]; then
  echo "[slurm] ERROR: $ADAPTER_DIR not found. Did the round-1 train job land on Tillicum?" >&2
  exit 2
fi
for split in "${SPLITS[@]}"; do
  jsonl="data/frozen/beads/sizes/full/${split}.jsonl"
  if [[ ! -f "$jsonl" ]]; then
    echo "[slurm] ERROR: $jsonl not found. Run scripts/freeze_splits.py --dataset beads or push-data." >&2
    exit 2
  fi
done

for split in "${SPLITS[@]}"; do
  cell_name="${ADAPTER_RUN}__on__beads_${split}"
  cell_dir="outputs/cross_eval/${cell_name}"
  test_jsonl="data/frozen/beads/sizes/full/${split}.jsonl"

  echo
  echo "[slurm] Predicting $cell_name"
  mkdir -p "$cell_dir"
  python scripts/eval_adapter.py \
    --adapter-path "$ADAPTER_DIR" \
    --test-jsonl   "$test_jsonl" \
    --eval-dataset beads \
    --output-dir   "$cell_dir" \
    --run-name     "$cell_name" \
    --eval-batch-size "$EVAL_BATCH_SIZE"
done

echo
echo "[slurm] Done. Output cells:"
ls -la outputs/cross_eval/${ADAPTER_RUN}__on__beads_{train,val}/predictions.jsonl 2>&1 || true
