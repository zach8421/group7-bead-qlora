#!/bin/bash
# QLoRA sweep launcher. Picks the training (DATASET, SIZE) cell.
#
#   DATASET ∈ {beads,babe,cajcodes,wnc}   (default: beads)
#   SIZE    — BEADs only: 100|500|1k|5k|full;  others: full
#   EPOCHS  — float (default 3)
#
# Submit examples:
#   sbatch --export=ALL,DATASET=beads,SIZE=100  --time=00:30:00 scripts/run_qlora.slurm
#   sbatch --export=ALL,DATASET=beads,SIZE=full --time=05:00:00 scripts/run_qlora.slurm
#   sbatch --export=ALL,DATASET=babe,SIZE=full  --time=01:00:00 scripts/run_qlora.slurm
#   sbatch --export=ALL,DATASET=cajcodes,SIZE=full --time=00:30:00 scripts/run_qlora.slurm
#   sbatch --export=ALL,DATASET=wnc,SIZE=full,MAX_TRAIN_ROWS=27263 --time=04:00:00 scripts/run_qlora.slurm
#
# MAX_TRAIN_ROWS caps the loaded train set to the first N rows after shuffle.
# Use for WNC (88k native) to match BEADs full (27,263) and keep cross-eval
# apples-to-apples — see docs/build_log.md 2026-05-19 entry.
#
# The launcher: prepares splits if missing, trains, evals on the same-dataset
# test set (cross-eval is run separately via scripts/cross_eval.py), and
# updates outputs/manifest.csv via scripts/update_manifest.py.

#SBATCH --job-name=bead-qlora-sweep
#SBATCH --account=group7
#SBATCH --qos=normal
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

set -euo pipefail
mkdir -p logs

# ---- Knobs from the submitter ----
DATASET="${DATASET:-beads}"
SIZE="${SIZE:-full}"
EPOCHS="${EPOCHS:-3}"
PER_DEVICE_BS="${PER_DEVICE_BS:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-512}"
EVAL_ROWS="${EVAL_ROWS:-0}"  # 0 = full test set; >0 = subset (smoke)
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"  # sequences per forward; 2 per test row
MAX_TRAIN_ROWS="${MAX_TRAIN_ROWS:-0}"  # 0 = full train.jsonl; >0 = cap (e.g. wnc apples-to-apples)

case "$DATASET" in
  beads|babe|cajcodes|wnc) ;;
  beads_cleaned_remove|beads_cleaned_remove_balanced|beads_cleaned_flip|beads_cleaned_flip_balanced) ;;
  beads_cleaned_v2_strict|beads_cleaned_v2_majority) ;;
  *) echo "Unknown DATASET='$DATASET'. Use beads|babe|cajcodes|wnc or beads_cleaned_{remove,remove_balanced,flip,flip_balanced,v2_strict,v2_majority}." >&2; exit 2 ;;
esac

# Layout convention:
#   beads + cleaned variants have multiple sizes (nested); the simple
#   datasets (babe/cajcodes/wnc) only have 'full'.
if [[ "$DATASET" == "beads" || "$DATASET" == beads_cleaned_* ]]; then
  case "$SIZE" in
    100|500|1k|5k|full) ;;
    *) echo "Unknown SIZE='$SIZE' for $DATASET. Use 100|500|1k|5k|full." >&2; exit 2 ;;
  esac
  SIZE_DIR="sizes/${SIZE}"
else
  if [[ "$SIZE" != "full" ]]; then
    echo "DATASET=$DATASET only has 'full' size; got SIZE=$SIZE." >&2; exit 2
  fi
  SIZE_DIR="full"
fi

FROZEN_ROOT="data/frozen/${DATASET}"
TRAIN_FILE="${FROZEN_ROOT}/${SIZE_DIR}/train.jsonl"
TEST_FILE="${FROZEN_ROOT}/${SIZE_DIR}/test.jsonl"
COMMITTED_MANIFEST="${FROZEN_ROOT}/splits_manifest.json"

# Intermediate checkpoints only for the longest runs (~5k steps+). Skip for
# small subsets since each is well under an hour. Saving every 800 steps gives
# roughly one checkpoint per ~30 min on the H200, so a node failure costs at
# most ~30 min of compute.
if [[ "$SIZE" == "full" && ("$DATASET" == "beads" || "$DATASET" == "wnc" || "$DATASET" == beads_cleaned_*) ]]; then
  SAVE_STEPS="${SAVE_STEPS:-800}"
  SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-2}"
else
  SAVE_STEPS="${SAVE_STEPS:-0}"
  SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-2}"
fi

RUN_NAME="qlora_${DATASET}_${SIZE}"
OUTPUT_DIR="outputs/${RUN_NAME}"
MANIFEST="outputs/manifest.csv"

# ---- Environment ----
module purge
module load conda
conda activate /gpfs/projects/imt526a/$USER/conda/envs/llm

# conda libstdc++ over /lib64's older copy (calibration run hit CXXABI_1.3.15).
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

# HF auth: prefer the on-disk token written by `hf auth login`.
if [[ -z "${HF_TOKEN:-}" && -z "${HUGGING_FACE_HUB_TOKEN:-}" ]]; then
  if [[ -r "$HOME/.cache/huggingface/token" ]]; then
    export HF_TOKEN="$(cat "$HOME/.cache/huggingface/token")"
    echo "[slurm] Loaded HF_TOKEN from ~/.cache/huggingface/token"
  else
    echo "WARN: HF_TOKEN unset and ~/.cache/huggingface/token missing — gated Llama will 401." >&2
  fi
fi

# Cache only the hub in the project dir so we don't fill $HOME quota.
# Do NOT set HF_HOME — that would also relocate the token file and break auth.
export HF_HUB_CACHE="${HF_HUB_CACHE:-$PWD/.hf_cache/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$PWD/.hf_cache/datasets}"
mkdir -p "$HF_HUB_CACHE" "$HF_DATASETS_CACHE"

echo "============================================================"
echo "Job:     $SLURM_JOB_ID   Node: ${SLURMD_NODENAME:-?}  GPU: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "PWD:     $PWD"
echo "Run:     $RUN_NAME  (SIZE=$SIZE EPOCHS=$EPOCHS)"
echo "Train:   $TRAIN_FILE"
echo "Output:  $OUTPUT_DIR"
echo "============================================================"

which python
python --version
nvidia-smi || true

# 1. Sanity: frozen splits exist and match the committed manifest.
#    If JSONLs are missing we regenerate them, but we always verify the on-disk
#    bytes hash to the sha256s recorded in the committed splits_manifest.json.
#    This is the only thing standing between us and a silently-drifted test set
#    (which would invalidate cross-method comparisons against TF-IDF / 3-shot).
if [[ -f "$COMMITTED_MANIFEST" ]]; then
  REF_MANIFEST_SNAPSHOT="$(mktemp -t splits_manifest_committed.XXXXXX.json)"
  cp "$COMMITTED_MANIFEST" "$REF_MANIFEST_SNAPSHOT"
  trap 'rm -f "$REF_MANIFEST_SNAPSHOT"' EXIT

  if [[ ! -f "$TRAIN_FILE" || ! -f "$TEST_FILE" ]]; then
    echo "[slurm] Frozen splits missing — running freeze_splits.py --dataset $DATASET ..."
    python scripts/freeze_splits.py --dataset "$DATASET"
  fi

  echo "[slurm] Verifying on-disk splits against committed manifest ..."
  python scripts/verify_splits_manifest.py \
    --frozen-dir "$FROZEN_ROOT" \
    --reference-manifest "$REF_MANIFEST_SNAPSHOT"
else
  echo "[slurm] WARNING: $COMMITTED_MANIFEST not present — building from scratch with no reference to verify against." >&2
  if [[ ! -f "$TRAIN_FILE" || ! -f "$TEST_FILE" ]]; then
    python scripts/freeze_splits.py --dataset "$DATASET"
  fi
fi

# 2. Train.
echo "[slurm] Training $RUN_NAME ..."
TRAIN_ARGS=( --train-jsonl "$TRAIN_FILE" \
             --output-dir "$OUTPUT_DIR" \
             --run-name "$RUN_NAME" \
             --train-dataset "$DATASET" \
             --splits-manifest "$COMMITTED_MANIFEST" \
             --per-device-batch-size "$PER_DEVICE_BS" \
             --grad-accum "$GRAD_ACCUM" \
             --max-seq-length "$MAX_SEQ_LEN" \
             --num-epochs "$EPOCHS" \
             --save-steps "$SAVE_STEPS" \
             --save-total-limit "$SAVE_TOTAL_LIMIT" )
if [[ "$MAX_TRAIN_ROWS" -gt 0 ]]; then
  echo "[slurm] Capping training rows at $MAX_TRAIN_ROWS"
  TRAIN_ARGS+=( --max-train-rows "$MAX_TRAIN_ROWS" )
fi
python scripts/train_qlora.py "${TRAIN_ARGS[@]}"

# 3. Eval on held-out test (same dataset). Cross-eval runs separately.
echo "[slurm] Evaluating $RUN_NAME on $TEST_FILE ..."
EVAL_ARGS=( --adapter-path "$OUTPUT_DIR/adapter" \
            --test-jsonl "$TEST_FILE" \
            --eval-dataset "$DATASET" \
            --output-dir "$OUTPUT_DIR" \
            --run-name "$RUN_NAME" \
            --eval-batch-size "$EVAL_BATCH_SIZE" )
if [[ "$EVAL_ROWS" -gt 0 ]]; then
  EVAL_ARGS+=( --max-test-rows "$EVAL_ROWS" )
fi
python scripts/eval_adapter.py "${EVAL_ARGS[@]}"

# 4. Update the sweep manifest with one row for this run.
echo "[slurm] Updating manifest ..."
python scripts/update_manifest.py --run-dir "$OUTPUT_DIR" --manifest "$MANIFEST"

echo "[slurm] Done."
