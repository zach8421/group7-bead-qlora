#!/bin/bash
# QLoRA sweep launcher. Pick the training size with SIZE=100|500|1k|5k|full
# and (optionally) EPOCHS=N. Defaults: SIZE=1k, EPOCHS=3.
#
# Submit examples:
#   sbatch --export=ALL,SIZE=100  --time=00:30:00 scripts/run_qlora.slurm
#   sbatch --export=ALL,SIZE=500  --time=00:45:00 scripts/run_qlora.slurm
#   sbatch --export=ALL,SIZE=1k   --time=01:00:00 scripts/run_qlora.slurm
#   sbatch --export=ALL,SIZE=5k   --time=02:00:00 scripts/run_qlora.slurm
#   sbatch --export=ALL,SIZE=full --time=05:00:00 scripts/run_qlora.slurm
#
# The launcher: prepares splits if missing, trains, evals on test_held_out,
# and updates outputs/manifest.csv via scripts/update_manifest.py.

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
SIZE="${SIZE:-1k}"
EPOCHS="${EPOCHS:-3}"
PER_DEVICE_BS="${PER_DEVICE_BS:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-512}"
EVAL_ROWS="${EVAL_ROWS:-0}"  # 0 = full test_held_out (6816); >0 = subset (smoke)
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"  # sequences per forward; 2 per test row

case "$SIZE" in
  100)   TRAIN_FILE="data/frozen/train_100.jsonl" ;;
  500)   TRAIN_FILE="data/frozen/train_500.jsonl" ;;
  1k)    TRAIN_FILE="data/frozen/train_1k.jsonl" ;;
  5k)    TRAIN_FILE="data/frozen/train_5k.jsonl" ;;
  full)  TRAIN_FILE="data/frozen/train_full.jsonl" ;;
  *)     echo "Unknown SIZE='$SIZE'. Use 100|500|1k|5k|full." >&2; exit 2 ;;
esac

# Intermediate checkpoints only for the longest run. 27,263 examples with
# effective batch 16 ⇒ ~1,700 steps/epoch × 3 epochs ≈ 5,100 steps. Saving
# every 800 steps gives roughly one checkpoint per ~30 min of wall-clock on
# the H200 (per the v1 calibration projection of ~2.5h total), so a node
# failure costs at most ~30 min of compute.
if [[ "$SIZE" == "full" ]]; then
  SAVE_STEPS="${SAVE_STEPS:-800}"
  SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-2}"
else
  SAVE_STEPS="${SAVE_STEPS:-0}"
  SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-2}"
fi

RUN_NAME="qlora_${SIZE}"
OUTPUT_DIR="outputs/${RUN_NAME}"
TEST_FILE="data/frozen/test_held_out.jsonl"
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
COMMITTED_MANIFEST="data/frozen/splits_manifest.json"
if [[ -f "$COMMITTED_MANIFEST" ]]; then
  REF_MANIFEST_SNAPSHOT="$(mktemp -t splits_manifest_committed.XXXXXX.json)"
  cp "$COMMITTED_MANIFEST" "$REF_MANIFEST_SNAPSHOT"
  trap 'rm -f "$REF_MANIFEST_SNAPSHOT"' EXIT

  if [[ ! -f "$TRAIN_FILE" || ! -f "$TEST_FILE" ]]; then
    echo "[slurm] Frozen splits missing — running freeze_splits.py ..."
    python scripts/freeze_splits.py --csv-dir data/bead --out-dir data/frozen
  fi

  echo "[slurm] Verifying on-disk splits against committed manifest ..."
  python scripts/verify_splits_manifest.py \
    --frozen-dir data/frozen \
    --reference-manifest "$REF_MANIFEST_SNAPSHOT"
else
  echo "[slurm] WARNING: $COMMITTED_MANIFEST not present — building from scratch with no reference to verify against." >&2
  if [[ ! -f "$TRAIN_FILE" || ! -f "$TEST_FILE" ]]; then
    python scripts/freeze_splits.py --csv-dir data/bead --out-dir data/frozen
  fi
fi

# 2. Train.
echo "[slurm] Training $RUN_NAME ..."
python scripts/train_qlora.py \
  --train-jsonl "$TRAIN_FILE" \
  --output-dir "$OUTPUT_DIR" \
  --run-name "$RUN_NAME" \
  --per-device-batch-size "$PER_DEVICE_BS" \
  --grad-accum "$GRAD_ACCUM" \
  --max-seq-length "$MAX_SEQ_LEN" \
  --num-epochs "$EPOCHS" \
  --save-steps "$SAVE_STEPS" \
  --save-total-limit "$SAVE_TOTAL_LIMIT"

# 3. Eval on held-out test.
echo "[slurm] Evaluating $RUN_NAME on $TEST_FILE ..."
EVAL_ARGS=( --adapter-path "$OUTPUT_DIR/adapter" \
            --test-jsonl "$TEST_FILE" \
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
