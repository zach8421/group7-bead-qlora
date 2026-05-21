#!/usr/bin/env bash
# Cross-eval the cleaned BEADs adapters on the other 3 datasets
# (babe, cajcodes, wnc). 20 cleaned adapters x 3 datasets = 60 new
# cells at outputs/cross_eval/qlora_beads_cleaned_<variant>_<size>__on__<ds>/.
#
# Submitted as 4 slurm jobs (one per cleaning variant). Each job runs
# its 5 adapters x 3 datasets = 15 cells sequentially via
# scripts/cross_eval.py.
#
# Use case
# --------
# Extends the cleaning result to the cross-dataset transfer question:
# does cleaning BEADs's noise also improve the BEADs-trained model's
# generalization to BABE / cajcodes / WNC? Or is cleaning making the
# model MORE specialized to its (now clean) source distribution?
#
# Per the 2026-05-19 cross-eval matrix, the *original* qlora_beads_full
# scored 0.31 / 0.67 / 0.46 on babe / cajcodes / wnc (off-diagonal,
# barely above chance). This sweep tells us whether the cleaned
# adapters do meaningfully better or worse.
#
# Caveat: babe/cajcodes/wnc gold labels are their own published
# labels, which we haven't validated. The comparison original-vs-cleaned
# on each target is still apples-to-apples (both face the same gold).
#
# Usage:
#   scripts/launch_cleaned_cross_eval.sh          # submit all 4 jobs
#   scripts/launch_cleaned_cross_eval.sh --dry    # print sbatch commands
#
# Environment overrides:
#   VARIANTS       space-separated subset of cleaning variants
#                  (default: all 4)
#   SIZES          space-separated subset (default: 100 500 1k 5k full)
#   EVAL_DATASETS  space-separated (default: babe cajcodes wnc)
#   EXTRA_SBATCH   appended to every sbatch
#
# Each slurm job uses run_cross_eval.slurm (already in repo) with
# ADAPTERS + EVAL_DATASETS env-var overrides. The fcntl manifest lock
# from update_manifest.py is unnecessary here since each cross_eval.py
# instance writes cells sequentially within its own job, but the
# 4 parallel jobs together could race on the manifest — the lock
# guards that anyway.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

DRY_RUN=""
for arg in "$@"; do
  case "$arg" in
    -n|--dry|--dry-run) DRY_RUN="1" ;;
    -h|--help)
      sed -n '2,38p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *) echo "[launch] Unknown arg: $arg" >&2; exit 2 ;;
  esac
done

VARIANTS="${VARIANTS:-beads_cleaned_remove beads_cleaned_remove_balanced beads_cleaned_flip beads_cleaned_flip_balanced}"
SIZES="${SIZES:-100 500 1k 5k full}"
EVAL_DATASETS="${EVAL_DATASETS:-babe cajcodes wnc}"
EXTRA_SBATCH="${EXTRA_SBATCH:-}"

# Per-job walltime budget. Per the 2026-05-16 calibration, eval throughput
# is ~37.5 rows/sec on H200. Per cell: ~9s model load + (n_rows / 37.5)s
# forward. For our datasets: babe (413 rows) ~20s, cajcodes (66 rows) ~5s,
# wnc (11041 rows) ~5min. Per adapter (3 datasets): ~6 min. Per job
# (5 adapters): ~30 min. Request 1 hour to be safe.
JOB_WALLTIME="${JOB_WALLTIME:-01:00:00}"

submit() {
  # Returns the sbatch job id on stdout; everything else to stderr.
  # Pre-exported env vars (ADAPTERS, EVAL_DATASETS) are forwarded via
  # sbatch --export=ALL.
  local label="$1"; shift
  {
    echo
    echo "[launch] $label"
    echo "          (ADAPTERS=$ADAPTERS)"
    echo "          (EVAL_DATASETS=$EVAL_DATASETS)"
    printf '          sbatch %s scripts/run_cross_eval.slurm\n' "$*"
  } >&2
  if [[ -n "$DRY_RUN" ]]; then
    echo "          (dry-run; no submission)" >&2
    echo "mock-$(LC_ALL=C tr -dc a-z0-9 </dev/urandom | head -c 6)"
    return
  fi
  local out
  # ADAPTERS + EVAL_DATASETS must be exported before this `sbatch` call.
  out="$(sbatch "$@" scripts/run_cross_eval.slurm)"
  echo "          $out" >&2
  awk '{print $NF}' <<<"$out"
}

# Pre-flight: confirm every adapter dir exists locally before submission.
for variant in $VARIANTS; do
  for sz in $SIZES; do
    run_dir="outputs/qlora_${variant}_${sz}"
    if [[ ! -d "$run_dir/adapter" ]]; then
      echo "[launch] ERROR: $run_dir/adapter not found." >&2
      echo "         Adapter weights may not have been pulled from Tillicum." >&2
      exit 2
    fi
  done
done

declare -a JOB_IDS
for variant in $VARIANTS; do
  # Build the space-separated adapter list for this variant.
  ADAPTERS_LIST=""
  for sz in $SIZES; do
    ADAPTERS_LIST+="outputs/qlora_${variant}_${sz} "
  done
  ADAPTERS_LIST="${ADAPTERS_LIST% }"  # trim trailing space

  # Export env vars so sbatch --export=ALL picks them up.
  export ADAPTERS="$ADAPTERS_LIST"
  export EVAL_DATASETS="$EVAL_DATASETS"

  label="${variant} (5 adapters x ${EVAL_DATASETS// /+} = $(echo $SIZES | wc -w | tr -d ' ')*$(echo $EVAL_DATASETS | wc -w | tr -d ' ') cells)"
  jid=$(submit "$label  (walltime $JOB_WALLTIME)" \
    --export=ALL \
    --time="$JOB_WALLTIME" \
    --job-name="bead-cleaned-xeval-${variant}" \
    $EXTRA_SBATCH)
  JOB_IDS+=("$jid")
done

echo
echo "[launch] Submitted ${#JOB_IDS[@]} jobs:"
i=0
for variant in $VARIANTS; do
  echo "          ${variant}: ${JOB_IDS[$i]}"
  i=$((i + 1))
done
echo
echo "          squeue -u \$USER -t PD,R | head -10  # to watch"
echo
echo "          After completion:"
echo "          scripts/tillicum_sync.sh pull-all"
echo "          # The new cells' accuracy/F1 vs each dataset's published gold"
echo "          # are in outputs/cross_eval/qlora_beads_cleaned_<v>_<sz>__on__<ds>/eval_metrics.json"
echo "          # Compare to the existing baselines at outputs/cross_eval/qlora_beads_full__on__<ds>/"
echo "          # outputs/manifest.csv will also get one new row per cell (cross_eval.py upserts)."
