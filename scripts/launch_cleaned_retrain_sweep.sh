#!/usr/bin/env bash
# Submit the cleaning + retrain sweep: 4 cleaned datasets x 5 train sizes
# = 20 QLoRA fine-tuning jobs.
#
# Each job uses scripts/run_qlora.slurm with DATASET set to one of:
#   beads_cleaned_remove           (~16,892 full rows, natural ~70/30 biased)
#   beads_cleaned_remove_balanced  (~10,404 full rows, 50/50 by undersample)
#   beads_cleaned_flip             (~27,263 full rows, natural ~74/26 biased)
#   beads_cleaned_flip_balanced    (~14,246 full rows, 50/50 by undersample)
#
# and SIZE in {100, 500, 1k, 5k, full}. Output dirs:
#   outputs/qlora_<dataset>_<size>/
#
# The slurm wrapper does an on-the-fly eval against BEADs's original
# (noisy) test set, which is still useful as a "did the cleaned model
# fit BEADs gold?" sanity check. The HEADLINE evaluation (against the
# 500 hand-labels) is a separate post-sweep step — see
# scripts/score_against_hand_labels.py.
#
# Usage:
#   scripts/launch_cleaned_retrain_sweep.sh          # submit all 20 jobs
#   scripts/launch_cleaned_retrain_sweep.sh --dry    # print the sbatch commands
#
# Environment overrides:
#   SIZES         space-separated subset (default: "100 500 1k 5k full")
#   VARIANTS      space-separated subset (default: all 4 cleaned variants)
#   EXTRA_SBATCH  appended to every sbatch invocation
#
# All jobs run concurrently subject to the H200 reservation capacity.
# The fcntl lock in scripts/update_manifest.py guards the shared manifest
# against concurrent writers — same protection that worked for the
# original 5-job sweep on 2026-05-19.

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

DRY_RUN=""
for arg in "$@"; do
  case "$arg" in
    -n|--dry|--dry-run) DRY_RUN="1" ;;
    -h|--help)          sed -n '2,30p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "[launch] Unknown arg: $arg" >&2; exit 2 ;;
  esac
done

SIZES="${SIZES:-100 500 1k 5k full}"
VARIANTS="${VARIANTS:-beads_cleaned_remove beads_cleaned_remove_balanced beads_cleaned_flip beads_cleaned_flip_balanced}"
EXTRA_SBATCH="${EXTRA_SBATCH:-}"

submit() {
  # Returns the sbatch job id on stdout; everything else to stderr.
  local label="$1"; shift
  local cmd=(sbatch "$@")
  {
    echo
    echo "[launch] $label"
    printf '          %s\n' "${cmd[*]}"
  } >&2
  if [[ -n "$DRY_RUN" ]]; then
    echo "          (dry-run; no submission)" >&2
    echo "mock-$(LC_ALL=C tr -dc a-z0-9 </dev/urandom | head -c 6)"
    return
  fi
  local out
  out="$("${cmd[@]}")"
  echo "          $out" >&2
  awk '{print $NF}' <<<"$out"
}

# Per-(size, dataset) walltime budget. Cleaning shrinks the train set for
# most variants vs the original beads sweep, but request generously so
# small jobs don't get killed for queue surprise.
walltime_for() {
  local sz="$1"; local ds="$2"
  case "$sz" in
    100|500) echo "00:30:00" ;;
    1k)      echo "01:00:00" ;;
    5k)      echo "01:30:00" ;;
    full)
      case "$ds" in
        beads_cleaned_remove)          echo "03:00:00" ;;  # ~17k rows
        beads_cleaned_remove_balanced) echo "02:00:00" ;;  # ~10k rows
        beads_cleaned_flip)            echo "04:00:00" ;;  # 27k rows
        beads_cleaned_flip_balanced)   echo "02:30:00" ;;  # ~14k rows
        *)                             echo "04:00:00" ;;
      esac
      ;;
    *) echo "01:00:00" ;;
  esac
}

declare -a JOB_IDS

for ds in $VARIANTS; do
  case "$ds" in
    beads_cleaned_remove|beads_cleaned_remove_balanced|beads_cleaned_flip|beads_cleaned_flip_balanced) ;;
    *) echo "[launch] ERROR: unknown variant '$ds'" >&2; exit 2 ;;
  esac
  if [[ ! -f "data/frozen/$ds/splits_manifest.json" ]]; then
    echo "[launch] ERROR: data/frozen/$ds/splits_manifest.json not found. " >&2
    echo "         Run scripts/make_cleaned_train.py first." >&2
    exit 2
  fi
done

for ds in $VARIANTS; do
  for sz in $SIZES; do
    case "$sz" in
      100|500|1k|5k|full) ;;
      *) echo "[launch] ERROR: unknown size '$sz'" >&2; exit 2 ;;
    esac
    label="$ds / size=$sz"
    walltime="$(walltime_for "$sz" "$ds")"
    jid=$(submit "$label  (walltime $walltime)" \
      --export=ALL,DATASET="$ds",SIZE="$sz" \
      --time="$walltime" \
      $EXTRA_SBATCH \
      scripts/run_qlora.slurm)
    JOB_IDS+=("$jid")
  done
done

echo
echo "[launch] Submitted ${#JOB_IDS[@]} jobs:"
i=0
for ds in $VARIANTS; do
  for sz in $SIZES; do
    echo "          ${ds} / size=${sz}: ${JOB_IDS[$i]}"
    i=$((i + 1))
  done
done
echo
echo "          squeue -u \$USER -t PD,R | head -25  # to watch"
