#!/usr/bin/env bash
# Submit the round-2 cleaning + retrain sweep: 2 variants x 5 sizes = 10
# QLoRA training jobs. Builds on top of the round-1 cleaning result by
# adding the round-1 winning cleaned model as a 4th voter, then either
# requiring strict unanimity or majority across the 4 voters.
#
# Variants (both are flip + balanced — the round-1 winning combo):
#   beads_cleaned_v2_strict     all 4 voters unanimous against gold
#   beads_cleaned_v2_majority   >=3 of 4 voters agree against gold
#
# Pre-registered success criterion (locked):
#   1. Within the cleaned_v2_strict family (or whichever wins), accuracy
#      should be MONOTONIC across sizes: acc(500) <= acc(1k) <= acc(5k)
#      <= acc(full), allowing +/-3 pp tolerance per step.
#   2. Peak round-2 accuracy >= peak round-1 accuracy (0.768) at any size.
#
# Usage:
#   scripts/launch_cleaned_retrain_v2_sweep.sh          # submit all 10 jobs
#   scripts/launch_cleaned_retrain_v2_sweep.sh --dry    # print sbatch lines
#
# Environment overrides:
#   SIZES         space-separated subset (default: "100 500 1k 5k full")
#   VARIANTS      space-separated subset (default: "v2_strict v2_majority")
#   EXTRA_SBATCH  appended to every sbatch invocation

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

DRY_RUN=""
for arg in "$@"; do
  case "$arg" in
    -n|--dry|--dry-run) DRY_RUN="1" ;;
    -h|--help)
      sed -n '2,22p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *) echo "[launch-v2] Unknown arg: $arg" >&2; exit 2 ;;
  esac
done

SIZES="${SIZES:-100 500 1k 5k full}"
VARIANTS="${VARIANTS:-v2_strict v2_majority}"
EXTRA_SBATCH="${EXTRA_SBATCH:-}"

submit() {
  local label="$1"; shift
  local cmd=(sbatch "$@")
  {
    echo
    echo "[launch-v2] $label"
    printf '            %s\n' "${cmd[*]}"
  } >&2
  if [[ -n "$DRY_RUN" ]]; then
    echo "            (dry-run; no submission)" >&2
    echo "mock-$(LC_ALL=C tr -dc a-z0-9 </dev/urandom | head -c 6)"
    return
  fi
  local out
  out="$("${cmd[@]}")"
  echo "            $out" >&2
  awk '{print $NF}' <<<"$out"
}

walltime_for() {
  local sz="$1"; local ds_short="$2"
  case "$sz" in
    100|500) echo "00:30:00" ;;
    1k)      echo "01:00:00" ;;
    5k)      echo "01:30:00" ;;
    full)
      # v2_strict pool is much smaller (most rows un-flagged → not re-balanced
      # cap kicks in less aggressively). v2_majority is closer to round-1
      # flip_balanced (14k rows). Both well under 3 hours; request 3 to be safe.
      echo "03:00:00"
      ;;
    *) echo "01:00:00" ;;
  esac
}

# Pre-flight: confirm the v2 manifests exist.
for variant in $VARIANTS; do
  ds="beads_cleaned_${variant}"
  if [[ ! -f "data/frozen/$ds/splits_manifest.json" ]]; then
    echo "[launch-v2] ERROR: data/frozen/$ds/splits_manifest.json not found." >&2
    echo "            Run scripts/make_cleaned_train_v2.py + push-data first." >&2
    exit 2
  fi
done

declare -a JOB_IDS
for variant in $VARIANTS; do
  ds="beads_cleaned_${variant}"
  for sz in $SIZES; do
    case "$sz" in
      100|500|1k|5k|full) ;;
      *) echo "[launch-v2] ERROR: unknown size '$sz'" >&2; exit 2 ;;
    esac
    walltime="$(walltime_for "$sz" "$variant")"
    label="${ds} / size=${sz}  (walltime $walltime)"
    jid=$(submit "$label" \
      --export=ALL,DATASET="$ds",SIZE="$sz" \
      --time="$walltime" \
      --job-name="bead-r2-${variant}-${sz}" \
      $EXTRA_SBATCH \
      scripts/run_qlora.slurm)
    JOB_IDS+=("$jid")
  done
done

echo
echo "[launch-v2] Submitted ${#JOB_IDS[@]} jobs:"
i=0
for variant in $VARIANTS; do
  ds="beads_cleaned_${variant}"
  for sz in $SIZES; do
    echo "            ${ds} / size=${sz}: ${JOB_IDS[$i]}"
    i=$((i + 1))
  done
done
echo
echo "            squeue -u \$USER -t PD,R | head -25  # to watch"
echo
echo "            After all jobs complete:"
echo "            scripts/tillicum_sync.sh pull-all"
echo "            python scripts/score_against_hand_labels.py \\"
echo "              --predictions outputs/qlora_beads_cleaned_v2_*/predictions.jsonl \\"
echo "                            outputs/qlora_beads_cleaned_flip_balanced_*/predictions.jsonl \\"
echo "                            outputs/cross_eval/qlora_beads_full__on__beads/predictions.jsonl"
