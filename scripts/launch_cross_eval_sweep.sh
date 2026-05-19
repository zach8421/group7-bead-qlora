#!/usr/bin/env bash
# Submit the full cross-eval sweep to slurm: 3 new train jobs (babe, cajcodes,
# wnc) plus one cross-eval matrix job that depends on all three.
#
# BEADs is already trained (outputs/qlora_beads_full/adapter/ from sweep
# 117771). The cross-eval matrix evaluates all four full-size adapters on
# all four datasets' test splits — 16 cells, ~30 min on H200.
#
# Usage:
#   scripts/launch_cross_eval_sweep.sh          # submit all 4 jobs
#   scripts/launch_cross_eval_sweep.sh --dry    # print the sbatch commands, submit nothing
#
# Environment overrides (optional):
#   WNC_TRAIN_ROWS    cap on WNC train rows. Default 27263 (= BEADs full).
#                     Set to 0 to use the full 88k.
#   SKIP_CROSS_EVAL   "1" to submit only the 3 train jobs.
#   EXTRA_SBATCH      extra args passed to every sbatch invocation
#                     (e.g. "--partition=h200 --reservation=foo").
#
# Why dependencies and not concurrent submission: the cross-eval cell
# matrix needs all four adapters on disk. The train jobs themselves can run
# concurrently — the manifest race is now guarded by an fcntl lock in
# scripts/update_manifest.py — but cross-eval must wait for all to complete.

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

DRY_RUN=""
for arg in "$@"; do
  case "$arg" in
    -n|--dry|--dry-run) DRY_RUN="1" ;;
    -h|--help)          sed -n '2,21p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "[launch] Unknown arg: $arg" >&2; exit 2 ;;
  esac
done

WNC_TRAIN_ROWS="${WNC_TRAIN_ROWS:-27263}"
SKIP_CROSS_EVAL="${SKIP_CROSS_EVAL:-0}"
EXTRA_SBATCH="${EXTRA_SBATCH:-}"

submit() {
  # Returns the sbatch job id on stdout; everything else goes to stderr so
  # `JID=$(submit ...)` captures only the id.
  local label="$1"; shift
  local cmd=(sbatch "$@")
  {
    echo
    echo "[launch] $label"
    printf '          %s\n' "${cmd[*]}"
  } >&2
  if [[ -n "$DRY_RUN" ]]; then
    echo "          (dry-run; no submission)" >&2
    # Stable, non-numeric placeholder so afterok=mock-...:mock-... doesn't
    # silently look like a real dep list. (LC_ALL=C makes BSD tr happy on
    # macOS for the random suffix.)
    echo "mock-$(LC_ALL=C tr -dc a-z0-9 </dev/urandom | head -c 6)"
    return
  fi
  local out
  out="$("${cmd[@]}")"
  echo "          $out" >&2
  # `sbatch` prints "Submitted batch job <id>" on success; extract the id.
  awk '{print $NF}' <<<"$out"
}

declare -a TRAIN_JOB_IDS

JID_BABE=$(submit "train babe (full)" \
  --export=ALL,DATASET=babe,SIZE=full \
  --time=01:00:00 \
  $EXTRA_SBATCH \
  scripts/run_qlora.slurm)
TRAIN_JOB_IDS+=("$JID_BABE")

JID_CAJ=$(submit "train cajcodes (full)" \
  --export=ALL,DATASET=cajcodes,SIZE=full \
  --time=00:30:00 \
  $EXTRA_SBATCH \
  scripts/run_qlora.slurm)
TRAIN_JOB_IDS+=("$JID_CAJ")

if [[ "$WNC_TRAIN_ROWS" -gt 0 ]]; then
  WNC_EXPORT="ALL,DATASET=wnc,SIZE=full,MAX_TRAIN_ROWS=$WNC_TRAIN_ROWS"
  WNC_LABEL="train wnc (capped at $WNC_TRAIN_ROWS rows)"
else
  WNC_EXPORT="ALL,DATASET=wnc,SIZE=full"
  WNC_LABEL="train wnc (full 88k)"
fi
JID_WNC=$(submit "$WNC_LABEL" \
  --export="$WNC_EXPORT" \
  --time=04:00:00 \
  $EXTRA_SBATCH \
  scripts/run_qlora.slurm)
TRAIN_JOB_IDS+=("$JID_WNC")

if [[ "$SKIP_CROSS_EVAL" == "1" ]]; then
  echo
  echo "[launch] SKIP_CROSS_EVAL=1 — not submitting cross-eval. Run later:"
  echo "          sbatch --export=ALL scripts/run_cross_eval.slurm"
  exit 0
fi

DEP_LIST="$(IFS=:; echo "${TRAIN_JOB_IDS[*]}")"
JID_XEVAL=$(submit "cross-eval (4×4 matrix, deps on $DEP_LIST)" \
  --dependency=afterok:"$DEP_LIST" \
  --time=02:00:00 \
  $EXTRA_SBATCH \
  scripts/run_cross_eval.slurm)

echo
echo "[launch] Summary:"
echo "          babe     -> $JID_BABE"
echo "          cajcodes -> $JID_CAJ"
echo "          wnc      -> $JID_WNC"
echo "          x-eval   -> $JID_XEVAL  (afterok on the 3 trains)"
echo
echo "          squeue -u \$USER -t PD,R  # to watch"
