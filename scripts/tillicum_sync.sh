#!/usr/bin/env bash
# rsync helper for local <-> Tillicum.
#
# Tillicum has no git, so all transfers go over ssh/rsync. The remote project
# dir is a single shared clone (no per-user copies), so be aware that
# push-code from two teammates simultaneously will silently overwrite each
# other's working tree on a per-file basis (mtime+size wins). Coordinate.
#
# Usage:
#   scripts/tillicum_sync.sh push-code      # local -> remote: scripts/, docs/, manifest, top-level files
#   scripts/tillicum_sync.sh push-data      # local -> remote: data/bead/*.csv + data/frozen/*.jsonl
#   scripts/tillicum_sync.sh pull-results   # remote -> local: outputs/ (adapters, predictions, metrics)
#   scripts/tillicum_sync.sh pull-logs      # remote -> local: logs/ (slurm .out/.err)
#   scripts/tillicum_sync.sh pull-all       # pull-results + pull-logs
#   scripts/tillicum_sync.sh status         # ssh + ls the remote project dir
#
# Flags:
#   -n, --dry-run    Show what rsync would transfer without writing anything.
#
# Environment (override with `export FOO=...` or inline `FOO=... scripts/...`):
#   TILLICUM_USER     Tillicum username. REQUIRED (local $USER often differs).
#   TILLICUM_HOST     Default: tillicum.hyak.uw.edu
#   TILLICUM_PROJECT  Default: /gpfs/projects/imt526a/group7

set -euo pipefail

TILLICUM_HOST="${TILLICUM_HOST:-tillicum.hyak.uw.edu}"
TILLICUM_PROJECT="${TILLICUM_PROJECT:-/gpfs/projects/imt526a/group7}"

LOCAL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

DRY_RUN=""
SUBCMD=""

usage() {
  sed -n '2,23p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

for arg in "$@"; do
  case "$arg" in
    -n|--dry-run)                                   DRY_RUN="--dry-run" ;;
    -h|--help)                                      usage; exit 0 ;;
    push-code|push-data|pull-results|pull-logs|pull-all|status)
                                                    SUBCMD="$arg" ;;
    *) echo "[sync] Unknown arg: $arg" >&2; usage; exit 2 ;;
  esac
done

if [[ -z "$SUBCMD" ]]; then
  usage; exit 2
fi

if [[ -z "${TILLICUM_USER:-}" ]]; then
  echo "[sync] ERROR: TILLICUM_USER is not set. Your Tillicum username typically differs from"  >&2
  echo "       your local \$USER. Add this to ~/.zshrc (or ~/.bashrc):"                          >&2
  echo "           export TILLICUM_USER=yourtillicumname"                                        >&2
  echo "       Then 'source ~/.zshrc' and re-run."                                               >&2
  exit 2
fi

REMOTE="${TILLICUM_USER}@${TILLICUM_HOST}:${TILLICUM_PROJECT}"

# rsync flags shared by every transfer:
#   -a          archive (preserves perms/times/symlinks, recursive)
#   -v          verbose (one line per file)
#   -z          compress in flight (cheap on text-heavy transfers)
#   -h          human-readable sizes
#   --progress  per-file progress (works on macOS's stock rsync 2.6.9; --info=progress2 does not)
RSYNC=(rsync -avzh --progress)
[[ -n "$DRY_RUN" ]] && RSYNC+=("$DRY_RUN")

echo "[sync] cmd    : $SUBCMD${DRY_RUN:+  (dry-run)}"
echo "[sync] local  : $LOCAL_ROOT"
echo "[sync] remote : $REMOTE"
echo

case "$SUBCMD" in

  push-code)
    # File list comes from git: tracked files + untracked-not-ignored files.
    # This mirrors .gitignore exactly, so adding/changing exclude rules in
    # .gitignore (e.g. `.mypy_cache/`, `*.zip`, `terminal output*.txt`) is
    # automatically respected here — no parallel exclude list to maintain.
    if ! ( cd "$LOCAL_ROOT" && git rev-parse --git-dir >/dev/null 2>&1 ); then
      echo "[sync] ERROR: $LOCAL_ROOT is not a git repo; push-code relies on git ls-files." >&2
      exit 2
    fi

    FILE_LIST="$(mktemp -t tillicum_sync.XXXXXX)"
    trap 'rm -f "$FILE_LIST"' EXIT

    # -z (NUL-terminated) handles filenames with spaces, like 'docs/Proposal v2.txt'.
    ( cd "$LOCAL_ROOT" && git ls-files --cached --others --exclude-standard -z ) > "$FILE_LIST"

    n_files=$(tr -cd '\0' < "$FILE_LIST" | wc -c | tr -d ' ')
    echo "[sync] $n_files files selected by 'git ls-files --cached --others --exclude-standard'"
    echo

    "${RSYNC[@]}" --from0 --files-from="$FILE_LIST" \
      "$LOCAL_ROOT/" "$REMOTE/"
    ;;

  push-data)
    # Raw BEAD CSVs (CC-BY-NC, not in git) + locally-frozen JSONLs.
    # Only run this if you generated splits locally and want to skip running
    # freeze_splits.py on Tillicum. Otherwise prefer downloading BEAD directly
    # on Tillicum and letting the slurm launcher regenerate the JSONLs.
    if [[ -d "$LOCAL_ROOT/data/bead" ]]; then
      echo "[sync] data/bead/ -> remote"
      "${RSYNC[@]}" --include='*.csv' --include='README.md' --exclude='*' \
        "$LOCAL_ROOT/data/bead/" "$REMOTE/data/bead/"
    fi
    if [[ -d "$LOCAL_ROOT/data/frozen" ]]; then
      echo "[sync] data/frozen/*.jsonl + manifest -> remote"
      "${RSYNC[@]}" --include='*.jsonl' --include='splits_manifest.json' --exclude='*' \
        "$LOCAL_ROOT/data/frozen/" "$REMOTE/data/frozen/"
    fi
    ;;

  pull-results)
    # Sweep outputs: adapters, predictions, metrics, run_meta. Keep the v1
    # calibration dir on local untouched (it's already committed and frozen).
    mkdir -p "$LOCAL_ROOT/outputs"
    "${RSYNC[@]}" \
      --exclude='tillicum_1k_calibration/' \
      --exclude='__pycache__/' \
      --exclude='_smoke_*/' \
      --exclude='_interactive_smoke/' \
      "$REMOTE/outputs/" "$LOCAL_ROOT/outputs/"
    ;;

  pull-logs)
    mkdir -p "$LOCAL_ROOT/logs"
    "${RSYNC[@]}" \
      --include='*.out' --include='*.err' --include='.gitkeep' \
      --exclude='*' \
      "$REMOTE/logs/" "$LOCAL_ROOT/logs/"
    ;;

  pull-all)
    "$0" ${DRY_RUN:+--dry-run} pull-results
    echo
    "$0" ${DRY_RUN:+--dry-run} pull-logs
    ;;

  status)
    ssh "${TILLICUM_USER}@${TILLICUM_HOST}" \
      "echo '--- ls ${TILLICUM_PROJECT}' && ls -la '${TILLICUM_PROJECT}' && \
       echo && echo '--- squeue -u \$USER' && squeue -u \$USER 2>/dev/null || true && \
       echo && echo '--- recent outputs/' && ls -lat '${TILLICUM_PROJECT}/outputs/' 2>/dev/null | head -20 || true"
    ;;
esac

echo
echo "[sync] done."
