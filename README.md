# BEAD QLoRA sweep — Group 7 (IMT 526 A, Spring 2026)

Fine-tune **meta-llama/Llama-3.1-8B-Instruct** with **QLoRA** on the
[BEAD](https://huggingface.co/datasets/shainar/BEAD) bias-classification
split, sweeping over five training-set sizes (100, 500, 1k, 5k, full) to
characterize the learning curve against few-shot prompting and a TF-IDF
baseline. Targets run on a single H200 via UW's Tillicum cluster.

> Project context: `Project Proposal v2` (shared Google Drive). This repo is
> the artifact half — scripts, frozen splits, and the v1 calibration metrics.
> The narrative writeup lives elsewhere.

## Repository layout

```text
.
├── README.md                              # this file
├── LICENSE                                # MIT, applies to code only — see data/bead/README.md for data terms
├── requirements.txt                       # pip pin-set mirroring the Tillicum `llm` env
├── scripts/                               # v2 sweep (current)
│   ├── freeze_splits.py                   # build nested stratified subsets + sha256 manifest
│   ├── train_qlora.py                     # QLoRA SFT (4-bit nf4 + LoRA r=16), auto-writes run_meta
│   ├── eval_adapter.py                    # likelihood-scored binary eval → metrics + predictions
│   ├── update_manifest.py                 # upsert one row into outputs/manifest.csv after each run
│   ├── run_qlora.slurm                    # one Slurm launcher; SIZE=100|500|1k|5k|full
│   └── legacy/                            # v1 1k calibration scripts (kept for reproducibility)
├── data/
│   ├── bead/                              # raw BEAD CSVs (CC BY-NC 4.0; .csv files gitignored — see README in folder)
│   └── frozen/                            # splits_manifest.json is tracked; the JSONLs are gitignored and regenerable
├── docs/
│   ├── proposal_v2.txt                    # the May 9 Project Proposal v2 (full narrative)
│   ├── v1_calibration_writeup.md          # the May 9 calibration run writeup
│   └── build_log.md                       # running log of architecturally meaningful changes (source material for the final writeup)
├── outputs/
│   └── tillicum_1k_calibration/           # v1 metrics + predictions (adapter weights gitignored)
└── logs/                                  # Slurm stdout/stderr (gitignored content; dir kept)
```

## First-time setup (after `git clone`)

The raw BEAD CSVs and derived JSONL splits are **not** in the repo (CC BY-NC
data is regenerable; large derived artifacts are reproducible from code).
`data/frozen/splits_manifest.json` ships with sha256s, so anyone with the
repo can verify they regenerated the exact bytes used for training.

```bash
# 1. Set up the env (mirrors the Tillicum `llm` env).
conda create -n llm python=3.12 -y
conda activate llm
pip install -r requirements.txt

# 2. Fetch the BEAD CSVs into data/bead/  (one option: huggingface-cli).
#    Or pull straight into freeze_splits.py via --from-hf in the next step.
mkdir -p data/bead
huggingface-cli download shainar/BEAD --repo-type dataset \
  --include "Bias_classification/*.csv" --local-dir data/bead
# (move files so they land at data/bead/bias-train.csv, bias-valid.csv)

# 3. Rebuild the frozen splits and confirm the hashes match the manifest in git.
python scripts/freeze_splits.py --csv-dir data/bead --out-dir data/frozen
# The script prints the manifest; compare to the committed
# data/frozen/splits_manifest.json — sha256s should be byte-identical.
```

## Quickstart

### Local smoke test (no GPU, no Llama download)

These run on macOS/Linux with the conda env from `requirements.txt`. They
use a ~5 MB Llama-shaped tiny model so the data path + training loop +
adapter save/load + likelihood eval are all exercised end-to-end.

```bash
conda create -n llm python=3.12 -y
conda activate llm
pip install -r requirements.txt

# Build a tiny synthetic split (no internet, no HF gating).
python scripts/freeze_splits.py --mock --sizes 16,32,64 --out-dir _smoke_frozen

# Train smoke (4 steps, MPS/CPU friendly).
python scripts/train_qlora.py --smoke-test \
  --train-jsonl _smoke_frozen/train_32.jsonl \
  --splits-manifest _smoke_frozen/splits_manifest.json \
  --output-dir _smoke_run --run-name _smoke \
  --per-device-batch-size 2 --grad-accum 1 --max-seq-length 128

# Eval smoke.
python scripts/eval_adapter.py --smoke-test \
  --adapter-path _smoke_run/adapter_smoke \
  --test-jsonl _smoke_frozen/test_held_out.jsonl \
  --output-dir _smoke_run --run-name _smoke --max-test-rows 8

# Manifest upsert.
python scripts/update_manifest.py --run-dir _smoke_run --manifest _smoke_run/manifest.csv

# Cleanup
rm -rf _smoke_frozen _smoke_run
```

The smoke numbers are meaningless — the goal is to confirm tokenization,
chat template, completion-only loss masking, adapter save/load, and
likelihood inference all wire up correctly.

### Tillicum sweep run

After cloning to `/gpfs/projects/imt526a/group7` and verifying `data/bead/`
+ `data/frozen/` are present:

```bash
# One Slurm submission per training size. Time budgets bake in a 25%
# contingency over the calibration-derived estimates.
sbatch --export=ALL,SIZE=100  --time=00:30:00 scripts/run_qlora.slurm
sbatch --export=ALL,SIZE=500  --time=00:45:00 scripts/run_qlora.slurm
sbatch --export=ALL,SIZE=1k   --time=01:00:00 scripts/run_qlora.slurm
sbatch --export=ALL,SIZE=5k   --time=02:00:00 scripts/run_qlora.slurm
sbatch --export=ALL,SIZE=full --time=05:00:00 scripts/run_qlora.slurm

squeue -u $USER
```

Each job: idempotently re-builds `data/frozen/` if missing → trains →
evals on `data/frozen/test_held_out.jsonl` → upserts one row into
`outputs/manifest.csv`. Outputs land in `outputs/qlora_<size>/`.

Per-run artifacts (regenerable, gitignored except for v1 calibration metrics):

```text
outputs/qlora_<size>/
├── adapter/                    # PEFT adapter (safetensors)
├── train_metrics.json          # wall-clock, peak mem, throughput, trainable params
├── eval_metrics.json           # accuracy, precision/recall/F1 (pos + macro)
├── predictions.jsonl           # per-example gold/pred + per-class log-likelihood
└── run_meta.json               # args, libs, host, slurm job id, git head, input sha256s
```

## Syncing files with Tillicum

Tillicum has no git, so all transfers go over `ssh`/`rsync`. The team shares a
single clone at `/gpfs/projects/imt526a/group7` — there are no per-user
copies, so simultaneous `push-code` from two teammates will silently overwrite
each other on a per-file basis. Coordinate before pushing.

`scripts/tillicum_sync.sh` wraps the common cases. One-time setup — put your
Tillicum username (typically different from your local `$USER`) in your shell rc:

```bash
# ~/.zshrc or ~/.bashrc
export TILLICUM_USER=your_tillicum_username
# Optional — these have sensible defaults:
# export TILLICUM_HOST=tillicum.hyak.uw.edu
# export TILLICUM_PROJECT=/gpfs/projects/imt526a/group7
```

Then from anywhere in the repo:

```bash
# Local -> Tillicum: code, docs, splits_manifest.json (excludes outputs/, logs/,
# raw CSVs, frozen JSONLs — these are either pulled DOWN later or regenerated).
scripts/tillicum_sync.sh push-code

# Local -> Tillicum: raw BEAD CSVs + locally-frozen JSONLs (only if you froze
# locally and want to skip running freeze_splits.py on Tillicum).
scripts/tillicum_sync.sh push-data

# Tillicum -> Local: outputs/qlora_*/ (adapters, predictions, metrics, run_meta).
# Skips outputs/tillicum_1k_calibration/ (v1, already on local + in git).
scripts/tillicum_sync.sh pull-results

# Tillicum -> Local: slurm .out/.err.
scripts/tillicum_sync.sh pull-logs

# Both pulls.
scripts/tillicum_sync.sh pull-all

# Sanity check what's on the remote (ssh + ls + squeue).
scripts/tillicum_sync.sh status

# Always preview a transfer first if you're unsure — rsync runs in --dry-run
# mode and prints what it would change.
scripts/tillicum_sync.sh --dry-run push-code
```

Typical workflow for a sweep:

1. Edit code locally → commit + push to GitHub (audit trail for teammates).
2. `scripts/tillicum_sync.sh push-code` → mirror working tree to Tillicum.
3. `ssh ${TILLICUM_USER}@tillicum.hyak.uw.edu` → `cd /gpfs/projects/imt526a/group7` → `sbatch --export=ALL,SIZE=100 scripts/run_qlora.slurm`.
4. After the job: `scripts/tillicum_sync.sh pull-all` from local.
5. `git add outputs/qlora_*/run_meta.json outputs/qlora_*/train_metrics.json outputs/qlora_*/eval_metrics.json outputs/manifest.csv outputs/manifest.json && git commit` (adapters / predictions / checkpoints stay gitignored).

## Frozen splits — what your teammates need

`data/frozen/splits_manifest.json` is the source of truth. Every split has
a sha256 over its newline-joined JSONL bytes; teammates running the
TF-IDF + logistic regression baseline and the 3-shot prompting baseline
should reference these exact files so all three models share the same
held-out test set.

| Split | Rows | Use |
|---|---|---|
| `train_100.jsonl` | 100 | Sweep run 1 |
| `train_500.jsonl` | 500 | Sweep run 2 |
| `train_1k.jsonl` | 1,000 | Sweep run 3 |
| `train_5k.jsonl` | 5,000 | Sweep run 4 |
| `train_full.jsonl` | 27,263 | Sweep run 5 |
| `val.jsonl` | 8,520 | Model selection / hparam tuning |
| `test_held_out.jsonl` | 6,816 | Final eval — shared across QLoRA, TF-IDF, 3-shot |

Nested by construction: `train_100 ⊂ train_500 ⊂ train_1k ⊂ train_5k ⊂ train_full`.
A larger run sees everything a smaller run saw, plus the new examples.

## Environment

The training/eval scripts target the `llm` env on Tillicum, which mirrors
the class-aligned baseline (Python 3.12, torch 2.11, transformers 5.5,
peft 0.19, trl 1.3, bitsandbytes). `requirements.txt` here matches that
set, plus the macOS-only `KMP_DUPLICATE_LIB_OK=TRUE` workaround for the
local smoke test (libomp shows up twice on Apple Silicon when torch is
pip-installed alongside conda-forge scientific libs).

For Tillicum specifics — module loads, HF token handling, conda
activation, `LD_LIBRARY_PATH` for libstdc++ — see `scripts/run_qlora.slurm`.

## v1 calibration (May 9 2026)

A separate 1k-example, 1-epoch run on H200 was used to derive the
compute budget in section 4 of the proposal. Documented in
[docs/v1_calibration_writeup.md](docs/v1_calibration_writeup.md). The
measured metrics (`outputs/tillicum_1k_calibration/calibration_metrics.json`
and `eval_metrics.json`) are committed; the 80 MB adapter is gitignored
but regeneratable from `scripts/legacy/run_tillicum_1k.slurm`.

Key calibration result (informs the time budgets above):

- 1k examples × 1 epoch on a single H200: **1.58 min wall-clock**, **14.6 GB peak VRAM**, **10.6 ex/sec sustained throughput**.

## License

Code: MIT (`LICENSE`).
Data: BEAD is CC BY-NC 4.0 — see `data/bead/README.md` and
[the BEAD dataset card](https://huggingface.co/datasets/shainar/BEAD).
Research use only; do not redistribute commercially.

## Team

Abrevaa E. Prihutama (Data) · Zachary Greenman (Training) · Ash Dhammani (Eval + Coordination).
