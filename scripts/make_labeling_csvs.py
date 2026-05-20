"""Build per-labeler CSVs for the BEADs hand-labeling task.

Background
----------
The cleaning experiment (see docs/build_log.md 2026-05-19 entries) needs
a held-out set of clean labels to serve as ground truth. This script
samples 500 rows from BEADs test, stratified 250 biased / 250 non-biased
to handle the asymmetric noise rate, then carves it into three labeler
tasks of 200 rows each.

Sampling
--------
- 500 random rows from data/frozen/beads/sizes/full/test.jsonl
- Stratified 250 gold=0 / 250 gold=1 (the noise asymmetry — BEADs missed
  bias ~5x more often than it over-called it — means uniform sampling
  would under-represent the noisier non-biased rows)
- 50 designated as the shared IAA block (all 3 labelers see them)
- Remaining 450 split 3 ways: 150 unique per labeler
- Per labeler: 50 IAA + 150 unique = 200, randomly shuffled so the
  labeler can't tell which rows are shared

Blinding
--------
Labeler CSVs contain only ``id`` (sequential 1..200) and ``text``. No
BEADs gold label, no model predictions, no IAA flag. The labeler fills
in the ``label`` column with exactly one of: ``biased`` / ``non-biased``.

A private ``_mapping.csv`` keeps the join back to BEADs row_idx + gold
+ non_beads_vote for the post-labeling analysis. This file is NOT
shared with labelers.

Reproducibility
---------------
Everything is seeded (default 42). Same seed -> same sampling. If the
team wants to redo the experiment with a different sample, change the
seed and re-run.

Usage
-----
    python scripts/make_labeling_csvs.py
    python scripts/make_labeling_csvs.py --names alice bob carol
    python scripts/make_labeling_csvs.py --seed 17 --output-dir labeling_v2
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BEADS_TEST = REPO_ROOT / "data" / "frozen" / "beads" / "sizes" / "full" / "test.jsonl"
DEFAULT_AUDIT_CSV = REPO_ROOT / "beads_label_audit.csv"


def normalize_whitespace(text: str) -> str:
    """Collapse embedded newlines, tabs, and multi-spaces to single spaces.

    BEADs source rows can contain newlines mid-sentence ("\\n"). CSV writers
    correctly quote those fields, but VS Code's line-based CSV viewer still
    sees one logical row as multiple physical lines and complains. Collapsing
    here means each CSV row maps to exactly one file line, no surprises for
    the labelers either.
    """
    return " ".join(text.split())


def load_test_rows(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            rows.append({
                "beads_row_idx": i,
                "text": normalize_whitespace(r["text"]),
                "gold_int": int(r["label_int"]),
                "gold_label": "biased" if int(r["label_int"]) == 1 else "non-biased",
            })
    return rows


def load_non_beads_vote(audit_csv: Path) -> dict[int, str]:
    """Map BEADs test row_idx -> non_beads_vote ('biased'|'non-biased'|'split')."""
    if not audit_csv.is_file():
        print(f"[make-labeling] note: {audit_csv} not found; mapping will skip non_beads_vote column.",
              file=sys.stderr)
        return {}
    out: dict[int, str] = {}
    with audit_csv.open() as f:
        for r in csv.DictReader(f):
            out[int(r["row_idx"])] = r["non_beads_vote"]
    return out


def stratified_sample(rows: list[dict], n_per_class: int, rng: random.Random) -> list[dict]:
    """Draw n_per_class rows from each gold class."""
    by_class: dict[int, list[dict]] = {0: [], 1: []}
    for r in rows:
        by_class[r["gold_int"]].append(r)

    out: list[dict] = []
    for g in (0, 1):
        pool = by_class[g]
        if len(pool) < n_per_class:
            sys.exit(
                f"[make-labeling] only {len(pool)} rows available for gold={g}, "
                f"need {n_per_class}. Lower --per-class-n or recheck the test split."
            )
        out.extend(rng.sample(pool, n_per_class))
    return out


def carve_assignment(sample: list[dict], n_iaa: int, n_labelers: int,
                     rng: random.Random) -> tuple[list[dict], list[list[dict]]]:
    """Split the 500-row sample into:
      - iaa_rows: n_iaa rows seen by all labelers
      - per_labeler_unique: n_labelers lists of unique rows, no overlap

    Stratification on gold is preserved: the IAA block is sampled with
    roughly the same class balance as the input, and each unique chunk
    is class-balanced too.
    """
    n_total = len(sample)
    n_unique_total = n_total - n_iaa
    if n_unique_total % n_labelers != 0:
        sys.exit(
            f"[make-labeling] (n_total - n_iaa) = {n_unique_total} doesn't divide "
            f"evenly across {n_labelers} labelers. Adjust n_iaa or n_total."
        )
    n_unique_per_labeler = n_unique_total // n_labelers

    # Sample IAA preserving rough class balance.
    by_class: dict[int, list[dict]] = {0: [], 1: []}
    for r in sample:
        by_class[r["gold_int"]].append(r)
    rng.shuffle(by_class[0])
    rng.shuffle(by_class[1])

    iaa_per_class = n_iaa // 2
    iaa_rows = by_class[0][:iaa_per_class] + by_class[1][:iaa_per_class]
    rng.shuffle(iaa_rows)

    # Remaining rows after IAA carve-out, by class.
    remaining_by_class = {
        0: by_class[0][iaa_per_class:],
        1: by_class[1][iaa_per_class:],
    }
    rng.shuffle(remaining_by_class[0])
    rng.shuffle(remaining_by_class[1])

    # Distribute remaining rows to each labeler preserving class balance.
    n_unique_per_class = n_unique_per_labeler // 2
    per_labeler: list[list[dict]] = [[] for _ in range(n_labelers)]
    for c in (0, 1):
        pool = remaining_by_class[c]
        for li in range(n_labelers):
            chunk = pool[li * n_unique_per_class : (li + 1) * n_unique_per_class]
            per_labeler[li].extend(chunk)
    return iaa_rows, per_labeler


def write_labeler_csv(path: Path, ordered_rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "text", "label"])
        for i, r in enumerate(ordered_rows, start=1):
            writer.writerow([i, r["text"], ""])


def write_mapping(path: Path, mapping_rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["labeler", "labeler_row_id", "is_iaa", "beads_row_idx",
                  "gold_label", "non_beads_vote"]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(mapping_rows)


LABELER_README = """\
# BEADs hand-labeling task

You've been assigned **200 rows** from the BEADs test set. Your job is to
label each row as **biased** or **non-biased** based on the protocol
below. You'll save the file back and we'll merge everyone's labels into
a single consensus set.

## What you see and what you don't

You see only `id` and `text`. You do **not** see:
- BEADs's official gold label
- Any model predictions
- Which of your 200 rows are shared with other labelers

This is intentional — about 50 of your 200 rows are shared with the
other labelers as an inter-annotator-agreement (IAA) check, but they're
randomly mixed in so we can measure agreement on labels that none of us
knew were being cross-checked.

## How to label

For each row, fill in the `label` column with exactly one of these
strings (case-sensitive, no quotes, no extra spaces):

- `biased`
- `non-biased`

Leave any rows you can't decide on **blank** — don't guess. We'll
treat blank labels as "abstain" in the analysis. Better to skip 5 than
to coin-flip 5.

## Working definition of "biased"

**A statement is biased if it carries a partisan, emotional, sarcastic,
or judgmental framing on a public-interest topic.** Otherwise it's
non-biased.

This includes:
- Sarcasm and ridicule ("Of COURSE she has!", "Great. More corporatism...")
- Rhetorical questions implying a stance ("Anyone notice that...?")
- Partisan name-calling ("crony capitalism", "leftist mob", "religion of
  peace" used pejoratively)
- Opinion presented as fact about a contested issue
- Emotional appeals on political/social topics

This does NOT include:
- Pure factual statements ("The committee released its annual report.")
- Personal logistics or off-topic content ("Anyone get mouth ulcers?")
- Questions seeking information ("Does anyone know where I can find...?")
- Plain opinions on uncontested topics ("I love this restaurant.")
- Quoting biased speech *neutrally* (depends on framing — if the quote
  is presented for analysis vs amplified, judge accordingly)

When in doubt: ask "would a careful editor remove this sentence from a
news article for being one-sided?" If yes, it's biased.

## Sanity rules

- Sarcasm = biased (the speaker is making a judgmental point under cover
  of irony)
- Rhetorical questions with implicit answers = biased
- Loaded vocabulary on contested topics = biased
- Single-word reactions ("Awesome!", "Sad!") on news topics = biased if
  there's a clear political/judgmental target, otherwise non-biased
- Text with profanity but no political/judgmental framing = non-biased
- Text in another language or gibberish = non-biased (we can't judge)

## Time estimate

About 30 seconds per row × 200 rows = **~1.5-2 hours**. Don't try to do
it in one sitting; calibration drifts after ~50 rows. Take breaks.

## Save format

Save the file as `labeler_<your_name>_labeled.csv` (add `_labeled` to
the filename so we know it's done). Send it back via [whatever your
team uses].

## What happens next

1. We compute IAA on the shared rows. If we agree on at least 70% of
   them, we proceed.
2. We compare your consensus labels against BEADs's official gold to
   get a mislabel rate.
3. We use the consensus labels to evaluate the current and the
   cleaned-data QLoRA models.
"""


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--beads-test", default=str(DEFAULT_BEADS_TEST))
    ap.add_argument("--audit-csv", default=str(DEFAULT_AUDIT_CSV))
    ap.add_argument("--output-dir", default="labeling",
                    help="Directory for the labeler CSVs + mapping (default: labeling/)")
    ap.add_argument("--per-class-n", type=int, default=250,
                    help="Rows to sample from each gold class (default: 250 -> 500 total)")
    ap.add_argument("--n-iaa", type=int, default=50,
                    help="Number of IAA rows seen by all labelers (default: 50)")
    ap.add_argument("--names", nargs="+", default=["a", "b", "c"],
                    help="Labeler names (default: a b c). N names = N labelers.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    n_labelers = len(args.names)

    beads_test = Path(args.beads_test)
    audit_csv = Path(args.audit_csv)
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir

    if not beads_test.is_file():
        sys.exit(f"[make-labeling] not found: {beads_test}")

    print(f"[make-labeling] reading {beads_test}")
    rows = load_test_rows(beads_test)
    print(f"[make-labeling]   {len(rows)} test rows loaded")
    nb_vote = load_non_beads_vote(audit_csv)
    if nb_vote:
        print(f"[make-labeling]   {len(nb_vote)} non_beads_vote entries loaded from audit CSV")

    print(f"[make-labeling] stratified sample: {args.per_class_n} per class -> "
          f"{2 * args.per_class_n} total")
    sample = stratified_sample(rows, args.per_class_n, rng)

    print(f"[make-labeling] carving assignment: {args.n_iaa} IAA + "
          f"{(len(sample) - args.n_iaa) // n_labelers} unique per labeler "
          f"= {(len(sample) - args.n_iaa) // n_labelers + args.n_iaa} per labeler")
    iaa_rows, per_labeler_unique = carve_assignment(sample, args.n_iaa, n_labelers, rng)

    # Build per-labeler order: shuffle (iaa + unique) together so the labeler
    # can't tell which is which from row position.
    mapping_rows: list[dict] = []
    out_dir.mkdir(parents=True, exist_ok=True)
    for li, name in enumerate(args.names):
        combined = list(iaa_rows) + list(per_labeler_unique[li])
        # Tag each with is_iaa BEFORE shuffling so we can preserve the flag.
        tagged = [(True, r) for r in iaa_rows] + [(False, r) for r in per_labeler_unique[li]]
        rng.shuffle(tagged)

        ordered_rows = [r for _, r in tagged]
        csv_path = out_dir / f"labeler_{name}.csv"
        write_labeler_csv(csv_path, ordered_rows)
        print(f"[make-labeling]   wrote {csv_path}  ({len(ordered_rows)} rows)")

        for labeler_row_id, (is_iaa, r) in enumerate(tagged, start=1):
            mapping_rows.append({
                "labeler": name,
                "labeler_row_id": labeler_row_id,
                "is_iaa": int(is_iaa),
                "beads_row_idx": r["beads_row_idx"],
                "gold_label": r["gold_label"],
                "non_beads_vote": nb_vote.get(r["beads_row_idx"], ""),
            })

    mapping_path = out_dir / "_mapping.csv"
    write_mapping(mapping_path, mapping_rows)
    print(f"[make-labeling]   wrote {mapping_path}  ({len(mapping_rows)} rows; PRIVATE — do not share)")

    readme_path = out_dir / "LABELER_README.md"
    readme_path.write_text(LABELER_README)
    print(f"[make-labeling]   wrote {readme_path}")

    # Summary.
    print()
    print(f"[make-labeling] sampled {len(sample)} rows total; "
          f"{args.n_iaa} IAA + {len(sample) - args.n_iaa} split across {n_labelers} labelers")
    print(f"[make-labeling] each labeler gets {(len(sample) - args.n_iaa) // n_labelers + args.n_iaa} rows")
    print(f"[make-labeling] share labeler_*.csv + LABELER_README.md with the team.")
    print(f"[make-labeling] keep _mapping.csv PRIVATE — it would deanonymize the IAA block.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
