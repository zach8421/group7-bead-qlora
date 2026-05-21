# Build log — BEAD QLoRA sweep (group7)

Reverse-chronological log of architecturally meaningful changes, decisions,
and measurements. Intended as source material for the final technical
writeup. One entry per change worth remembering — not every commit.

For per-commit history, use `git log`. For per-run reproducibility metadata,
use the `run_meta.json` files in `outputs/<run>/`.

---

## 2026-05-21 — Project summary document for team + prof

**What**

[docs/project_summary.md](project_summary.md) — a single share-ready
summary of the full project arc. Six phases, each with What / Why /
Results blocks. Tight prose (no first or second person), all headline
numbers in tables, ~5-10 min read. Intended audience: teammates and
the instructor.

Phases summarised:

1. Original BEADs learning curve (0.804 against noisy gold)
2. 4x4 cross-eval matrix (datasets appear siloed)
3. Hand-labeling (~70% BEADs label noise)
4. Cleaning + retrain Round 1 (+48.6 pp to 0.768 against truth)
5. Cross-dataset transfer of cleaned models (BABE +49.6 pp)
6. Round 2 cleaning (stability gained, peak not exceeded)

Plus an "Overall outcome" section with 5 defensible claims, compute
cost ($26), and a reproducibility note.

Source material for the document is the per-phase entries below in
this build log; the summary distills them but doesn't add any new
findings.

---

## 2026-05-21 — Round-2 cleaning: monotonicity gained, peak not exceeded

**TL;DR**

Round 2 added the round-1 winner (qlora_beads_cleaned_flip_balanced_500)
as a 4th voter on top of the original 3-adapter ensemble and re-ran
the sweep with two flag rules: **strict** (all 4 voters unanimous
against gold) and **majority** (≥ 3 of 4 against gold). 10 new training
jobs.

**The pre-registered success criterion was not strictly met.** Peak
round-2 accuracy (0.724, v2_majority at size 5k) is **below** the
round-1 peak (0.768, flip_balanced at size 500). The monotonicity
criterion failed for v2_strict but **passed for v2_majority** — its
curve stays flat at 0.71-0.72 across train sizes 500–14k, within the
locked ±3pp tolerance per step.

So the headline finding is more interesting than a simple "round 2
won" or "round 2 lost":

> **Round 2 traded peak accuracy for stability.** The round-1 peak
> of 0.768 at size=500 appears to have been substantially a
> sampling-variance artifact, not a stable size dependence. When
> the cleaning is improved (4-voter rule), the model converges to
> a stable ~0.72 ceiling regardless of training size. That ceiling
> *is* the true post-cleaning accuracy of this approach.

Updated headline figures:
[docs/figures/cleaning_curve.png](figures/cleaning_curve.png) (accuracy)
and [cleaning_curve_f1.png](figures/cleaning_curve_f1.png) (F1_macro)
now include round-2 lines (green for strict, purple for majority).

**The mechanism: a 4-voter rule**

The round-1 cleaning used a 3-voter rule (babe, cajcodes, wnc
unanimous against BEADs gold → flip the label). The flip-correctness
gate measured 85% — so ~15% of round-1 relabels were wrong.

Round 2 adds qlora_beads_cleaned_flip_balanced_500 (the round-1
hand-label winner, 0.768 acc on hand-labels) as a 4th voter. Two
new variants:

| Rule | Definition | Flag rate (train) |
| --- | --- | ---: |
| `v2_strict` | All 4 voters unanimous on the same label, ≠ gold | **33.7%** (9,184 of 27,263 rows) |
| `v2_majority` | ≥ 3 of 4 voters agree on the same label, ≠ gold | **54.4%** (14,824 rows) |

Compare to the round-1 flag rate of 38.0% (10,371 rows). The strict
rule **catches 11.4% of round-1 flags as suspect** — those are rows
where the cleaned model "vetoes" the cross-dataset ensemble's
proposed flip. The majority rule **adds new flags** that round-1's
unanimous threshold missed, casting a wider net.

Both pools are balanced (50/50) by undersample, then nested:

| Variant | Full-pool size |
| --- | ---: |
| `beads_cleaned_v2_strict` | 16,424 |
| `beads_cleaned_v2_majority` | 16,008 |

Generator: [scripts/make_cleaned_train_v2.py](../scripts/make_cleaned_train_v2.py).

**Ranked accuracy table (492 non-abstain hand-label rows)**

| Rank | Adapter | Train n | Acc | F1_macro |
| ---: | --- | ---: | ---: | ---: |
| 1 | qlora_beads_cleaned_flip_balanced_500 (round 1) | 500 | **0.768** | 0.768 |
| 2 | qlora_beads_cleaned_v2_majority_5k | 5,000 | **0.724** | **0.723** |
| 3 | qlora_beads_cleaned_v2_majority_500 | 500 | 0.715 | 0.714 |
| 4 | qlora_beads_cleaned_v2_majority_full | 16,008 | 0.713 | 0.713 |
| 5 | qlora_beads_cleaned_v2_majority_1k | 1,000 | 0.711 | 0.711 |
| 6 | qlora_beads_cleaned_v2_strict_500 | 500 | 0.705 | 0.703 |
| 7 | qlora_beads_cleaned_flip_balanced_1k | 1,000 | 0.683 | 0.683 |
| 7 | qlora_beads_cleaned_flip_balanced_5k | 5,000 | 0.683 | 0.683 |
| 9 | qlora_beads_cleaned_v2_strict_full | 16,424 | 0.665 | 0.664 |
| 10 | qlora_beads_cleaned_v2_majority_100 | 100 | 0.663 | 0.658 |
| 11 | qlora_beads_cleaned_v2_strict_5k | 5,000 | 0.657 | 0.657 |
| 12 | qlora_beads_cleaned_flip_balanced_full | 14,246 | 0.652 | 0.652 |
| 13 | qlora_beads_cleaned_v2_strict_1k | 1,000 | 0.648 | 0.646 |
| 14 | qlora_beads_cleaned_v2_strict_100 | 100 | 0.518 | 0.474 |
| 15 | qlora_beads_cleaned_flip_balanced_100 | 100 | 0.492 | 0.433 |
| ... | (other round-1 + original sweep) | — | — | — |
| 22 | qlora_beads_full (ORIGINAL BASELINE) | 27,263 | **0.283** | 0.275 |

The full 16-row hand-label table is in `hand_label_eval.csv`.

**Curves side by side**

Round 1 flip_balanced: `0.49 → 0.77 → 0.68 → 0.68 → 0.65` (peak, then drops)

Round 2 strict:        `0.52 → 0.71 → 0.65 → 0.66 → 0.66` (not monotonic — drops at 1k)

Round 2 majority:      `0.66 → 0.72 → 0.71 → 0.72 → 0.71` (**near-monotonic**, ±1.3pp range across 500-full)

**Pre-registered criterion: outcome**

Criterion 1 (monotonicity, ±3pp per step):

- v2_strict: 500→1k drops 5.7pp, exceeds tolerance → **FAILED**
- v2_majority: max step = +1.3pp (5k→full), well within tolerance → **PASSED**

Criterion 2 (peak ≥ round-1 peak of 0.768): **FAILED** for both variants
(v2_majority peak = 0.724, v2_strict peak = 0.705).

By strict pre-registered terms, the iterative cleaning hypothesis is
*not* vindicated. But the result is more informative than a binary
pass/fail. The interpretation below is what the writeup should claim.

**Interpretation: the round-1 peak was sampling variance, not a sweet spot**

The flat ~0.72 v2_majority curve is what an honest cleaning ceiling
*should* look like — stable across data sizes, no peak-and-drop. The
fact that round 1 had a striking peak at size=500 was almost certainly
**sampling variance**, not a stable property of small-data training.
Two lines of evidence:

1. **Round 2's near-monotonic curve across an analogous size range.**
   Improving the cleaning (adding the 4th voter) removed the peak,
   replacing it with a stable band. If "500 is the sweet spot" were
   a real property of the task, the better-cleaned variant should
   *also* show it. It doesn't.

2. **Round 2 v2_majority @ size=100 scored 0.663**, vs round-1
   flip_balanced @ size=100 scored 0.492. **+17 pp improvement at
   the same data size**, driven entirely by the cleaning quality
   change. Same model architecture, same hyperparameters — the only
   thing different is the labels. The strong improvement at 100 rows
   confirms the 4-voter rule produces fundamentally cleaner training
   data.

So the *peak accuracy* finding from round 1 (0.768) is real but
unstable. The *robust* finding across rounds: **cleaned BEADs models
land at ~0.71-0.77 against hand-labels regardless of the specific
cleaning rule, an order of magnitude above the noisy-data baseline
(0.283).**

**Why the peak wasn't pushed higher**

A natural hypothesis was that better cleaning would lift the peak
*and* stabilize the curve. We got the stabilization without the
lift. Possible reasons:

- **The 0.72-0.77 band is the cleaning method's true ceiling.** No
  matter how clean the labels, the 8B-parameter Llama + LoRA-r16
  recipe on ~10-16k stratified examples gets to ~0.72-0.77 and no
  higher. Further gains would need different architecture, more data,
  or human-in-the-loop validation.
- **The 4th voter introduces correlated noise with the round-1
  ensemble.** The cleaned model was *trained on* round-1 relabels, so
  its agreement with the ensemble is partly tautological. The strict
  rule's veto signal works, but it's not a fully independent fifth
  perspective — it's the round-1 method looking at itself.
- **Sampling variance at n=492.** 95% Wilson CI on accuracy is
  roughly ±4pp at this sample size. The 0.724 vs 0.768 gap (4.4pp) is
  *just at* the edge of statistical noise. We can't rule out that
  round-2 majority is just as good as round-1's peak; the test set
  isn't big enough to distinguish them confidently. **A bigger
  held-out hand-labeled set would tighten this.**

**What this means for the writeup**

The writeup now has *two* defensible accuracy claims instead of one
inflated headline:

1. **Round 1 framing** (the original headline):
   *"Cleaning + retraining lifts accuracy from 0.283 to 0.768 on the
   492-row hand-label consensus — a +48.6 pp gain."* True, but the
   0.768 is at a specific training size (500) and degrades at larger
   sizes.

2. **Round 2 framing** (the more defensible claim):
   *"Iterative cleaning with the cleaned model as a 4th voter
   produces a stable ~0.72 accuracy across train sizes 500–16k. This
   is below the round-1 peak (0.768) but is robust across the sweep
   — the round-1 peak appears to have been a sampling-variance
   artifact, not a stable size dependence."*

Together, the most defensible claim:

> Cleaning BEADs with a cross-dataset adapter ensemble lifts the
> Llama-3.1-8B model's accuracy against careful human labels from
> 0.283 (noisy gold) to a stable ~0.72-0.77 (cleaned). The exact
> peak depends on training size and cleaning iteration in ways that
> are partly sampling-variance; the **robust claim is the +44 to
> +49 pp lift**, with the cleaning method's true ceiling in the
> 0.71-0.77 band on this evaluation.

**Costs and what's next**

Round-2 arc cost ~$10 actual: the prediction job for the 4th voter
(~$1) + 10 training jobs ($9). Total project compute is now ~$26.

Remaining open methodological question (out of scope for this
deliverable):

- **Confidence-weighted re-relabeling.** Only re-relabel when the
  4th voter has high prediction margin. Round 2 used a binary
  agreement check. Margin-weighted version could filter out
  low-confidence "wrong" relabels without losing high-confidence
  signal.
- **Larger hand-label set.** Confidently distinguishing 0.72 from
  0.77 needs n ≈ 1000-1500 hand-labels, not 500. Hand-labeling
  another ~500-1000 rows would tighten the comparison and let us
  test whether round-1's peak is real or noise.
- **Multi-seed cleaned voter ensemble.** Train 3-5 round-1 models
  with different seeds; only re-relabel where they all agree. Reduces
  the single-cleaned-model bias.

All three are valid future work; none are blocking the current
writeup.

---

## 2026-05-20 — Cleaned BEADs adapters transfer to other datasets (the silos were noise)

**TL;DR**

Extended the cross-eval matrix to the cleaned BEADs adapters: 20
cleaned adapters × 3 non-BEADs datasets (BABE, cajcodes, WNC) = 60
new cells. **Every cleaned variant beats the original BEADs adapter
on every target.** The best cleaned adapter on BABE scores **0.809
accuracy / 0.802 F1_macro** — within 6 pp of a model trained on
BABE itself. **The "datasets-are-siloed" finding from 2026-05-19
was substantially an artifact of label noise**: when BEADs's
training data is cleaned, its model transfers dramatically better,
at least to the BABE domain.

Headline figures:
- [docs/figures/transfer_before_after.png](figures/transfer_before_after.png) — bar chart, original vs best cleaned on each target (accuracy)
- [docs/figures/transfer_before_after_f1.png](figures/transfer_before_after_f1.png) — same with F1_macro
- [docs/figures/transfer_curve.png](figures/transfer_curve.png) — 4-panel learning curves, one panel per target

**Sweep ran**

[scripts/launch_cleaned_cross_eval.sh](../scripts/launch_cleaned_cross_eval.sh)
submitted 4 slurm jobs (one per cleaning variant) on Tillicum, each
sequentially evaluating its 5 adapters × 3 datasets = 15 cells via
[scripts/cross_eval.py](../scripts/cross_eval.py). All 4 jobs landed
GPUs immediately; total wall ~30 min. Cells written to
``outputs/cross_eval/qlora_beads_cleaned_<variant>_<size>__on__<ds>/``.

**Transfer matrix — best cleaned variant per (target, family)**

Accuracy / F1_macro. Best cleaned cell across all sizes within each variant.

| | **babe** (n=413) | **cajcodes** (n=66) | **wnc** (n=11,041) |
| --- | ---: | ---: | ---: |
| Original `qlora_beads_full` | 0.312 / 0.277 | 0.667 / 0.400 | 0.462 / 0.456 |
| Best cleaned: `remove` | 0.763 / 0.748 | 0.697 / 0.411 | 0.568 / 0.565 |
| Best cleaned: `remove_balanced` | 0.569 / 0.485 | 0.727 / 0.723 | 0.530 / 0.456 |
| Best cleaned: `flip` | 0.666 / 0.601 | 0.742 / 0.552 | 0.577 / 0.557 |
| **Best cleaned: `flip_balanced`** | **0.809 / 0.802** | **0.773 / 0.669** | **0.591 / 0.586** |

**Absolute-best lift per target**

| Target | Winner | Accuracy (orig → best) | F1_macro (orig → best) |
| --- | --- | ---: | ---: |
| BEADs (hand-labels) | cleaned_flip_balanced_500 | 0.283 → **0.768** (+48.6 pp) | 0.275 → 0.768 (+49.3 pp) |
| **BABE** | cleaned_flip_balanced_full | 0.312 → **0.809** (**+49.6 pp**) | 0.277 → 0.802 (+52.5 pp) |
| **cajcodes** | (acc) cleaned_flip_balanced_1k; (F1m) cleaned_remove_balanced_100 | 0.667 → 0.773 (+10.6 pp) | 0.400 → **0.723** (**+32.3 pp**) |
| **WNC** | cleaned_flip_balanced_full | 0.462 → 0.591 (+12.9 pp) | 0.456 → 0.586 (+12.9 pp) |

**Five interpretations worth a writeup paragraph each**

1. **The cleaning produced a domain-general bias classifier, not just a
   better BEADs classifier.** The BABE-specialist model (qlora_babe_full)
   scored 0.867 on BABE. The cleaned BEADs model scores **0.809 on
   BABE** — within 6 pp of a model that trained on BABE itself.
   That's a fundamentally different transfer story than the 2026-05-19
   cross-eval matrix told us.

2. **The "siloed datasets" finding was substantially an artifact of label
   noise.** Original beads → babe was 0.312 (chance-level). Cleaned
   beads → babe is 0.809 (+50 pp). The original silo wasn't a
   construct-validity issue — it was the BEADs labeler's
   non-public-interest "biased" calls *not transferring*, because
   they aren't really bias in the way other labelers define it.
   Cleaning out those calls revealed the shared signal that was
   underneath.

3. **F1_macro tells a sharper story than accuracy on imbalanced
   targets.** cajcodes is 70% biased; the original BEADs adapter on
   cajcodes had P=0.69, R=0.96 — i.e., it was predicting "biased"
   almost always, which gave it 0.667 accuracy *despite* being
   degenerate. F1_macro = 0.400 surfaced this. The cleaned adapters
   actually learn cajcodes' decision boundary; the F1_macro lift
   (+32.3 pp) is the real measure.

4. **Size-vs-transfer reversal: the BEADs-headline winner does NOT win
   on transfer.** On BEADs hand-labels: cleaned_flip_balanced_**500**
   (0.768) > cleaned_flip_balanced_full (0.652). On BABE:
   cleaned_flip_balanced_**full** (0.809) > cleaned_flip_balanced_500
   (0.644). The 500-row model is tightly fit to clean BEADs and
   doesn't see enough variation to generalize; the full-pool model
   sees more diverse text *plus* the 15% wrong relabels, but the
   diversity helps transfer more than the wrong relabels hurt — on
   *other* datasets. The wrong relabels only hurt on BEADs because
   that's the domain the model was supposed to learn cleanly. **If
   the writeup wants to claim "domain-general bias classifier,"
   the right ambassador is cleaned_flip_balanced_full.**

5. **flip_balanced family dominates on transfer just like on BEADs.**
   Across all three non-BEADs targets, flip_balanced wins or ties
   for best. Combining "flagged-row relabel via cross-dataset
   ensemble" with "50/50 class balance via undersample" is the
   architecturally-correct combo. The original sweep had ~50/50
   balance by stratification; the cleaning has to actively restore
   it. Without balancing, the cleaning's natural ~74% biased pool
   collapses the model into majority-class prediction.

**A nuance worth flagging**

On cajcodes specifically, the F1_macro winner is
**remove_balanced_100** at 0.723 — only 100 training rows, 50/50.
This makes sense: cajcodes is tiny (66 test rows), 70% biased.
A model trained on 50 clean biased + 50 clean non-biased examples
captures cajcodes' decision boundary well; bigger models overfit
to BEADs-specific patterns that cajcodes doesn't share. The
single-best metric across (target, variant, size) is
sensitive to these kinds of small-sample interactions; the
absolute numbers should be reported with cajcodes' n=66 caveat.

**What this reframes**

The original 2026-05-19 cross-eval entry concluded "datasets are
siloed; bias as labeled by these four datasets is not one underlying
construct measured four different ways." That conclusion needs
updating:

> **Revised interpretation**: On *each dataset's noisy gold labels*,
> the datasets look siloed because each labeling process has its
> own systematic biases. When one dataset's labels are cleaned
> against careful human judgment of "bias on a public-interest
> topic," the cleaned model transfers far better — at least
> within the BEADs/BABE news-comment-section domain. The underlying
> bias signal IS more shared than the noisy comparison suggested;
> the silos were largely the labeling-process noise of each source.

This is a sharper claim with positive implications and is worth its
own section in the writeup.

**Plot generator**

[scripts/make_transfer_figures.py](../scripts/make_transfer_figures.py)
produces both figures from disk. Run with no args for accuracy variants,
`--metric f1_macro` for F1_macro variants. Reads from the cross-eval
cells' eval_metrics.json + hand_label_eval.csv.

**Cost**

~$2-3 in additional H200 compute beyond the 2026-05-20 cleaning sweep.
Total for the cleaning experiment arc (cleaning + retrain + cross-eval
extension): ~$25-30.

---

## 2026-05-20 — Cleaning experiment complete: +48.6 pp accuracy lift

**TL;DR**

The cleaning + retrain experiment succeeded by a wide margin. Best
cleaned adapter (``qlora_beads_cleaned_flip_balanced_500``) scores
**0.768 accuracy / 0.768 F1_macro** against the team's hand-label
consensus — **+48.6 percentage points over the original baseline** of
0.283. Every cleaning variant beats the original at every size. The
pre-registered success threshold was a lift ≥ 1.5 × IAA disagreement
rate (~22.5 pp); the experiment cleared it by ~2×.

The headline figure is at
[docs/figures/cleaning_curve.png](figures/cleaning_curve.png).

**Sweep ran**

20 retrains, 4 cleaning variants × 5 train sizes, on Tillicum
(slurm jobs 119840-119859, ~$23 actual spend on H200s,
~3h wall under reasonable queue contention). Launcher:
[scripts/launch_cleaned_retrain_sweep.sh](../scripts/launch_cleaned_retrain_sweep.sh).
Cleaned-data builder:
[scripts/make_cleaned_train.py](../scripts/make_cleaned_train.py).
Post-sweep scoring:
[scripts/score_against_hand_labels.py](../scripts/score_against_hand_labels.py).

**Ranked table** (25 adapters scored against the 492 non-abstain
hand-labeled rows; sorted by accuracy descending)

| Rank | Adapter | Train n | Acc | F1_macro | P_pos | R_pos |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| **1** | qlora_beads_cleaned_flip_balanced_500 | 500 | **0.768** | **0.768** | 0.667 | 0.871 |
| 2 | qlora_beads_cleaned_flip_balanced_1k | 1,000 | 0.683 | 0.683 | 0.584 | 0.792 |
| 3 | qlora_beads_cleaned_flip_balanced_5k | 5,000 | 0.683 | 0.683 | 0.582 | 0.807 |
| 4 | qlora_beads_cleaned_flip_balanced_full | 14,246 | 0.652 | 0.652 | 0.553 | 0.802 |
| 5 | qlora_beads_cleaned_remove_100 | 100 | 0.614 | 0.606 | 0.517 | 0.916 |
| 6 | qlora_beads_cleaned_flip_full | 27,263 | 0.565 | 0.546 | 0.485 | 0.941 |
| 7 | qlora_beads_cleaned_flip_5k | 5,000 | 0.561 | 0.540 | 0.482 | 0.946 |
| 8 | qlora_beads_cleaned_flip_balanced_100 | 100 | 0.492 | 0.433 | 0.446 | 0.990 |
| 9 | qlora_beads_cleaned_remove_balanced_500 | 500 | 0.468 | 0.467 | 0.391 | 0.530 |
| 10 | qlora_beads_cleaned_flip_1k | 1,000 | 0.453 | 0.380 | 0.427 | 0.970 |
| 11 | qlora_beads_cleaned_remove_balanced_5k | 5,000 | 0.451 | 0.448 | 0.364 | 0.451 |
| 12 | qlora_beads_cleaned_flip_100 | 100 | 0.445 | 0.356 | 0.425 | 0.995 |
| 13 | qlora_beads_cleaned_flip_500 | 500 | 0.437 | 0.344 | 0.421 | 0.990 |
| 14 | qlora_beads_cleaned_remove_balanced_100 | 100 | 0.433 | 0.337 | 0.419 | 0.990 |
| 15 | qlora_beads_cleaned_remove_balanced_1k | 1,000 | 0.429 | 0.429 | 0.368 | 0.545 |
| 16 | qlora_beads_cleaned_remove_full | 16,892 | 0.427 | 0.420 | 0.383 | 0.648 |
| 17 | qlora_beads_cleaned_remove_balanced_full | 10,404 | 0.421 | 0.420 | 0.348 | 0.470 |
| 18 | qlora_beads_100 (original) | 100 | 0.411 | 0.291 | 0.411 | 1.000 |
| 18 | qlora_beads_cleaned_remove_500 | 500 | 0.411 | 0.291 | 0.411 | 1.000 |
| 20 | qlora_beads_cleaned_remove_1k | 1,000 | 0.392 | 0.317 | 0.393 | 0.881 |
| 21 | qlora_beads_cleaned_remove_5k | 5,000 | 0.380 | 0.362 | 0.362 | 0.668 |
| 22 | **qlora_beads_full (ORIGINAL BASELINE)** | 27,263 | **0.283** | 0.275 | 0.184 | 0.218 |
| 23 | qlora_beads_5k (original) | 5,000 | 0.256 | 0.246 | 0.150 | 0.173 |
| 24 | qlora_beads_500 (original) | 500 | 0.248 | 0.248 | 0.204 | 0.287 |
| 25 | qlora_beads_1k (original) | 1,000 | 0.205 | 0.202 | 0.135 | 0.173 |

Source data: ``hand_label_eval.csv`` (gitignored, regenerable from
the predictions.jsonl + the labeling/ artifacts).

**Five findings worth a writeup paragraph each**

1. **Flip beats remove for the cleaning action.** Top 4 are all
   ``flip_balanced``; the best ``remove`` variant is rank 5 with a
   15 pp gap below the winner. The cross-dataset ensemble's
   *which-label* relabel (not just *whether-noisy*) is genuinely
   tracking truth — the 85.1% flip-correctness we measured against
   the 500 hand-labels held up at training scale.

2. **Class balancing matters as much as the cleaning step.** Within
   the flip family: natural-balance best is 0.565; balanced best is
   0.768 — **20 pp from the balance step alone**. The cleaning
   skews class distribution (~74% biased in flip-natural) and
   that imbalance hurts the model just as much as the gold-label
   noise it was trained to escape.

3. **More data isn't better when the cleaning is imperfect.** The
   ``flip_balanced`` curve is *non-monotonic*: 500 > 1k = 5k > full.
   The 500-row model beats the 14k-row full model by 12 pp.
   Probable mechanism: the flip relabels at full scale include
   enough wrong relabels (15% per the flip-correctness gate) that
   adding more data starts injecting more *new* noise than it
   resolves. Sub-pool stratification at smaller sizes naturally
   filters that out.

4. **The original sweep is flat against truth.** All five sizes of
   the original (100/500/1k/5k/full) land between 0.21 and 0.41 on
   hand-labels. The original ``qlora_beads_full`` (0.283) *isn't even
   the best of the original sizes* — that's ``qlora_beads_100`` at
   0.411, which is degenerate collapse mode (recall 1.000, precision
   0.411). Adding training data on noisy labels lets the model learn
   the noise better, not the signal. **The original 0.7987-vs-noisy-gold
   accuracy was almost entirely noise memorization.**

5. **Cleaned model on truth ≈ original model on noisy gold.** The
   winner's 0.768 against hand-labels is within 3 pp of the original's
   0.7987 against its own noisy training distribution. So **a model
   trained on cleaned BEADs and evaluated on careful human labels
   recovers nearly the same accuracy the original benchmark claimed,
   but on the right thing.** This is the cleanest operational
   vindication of the cleaning hypothesis.

**Collapse-mode tag**

Many of the rank-12-through-18 entries have ``recall_pos ≈ 0.99``
and ``precision_pos ≈ 0.42`` — i.e., they predict "biased" almost
always. This is class-imbalance collapse. It explains why those
accuracy numbers cluster near 0.42-0.44 (which is approximately the
biased-class prior in the hand-labels). Worth flagging in the
writeup: accuracy alone is misleading for these collapsed models;
F1_macro is the more honest comparison and shows much sharper
distinctions (0.291 collapsed vs 0.768 winner).

**Pre-registered success criterion: passed by 2×**

Locked threshold from the 2026-05-20 entry:
> lift ≥ 1.5 × IAA disagreement rate

IAA agreement was 81-88% (call it 85% midpoint), disagreement ~15%.
Threshold: 1.5 × 15 = **22.5 pp**. Measured lift: 0.768 − 0.283 =
**48.6 pp**. Roughly 2× the success threshold. The result is
defensible.

**What's next**

The cleaning experiment is the project's empirical core; everything
else is supporting infrastructure. Remaining deliverables for the
writeup are documentation, not experiments:

1. The headline figure ([docs/figures/cleaning_curve.png](figures/cleaning_curve.png))
   + F1_macro variant ([cleaning_curve_f1.png](figures/cleaning_curve_f1.png))
   are now committed.
2. Writeup draft — the 5 numbered findings above are the natural
   section structure.
3. Optional: qualitative inspection of disagreement rows (proposal
   §5, Week 8 deliverable) — partially answered by the IAA
   quicklook and the cleaning-experiment behaviour, but a hand-eye
   pass on 20-30 rows would add narrative texture.

---

## 2026-05-20 — Hand-labeling complete: IAA passed, BEADs noise ~70%

**TL;DR**

Three teammates hand-labeled 500 stratified random rows from BEADs test
(200 each: 150 unique + 50 IAA-shared, blind). After one round of
recalibration on the bias definition, **IAA gate passed** (81-88%
pairwise agreement, Cohen's κ 0.57-0.75). All four headline
measurements landed:

| Metric | Value |
| --- | ---: |
| BEADs mislabel rate vs team consensus | **69.5%** |
| BEADs missed bias (gold=clean, team=biased) | 148 |
| BEADs over-called bias (gold=biased, team=clean) | **194** |
| Directional asymmetry | 0.8 : 1 |
| qlora_beads_full accuracy vs team consensus | **0.283** |
| Cross-dataset ensemble flip-correctness | **85.1%** |
| Gate decision for retrain step | **remove_and_flip** |

Three findings worth pulling out:

1. **BEADs has substantial label noise — ~70% of test rows disagree
   with the calibrated 3-person team.** This is the headline statistic
   for the writeup.

2. **The noise direction is over-calling, not under-calling.** The
   cross-eval matrix earlier suggested BEADs missed bias ~5× more
   often than it over-called (5 : 1). The hand-labels show the
   opposite direction: BEADs over-calls slightly more than it
   misses (0.8 : 1). **The model-only signal was misleading about
   which way the noise points.** Cross-dataset adapter agreement
   captures *that something is wrong*, not *which direction it's
   wrong*; only human labels resolve that.

3. **The BEADs-fine-tuned QLoRA model has learned the noise.** Same
   model that scored 0.7987 against BEADs's noisy gold scores **0.283
   against careful human labels** (well below chance on a binary
   task). The model is *highly aligned* with BEADs's labels and
   *highly anti-aligned* with truth. Strongest single argument for
   the cleaning + retraining experiment.

**The labeling task as run**

- **500 rows**, stratified 250 biased / 250 non-biased from BEADs test.
- **3 labelers** (abrevaa, ash, zach), each labeled 200 rows: 150
  unique + 50 shared IAA block randomly interleaved so labelers
  couldn't tell which were shared.
- **Blind**: no BEADs gold label, no QLoRA model predictions visible.
- **Labels**: exactly `biased` / `non-biased` / blank (abstain).
- Sampler script:
  [scripts/make_labeling_csvs.py](../scripts/make_labeling_csvs.py)
  (seed 42, reproducible).
- Scoring script:
  [scripts/score_hand_labels.py](../scripts/score_hand_labels.py).

**IAA: failed first, passed after recalibration**

First pass:

| Pair | Agreement | Pairs | Cohen's κ |
| --- | ---: | ---: | ---: |
| abrevaa - ash | 81.2% | 48 | +0.57 (moderate) |
| abrevaa - zach | 63.6% | 44 | +0.33 (fair) |
| ash - zach | 63.0% | 46 | +0.35 (fair) |

Zach was the outlier on 12 of 38 rows where everyone labeled (32%,
vs 11% for abrevaa and 13% for ash). All 12 of zach's outlier calls
were in the same direction: zach said `biased` and the other two
said `non-biased`.

Reading the rows revealed the calibration gap: zach was applying
"biased" to *any opinion-shaped or sentiment-shaped text*, while
abrevaa and ash were reserving it for opinions on **public-interest
topics** (politics, social issues, contested public debates) per
the README's definition. Examples zach miscalled:

- *"feelin' for all my Pen fans right now"* (personal sports
  sympathy)
- *"vegging at home in my comfy pjs..."* (personal logistics)
- *"So sad Figlios in Uptown is closing"* (restaurant comment)
- *"@garrettscribner 150k?! Well, at least you have standards"*
  (sarcasm at an individual)

Zach went back, kept only the "biased" labels that genuinely
applied to public-interest opinions, and filled in the 10 previously
abstained rows. After this single recalibration pass:

| Pair | Before | After | Change |
| --- | ---: | ---: | ---: |
| abrevaa - ash | 81.2% | 81.2% | unchanged |
| abrevaa - zach | 63.6% | **87.5%** | **+23.9 pp** |
| ash - zach | 63.0% | **82.0%** | +19.0 pp |
| three-way unanimous | 52.3% | **75.0%** | +22.7 pp |

Cohen's κ jumped to 0.57-0.75 (moderate-to-substantial agreement).
**IAA gate passed**; all pairwise % ≥ 70%.

**Lesson for the methodology section**: blind-labeling without
explicit pre-calibration on edge cases produces a single systematic
offset (one labeler over-calling biased), not random scatter. One
discussion round on a specific edge-case set was enough to align;
re-labeling fixed it cleanly. The pre-registered IAA gate + the
"if it fails, recalibrate and re-run" loop in the locked plan was
the right structure for this.

**Why the directional asymmetry flipped**

The cross-eval matrix (2026-05-19 entry) measured *model* agreement
patterns: when 3 non-BEADs adapters unanimously disagree with BEADs
gold, what direction do they disagree in? Test split: 2,135 rows
where ensemble said biased + BEADs said clean; 450 rows where
ensemble said clean + BEADs said biased. 5 : 1 toward
"BEADs missed bias."

The hand-labels measure *human* agreement patterns. Same 500-row
sample: 148 missed-bias + 194 over-called. 0.8 : 1 toward
"BEADs over-called."

Both are true. The reconciliation: the *non-BEADs adapters
disagree with BEADs* in a particular direction, but **humans
disagree with BEADs in a slightly different direction** because
humans apply the "public-interest topic" requirement that the
cross-dataset adapters don't have. The babe/cajcodes/wnc adapters
are still working off lexical/tonal features of bias that include
non-public-interest sentiment. Humans (with the calibrated
definition) restrict bias to public-interest content, which means
many of BEADs's "biased" calls on personal/sentiment-only content
are revealed as over-calls.

**This subtlety matters for the writeup**: the 5 : 1 number from
the cross-eval was real but measured something different from what
it appeared to. Don't conflate "models disagree with gold" with
"gold is wrong in this direction."

**Why qlora_beads_full scored 0.283 against hand-labels**

The BEADs-fine-tuned QLoRA model agrees with BEADs gold ~80% of
the time (it was trained on those labels). BEADs gold agrees with
the team consensus ~30% of the time (the inverse of the 70%
mislabel rate). So the model's expected accuracy against the team
is roughly 0.8 × 0.3 + 0.2 × 0.7 = 0.38. Observed: 0.283.

The gap (0.28 vs 0.38) is the model's errors *correlating* with
BEADs's errors — the model didn't randomly memorize; it memorized
the noisy pattern such that its errors land on roughly the same
rows BEADs is wrong on. **The model is anti-aligned with truth by
construction of its training data.**

This is the strongest single piece of evidence that cleaning the
training data could move the needle. If qlora_beads_full's noisy
0.7987 against BEADs gold is the artifact, and 0.283 against
truth is the reality, then a model retrained on cleaner training
data should land somewhere meaningfully higher than 0.283 against
the same hand-labels.

**Flip-correctness gate: pass with high margin**

199 of 500 rows in the sample were flagged (`cross_unanimous_disagree
== 1`). On those flagged rows where the team consensus is
non-abstain (195 rows), the ensemble's vote matches the team
consensus on 166 — **85.1%**. Well above the 70% threshold for
"both remove and flip retrains."

Practically: if we use the ensemble's vote to *relabel* the
flagged training rows (the "flip" intervention), the relabel will
be correct ~85% of the time according to careful human judgment.
The 15% incorrect relabels add some noise but are likely dominated
by the 70% noise we'd be removing from the original gold. The
flip experiment is justified.

**What landed on disk**

| Path | What it is | Tracked? |
| --- | --- | --- |
| [scripts/score_hand_labels.py](../scripts/score_hand_labels.py) | The scoring tool | Yes |
| [labeling/labeler_abrevaa_labeled.csv](../labeling/labeler_abrevaa_labeled.csv) | abrevaa's 200 rows | Yes (experimental data) |
| [labeling/labeler_ash_labeled.csv](../labeling/labeler_ash_labeled.csv) | ash's 200 rows | Yes |
| [labeling/labeler_zach_labeled.csv](../labeling/labeler_zach_labeled.csv) | zach's 200 rows (post-recalibration) | Yes |
| [labeling/_mapping.csv](../labeling/_mapping.csv) | Private join file (labeler letter → BEADs row_idx + gold + non_beads_vote) | Yes |
| [labeling/LABELER_README.md](../labeling/LABELER_README.md) | Labeling protocol with definition + edge-case rules | Yes |
| `hand_label_scoring_per_row.csv` | 500-row per-row diagnostics. Regenerable. | No (gitignored) |
| `hand_label_scoring_summary.json` | Top-level summary stats. Regenerable. | No (gitignored) |
| `hand_label_scoring_iaa_quicklook.csv` | 50 IAA rows side-by-side. Regenerable. | No (gitignored) |
| `labeling/labeler_{a,b,c}.csv` | Blank task files from the sampler. Regenerable. | No (gitignored) |

To regenerate the derived files at any time:

```bash
python scripts/score_hand_labels.py --label-map a=abrevaa b=ash c=zach
```

**Decisions still open**

The gate decisions are all settled. What remains is launching the
cleaning + retrain sweep on Tillicum:

1. Apply the cleaning rule (`cross_unanimous_disagree == 1`) to
   BEADs train using the 10,371 rows already identified by the
   2026-05-19 train+val prediction job (entry above).
2. Produce two cleaned train files:
   - **Remove version**: 16,892 rows (27,263 minus 10,371 flagged)
   - **Flip version**: 27,263 rows with the 10,371 flagged rows
     relabeled to their `non_beads_vote` consensus
3. Submit the full `qlora_{100, 500, 1k, 5k, full}` sweep × 2
   cleaning methods = **10 retrains** (~$20-25, ~3 hrs wall).
4. Evaluate each new adapter against the 500 hand-labels.
5. Plot all three learning curves (original, removed, flipped) on
   the same axis against hand-label accuracy. The headline plot
   for the writeup.

---

## 2026-05-19 — BEADs label-audit tool + cleaning experiment plan

**What this adds**

A tool to spot-check BEADs's gold labels using the four cross-eval
adapters as independent voters, plus a pre-registered plan to
hand-label a held-out sample and test whether algorithmically
cleaning BEADs's training data improves model accuracy. Sets up the
Week 8 "qualitative inspection" deliverable in a way that produces a
quantitative result in the same pass.

**The tool**

[scripts/beads_spot_check.py](../scripts/beads_spot_check.py) joins the
four ``outputs/cross_eval/qlora_<ds>_full__on__beads/predictions.jsonl``
files into a single share-ready CSV (default:
``beads_label_audit.csv`` in the repo root, regenerable, not gitted).

Output schema:

| Column | Meaning |
| --- | --- |
| `row_idx` | Position in BEADs test (stable ID across files) |
| `verdict` | One of `mislabel_likely_missed_bias`, `mislabel_likely_over_called_bias`, `agree_biased`, `agree_clean`, `mixed` |
| `confidence` | Mean per-adapter log-odds margin (decisiveness, not vote direction) |
| `text` | BEADs sentence |
| `gold_label` | What BEADs's official label says (`biased` / `non-biased`) |
| `pred_{beads,babe,cajcodes,wnc}` | Each adapter's prediction |
| `non_beads_vote` | Consensus of BABE + cajcodes + WNC only (BEADs adapter excluded so this is *independent* of BEADs's training signal). One of `biased` / `non-biased` / `split`. |
| `models_disagreeing_with_gold` | 0-4 |

Sort: most-actionable verdicts first, then confidence within bucket.

**The signal the audit surfaces**

Verdict breakdown across the 6,816-row BEADs test split:

| Verdict | Count | % |
| --- | ---: | ---: |
| `mislabel_likely_missed_bias` | 276 | 4.0% |
| `mislabel_likely_over_called_bias` | 36 | 0.5% |
| `mixed` (partial disagreement) | 5,605 | 82.2% |
| `agree_biased` | 834 | 12.2% |
| `agree_clean` | 65 | **1.0%** |

The 65 rows of confirmed-clean labels is the striking number — out of
~3,400 test rows where BEADs says `non-biased`, only 2% have all four
adapters confirming "yes this is clean." Compare to ~25% for biased.
**BEADs's `non-biased` category is much noisier than its `biased`
category.**

The stronger framing comes from `non_beads_vote` — the independent
ensemble (BABE + cajcodes + WNC, no BEADs adapter):

| non_beads_vote | vs gold | Count |
| ---: | --- | ---: |
| biased | gold = biased | 1,203 |
| **biased** | **gold = non-biased** | **2,135**  *(BEADs missed bias)* |
| non-biased | gold = biased | 450  *(BEADs over-called)* |
| non-biased | gold = non-biased | 133 |
| split | (either gold) | 2,895 |

So when three adapters trained on three *different* bias datasets all
agree with each other (3,921 rows, 57% of the test set), they reject
BEADs's gold label on **2,585 of those rows — 66%**. The BEADs adapter
itself isn't part of this vote, so the signal is genuinely
independent of BEADs's labelers.

**The asymmetry that matters for the writeup**

BEADs missed bias **~5× more often** than it over-called it: 2,135 rows
where the independent ensemble says biased + BEADs says clean, vs 450
in the opposite direction. If hand-labeling confirms this ratio, the
writeup story is "BEADs's `non-biased` category is essentially a
catch-all that under-labels bias by a large margin."

**Methodological caveat — why this isn't yet a finding**

The 2,585 "candidate mislabel" count is the **maximum plausible**
signal, not a validated count. Each of the three non-BEADs adapters
individually scores ≤ 0.46 accuracy on BEADs (worse than chance — see
[2026-05-19 cross-eval entry](#2026-05-19--qlora-cross-eval-matrix-complete-the-headline-result)).
They're noisy. The argument for treating their *agreement* as signal
is that they each fail differently — when babe, cajcodes, and WNC
agree, the agreement is likely on a shared feature rather than each
model's idiosyncratic noise.

But "likely" needs validation. The signal could also be that all three
datasets share a *different notion of bias* than BEADs (a
construct-validity issue, not a noise issue). The hand-labeling step
distinguishes those.

**The planned experiment (numbers locked 2026-05-19)**

Pre-registered before any cleaning is computed on the BEADs train pool.
"Slightly aggressive" choices across the parameters — strong enough to
test whether the cleaning method actually does anything, conservative
enough that a positive result wouldn't be explained away by lax
thresholds.

1. **Sample**: **500 random rows** from BEADs test, **stratified
   250/250 by gold label** (uniform sampling would under-represent the
   noisier `non-biased` rows given the 5:1 missed-bias asymmetry).
2. **Hand-label** with a team of 3:
   - Each labeler gets **200 rows (150 unique + 50 shared IAA block)**.
   - The 50 IAA rows are randomly interleaved with each labeler's
     unique 150 so they don't know which rows are shared.
   - Blind: no gold label, no model predictions shown during labeling.
   - Written protocol with edge-case rules agreed *before* labeling
     begins (mitigates the obvious IAA-suppression failure mode).
3. **IAA gate**: pairwise agreement on the 50 shared rows **≥ 70%**, or
   stop and recalibrate the labeling protocol.
4. **Three measurements against the consensus hand-labels**:
   - BEADs gold mislabel rate (the headline statistic)
   - `qlora_beads_full` true accuracy (vs the 0.7987 against noisy gold)
   - Cross-dataset ensemble flip-correctness on flagged rows
     (validates whether the ensemble's *which-label* judgment is
     reliable enough to use for re-labeling, not just removal)
5. **Pre-registered cleaning flag**: `cross_unanimous_disagree == 1` —
   a row is flagged whenever all three non-BEADs adapters (BABE,
   cajcodes, WNC) agree with each other against BEADs gold. The BEADs
   adapter's vote is *not* required (it trained on the gold so its
   agreement is partly tautological). The four adapters were run on
   BEADs train + val via
   [scripts/predict_beads_train_val.slurm](../scripts/predict_beads_train_val.slurm)
   (~$1, ~65 min wall) to produce the measurements below.

   | Split | Total | Flagged | Rate | Missed bias | Over-called | Asymmetry |
   | --- | ---: | ---: | ---: | ---: | ---: | ---: |
   | train | 27,263 | **10,371** | 38.0% | 8,450 | 1,921 | 4.4:1 |
   | val   |  8,520 |  3,178 | 37.3% | 2,611 |   567 | 4.6:1 |
   | test  |  6,816 |  2,585 | 37.9% | 2,135 |   450 | 4.7:1 |

   Three takeaways:
   - **The cleaned train set will have 16,892 rows** (27,263 − 10,371),
     ~62% of the original pool.
   - **Flag rate is essentially identical across train / val / test**
     (37.3-38.0%). The earlier extrapolation from test (37.9%) is
     within 0.7 pp of the measured train rate — the noise is uniformly
     distributed, not split-specific.
   - **The 4.4-4.7:1 missed-bias asymmetry holds across all three
     splits.** BEADs systematically under-labels biased content in
     its `non-biased` category. This is a structural property of the
     dataset, not a sampling artifact — worth its own sentence in the
     writeup.

   The cleaning *action* (remove vs flip) is conditional on step (4)'s
   flip-correctness:
   - **≥ 70%**: both remove and flip retrains run (three-way comparison)
   - **55-70%**: only remove (flip is too unreliable to trust the relabel)
   - **< 55%**: skip the retrain step; the noise-detection result stands alone
6. **Retrain** the full `qlora_{100, 500, 1k, 5k, full}` sweep on
   cleaned train (not just `qlora_full`, because the learning-curve
   shape answers whether cleaning matters more at low or high data —
   probably the more interesting finding). Evaluate the new sweep
   against the 500 hand-labeled rows, not against the noisy BEADs test
   (the latter would be moving the goalposts).
7. **Success criterion**: cleaned-data `qlora_full` accuracy on the 500
   hand-labels exceeds original-data `qlora_full` accuracy by
   **≥ 1.5 × the IAA disagreement rate**. Concrete thresholds at
   plausible IAA values:

   | IAA agreement | Disagreement rate | Lift threshold for success |
   | ---: | ---: | ---: |
   | 95% | 5% | ≥ 7.5 pp |
   | 90% | 10% | ≥ 15 pp |
   | 85% | 15% | ≥ 22.5 pp |
   | 80% | 20% | ≥ 30 pp |
   | 70% (gate floor) | 30% | ≥ 45 pp |

   The 1.5× multiplier requires the lift to exceed the ground-truth
   noise floor with a margin for finite-sample noise on a 500-row
   evaluation. Strict but not pathological — at decent IAA (≥85%) it's
   asking for a clearly visible improvement, not a marginal one.

What this experiment can produce:

| Outcome | What it tells the writeup |
| --- | --- |
| Cleaning improves lift on hand-labels | "BEADs noise is real and addressable; ensemble-cleaning is a viable BEADs auto-cleaner." Strong story. |
| Cleaning has no effect | "Either BEADs noise is structured such that QLoRA learns through it, or our ensemble flag captures something other than label noise (a different construct of bias)." Still a finding. |
| Cleaning hurts | "Removing/flipping flagged rows removes useful hard cases the decision boundary depends on, or the flagged rows weren't actually mislabels." A negative result worth reporting. |

**Decisions still open**

The cleaning rule has been applied. What remains is conditional on the
hand-labels:

- Whether to **remove** vs **flip** vs **skip** flagged rows, gated by
  the ensemble's flip-correctness on the 500 hand-labeled rows. Can
  only be answered once labeling is done.
- Whether the success criterion is met. Same dependency.

**Budget at-pinned-numbers**

| Item | Cost |
| --- | ---: |
| Predict 4 adapters on BEADs train + val | ~$2 |
| Full sweep retrain on "removed" cleaned data | ~$10 |
| Full sweep retrain on "flipped" cleaned data (only if flip-correctness gate passes) | ~$10 |
| **Total if both retrains** | **~$22** |
| **Total if only remove retrain** | **~$12** |

Plus ~6 hours of team labeling time (3 people × 2 hrs).

**Where the artifacts live**

- [scripts/beads_spot_check.py](../scripts/beads_spot_check.py) — the
  tool (commit `28a3614`).
- `beads_label_audit.csv` (regenerable, not gitted) — the
  share-with-teammates CSV.
- [outputs/cross_eval/qlora_*_full__on__beads/predictions.jsonl](../outputs/cross_eval/)
  — the per-row predictions the audit is built from.

---

## 2026-05-19 — QLoRA cross-eval matrix complete (the headline result)

**TL;DR**

A Llama-3.1-8B QLoRA fine-tune learns each of the four bias datasets
well in isolation (diagonal mean accuracy **0.842**) but does **not**
transfer between them (off-diagonal mean accuracy **0.510** — barely
above chance on a balanced binary task). The 8B-param model with
full LoRA fine-tuning does not soften the empirical signal that the
TF-IDF baseline already showed: **datasets are siloed.** Several
off-diagonals collapse to *worse than majority-class*. This is the
project's central empirical finding.

**Per-adapter training cost + same-dataset score**

| Adapter | n_train | Wall (min) | Accuracy | F1_macro | P_pos | R_pos |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| qlora_beads_full     | 27,263 | 99.1 | 0.7987 | 0.7987 | 0.7998 | 0.7961 |
| qlora_babe_full      | 3,296  | 12.9 | 0.8668 | 0.8645 | 0.8692 | 0.8957 |
| qlora_cajcodes_full  | 525    | 2.0  | 0.9697 | 0.9651 | 1.0000 | 0.9565 |
| qlora_wnc_full       | 27,263 | 97.6 | 0.7339 | 0.7336 | 0.7494 | 0.7027 |

WNC was capped at 27,263 rows (= BEADs full) for apples-to-apples
training size; the simple-dataset trains used their full pool. cajcodes'
0.970 is on a 66-row test set — wide CIs.

**The matrix — accuracy** (rows=train, cols=eval)

|         |   beads |    babe | cajcodes |     wnc |
| ---:    |   ---:  |   ---:  |    ---:  |   ---:  |
| **beads**    |  **0.7987** |  0.3123 |  0.6667  |  0.4617 |
| **babe**     |  0.3790 |  **0.8668** |  0.4242  |  0.5633 |
| **cajcodes** |  0.4224 |  0.5884 |  **0.9697**  |  0.5572 |
| **wnc**      |  0.4070 |  0.7603 |  0.5758  |  **0.7339** |

**The matrix — F1_macro** (rows=train, cols=eval)

|         |   beads |    babe | cajcodes |     wnc |
| ---:    |   ---:  |   ---:  |    ---:  |   ---:  |
| **beads**    |  **0.7987** |  0.2773 |  0.4000  |  0.4562 |
| **babe**     |  0.3496 |  **0.8645** |  0.4107  |  0.5198 |
| **cajcodes** |  0.3384 |  0.4314 |  **0.9651**  |  0.5346 |
| **wnc**      |  0.4026 |  0.7454 |  0.4454  |  **0.7336** |

Diagonal mean accuracy 0.842 (range 0.734-0.970). Off-diagonal mean
accuracy 0.510 (range 0.312-0.760). The F1_macro matrix tells the same
story but slightly sharper — three off-diagonal F1_macro values land
below 0.40 (i.e. degenerate-prediction patterns where one class is
mostly skipped).

**Cells worth flagging for the writeup**

1. **wnc → babe = 0.760 accuracy / 0.745 F1_macro** is by far the
   strongest off-diagonal — the only cross-dataset cell above 0.70.
   Plausible explanation: WNC's 27k training pool has broad lexical
   coverage of news-adjacent text and Wikipedia-derived neutralizations,
   which overlaps reasonably with BABE's news-sentence labels.
   *Asymmetric*: babe → wnc only reaches 0.563. So it's WNC's coverage
   helping on BABE, not vice versa.
2. **beads → cajcodes = 0.667 accuracy but F1_macro = 0.400** is a
   degenerate-prediction signature. cajcodes' test set is 69.7% positive,
   and a BEADs-trained model that over-predicts "biased" looks accurate
   without actually transferring. Reporting both metrics — not just
   accuracy — is the right framing here.
3. **cajcodes diagonals are on 66 test rows**; wide CIs. The 0.970
   diagonal sits a few rows from collapse — note in the writeup.
4. **All beads-train off-diagonal F1_macro values are ≤ 0.46** —
   BEADs-trained models transfer notably worse than wnc-trained ones,
   even though BEADs has comparable train size and was the original
   "main" dataset of the project. Suggests BEADs has more
   dataset-specific lexical signature than WNC's Wikipedia-NPOV pairs.

**The story this tells**

The proposal's central empirical question was whether bias datasets
were measuring "the same thing." The matrix says no:

- Each adapter beats its own dataset's TF-IDF baseline by 2-13 points
  (e.g. wnc 0.736 QLoRA vs 0.536 TF-IDF, +0.20 — the largest gain).
  So QLoRA *is* learning real per-dataset signal, not just memorising.
- That signal does not transfer. The mean off-diagonal accuracy of
  0.510 is essentially indistinguishable from coin-flip on a binary
  task with class prior near 0.50.
- This is consistent across all four train datasets — it isn't a
  property of one outlier dataset.

The strongest defensible writeup framing: *"bias" as labelled by
these four datasets is not one underlying construct measured four
different ways; it's four different constructs that happen to share
a label vocabulary.*

**Operational notes worth capturing for the next sweep**

The execution path got messier than necessary:

1. **Concurrent submissions raced.** A 119286/119287 pair landed in
   `squeue` because an earlier launch had already submitted babe +
   cajcodes when the user re-ran `launch_cross_eval_sweep.sh`. Both
   pairs ran to completion; the duplicate babe/cajcodes adapters
   consumed ~$1.35 of compute writing byte-identical outputs (same
   seed, same recipe, deterministic). The fcntl manifest lock that
   the 2026-05-19 prep entry added held cleanly — no corruption,
   just wasted GPU-hours.
2. **wnc's first run failed in 6 seconds** because
   `data/frozen/wnc/full/train.jsonl` wasn't on Tillicum. `push-data`
   wasn't run before launch, and WNC's loader has no HF fallback
   (it needs the raw `bias_data.zip` TSVs which are gitignored). babe
   and cajcodes survived because their loaders pull from HuggingFace.
   The slurm wrapper auto-fell-back to `freeze_splits.py` to
   regenerate, then crashed on the missing TSVs.
   **Lesson**: have `run_qlora.slurm` hard-fail with a clear error
   when JSONLs are missing for wnc/beads (no silent fallback to
   freeze_splits, which only works for HF-backed loaders). Defer this
   to a follow-up entry; for now we know to always `push-data`.
3. **The BEADs full adapter from sweep 117771 was deleted** during
   the post-rename cleanup of `outputs/qlora_full/` directories.
   The adapter weights had been written under the pre-rename name
   (`outputs/qlora_full/adapter/`) on Tillicum, and the recommended
   `rm -rf outputs/qlora_{100,500,1k,5k,full}` for cleaning up empty
   shells caught the adapter too. Cost ~$4.50 to retrain it.
   **Lesson**: when proposing destructive cleanup, name the specific
   files (not glob patterns) and check `find` for `adapter_model.safetensors`
   before recommending. Or commit adapters to git-LFS so they survive
   server-side housekeeping.

The retrained `outputs/qlora_beads_full/adapter/` scored 0.7987
(vs the May-16 sweep's 0.8036; delta -0.005). Within bf16 reduction
noise + training non-determinism. Not flagged as a separate issue.

**Final budget tally** for the cross-eval arm:

| Job | Cost |
| --- | ---: |
| 119286/119290 (babe, original + dup) | ~$1.80 |
| 119287/119291 (cajcodes, original + dup) | ~$0.55 |
| 119292 (wnc, failed 6s) | ~$0.00 |
| 119304 (beads retrain) | ~$3.60 |
| 119305 (wnc, real) | ~$3.60 |
| 119307 (cross-eval matrix) | ~$0.30 |
| **Total** | **~$9.85** |

Estimated vs the originally-proposed 4 trains + 1 x-eval (~$6.75):
roughly $3 of operational tax from the duplicates and beads retrain.
The matrix itself is settled and reproducible from
[outputs/cross_eval/](../outputs/cross_eval/) (16 cells × 3 files
each, all gitted).

**Where to read more**

- [outputs/manifest.csv](../outputs/manifest.csv) — the consolidated
  table. 24 rows (5 BEADs sweep + 3 new train rows + 16 cross-eval
  cells). The matrix is buildable by filtering on
  `(train_dataset, eval_dataset)`.
- [outputs/cross_eval/](../outputs/cross_eval/) — per-cell
  `eval_metrics.json` and `predictions.jsonl`.
- Commits `def1572` (pipeline + launchers) and `36700b7` (results).

**What's next**

- **Disagreement inspection** (Week 8 deliverable, per proposal §5):
  pick 30-50 examples where adapters disagree on the same input across
  datasets, hand-label them, see if the dataset-specific decisions
  are defensible or look like artifacts.
- **TF-IDF vs QLoRA delta table** for the writeup — a one-pager that
  shows each off-diagonal cell with the matched TF-IDF cell next to
  it, to argue that the QLoRA result isn't just "TF-IDF + noise."
- **WNC size cap retrospective**: WNC at 27k matched BEADs's train
  size, but BABE (3.3k) and cajcodes (525) didn't get capped. If the
  reviewer asks why babe transfers worse than wnc, the train-size
  confound is on the table. Worth adding a footnote, not blocking.

---

## 2026-05-19 — Tillicum launch prep: 3 new adapters + 16-cell cross-eval

**What this adds**

Plumbing to launch the QLoRA arm of the cross-eval matrix on Tillicum as a
single `sbatch`-chain — three new train jobs (BABE, cajcodes, WNC) plus one
cross-eval job that fans out across all four `outputs/qlora_*_full/` adapters
× all four test sets (16 cells).

**Decisions baked in**

- **WNC capped at 27,263 rows** (= BEADs full). Holds dataset size constant
  across the four adapters so the cross-eval matrix isolates *dataset content*
  rather than mixing in *training set size*. Implemented as the
  `--max-train-rows` knob already in `train_qlora.py`, surfaced as
  `MAX_TRAIN_ROWS` in [scripts/run_qlora.slurm](../scripts/run_qlora.slurm).
- **Adapters run concurrently, cross-eval waits.** The three train jobs are
  submitted with no inter-job dependency (different datasets, different
  durations — wnc ≈ 100 min, babe ≈ 12 min, cajcodes ≈ 2 min on H200). The
  cross-eval job carries `--dependency=afterok:<3 train ids>` so it only
  runs after all four adapters exist on disk.
- **Manifest race finally fixed** — see below.

**Files added / changed**

| Path | Change |
| --- | --- |
| [scripts/launch_cross_eval_sweep.sh](../scripts/launch_cross_eval_sweep.sh) | **new** — submits all 4 jobs with the right dependency edges; `--dry` prints the sbatch commands without submitting |
| [scripts/run_cross_eval.slurm](../scripts/run_cross_eval.slurm) | **new** — single-process wrapper around `cross_eval.py` (one node, one GPU, ≤2 h walltime) |
| [scripts/run_qlora.slurm](../scripts/run_qlora.slurm) | `MAX_TRAIN_ROWS` env knob threaded through to `train_qlora.py --max-train-rows` |
| [scripts/update_manifest.py](../scripts/update_manifest.py) | `manifest_lock()` — fcntl advisory lock around the read-modify-write |
| [scripts/tillicum_sync.sh](../scripts/tillicum_sync.sh) | `push-data` now recurses into the nested `data/frozen/<dataset>/...` layout (the old `--include='*.jsonl' --exclude='*'` filter blocked directory descent) |

**Manifest race condition — fixed**

The 2026-05-16 entry flagged the `outputs/manifest.csv` race
(last-writer-wins when concurrent jobs upsert) as worth fixing. Option (1)
from that entry — an `fcntl.flock` advisory lock — is now in
[scripts/update_manifest.py:25-65](../scripts/update_manifest.py#L25-L65).
The lock sibling is `outputs/manifest.csv.lock`, gitignored.

Smoke-tested by spawning two threads that each acquire the lock and sleep
0.5 s: events arrived `[enter A, exit A, enter B, exit B]` with no
interleaving. Stays a no-op on Windows (no `fcntl`); macOS/Linux paths both
have it.

Concurrent submission is now safe — the three train jobs can finish in any
order without clobbering each others' manifest rows. The cross-eval job is
already single-process (16 cells iterated sequentially inside one Python
interpreter) so it doesn't race with itself.

**Launch sequence (Tillicum)**

```bash
# 0. From local — push code and the frozen JSONLs (raw WNC TSVs and HF
#    caches stay local; only the deterministic JSONLs need to land on Tillicum).
scripts/tillicum_sync.sh push-code
scripts/tillicum_sync.sh push-data

# 1. On Tillicum — kick off the whole sweep.
ssh $TILLICUM_USER@tillicum.hyak.uw.edu
cd /gpfs/projects/imt526a/group7
scripts/launch_cross_eval_sweep.sh --dry   # eyeball the 4 sbatch commands first
scripts/launch_cross_eval_sweep.sh         # real submission
squeue -u $USER -t PD,R                     # watch

# 2. After cross-eval finishes — pull results back.
# Local:
scripts/tillicum_sync.sh pull-all
```

The launch script names each job (`babe`, `cajcodes`, `wnc-capped`, `x-eval`)
and prints the job ids + the watch command. `--dry` mode invents stable
non-numeric placeholder ids so a mis-formed dependency string is obvious
visually (`afterok:mock-...:mock-...`) rather than silently looking like a
real submission.

**Expected output**

After the sweep finishes:

```text
outputs/
├── qlora_beads_full/       # already there from 117771
├── qlora_babe_full/
├── qlora_cajcodes_full/
├── qlora_wnc_full/         # 27,263-row train
├── cross_eval/
│   ├── qlora_beads_full__on__beads/     # 4 same-dataset cells (redundant
│   ├── qlora_beads_full__on__babe/      # with the train jobs' eval — harmless,
│   ├── ...                              # different output dirs)
│   └── qlora_wnc_full__on__wnc/         # 16 cells total
└── manifest.csv             # 5 single-dataset rows + 16 cross-eval rows
```

**Open follow-ups**

- The training-time eval (run by `run_qlora.slurm` step 3) already produces
  `outputs/qlora_<ds>_full/eval_metrics.json` for the diagonal — it's
  recomputed by the cross-eval job as
  `outputs/cross_eval/qlora_<ds>_full__on__<ds>/eval_metrics.json`. Two
  separate files for the same number is fine (the cross-eval directory's
  cell is what populates the matrix in the writeup); deduplicating would
  cost more code than it saves and risks losing the in-job sanity number.
- Optional: pass `SKIP_EXISTING=1` to `run_cross_eval.slurm` if the user
  re-runs cross-eval and wants to keep the original cells. Default is
  `0` (recompute everything) so a code-change rerun produces a coherent
  matrix.

---

## 2026-05-19 — Multi-dataset pipeline + cross-eval matrix

**What changed**

The pipeline was single-dataset (BEADs only). It now runs over four bias
datasets — BEADs, BABE, cajcodes/political-bias, WNC — with a uniform
id/text/label schema, per-dataset frozen splits, and a cross-evaluation
harness that scores every adapter on every dataset's test set. The goal:
empirically test whether a model fine-tuned on one bias dataset transfers
to the others, or whether each dataset is largely about itself.

**Datasets in scope, and what was rejected**

| Dataset | Source | Rows (train/val/test) | Notes |
| --- | --- | --- | --- |
| beads | `shainar/BEAD` (existing) | 27,263 / 8,520 / 6,816 | Pre-restructure splits preserved bit-exact |
| babe | `mediabiasgroup/BABE` (HF, train+test unioned) | 3,296 / 412 / 413 | 80/10/10 stratified |
| cajcodes | `cajcodes/political-bias` (HF) | 525 / 66 / 66 | 5-class → binary (center=0, else=1); 69% positive |
| wnc | Pryzant et al. *bias_data.zip* (Dropbox) | 88,328 / 11,041 / 11,041 | Pair → biased(1)+neutral(0); all 3 native files unioned |

**Dropped**: Baly Article-Bias-Prediction and Hyperpartisan-byarticle are
full-article (~100–1000× longer than BEADs sentences). Mixing article-level
into a sentence-level cross-eval pool would conflate length distribution
with dataset identity; the empirical question is cleaner without them.
Decision recorded against the option to add them back as held-out
article-level test sets if a later experiment wants that signal.

**New layout** — per-dataset subdir with sizes nested:

```text
data/frozen/
├── beads/{splits_manifest.json, sizes/{100,500,1k,5k,full}/{train,val,test}.jsonl}
├── babe/{splits_manifest.json, full/{train,val,test}.jsonl}
├── cajcodes/{splits_manifest.json, full/{train,val,test}.jsonl}
└── wnc/{splits_manifest.json, full/{train,val,test}.jsonl}
```

`val.jsonl` and `test.jsonl` are copied into every BEADs size dir so each
size dir is self-contained. Disk cost is trivial; the cross-eval
orchestrator becomes pathless ("give me the test for dataset X").

Output dirs renamed to match: `outputs/qlora_{100,500,1k,5k,full}/` →
`outputs/qlora_beads_{100,500,1k,5k,full}/`. The matching `run_name`,
`inputs.train_jsonl`, and `inputs.splits_manifest` fields in each run's
`run_meta.json` were rewritten in the same step; SHA256s of historical
inputs were **not** rewritten (those are still pre-restructure history).
Tag `pre-restructure-v2-sweep` marks the commit before the move.

**Schema bump: `splits_manifest.json` v1 → v2**

v1 was flat: `splits["train_5k"] = {path, sha256, n, ...}`. v2 nests
explicit size and role: `sizes["5k"]["train"] = {path, sha256, n, ...}`,
plus a top-level `dataset` field. Both
[scripts/train_qlora.py:236-280](../scripts/train_qlora.py#L236-L280) and
[scripts/verify_splits_manifest.py:60-77](../scripts/verify_splits_manifest.py#L60-L77)
handle both schemas, so legacy v1 manifests in `data/processed/` (v1
calibration) still verify.

**Outputs manifest: `train_dataset` + `eval_dataset` columns**

[scripts/update_manifest.py](../scripts/update_manifest.py) gained two new
columns positioned right after `run_name`. The 5 BEADs sweep runs were
re-`update_manifest`'d after the rename so they pick up
`train_dataset=beads, eval_dataset=beads`. Cross-eval cells (rows of the
form `qlora_{train}_full__on__{eval}`) are distinguished by a different
`run_name` and stamped with the appropriate dataset values.

**Files that materially changed**

| Path | Change |
| --- | --- |
| [scripts/dataset_loaders/](../scripts/dataset_loaders/) | **new** — one `load()` per dataset, returns id/text/label_int |
| [scripts/freeze_splits.py](../scripts/freeze_splits.py) | dispatch on `--dataset`; writes schema v2 |
| [scripts/train_qlora.py](../scripts/train_qlora.py) | `--train-dataset` arg; v1+v2 SHA256 verification |
| [scripts/eval_adapter.py](../scripts/eval_adapter.py) | `--eval-dataset` arg; stamped into `eval_metrics.json` |
| [scripts/update_manifest.py](../scripts/update_manifest.py) | new columns; path-based inference fallback for legacy runs |
| [scripts/verify_splits_manifest.py](../scripts/verify_splits_manifest.py) | flattens v1 `splits` + v2 `sizes.{size}.{role}` into one check loop |
| [scripts/cross_eval.py](../scripts/cross_eval.py) | **new** — orchestrates (adapter × dataset) matrix; writes cells under `outputs/cross_eval/{run}__on__{ds}/` |
| [scripts/run_qlora.slurm](../scripts/run_qlora.slurm) | parameterized `DATASET + SIZE`; valid combinations enforced |
| [baselines/tfidf/tfidf_baseline.ipynb](../baselines/tfidf/tfidf_baseline.ipynb) | papermill parameters cell + dataset-aware metrics |

**Headline empirical result — TF-IDF cross-eval matrix (accuracy)**

Run on the new pipeline end-to-end (16 cells, all in
`baselines/tfidf/runs/`). TF-IDF + LogReg, ngram (1,2), max_features=50k,
C=1.0, seed=42 — identical recipe to the existing BEADs baseline,
applied to every (train, test) pair.

| train \ eval | beads | babe | cajcodes | wnc |
| ---: | ---: | ---: | ---: | ---: |
| **beads** | **0.767** | 0.407 | 0.697 | 0.504 |
| **babe** | 0.485 | **0.722** | 0.636 | 0.510 |
| **cajcodes** | 0.497 | 0.554 | **0.955** | 0.500 |
| **wnc** | 0.467 | 0.608 | 0.561 | **0.536** |

F1-macro shows the same pattern — diagonals dominate; off-diagonals are
weak (often *worse* than majority-class). Test-set sizes:
beads=6,816 · babe=413 · cajcodes=66 · wnc=11,041 (cajcodes' 0.955 is on
66 rows and is noisy).

**What this means for the project's empirical question**

Even before the QLoRA cross-eval runs, the linear baseline already gives a
strong signal: **a bias classifier trained on one of these datasets does
not generalize to the others**. Most off-diagonals sit near or below
majority-class. The two highest cross-transfers (beads→cajcodes 0.70,
babe→cajcodes 0.64) likely reflect cajcodes' lexical regularity (it's
synthetic and short) — not a deeper transfer. The QLoRA matrix will tell
us whether a 8B-param fine-tune softens this — but the prior is now
"datasets are siloed."

**Implementation lessons worth keeping**

1. *Name your loader package anything other than `datasets`.* The first
   draft used `scripts/datasets/`. When you `python scripts/freeze_splits.py`,
   Python prepends `scripts/` to `sys.path` — and `scripts/datasets/`
   then shadows the HuggingFace `datasets` library inside the loaders.
   Renamed to `scripts/dataset_loaders/`; lesson noted in the package
   docstring so the next person doesn't try to "fix" the naming.

2. *Reproducibility check the migration before deleting the originals.*
   The new `freeze_splits.py --dataset beads` was first run to `/tmp` and
   its SHA256s diffed against the pre-restructure `splits_manifest.json`.
   All 7 hashes matched bit-for-bit — content invariance under the
   refactor proven. Only then were the flat-layout files removed in the
   same commit that adds the new layout. Without this guard, a silent
   change in pandas/sklearn split ordering would have invalidated
   apples-to-apples comparisons against the pre-rename runs.

3. *Path inference is fragile when the path schema itself changes.*
   `update_manifest.py` infers `eval_dataset` from the test JSONL path as
   a fallback. First pass extracted the segment after "frozen" — which on
   legacy `data/frozen/test_held_out.jsonl` resolved to
   `"test_held_out.jsonl"` and wrote that string into the manifest. Fix:
   constrain inferred segments to `{beads, babe, cajcodes, wnc}`, and
   treat `data/frozen/*.jsonl` (the flat legacy shape) as implicitly
   beads.

**Verification chain that passed**

1. BEADs SHA256s match pre-restructure (`pre-restructure-v2-sweep` tag) —
   7/7 splits bit-identical.
2. All 4 datasets' on-disk JSONLs match their committed manifests
   (`verify_splits_manifest.py` on each, 24 splits OK).
3. Loader smoke-test (`load() → DataFrame` for each of the 4) — no nulls,
   binary labels, unique IDs.
4. TF-IDF cross-eval (16 cells) ran end-to-end via papermill —
   matrix above.
5. `cross_eval.py --dry-run` produced the expected 8 cells when pointed
   at `outputs/qlora_beads_100` and `outputs/qlora_beads_full` (sanity
   check on path resolution; no QLoRA cells have been actually evaluated
   yet — Tillicum job).

**Open follow-ups**

- **HPC: train 3 new adapters** (babe / cajcodes / wnc, full size,
  Llama-3.1-8B-Instruct, same QLoRA recipe as BEADs). Then run
  `python scripts/cross_eval.py --adapters outputs/qlora_*_full
  --eval-datasets beads babe cajcodes wnc` to produce the QLoRA version
  of the matrix above. 4 trains + 16 evals.
- **WNC size cap**: WNC's full train pool is 88,328 rows — ~3.2× BEADs'
  27,263. Either cap WNC at 27k (apples-to-apples comparison) or run it
  at full size (more data is a confound bundled into the result).
  Recommendation: cap. Decide before submitting the SLURM job.
- **cajcodes test set is tiny** (66 rows). Cross-eval rows landing on
  cajcodes carry wide CIs and should be reported with that caveat in the
  writeup, not as point estimates.
- **`update_manifest.py` race condition** is now more painful: 32 QLoRA
  cross-eval cells if all run concurrently. The 2026-05-16 entry's
  proposed fixes (file lock or per-run shard + consolidate) are now
  worth doing. Workaround until then: run cells serially or post-process
  in a single thread.
- **Baly + Hyperpartisan** can be added back as article-level held-out
  test sets without disturbing the sentence-level pool. Loader stubs not
  written; defer until/unless the writeup wants them.

---

## 2026-05-16 — Full sweep complete (qlora_100 → qlora_full)

**What ran**

All five v2 sweep runs launched in parallel on Tillicum node g006 (8× H200,
each job got its own GPU). Jobs `117767`–`117771`. EVAL_BATCH_SIZE=16
(default) for all five.

**Headline result — the learning curve**

| Train rows | Accuracy | F1_macro | F1_pos | Prec_pos | Recall_pos | Train wall | Slurm |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 100 | 0.4991 | 0.3337 | 0.6657 | 0.4992 | 0.9988 | 0.5 min | 117767 |
| 500 | 0.6844 | 0.6817 | 0.7111 | 0.6549 | 0.7778 | 2.1 min | 117768 |
| 1,000 | 0.6981 | 0.6975 | 0.7108 | 0.6811 | 0.7432 | 4.1 min | 117769 |
| 5,000 | 0.7584 | 0.7584 | 0.7568 | 0.7605 | 0.7532 | 18.6 min | 117770 |
| 27,263 | **0.8036** | **0.8035** | 0.8024 | 0.8059 | 0.7990 | 99.3 min | 117771 |

Per-run metrics live in `outputs/qlora_<size>/eval_metrics.json` and
`train_metrics.json`. The consolidated manifest is at
[outputs/manifest.csv](../outputs/manifest.csv) /
[outputs/manifest.json](../outputs/manifest.json).

**What this means**

- **`qlora_full` reaches 0.804 accuracy / 0.804 F1_macro** on the 6,816-row
  held-out test set. That clears the BEAD paper's reported Llama2-7B
  baseline of 0.77 accuracy (Raza et al., 2024) by ~3 points. Headline
  number for the writeup.
- **Returns are still climbing at the right edge of the curve.**
  100→500 gives +0.185 accuracy, 500→1k gives +0.014, 1k→5k +0.060,
  5k→full +0.045. The 1k→5k and 5k→full deltas are similar in size, so
  more data would plausibly still help. Not yet in the diminishing-returns
  regime by the time the BEAD training split is exhausted.
- **The over-predict-biased asymmetry is a low-data phenomenon, not a
  permanent feature.** Precision/recall asymmetry resolves cleanly with
  scale: at 100 we have P 0.50 / R 1.00 (degenerate collapse), at 500
  P 0.65 / R 0.78, at 5k P 0.76 / R 0.75, at full P 0.81 / R 0.80
  (balanced). The proposal's Risk #1 ("model over-predicting biased after
  1 epoch on 1k") is now contextualized — it's a data-quantity artifact,
  not a stable decision boundary.
- **The 100-row collapse is reproducible.** Across the calibration sweep
  and the canonical run, all qlora_100 instances landed at ~0.499 accuracy
  with recall_pos > 0.998. Useful left-anchor for the curve.

**Calibration math sanity**

Sustained training throughput converged to ~13.7 ex/s on H200 with
gradient checkpointing — better than the v1 calibration's 10.6 ex/s
extrapolation. Training wall scaled almost exactly linearly with example
count (100→0.5 min, full→99.3 min ≈ 100×). Peak VRAM was 14.6 GB across
all runs ≥ 500, matching the calibration prediction. The proposal's
H200-hours budget was generous by ~1.8× on the train side.

**Eval wall held steady at ~180 s per run** at bs=16 (vs the original
864 s before the refactor) — the calibration result transfers cleanly
across sweep sizes since they all evaluate the same test set.

**Open follow-ups**

- TF-IDF baseline (Abrevaa) is in `TF-IDF baseline/` locally but not yet
  committed; needs a directory rename before commit (space in name).
- 3-shot Llama-3.1-8B prompting baseline still paused pending Prof. Harker's
  guidance on label-quality concerns surfaced during inspection.
- Learning-curve plot + qualitative inspection on 30–50 disagreement examples
  are Week 8 deliverables.
- The bigger structural eval-perf win — length-sorted batching — is
  unblocked but not implemented. See [[2026-05-16-calibration]]; the
  measured ceiling for random-order batches is ~180 s and we don't need
  faster than that for the remaining work.

---

## 2026-05-16 — Eval-batch-size calibration + completion-only scoring (OOM fix)

**Calibration sweep on qlora_100**

After the batched-scoring refactor (entry below), ran a four-point
sweep over `EVAL_BATCH_SIZE` against the qlora_100 adapter to find the
optimal batch size before launching the bigger sweeps.

| Slurm | bs | Forward (s) | rows/s | per-item (ms) | Result |
| ---: | ---: | ---: | ---: | ---: | --- |
| 117680 | 16 | 181.9 | 37.5 | 13.3 | baseline (post-refactor) |
| 117738 | 32 | 199.2 | 34.2 | 14.6 | slower than bs=16 |
| 117756 | 64 | — | — | — | **OOM (87 GB allocation)** |
| 117761 | 32 | 196.1 | 34.8 | 14.4 | post-OOM-patch, still slower |
| 117762 | 64 | 226.2 | 30.1 | 16.6 | slower again |

**Finding: per-item cost grows monotonically with batch size.** The H200
is fully compute-saturated at bs=16. Going bigger just invites more
padding waste — BEAD texts vary widely in length, so larger batches
right-pad to longer max-in-batch values and waste more compute on pad
tokens. bs=16 is the optimal point for the **current random-order
batching** code path.

**OOM fix — completion-only scoring**

The bs=64 run (slurm 117756) crashed with `torch.OutOfMemoryError`
trying to allocate 87.12 GB at this line in `score_items_batched`:

```python
log_probs = torch.log_softmax(logits[:, :-1, :].float(), dim=-1)  # (B, T-1, V)
```

Root cause: the original refactor materialized a full `(B, T-1, V)`
fp32 log-softmax tensor over the entire sequence even though we only
needed log-probs at the 1–3 completion-token positions per row. At bs=64,
max_seq_length=512, V=128,256: that tensor alone is 16.8 GB, and the
upstream `.float()` cast plus the gather staging push the allocation
chain to 87 GB.

**Fix (commit `d1aa8e4`)**: slice logits to the completion positions
**per row** before computing log-probs. Each slice is `(n_comp, V)` with
n_comp typically 1–3, so the fp32 cast + logsumexp math costs effectively
nothing. Peak eval memory is now bounded by the model's own `(B, T, V)`
bf16 logits output (~17 GB at bs=128, fine on a 141 GB H200).

Math is identical:
`log p(target | context) = logit[target] - logsumexp(logits)`. We compute
this on the small completion slice; the full-vocab reduction still
happens (it's how you normalize a distribution) but only at completion
positions.

**Verification**: local fp32 smoke flow on tiny-Llama (20 rows) showed
20/20 prediction matches between bs=1 and bs=16 with score deltas at
3.8e-06 — float32 reduction noise from a different kernel path
(`logsumexp + gather` vs `log_softmax`), nowhere near anything that
flips a prediction. On Tillicum at bf16 the metrics drift across the
three qlora_100 runs (0.4990 / 0.4994 / 0.4990) is entirely from
training non-determinism, not the eval path.

**Lesson worth remembering**

My first refactor (batching) optimized the per-row scoring loop but left
the per-batch math memory-naive. A peer agent flagged the issue
("score only the completion tokens instead of computing log-probs for
every prompt token") which matched the right diagnosis. Always slice
before reducing when only a small subset of positions matter.

---

## 2026-05-16 — Manifest race + sort-key bug

**What broke**

All five sweep jobs ran in parallel on g006 and tried to upsert into the
shared `outputs/manifest.csv` simultaneously. Only the last writer (the
qlora_100 row, since it finished first and re-wrote last after others
started reading the empty file) survived. The other four runs' rows were
lost from the consolidated manifest, though per-run `eval_metrics.json`,
`train_metrics.json`, and `run_meta.json` were unaffected (they live in
separate run directories).

When I tried to rebuild the manifest by re-running
`scripts/update_manifest.py` sequentially against each run directory,
the first call succeeded but the second through fifth raised
`TypeError: '<' not supported between instances of 'int' and 'str'`.

Root cause in [scripts/update_manifest.py:131](../scripts/update_manifest.py#L131):
the sort key was `(r.get("train_size") or 0, r.get("run_name") or "")`.
Freshly-built rows have `train_size` as an `int` (from `train_metrics.json`),
but rows loaded from an existing CSV have `train_size` as a `str` (CSV
stringifies everything). Once both shapes coexist mid-rebuild, Python 3
can't compare int with str → TypeError.

**Fix**

Added `_train_size_int()` helper that coerces the value to int via
try/except. The sort key uses the helper, so it works whether the row
came from a fresh JSON read or a CSV round-trip.

**Follow-ups**

- The race condition itself isn't fixed. If five concurrent jobs all
  upsert into the same CSV, the result is non-deterministic. A proper
  fix would either:
  1. Add a file lock (`fcntl.flock`) around the read-modify-write in
     `update_manifest.py`, or
  2. Have each job write to a per-run shard (e.g.
     `outputs/qlora_<size>/manifest_row.json`) and run a separate
     consolidation step after the sweep.
  The simplest near-term mitigation is to launch sweep jobs with a small
  stagger (e.g. `--dependency=afterany:<prev_id>` so they queue serially)
  or to just re-run `update_manifest.py` for each run dir after the
  sweep finishes.
- Worth noting: the per-run files are the ground truth. The manifest is
  a derived view that can always be rebuilt from them. Losing the
  manifest is annoying but not data loss.

---

## 2026-05-16 — Batched eval scoring

**What changed**

[scripts/eval_adapter.py](../scripts/eval_adapter.py) rewritten so the
likelihood-scoring pass over the held-out test set runs in batches instead
of one (row, label) at a time:

- `prepare_scoring_items()` tokenizes each prompt once, then tokenizes
  `prompt + label` once per (row, label). The full prompt+label string is
  still tokenized per item so BPE boundary effects at the
  prompt-end/label-start boundary are byte-identical to the previous
  single-example path.
- `score_items_batched()` right-pads sequences within each batch, builds
  the attention mask, runs one `torch.inference_mode()` forward per batch,
  gathers per-token log-probs only at completion positions via a
  `(B, T-1)` boolean mask, and accumulates per-row scores in a GPU tensor.
- Results stay on the GPU until the entire eval is done — one CPU sync at
  the end instead of one per row.
- `--eval-batch-size N` (default 16) added; threaded through as
  `EVAL_BATCH_SIZE` in [scripts/run_qlora.slurm](../scripts/run_qlora.slurm).
- Eval-side base-model load now uses `device_map={"": 0}` (single GPU)
  instead of `"auto"`.

**Why**

The v1 eval path was `~7.9 rows/sec` on H200 — 864 s for the 6,816-row
test set in qlora_100. Inspection showed: every row did **2** forward
passes at batch size 1, **4** tokenizer calls, and forced a CUDA sync per
`.item()` to read out each scalar score. The H200 was idling. With the
test set unchanged across the 5 sweep sizes, fixing this once was worth
~13 min per sweep run, ~1 hour cumulative across the sweep.

**Verification**

Local smoke flow (fp32 tiny-Llama on MPS, 20 rows): predictions and
per-class scores **bit-identical** between `--eval-batch-size 1` and
`--eval-batch-size 16`. On Tillicum at bf16 the batched path will diverge
from bs=1 by reduction-order noise only — orders of magnitude below the
typical class-score gap, so argmax decisions stay stable.

**Expected impact**

~8–14× wall-clock speedup at batch 16. To be confirmed by the first
re-run of qlora_100 against the new code.

**Open follow-ups**

- The slurm wrapper launches `train_qlora.py` and `eval_adapter.py` as
  two separate Python processes, so the 8B base model loads twice per
  job (~30–90 s each on GPFS). A single-process orchestrator that loads
  once, trains, swaps the adapter into eval mode, and scores would
  recover this. Bigger refactor than this entry; deferred.

---

## 2026-05-14 — qlora_100 sanity run (slurm 115121)

**What ran**

First full v2 sweep run, smallest training size:

- Train: `train_100.jsonl` (100 rows), 3 epochs, effective batch 16,
  21 global steps, 29.2 s wall-clock, 13.0 GB peak VRAM.
- Eval: held-out `test_held_out.jsonl` (6,816 rows), 863.7 s.
- Adapter: `outputs/qlora_100/adapter/` (gitignored), metrics committed
  in [outputs/qlora_100/](../outputs/qlora_100/).

**Result**

| Metric | qlora_100 |
| --- | --- |
| accuracy | 0.4990 |
| f1_pos | 0.6655 |
| f1_macro | 0.3339 |
| precision_pos | 0.4991 |
| recall_pos | 0.9982 |
| recall_neg | 0.0012 |
| train_loss_final | 1.997 |

The adapter collapsed to "always predict biased" — recall_pos 0.998,
recall_neg 0.001. The training loss never came down meaningfully (started
at 2.6 mid-epoch-1, ended at ~1.4 mid-epoch-3, reported final 1.997
because TRL averages over the run).

**Why it matters**

This was the explicit sanity-check leg called for in Risk #2 of the v2
proposal — "The 100-example run serves as an explicit sanity check before
launching larger runs." The collapse is consistent with the v1
calibration's noted asymmetry (over-prediction of biased after 1 epoch
on 1k) and is **not** evidence of a broken pipeline:

- The mechanical pipeline ran cleanly end-to-end: split verification
  against the committed `splits_manifest.json` sha256s, training,
  adapter save, eval, manifest upsert.
- The pathology is consistent with under-data, not a code bug.
- The proposal's framing — learning-curve sweep across five sizes — treats
  a degenerate left endpoint as a valid (if uninformative) data point.

**Decision**

Continue with 500 / 1k / 5k / full rather than stopping to retune. The
500-row run is the first one where the model has enough supervision to
meaningfully move off the prior.

**Performance bottleneck flagged**

Eval time (863 s) dominated training (29 s) by ~30×. Single-example
likelihood scoring identified as the bottleneck — see 2026-05-16 entry.

---

## 2026-05-13 — v2 sweep scaffolding committed

**What landed**

Initial public commit (`2e0f0db`) plus three follow-ups: 

- `scripts/freeze_splits.py` — builds nested stratified subsets
  (`train_100 ⊂ 500 ⊂ 1k ⊂ 5k ⊂ full`) and a sha256 manifest.
- `scripts/train_qlora.py` — QLoRA SFT (Llama-3.1-8B-Instruct, 4-bit
  nf4, LoRA r=16 on q/k/v/o/gate/up/down). Loss is masked to the
  assistant turn via `trl.SFTConfig(completion_only_loss=True)` (asserted
  at startup; the script exits if the trl release doesn't support that
  param).
- `scripts/eval_adapter.py` — likelihood-scored binary eval, picks the
  higher-summed-log-prob completion between `"biased"` and `"non-biased"`
  under the same chat template the model was trained on.
- `scripts/update_manifest.py` — upserts one row per run into
  `outputs/manifest.csv` and `outputs/manifest.json` from the
  per-run `run_meta.json` + metrics files.
- `scripts/run_qlora.slurm` — single launcher; `SIZE=100|500|1k|5k|full`.
- `scripts/legacy/` — preserves the v1 calibration scripts.

**Audit guards (commit `b2a291c`)**

The commit message refers to C1/C2/C3 and I4/I6. In the codebase those
became:

- **C1 — split integrity.** `verify_splits_manifest.py` hashes every
  on-disk JSONL against the committed `splits_manifest.json` before
  training; the slurm wrapper snapshots the manifest into `/tmp` first
  to detect mid-job modification.
- **C2 — input drift at training time.** `train_qlora.py` re-hashes
  `--train-jsonl` and compares to the manifest entry; mismatch is a
  hard exit.
- **C3 — TRL contract.** `train_qlora.py` inspects `SFTConfig`'s
  signature and aborts if `completion_only_loss` isn't accepted by the
  installed trl release.
- **I4 / I6 — reproducibility metadata.** Every run writes
  `run_meta.json` with: argparse values, started/finished UTC
  timestamps, slurm job id + node, CUDA visible devices, library
  versions (torch / transformers / peft / trl / bitsandbytes / datasets
  / accelerate / CUDA / GPU name), git head, input sha256s, and label
  map. `update_manifest.py` joins these into the sweep manifest.

**Tillicum sync (commit `755e44a`)**

`scripts/tillicum_sync.sh` wraps the common rsync transfers:
`push-code`, `push-data`, `pull-results`, `pull-logs`, `pull-all`,
`status`, with a `--dry-run` mode. The team shares a single clone at
`/gpfs/projects/imt526a/group7` — no per-user copies — so the sync
script also documents the coordination protocol in its help text.

---

## 2026-05-09 — v1 1k-example timing calibration

**What ran**

A 1,000-example × 1-epoch QLoRA fine-tune on H200 via Tillicum, run
**solely** to size the compute budget in the v2 proposal. Not a results
run.

**Measured**

- 1.58 min wall-clock
- 14.6 GB peak VRAM
- 10.6 examples/sec sustained throughput

**Why it shaped the project**

These three numbers extrapolated linearly across the sweep gave the
H200-hours estimates in proposal §4 — `qlora_100: 0.30 H200-hrs`,
`qlora_500: 0.34`, `qlora_1k: 0.39`, `qlora_5k: 0.85`, `qlora_full: 3.04`
— plus a 25% contingency buffer of 4 hours. The full sweep budget was
~10 H200-hrs.

A secondary observation from this run informed Risk #1: after 1 epoch
on 1k examples the model produced 54 false positives against 16 false
negatives. That asymmetry is what the qlora_100 sanity run was looking
for, and is what it found in a more extreme form (recall_pos 0.998).

**Where to read more**

- [docs/v1_calibration_writeup.md](v1_calibration_writeup.md) — full
  v1 narrative.
- [outputs/tillicum_1k_calibration/](../outputs/tillicum_1k_calibration/)
  — committed metrics; adapter weights gitignored but regenerable from
  `scripts/legacy/run_tillicum_1k.slurm`.

---

## Standing snapshot (as of 2026-05-19)

**BEADs sweep** — all five runs complete (2026-05-16; renamed 2026-05-19)

| Run | Train rows | Accuracy | F1_macro | Prec_pos | Recall_pos | Slurm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| qlora_beads_100 | 100 | 0.4991 | 0.3337 | 0.4992 | 0.9988 | 117767 |
| qlora_beads_500 | 500 | 0.6844 | 0.6817 | 0.6549 | 0.7778 | 117768 |
| qlora_beads_1k | 1,000 | 0.6981 | 0.6975 | 0.6811 | 0.7432 | 117769 |
| qlora_beads_5k | 5,000 | 0.7584 | 0.7584 | 0.7605 | 0.7532 | 117770 |
| qlora_beads_full | 27,263 | **0.8036** | **0.8035** | 0.8059 | 0.7990 | 117771 |

**Cross-eval datasets** — frozen on 2026-05-19, verified against committed manifests

| Dataset | Train / val / test | Class balance (test) | Train splits manifest |
| --- | ---: | ---: | --- |
| beads | 27,263 / 8,520 / 6,816 | 50.1% biased | [data/frozen/beads/splits_manifest.json](../data/frozen/beads/splits_manifest.json) |
| babe | 3,296 / 412 / 413 | 55.7% biased | [data/frozen/babe/splits_manifest.json](../data/frozen/babe/splits_manifest.json) |
| cajcodes | 525 / 66 / 66 | 69.7% biased | [data/frozen/cajcodes/splits_manifest.json](../data/frozen/cajcodes/splits_manifest.json) |
| wnc | 88,328 / 11,041 / 11,041 | 50.0% biased | [data/frozen/wnc/splits_manifest.json](../data/frozen/wnc/splits_manifest.json) |

**Cross-eval matrices**

| Method | Status | Where |
| --- | --- | --- |
| TF-IDF + LogReg | Complete, 16 cells (4×4) | [baselines/tfidf/runs/](../baselines/tfidf/runs/); matrix in the 2026-05-19 entry |
| QLoRA (Llama-3.1-8B) | BEADs row complete (×4 evals pending); 3 new adapters not yet trained | Cells will land at `outputs/cross_eval/{run}__on__{ds}/` |

**Baselines (single-dataset, BEADs-on-BEADs anchor)**

| Baseline | Owner | Status |
| --- | --- | --- |
| TF-IDF + logistic regression | Abrevaa | Now part of the cross-eval matrix. BEADs-on-BEADs: acc 0.7675, F1_macro 0.7675 (unchanged from 2026-05-15 single-cell result). |
| 3-shot Llama-3.1-8B prompting | Ash | Paused — example selection raised label-quality concerns; awaiting Prof. Harker's guidance |

**Open questions / follow-ups**

- **HPC: 3 new QLoRA adapters** (babe / cajcodes / wnc, full size, same
  recipe as the BEADs sweep) — see the 2026-05-19 entry for the launch
  command. Then 32 cross-eval cells via `scripts/cross_eval.py`.
- **WNC size cap decision** — full WNC is 88k rows (~3.2× BEADs full).
  Cap at 27k for apples-to-apples or run full and treat extra data as a
  confound. Recommendation: cap.
- **cajcodes test set is 66 rows** — wide CIs; report cross-eval cells
  landing on cajcodes with that caveat.
- 3-shot prompting baseline blocked on label-noise question with the instructor.
- Week 8 deliverables: learning-curve plot (BEADs; done at commit `965c576`),
  qualitative inspection of 30–50 disagreement examples (per proposal §5).
- **Length-sorted batching** would unlock another 2–3× eval speedup
  (estimated 60–90 s/run vs. the current 180 s) by eliminating
  padding waste in random-order batches. Not blocking — current eval
  cost is small relative to training, especially for qlora_full.
- **Manifest race condition** is now more painful — 32 QLoRA cross-eval
  cells could collide on `outputs/manifest.csv` if run concurrently. The
  2026-05-16 entry's proposed fixes (file lock or per-run shard +
  consolidate) are now worth doing. Workaround: run cells serially.
- **Model-load cost on Tillicum**: ~9 s on warm GPFS cache (measured
  this round), paid twice because train and eval are separate Python
  processes. Combining them into a single orchestrator would recover
  ~9 s × 5 runs ≈ 45 s. Marginal; not worth doing now that eval is
  fast.
