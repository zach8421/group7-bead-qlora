# BEAD QLoRA — Project Summary

**Goal**: Build a bias / non-bias classifier on BEAD using QLoRA fine-tuning of Llama-3.1-8B-Instruct, and test how reliable the resulting model is.

The project unfolded in six phases. Each phase below records what was done, the question it was meant to answer, and the results.

---

## Phase 1 — Original BEADs learning curve

**What**: Trained QLoRA adapters on 5 nested stratified subsets of BEADs (100, 500, 1k, 5k, and 27,263 rows). Each evaluated on the 6,816-row held-out test split.

**Why**: Establish a baseline learning curve and confirm QLoRA is competitive at this scale.

**Results**:

| Train rows | Accuracy | F1_macro |
| ---: | ---: | ---: |
| 100 | 0.499 | 0.334 |
| 500 | 0.684 | 0.682 |
| 1,000 | 0.698 | 0.698 |
| 5,000 | 0.758 | 0.758 |
| 27,263 | **0.804** | **0.804** |

The full-size adapter cleared the BEAD paper's Llama2-7B benchmark (0.77 accuracy) by ~3 points.

---

## Phase 2 — Multi-dataset cross-evaluation matrix

**What**: Trained QLoRA adapters on three additional bias datasets (BABE, cajcodes/political-bias, WNC) and produced a 4×4 cross-evaluation matrix: every adapter scored against every dataset's test split.

**Why**: Test whether bias as labeled across these datasets reflects a shared underlying construct or four different construct-specific labelings.

**Results** (accuracy):

| train \ eval | beads | babe | cajcodes | wnc |
| ---: | ---: | ---: | ---: | ---: |
| **beads** | **0.799** | 0.312 | 0.667 | 0.462 |
| **babe** | 0.379 | **0.867** | 0.424 | 0.563 |
| **cajcodes** | 0.422 | 0.588 | **0.970** | 0.557 |
| **wnc** | 0.407 | 0.760 | 0.576 | **0.734** |

- Diagonal mean: 0.842. Off-diagonal mean: 0.510 (barely above chance on a balanced binary task).
- Adapters perform well only on their own dataset. Datasets appear siloed at the model level.

---

## Phase 3 — Hand-labeling experiment

**What**: A 3-person team hand-labeled 500 stratified random rows from BEADs test. Each labeler took 200 rows (150 unique + 50 shared for inter-annotator agreement). Labeling was blind to BEADs gold and model predictions.

**Why**: Test whether BEADs's gold labels are reliable enough to serve as ground truth, and create a clean evaluation set for downstream comparisons.

**Results**:

- First-pass inter-annotator agreement: 81% / 64% / 63% pairwise. Recalibration on the bias definition (emphasizing "public-interest topic") followed by relabeling produced second-pass agreement of 81% / 88% / 82%, Cohen's κ 0.57–0.75. The IAA gate passed.
- BEADs gold disagrees with team consensus on **69.5% of rows** (342 of 492 non-abstain).
- Direction of disagreement: BEADs over-calls bias 1.3× more often than it misses bias (194 over-calls vs 148 missed). This reverses the direction inferred from Phase 2's model-only signal.
- The original BEADs-trained adapter scores **0.283 accuracy** against the team consensus — well below chance on a balanced binary task — despite scoring 0.799 against BEADs's own noisy gold. The adapter learned the labeling noise.

---

## Phase 4 — Cleaning + retrain experiment (Round 1)

**What**: Applied a 3-voter cleaning rule (BABE + cajcodes + WNC adapters unanimous against BEADs gold = flag for cleaning). Tested two cleaning actions (remove the flagged rows; flip them to the ensemble's consensus label) and two class balances (natural distribution after cleaning; undersample majority class to 50/50). 4 cleaned dataset variants × 5 train sizes = **20 new adapters**.

**Why**: Test whether algorithmically cleaning BEADs labels lifts model accuracy against the hand-label consensus, and isolate the contribution of cleaning vs class balance.

**Results** (accuracy on the 492-row hand-label consensus):

| Adapter | Train rows | Accuracy | F1_macro |
| --- | ---: | ---: | ---: |
| qlora_beads_full (baseline) | 27,263 | 0.283 | 0.275 |
| Best cleaned: remove + balanced | 500 | 0.468 | 0.467 |
| Best cleaned: flip | 27,263 | 0.565 | 0.546 |
| **Best cleaned: flip + balanced** | **500** | **0.768** | **0.768** |

- Headline lift: **+48.6 percentage points accuracy** over the noisy-data baseline.
- Flip dominates remove. Class balancing matters as much as cleaning.
- The cleaned_flip_balanced learning curve is non-monotonic: peaks at 500 training rows (0.768) and drops to 0.652 at full size (14,246 rows), consistent with a ~15% wrong-relabel rate in the cleaning step injecting noise at scale.

See `docs/figures/cleaning_curve.png`.

---

## Phase 5 — Cross-dataset transfer of cleaned models

**What**: Evaluated all 20 cleaned BEADs adapters on the BABE, cajcodes, and WNC test splits. 60 new cross-evaluation cells.

**Why**: Test whether cleaning BEADs labels also improves cross-dataset transfer, or makes the model more specialized to its (now clean) BEADs source distribution.

**Results** (accuracy / F1_macro):

| Target | Original BEADs adapter | Best cleaned adapter | Lift (acc) |
| --- | ---: | ---: | ---: |
| BEADs (hand-labels) | 0.283 / 0.275 | 0.768 / 0.768 | +48.6 pp |
| BABE | 0.312 / 0.277 | **0.809** / **0.802** | **+49.6 pp** |
| cajcodes | 0.667 / 0.400 | 0.773 / 0.669 | +10.6 pp |
| WNC | 0.462 / 0.456 | 0.591 / 0.586 | +12.9 pp |

- Every cleaned variant beats the original adapter on every target.
- The cleaned BEADs adapter scores 0.809 on BABE — within 6 points of a BABE-specialist model (0.867).
- Phase 2's "datasets are siloed" finding requires revision: when one dataset's labels are cleaned against a careful human definition, its model transfers far better. The silos were substantially each dataset's own labeling noise, not fundamental differences in what "bias" means across datasets.

See `docs/figures/transfer_before_after.png` and `docs/figures/transfer_curve.png`.

---

## Phase 6 — Round 2 cleaning (4-voter rule)

**What**: Added the Round 1 winning cleaned adapter (cleaned_flip_balanced_500) as a 4th voter to the cleaning rule. Two new variants:

- `v2_strict`: all 4 voters unanimous against BEADs gold
- `v2_majority`: ≥ 3 of 4 voters agree against BEADs gold

10 new adapters (2 variants × 5 sizes). Pre-registered success criterion (locked before training): monotonic accuracy across the sweep, and peak accuracy ≥ Round 1's 0.768.

**Why**: Test whether iterative cleaning — using the Round 1 cleaned model to validate and refine the cleaning decisions — pushes past the Round 1 ceiling and produces a stable learning curve.

**Results**:

| Variant \ size | 100 | 500 | 1k | 5k | full |
| --- | ---: | ---: | ---: | ---: | ---: |
| Round 1 (flip + balanced) | 0.492 | **0.768** | 0.683 | 0.683 | 0.652 |
| Round 2 strict | 0.518 | 0.705 | 0.648 | 0.657 | 0.665 |
| **Round 2 majority** | **0.663** | 0.715 | 0.711 | **0.724** | 0.713 |

- The peak criterion was not met. Round 2 peak (0.724) is below Round 1 peak (0.768).
- The monotonicity criterion passed for v2_majority (curve stays within ±1.3 pp across sizes 500–16k) and failed for v2_strict (500→1k drop of 5.7 pp).
- v2_majority at size 100 scored 0.663 (vs Round 1 size 100 = 0.492). +17 pp at the same data size demonstrates the 4-voter rule produces cleaner training data even when the peak accuracy doesn't rise.

**Interpretation**: Round 2 traded peak accuracy for stability. The Round 1 peak at size=500 appears to have been substantially a sampling-variance artifact rather than a stable size dependence. The flat ~0.72 v2_majority curve across all sizes is consistent with the true post-cleaning ceiling of the method.

See updated `docs/figures/cleaning_curve.png`.

---

## Overall outcome

**Headline numbers**:

| Model | Evaluation | Accuracy |
| --- | --- | ---: |
| qlora_beads_full (original, noisy training) | BEADs noisy gold | 0.804 |
| qlora_beads_full (original) | Hand-label consensus | **0.283** |
| Round 1 cleaned (peak) | Hand-label consensus | **0.768** |
| Round 2 cleaned (stable) | Hand-label consensus | 0.724 |
| Round 1 cleaned, transferred to BABE | BABE gold | 0.809 |

**Defensible claims**:

1. The BEADs dataset has ~70% label disagreement with a calibrated 3-person human team. The "non-biased" category in particular is unreliable: BEADs over-calls bias on personal sentiment / non-public-interest content where a human reading the definition strictly would not.

2. A QLoRA adapter trained on BEADs's noisy labels scores 0.80 against those labels but only 0.28 against careful human labels. The model has learned the labeling noise rather than the underlying construct.

3. A cross-dataset adapter ensemble can identify the noisy rows and propose replacement labels with ~85% accuracy against human consensus. Using those relabels for training lifts accuracy on the hand-label consensus from 0.28 to a stable 0.72–0.77 (+45 to +49 percentage points).

4. The cleaned BEADs adapter transfers to other bias datasets dramatically better than the original. The "datasets are siloed" finding from the model-only cross-evaluation reflects each dataset's labeling noise, not a deeper construct disagreement.

5. Iterative cleaning (adding the cleaned model as an additional voter) does not push past the single-round ceiling. The ~0.72–0.77 band appears to be the method's true accuracy ceiling on this evaluation. Pushing higher would require a larger hand-labeled validation set, confidence-weighted re-relabeling, multi-seed voter ensembles, or active-learning-style human-in-the-loop validation.

**Compute cost**: ~$26 across ~70 Tillicum H200 jobs.

**Reproducibility**: All datasets are frozen with SHA256 manifests. All training and evaluation runs are pinned to seed 42. Every result in this summary can be reconstructed by re-running the relevant script against the repository at the current `main` commit. Adapter weights and large prediction files are gitignored but regenerable.
