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
