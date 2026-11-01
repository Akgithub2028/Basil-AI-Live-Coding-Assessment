# One bet: shared local temporal tokenization

Written before implementing or evaluating this model. No custom-model results
exist when this plan is written. The baseline test results have already been
seen, so this is an exploratory follow-up, not an untouched confirmatory test.

## What the imported T4 audit establishes

The supplied `colab_20261101T083535270691Z.zip` completed successfully and exact
repeat checks passed. Vanilla: train 99.07%, test 50.74% ± 4.48 percentage points;
EEGNet with Part 1 architecture: train 83.15%, test 57.04% ± 2.28 points.
The mean train/test gaps are about 48.33 and 26.11 points respectively. This
supports severe vanilla overfitting on this split; it does not identify its
sole cause. Three seeds are a weak estimate of optimization variability, and
two test subjects are a weak basis for population-level claims. Neither model
is proven to use physiological motor signals rather than cue-related features.

## Hypothesis (fixed before coding)

Replacing per-sample tokens with overlapping tokens built from short temporal
filters shared across electrodes will reduce attention cost and encourage
local EEG features, improving held-out-subject balanced accuracy and seed
stability without changing the attention backbone or total parameter count.

This is one architectural bet: **factorized local temporal tokenization**.
We do not add band selection, geometry, augmentation, new losses, extra
regularization or tune model size at the same time.

## Mechanism

At 160 Hz, apply eight learned 32-sample (200 ms) filters, stride 16 (100 ms),
identically to every electrode. This yields 29 overlapping windows. For each
window, linearly mix 64 electrodes × 8 filter outputs into a 64-dimensional
token. Add CLS and learned temporal positions, then use the same three-layer,
four-head, width-64 transformer, dropout 0.3 and classifier as vanilla.

The shared temporal basis is similar to EEGNet's first stage: the network need
not relearn each elementary temporal filter independently at every electrode.
The spatial projection can use all electrode slots; no left/right assumption
is imposed. The task is bilateral fists versus feet, so a left/right-only prior
would be inappropriate. Temporal filters may learn useful frequency responses,
but neither band selectivity nor motor localization is guaranteed.

The stem has no extra nonlinearity. Its factorization restricts the temporal
basis of the learned patch projection. This is a stronger, explicit sharing
assumption than the starter's unrestricted 64×32-to-64 patch convolution.

For the prescribed 64×481 input, the model has **135,746 parameters**, exactly
the vanilla count. The additional stem/projection weights happen to offset
the shorter positional embedding. This equality follows from the fixed
eight-filter configuration, not a post-result size search. Equal counts do
not imply equal effective capacity; the difference in parameter structure is
the proposed prior.

Sequence lengths including CLS: 482 → 30. Attention scores per head per layer:
232,324 → 900, about **258× fewer**. Actual runtime and allocated memory are
measured separately; total compute is not reduced by that same factor.

## Four configurations, one hypothesis and one ablation

1. EEGNet with Part 1 settings (kernel 80, dropout 0.5): trusted convolutional reference.
2. Unchanged VanillaEEGTransformer: attention without the proposed prior.
3. `YourModel`: the factorized local temporal stem and unchanged attention backbone.
4. `ScatteredTimeControl`: the identical stem and backbone, but a fixed random
   permutation scatters the first 480 time samples before tokenization. The
   final unused sample remains in place. The permutation is the same across
   electrodes, trials and seeds; all included values and instantaneous scalp
   vectors are preserved. Parameters, token counts and initial learned tensors
   exactly match the local model for each seed.

Local versus vanilla tests the practical package of temporal sharing and
compression, at equal parameter count. Local versus scattered tests ordered,
contiguous tokenization versus arbitrary temporal grouping, at equal parameters
and attention cost. The latter also changes the meaning of positional order;
it is not a proof of a particular learned frequency mechanism.

## Locked protocol

- Five seeds: 0, 1, 2, 3, 4. No seed selection.
- 40 epochs, fixed final checkpoint, AdamW lr=0.001, weight decay=0.001.
- Effective batch 64; transformer microbatch 16, EEGNet full batches.
- FP16 AMP training, FP32 evaluation, strict determinism, identical minibatch
  order across configurations for each seed. Repeat seed 0 for all models.
- Development: train subjects 1–6, evaluate subjects 7–8. Refit scaling on 1–6
  only. Evaluate development loss/accuracy every epoch to diagnose learning.
- Final: refit scaling on subjects 1–8, retrain all four configurations, then
  evaluate subjects 9–10 at the final epoch only. Same fixed settings regardless
  of development outcome. Do not use final scores to choose models or epochs.
- Same 7–30 Hz filter, 0–3 s epochs and normalization epsilon as Part 2.
- Both stages run automatically with these settings; there is no sweep or
  automatic selection of a winner. Earlier three-seed audit is historical
  context, not pooled with the new five-seed comparison.

## What counts as success?

**A successful research exercise:** the experiment runs reproducibly, isolates
the proposed change, measures the prespecified outcomes and reaches an honest
conclusion. A rejected hypothesis can satisfy the assessment completely.

**Mechanical prediction:** model shape, exact parameter equality and theoretical
attention-score reduction pass; timing/memory demonstrate their practical cost.
These alone do not establish better EEG decoding.

**Primary empirical endpoints:** mean and population SD of held-out-subject
balanced accuracy across five seeds. Support for the full stability/locality
hypothesis requires all of:

- local mean balanced accuracy > vanilla and > scattered control;
- local seed SD <= vanilla seed SD;
- local mean balanced accuracy > 50%;
- no local run predicts either class on fewer than 10% of evaluation trials.

These are descriptive pilot criteria, not a significance test. Report exact
differences, all seeds and both subjects, even if a gate fails. A smaller SD
from a collapsed constant classifier is not success. Outcomes are reported
separately for development and final stages. A single-point gain is weak
evidence, not a discovery. Five seeds do not create five independent cohorts.

**Secondary endpoints:** ordinary accuracy, best/worst seed, train/evaluation
gap, cross-entropy, class collapse, per-subject results, time, GPU memory and
the local/scattered paired-seed difference. Do not redefine the primary endpoint
after seeing which metric improves.

## Useful negative outcomes

- Local and scattered both help equally: compression/factorization may help,
  but this control does not support an additional benefit from temporal locality.
- Local trains well but still fails on new subjects: locality alone does not
  solve subject shift or cue confounding.
- Local does not fit training: excessive compression, the temporal basis or
  optimization can be limiting; the exact cause needs another experiment.
- Seed SD drops but prediction collapses: the model is consistently wrong,
  which fails the stability claim.
- Accuracy stays similar while attention work and runtime fall: useful efficiency
  evidence, with the generalization hypothesis unsupported or mixed.

Inspect learned temporal spectra and attention summaries as diagnostics, not
proof of physiology. No t-SNE plot or attention heatmap establishes motor-cortex
decoding or causal feature importance.

## Other ideas considered, not added to this experiment

- Local log-power tokens: physiologically motivated, but impose a different
  phase-insensitivity assumption and need their own controlled study.
- Electrode geometry attention: relevant to this bilateral task, but changes
  the attention axis and spatial prior together if introduced from this baseline.
- More dropout or a smaller transformer: useful scarcity controls, but less
  direct tests of EEG temporal structure. No hyperparameter hunt is planned.

Relevant precedents, not performance guarantees for this dataset:
[Schirrmeister et al., 2017](https://pmc.ncbi.nlm.nih.gov/articles/PMC5655781/)
and [EEG Conformer](https://bingchuanliu.github.io/assets/pdf/song_tnsre.pdf).
