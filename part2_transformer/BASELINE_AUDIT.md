# Part 2 baseline audit

The active two-model Colab workflow is documented in [COLAB.md](COLAB.md).
This file also preserves notes from the earlier CPU exploration, which included
the starter EEGNet as an extra reference. That CPU experiment was interrupted
before the vanilla transformer's full runs completed; its partial artifacts
are not a completed comparison. Colab trains only the Part 1 EEGNet configuration
and the unchanged vanilla transformer, as requested.

This audit runs before designing our model. `models.py`, `train.py`, `main.py`
and the example `YourModel` are preserved. The only starter change is an explicit
Part 2 download directory in `data.py` (no preprocessing changes).

## Reproduce

From `part2_transformer/`, using the existing repository environment:

```bash
../.venv/bin/python -B -u prepare_data.py
../.venv/bin/python -B -u audit_baselines.py
```

Preparation reuses the identical source recordings from Part 1 when available,
otherwise downloads from PhysioNet. Each of the 30 files is checked against the
SHA-256 registry shipped with installed MNE, even when already cached. The
data, temporary files, plotting cache, preprocessed arrays and checkpoints all
stay under Part 2 `.cache/`. The shared environment is only used to execute.
No downloads or checkpoints need to be committed.

Artifacts: `dataset_manifest.json`, `data_summary.json`, and one timestamped
`artifacts/audit_*/` directory per audit. Each audit includes source hashes,
architecture traces, epoch histories, predictions, per-seed results, aggregate
statistics and `baseline_audit.png`. The latter is an audit figure, not the
eventual custom-model submission figure.

## Data and protocol

- PhysioNet EEGBCI v1.0.0; subjects 1–10, imagery runs 6, 10 and 14.
- Both fists (T1, label 0) versus both feet (T2, label 1); rest excluded.
- 30 EDF files, 77,040,000 bytes (77.04 MB / 73.47 MiB).
- 64 electrodes, 160 Hz, 0–3 seconds inclusive: 481 samples per trial.
- Training: 360 trials from subjects 1–8 (178 fists / 182 feet).
- Test: 90 trials from subjects 9–10 (46 fists / 44 feet), 45 per subject.
- Continuous-recording FIR bandpass: 7–30 Hz, then epoching, no baseline.
- Per-channel scaling uses training statistics only. Float32 trial arrays
  occupy 55,411,200 bytes (52.84 MiB), excluding raw objects and training memory.
- Montage coordinates are finite and aligned to the 64 input channels.

The test majority-class baseline is 46/90 = 51.11%, rather than exactly 50%.
For balanced accuracy the constant-class baseline is 50%.

Normalization nuance: the starter adds `1e-6` to standard deviations in volts.
That epsilon is not negligible here; normalized training-channel standard
deviations span about 0.886–0.952 instead of one. Training-only fitting is
correct; the scale is slightly attenuated. This behavior is preserved for all
audited configurations and recorded in `data_summary.json`.

## EEGNet: what each stage learns

Input is `(batch, 64 electrodes, 481 time samples)`.

| Stage | Output excluding batch | Interpretation |
| --- | --- | --- |
| Unsqueeze | 1 × 64 × 481 | Add convolutional feature axis |
| Temporal conv + BN | 8 × 64 × 482 | Eight shared time filters, applied separately at every electrode |
| Spatial depthwise conv + BN + ELU + pool + dropout | 16 × 1 × 120 | Two learned full-scalp spatial filters per temporal filter; downsample time by four |
| Separable temporal conv + BN + ELU + pool + dropout | 16 × 1 × 15 | Temporal refinement within features, then pointwise mixing; downsample by eight |
| Flatten + linear head | 2 | Class logits from 240 features |

The first kernel covers 64 samples / 400 ms in the starter, versus 80 samples /
500 ms in the Part 1 configuration. Even-sized padded convolutions increase
length by one before pooling; the measured trace is intentional, not a shape
error. Spatial filters span all electrodes; they are learned scalp mixtures,
not hardcoded neighbourhood filters. The separable block first filters each
feature independently, then mixes features with a 1×1 convolution.

The priors are temporal locality, weight sharing across time and electrodes,
frequency-selective spatial filtering, and temporal compression. Learned time
filters can become band selective, but this audit does not assume their actual
frequency responses without inspecting fitted weights.

Before the classifier, consecutive features are spaced 32 input samples
(200 ms) apart. Their nominal temporal receptive field is 155 samples for the
starter and 171 for the Part 1 settings, including convolutions and pooling;
padding changes the amount of real data available at the edges. The flattened
classifier sees all 15 temporal positions, so EEGNet is not fully invariant to
when a feature occurs within the trial.

Two configurations are explicitly separated:

| Configuration | Temporal kernel | Dropout | Trainable parameters |
| --- | --- | --- | --- |
| Part 2 starter EEGNet | 64 | 0.25 | 2,610 |
| EEGNet with Part 1 settings | 80 | 0.50 | 2,738 |

The second uses Part 1's architectural settings in the provided Part 2 class.
It is a freshly trained model, not a loaded Part 1 checkpoint. Its result is
not a reproduction of Part 1's reported accuracy: Part 1 used a 1–40 Hz filter,
a 1–4 s crop, validation-based selection and a different training allocation.
Here all models share Part 2 preprocessing and the same fixed training budget.

An implementation detail: EEGNet's constructor probes its feature dimension
using a zero input while in training mode. That increments BN running-stat
counters and consumes dropout RNG. It is preserved, applied consistently,
and is not evidence of test leakage.

## Vanilla transformer: what attention actually sees

1. Transpose `(B,64,481)` to `(B,481,64)`.
2. A shared linear projection mixes the 64 electrodes into a 64-dimensional
   representation for each individual timestep.
3. Prepend a learned CLS token and add learned absolute positional embeddings:
   `(B,482,64)`. The model does have positional information; it lacks an
   EEG-specific locality or geometry prior.
4. Three transformer encoder layers, each with four attention heads (head
   dimension 16), a 64→128→64 GELU feed-forward network, residual connections,
   layer normalization and dropout 0.3. PyTorch's default here is post-norm.
5. Layer-normalize the final CLS representation and map 64 features to two logits.

The transformer has **135,746 trainable parameters**, about 52 times the starter
EEGNet's count. Any performance comparison therefore combines architectural
priors with a substantial capacity difference; it does not isolate the causal
effect of attention alone.

For each head: `softmax(Q Kᵀ / sqrt(16)) V`. Division by four controls dot-product
scale so softmax does not saturate merely as head dimension increases.

There are **482 tokens including CLS**, hence **232,324 attention scores per
head per layer**. At batch 64 with four heads the conceptual score tensor alone
contains 59,474,944 float32 entries, about 226.88 MiB per layer. This is not a
measurement of actual allocated memory: optimized attention can avoid
materializing the full tensor. QK and attention-times-V together require about
89.2 million multiply-accumulates per trial across three layers, excluding
projections, feed-forward blocks and backpropagation.

Its attention axis is time, not electrodes. Spatial information enters via
the linear projection; token attention plots cannot directly identify which
electrodes drive the prediction. The input is already bandpass filtered, but
the architecture has no explicit band decomposition, local temporal filtering
or scalp geometry. Learning useful temporal structure from only eight people
may be difficult. That is a hypothesis, not proof that every failure is overfit.

Electrode identity is still available through fixed input slots in the linear
projection; absence of geometry is not absence of channel information. Also,
PyTorch clones the supplied encoder layer: the three layers start with equal
values but have separate parameters, and can diverge during training.

## Training and evaluation audit

The untouched starter uses AdamW, learning rate 0.001, weight decay 0.001,
batch 64, cross-entropy, 40 epochs and NumPy/PyTorch seed setting. It returns
both final test accuracy and the peak test accuracy seen during training;
`main.py` reports the latter. Even though peak scores do not restore a best
checkpoint, selecting the reported epoch with test labels is optimistic.

Our separate audit runner preserves the models and optimizer settings but
uses a **prespecified final epoch**, identical seeds 0/1/2 for all configurations,
two CPU threads, deterministic algorithms, and test evaluation after training.
There is no search for a winning seed or configuration. Epoch progress contains
training loss and online training accuracy only. Final evaluation reports
training/test accuracy, balanced accuracy, class counts, confusion matrices,
per-subject scores, finite initial gradients and wall-clock timings.

This fixed-budget design does not need a validation split. If we later select
epochs or hyperparameters, selection must use training-subject validation data;
the test subjects must not drive those decisions. These exploratory baseline
test results should not be used to tune our eventual custom model.

Three seeds describe optimization variability on the same two test subjects.
They do not create additional independent subjects or establish population-level
significance. A one-trial change moves pooled accuracy by 1.11 percentage points.

## Interpretation boundaries

- Both fists versus both feet is bilateral; a left/right-only prior misses the
  contrast emphasized by the assessment. Hand/foot topography motivates spatial
  structure, but scalp signals are mixtures and do not localize cortical sources.
- High accuracy alone cannot demonstrate motor-cortex decoding. The visual cue /
  eye-position confound identified in Part 1 remains a concern; 7–30 Hz filtering
  and cross-subject separation do not establish its removal.
- A training/test gap can support an overfitting interpretation. Low scores on
  both sets can also reflect optimization difficulty or insufficient features.
  A near-constant predictor should be described explicitly.
- The starter `figure.png` is t-SNE of CLS embeddings, not attention weights.
  Class separation in t-SNE is not proof of physiological validity.
- The starter custom model already provides temporal patches. We have not
  changed or evaluated it in this baseline audit.

The completed GPU outcome tables are preserved in
[Experiment 00](experiments/00_baseline_audit/README.md) and the subsequent
[controlled study](experiments/01_shared_temporal_tokens/README.md).
