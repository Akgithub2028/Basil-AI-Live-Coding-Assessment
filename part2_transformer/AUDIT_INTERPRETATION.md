# Interpreting the uploaded T4 audit

Source: [original baseline ZIP](experiments/00_baseline_audit/colab_20261101T083535270691Z.zip).
Its metadata reports completion on
Tesla T4, strict determinism and exact repeated-seed matches. Baseline source
hashes match the unmodified `models.py` currently used for the new study.

| Observation | EEGNet with Part 1 settings | Vanilla transformer |
| --- | ---: | ---: |
| Mean train accuracy | 83.15% | 99.07% |
| Mean held-out accuracy | 57.04% | 50.74% |
| Mean train/test gap | 26.11 pp | 48.33 pp |
| Seed SD, held-out accuracy | 2.28 pp | 4.48 pp |
| Mean training time, 40 epochs | 7.86 s | 28.46 s |
| Approximate test cross-entropy from saved probabilities | 0.719 | 1.812 |
| Confidence on incorrect predictions, range across seeds | 65–67% | 89–94% |

The approximate cross-entropy uses clipped saved probabilities (floor 1e-12),
not original logits. A uniform 50/50 predictor has cross-entropy ln(2) ≈ 0.693.
Thus vanilla's problem includes confidently wrong predictions, not just a
slightly lower count of correct predictions. Both models show a train/test gap;
EEGNet's probability quality is also imperfect despite its higher accuracy.

The dramatic training-loss drop after roughly epoch 20 does not demonstrate
better generalization, because that plot contains training loss only. The new
development-stage evaluation-loss curves help distinguish fitting training
data from improving held-out performance. We will not select a test epoch.

The larger seed spread is descriptive evidence from only three initializations.
It does not prove a precise variance ratio or a population-level difference.
Changing a seed changes initialization/dropout/order; exact reruns of one seed
being reproducible is fully compatible with variation between different seeds.

The task has only two test people. A 10-point range is nine of the 90 trials.
Earlier baseline results already expose test performance, so the follow-up
cannot honestly be described as a completely untouched confirmatory experiment.
Raw calculations and archive SHA-256 are in
[imported_gpu_audit_analysis.json](experiments/00_baseline_audit/results/imported_gpu_audit_analysis.json).

## Why the chosen follow-up is a useful bet

The audit supports overfitting, but does not prove which architectural weakness
caused it. We test one proposed remedy: local temporal weight sharing before
attention. This is directly related to what EEGNet supplies cheaply, while
keeping the attention component visible and unchanged.

For an electrode c and window n, the temporal bank produces:

`u[c,f,n] = sum_tau h[f,tau] * x[c,16*n+tau]`, for tau=0..31.

The same eight h filters are used for every electrode. A learned projection
then mixes the 64×8 features into the window token:

`token[n,d] = bias[d] + sum_c,f P[d,c,f] * u[c,f,n]`.

The factorization constrains local patch weights to use a common eight-filter
temporal basis. It retains a learned spatial mixture across all electrodes;
it imposes neither a left/right contrast nor a claim of source localization.
There is no extra nonlinearity, extra dropout, augmentation or loss change.

| Parameter group | Vanilla | Proposed local stem |
| --- | ---: | ---: |
| Shared temporal filters | 0 | 256 |
| Projection | 4,160 | 32,832 |
| Positional embeddings | 30,848 | 1,920 |
| CLS | 64 | 64 |
| Encoder | 100,416 | 100,416 |
| Final norm | 128 | 128 |
| Classifier | 130 | 130 |
| **Total** | **135,746** | **135,746** |

The equality is useful for this comparison. It does not make model capacity
identical: arranging parameters differently is the inductive bias. The
scattered-time control has identical learned tensors at initialization and
identical token/parameter counts, but groups time samples arbitrarily rather
than contiguously. It helps distinguish a benefit from ordered local structure
from a benefit shared by both short-token variants.

This is not a claim that shorter attention automatically prevents overfitting.
A small sequence can still memorize training data. The matched control,
training/evaluation gaps, cross-entropy and class-collapse diagnostics make
that possible failure visible.

## What the assessment wants from the outcome

A valid research exercise can reject its hypothesis. We earn a defensible
conclusion by stating the mechanism first, changing one coherent component,
using fair controls, reproducing all seeds and explaining the evidence.

The full empirical hypothesis is supported only when the plan's generalization,
stability and non-collapse checks hold. Efficiency alone supports a narrower
claim. Beating EEGNet is optional; measuring and explaining the comparison is
required. The task is not to invent a never-before-seen architecture or to
keep trying ideas until test accuracy happens to rise.

For the live defence, be ready to derive the 29 windows plus CLS, 482² versus
30² attention scores, the parameter table, why filters are shared across
electrodes, what the scattered control changes, and why seed variability is
different from nondeterministic execution. With 80 subjects, the data-scarcity
motivation may weaken, while the compute advantage remains; the actual
generalization effect would need another experiment.
