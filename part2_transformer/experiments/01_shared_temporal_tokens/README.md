# Experiment 01 — shared local temporal tokenization

**Outcome: mixed evidence; the final-stage stability prediction is unsupported.**
The experiment completed, was reproducible, and produced useful measurements.
It should not be described as either a complete success or a complete failure.

Original archive: [study_20261101T091133698074Z.zip](study_20261101T091133698074Z.zip).
Original report: [EXPERIMENT_REPORT.md](results/EXPERIMENT_REPORT.md).
Hypothesis snapshot: [EXPERIMENT_PLAN.md](results/EXPERIMENT_PLAN.md).
Final figure: [figure.png](results/figure.png).
Integrity/source provenance: [PROVENANCE.json](PROVENANCE.json).

## Design

Eight learned 32-sample temporal filters are shared across all 64 electrodes,
at stride 16. A spatial projection produces 29 local tokens, then CLS and the
unchanged attention backbone. Proposed model, vanilla and scattered-time
control each have 135,746 parameters. The control scatters the first 480 samples
with a fixed permutation, preserving the included sample values and scalp
vectors while changing contiguous temporal grouping and positional meaning.

Five seeds, fixed 40 epochs, matched minibatch orders. Development trains 1–6
and evaluates 7–8; final retrains 1–8 and evaluates 9–10. Scaling is refitted
using training subjects only. Exact repeated-seed checks pass for all four
models in both stages. The final checkpoint is specified before training.

## What happened

| Stage | EEGNet BA mean ± SD | Vanilla BA mean ± SD | Local BA mean ± SD | Scattered BA mean ± SD |
| --- | ---: | ---: | ---: | ---: |
| Development, subjects 7–8 | 60.67% ± 5.14 pp | 51.78% ± 3.95 pp | 58.22% ± 1.81 pp | 56.44% ± 4.30 pp |
| Final, subjects 9–10 | 54.95% ± 3.15 pp | 50.65% ± 3.06 pp | 54.02% ± 5.32 pp | 51.32% ± 3.01 pp |

BA is balanced accuracy. SD is optimization variability on the same people,
not uncertainty over independent subject cohorts. Development met all original
descriptive support criteria. The final stage failed the seed-stability gate:
local SD was 5.32 pp, above vanilla's 3.06 pp. Other final gates passed.

| Final metric | EEGNet | Vanilla | Local | Scattered |
| --- | ---: | ---: | ---: | ---: |
| Mean ordinary accuracy | 54.67% | 50.44% | 54.00% | 51.33% |
| Best / worst ordinary accuracy | 58.89% / 50.00% | 55.56% / 45.56% | 62.22% / 47.78% | 55.56% / 47.78% |
| Mean train accuracy | 82.39% | 99.56% | 99.89% | 99.28% |
| Mean training time | 7.77 s | 28.31 s | 18.32 s | 18.55 s |
| Peak allocated GPU memory | 213.0 MiB | 649.5 MiB | 128.9 MiB | 128.9 MiB |

Local beat vanilla's balanced accuracy in four of five paired seeds, averaging
+3.37 pp, and beat scattered in three of five, averaging +2.70 pp. These are
small exploratory differences, not proof of a reliable population advantage.
Local did not beat EEGNet on mean accuracy, and matching EEGNet was never a
pass condition in the assessment.

The structural efficiency prediction held: scores per attention head/layer fell
from 232,324 to 900 (258× fewer). Practical improvements were smaller and measured:
about 35% less training time and 80% lower peak allocated GPU memory than vanilla.
The stem/projection and other operations still cost time; fewer attention scores
do not imply an equally large end-to-end speedup.

## What the evidence supports

- Temporal locality helped mean scores modestly on these splits, at matched
  parameter count. The control provides limited supporting evidence, with
  substantial seed variation.
- Attention cost and measured memory/runtime decreased.
- The prior did **not** prevent overfitting: local train accuracy reached 99.89%,
  while final mean balanced accuracy was 54.02%.
- The stability improvement seen during development did **not** transfer to
  subjects 9–10. This limitation must stay in the write-up.
- Both attention variants remain overconfident. Mean final cross-entropy is
  about 2.09 for local versus 2.17 for vanilla; both exceed 0.693 for uniform
  predictions, despite their different accuracy counts.

This does not show all locality priors fail, or that no EEG information exists.
It shows this particular raw-waveform temporal prior was insufficient to deliver
the full generalization-and-stability hypothesis under this protocol.

The task asks for one defensible idea and honest evaluation; it explicitly
allows negative results. This completed experiment already supplies a valid
research narrative. Further work must preserve it rather than replace it with
only whichever run eventually scores highest.

The baseline test scores were already known before this hypothesis was chosen.
Repeated use of subjects 9–10 makes follow-ups exploratory. Claims about a
physiological signal or robust cross-subject improvement need more independent
evidence. These results do not remove the cue/ocular confound from Part 1.
