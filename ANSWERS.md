# EEG assessment answers

## Part 1 — Task 1: diagnose and correct

Scope: Part 1 Tasks 1–4. Part 2 is unchanged. Numerical experiments and
data-dependent interpretations are pending a successful run; code preparation
and analytical short answers do not substitute for measured results.

### Original baseline

The untouched starter is preserved in commit `85b84b8`. Intended baseline:
`python train.py --channels all --seed 0 --epochs 30`, from `part1_diagnose/`.
The local attempt failed before loading data because PyTorch is unavailable;
see `artifacts/part1/original_attempt.log`. Installing the requirements also
failed because external DNS resolution is unavailable in the terminal.
CPU dependencies were subsequently installed and all seven offline checks
now pass, including checkpoint selection; the recordings still need downloading.
Missing results must not be interpreted as chance accuracy or zero accuracy.

<!-- PART1_RESULTS_START -->
Run: seed=0, epochs=30, CPU threads=2.

| Experiment | Accuracy / output |
| --- | --- |
| Original starter (invalid protocol) | not measured (run failed) |
| Corrected all channels | not measured (run failed) |
| Corrected frontal only | not measured (run failed) |
| Mu ERD | not generated (run failed) |

Full records: `results.jsonl`; per-run logs: `artifacts/part1/attempts/`.
<!-- PART1_RESULTS_END -->

### Defect accounting

The six demonstrated defects are: wrong runs, unnecessarily long epochs,
post-cue ERD baseline, trial-level splitting, all-data normalization, and
test-based checkpoint selection. Cue-onset exclusion is an additional
confound-reduction choice, documented separately rather than counted as a
planted defect. We do not count the intentionally empty frontal group (Task 3)
as a silent planted defect.

### Fix 1 — actual movement loaded as imagery (`data.py`)

Wrong: runs 5/9/13 contain executed bilateral movements, while the task asks
for imagined movements (6/10/14).

Effect: the original number describes a different task and can be inflated by
actual movement, muscle activity and stronger movement-related signals; the
direction of the measured accuracy change is not guaranteed. Fix: select
6/10/14 without changing the T1/T2 hands/feet label mapping.

### Fix 2 — unused epoch tail discards valid trials (`data.py`)

Wrong: `Epochs` rejected the full −2 to +5 s interval before cropping it to the
network's shorter window. The unused fifth second can cross a run boundary
or the end of the recording even when the entire requested crop is available.

Effect: unnecessary, potentially condition-dependent trial loss reduces sample
size and can change class balance/accuracy in either direction. In the cached
original S001 runs there are 45 events: −2..5 s keeps 42, rejecting the final
trial of each run (two `BAD boundary`, one `TOO_SHORT`); −2..4 s keeps all 45.
Fix: end the shared epoch at +4 s, retaining pre-cue samples for ERD without
requiring an unused fifth second. This also avoids the same loss in ERD.

### Additional control — exclude cue-onset activity (`data.py`)

Wrong: the original 0–4 s crop includes the immediate response to a visual cue
whose position predicts the class. This weakens an interpretation as decoding
motor imagery, even if the prediction itself is correct.

Effect: cue-evoked visual/ocular information can raise accuracy above that
attributable to imagery; dropping it can lower accuracy, but the direction is
data-dependent. Fix: decode 1–4 s (481 samples at 160 Hz, inclusive endpoint).
This is a confound-reduction choice, not a mathematical error or a complete
ocular-artifact removal method: sustained gaze and zero-phase filter smearing
can remain. Keep the broadband filter for the later frontal-channel audit.

### Fix 3 — ERD uses post-cue activity as rest (`erd.py`)

Wrong: the baseline was 0–0.5 s after the cue although its comment called it
pre-imagery rest.

Effect: normalizing to a task-contaminated reference can attenuate, amplify or
reverse ERD/ERS; there is no universal direction. It does not change classifier
accuracy because `train.py` does not use `erd.py`. Fix: reference power to
−1.5 to −0.5 s pre-cue and reject missing/nonpositive reference power.

### Fix 4 — trial split masquerades as unseen-subject testing (`train.py`)

Wrong: `train_test_split` stratified trials by class and ignored `groups`, so
the same person's trials could occur in training, validation and testing.

Effect: usually optimistic for unseen-person generalization because a network
can exploit subject-specific signal patterns; the exact change is not fixed.
Fix: partition subject IDs first and assign all their trials together. Test
subjects are always 9–10 for the full dataset; a seeded two-subject validation
set is drawn from 1–8, leaving six training subjects. No labels select splits.
This is not an eight-subject training result: validation is held out separately.

### Fix 5 — preprocessing sees validation and test (`train.py`)

Wrong: channel means and standard deviations were fitted to all trials before
the split, leaking the distribution of held-out people into preprocessing.

Effect: evaluation is contaminated and may be optimistic through adaptation to
the held-out distribution, although unsupervised normalization leakage can
move a particular accuracy either way. Fix: split first, fit statistics using
training trials only, then apply those frozen statistics to every partition.

### Fix 6 — checkpoint chosen on test performance (`train.py`)

Wrong: the loop evaluated test accuracy every epoch and kept the checkpoint
with the largest test score; the printed validation accuracy was not used.

Effect: taking the maximum of noisy test scores is optimistically biased and
turns the test set into a selection set. Fix: choose the first checkpoint with
the highest validation accuracy, then evaluate the restored model on the test
set exactly once. Record the split IDs and validation selection criterion.
Removing this bias does not guarantee a lower score in every individual run.

### Sources and interpretation

- [MNE's EEGBCI reference example](https://mne.tools/stable/auto_examples/decoding/decoding_csp_eeg.html)
  identifies runs 6, 10 and 14 as hands-versus-feet imagery and explains excluding
  the first second to avoid classifying cue-evoked responses.
- [PhysioNet dataset documentation](https://physionet.org/content/eegmmidb/1.0.0/)
  describes actual versus imagined tasks and class-dependent top/bottom cues.

Changes above have directional explanations, not invented per-fix accuracy
effects. Several corrections need not monotonically lower accuracy on one seed.

## Part 1 — Task 2: mu ERD

<!-- PART1_ERD_START -->
**Awaiting real EEG data and `erd_mu.png`; no plot observations are claimed.**
The comparison will use mu-band power (8–13 Hz) relative to −1.5 to −0.5 s
pre-cue rest, with equal subject weights. The anatomical expectation is a
bilateral hand-related decrease near C3/C4 and a foot-related effect nearer Cz,
rather than a left-versus-right hand contrast. A weak Cz effect would be
plausible because medial foot sources lie in the interhemispheric fissure,
their orientation/depth limits scalp sensitivity, and subject variability can
attenuate the group average; it would not by itself establish a processing bug.
<!-- PART1_ERD_END -->

## Part 1 — Task 3: frontal-channel control

The prespecified subset is `Fp1, Fpz, Fp2, AF3, AFz, AF4, F7, F3, Fz, F4, F8`.
Use the same seed, subject partitions, normalization procedure, epoch window,
training budget and validation criterion for both channel configurations.
Spatial kernel size and consequently parameter count change with channel count;
the result is a control experiment, not a perfectly capacity-matched comparison.

<!-- PART1_FRONTAL_START -->
**Awaiting measured all-channel and frontal-only accuracies; no comparison is claimed.**
If frontal-only accuracy remains comparable and clearly above an appropriate
null, it would be consistent with a frontal cue-correlated shortcut, though
frontal neural activity and volume conduction are alternative explanations.
Vertical eye movements and gaze shifts toward the top (fists) or bottom (feet)
target can create electrooculographic (EOG) potentials strongest frontally.
Lower frontal accuracy would weaken that particular observation, but neither
result alone proves that the full model exclusively uses motor-cortex activity.
<!-- PART1_FRONTAL_END -->

## Part 1 — Task 4: short answers

1. **58% of 45 trials:** this is approximately 26/45; a prespecified one-sided
   exact binomial test against 0.5 gives **p = 0.18565**, so it is not persuasive
   evidence above chance at 5%. This assumes independent trials and a balanced
   null; with imbalance, temporal dependence or model selection, use an
   appropriate within-subject/run restricted permutation test rerunning the
   whole training/selection procedure.
2. **Depthwise spatial layer:** for each learned temporal filter, it learns
   `D` weighted combinations across electrodes—frequency-specific spatial
   filters analogous in purpose to CSP. These are discriminative scalp weights,
   not direct cortical source locations or proof that artifacts were excluded.
3. **Remove the cue shortcut:** use an identical centrally presented visual cue
   with its imagery instruction delivered through a counterbalanced auditory
   mapping, and monitor fixation with eye tracking/EOG. This removes the
   class-dependent top/bottom target and lets us verify that class-dependent
   gaze is absent instead of assuming cropping removed it.

Statistical calculation: `scipy.stats.binomtest(26, 45, 0.5, alternative="greater")`.
Sources: [SciPy binomial test](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binomtest.html),
[EEGNet paper](https://arxiv.org/abs/1611.08024),
[MNE ocular-artifact tutorial](https://mne.tools/stable/auto_tutorials/preprocessing/35_artifact_correction_regression.html).

## Baseline to carry into Part 2

EEGNet `F1=8, D=2, F2=16, kern=80, drop=0.5`; continuous 1–40 Hz filtering;
imagery runs 6/10/14; 1–4 s crops; 64 channels; training-only channel z-scoring;
AdamW lr=0.001, weight decay=0.001, batch size=64, 30 epochs, seed=0.
Checkpoint selection uses validation subjects; test subjects 9–10 are opened
after selection. The original execution-data/trial-split accuracy and corrected
imagery-data/subject-split accuracy estimate different things, so their numerical
difference must not be described as a controlled improvement in model quality.
Part 2 must harmonize its data window, filter, model settings and selection
protocol with this reference before claiming a fair comparison.
