# EEG assessment answers

## Part 1 — Task 1: diagnose and correct

Scope: Task 1 only. ERD interpretation, the frontal-channel experiment and
Tasks 2–4 remain for subsequent work. Part 2 is unchanged.

### Original baseline

The untouched starter is preserved in commit `85b84b8`. Intended baseline:
`python train.py --channels all --seed 0 --epochs 30`, from `part1_diagnose/`.
The local attempt failed before loading data because PyTorch is unavailable;
see `artifacts/part1/original_attempt.log`. Installing the requirements also
failed because external DNS resolution is unavailable in the terminal.
**Original accuracy: not measured. Corrected accuracy: not measured.**
Missing results must not be interpreted as chance accuracy or zero accuracy.

### Defect accounting

The PDF states six defects but supplies no answer key. The audit identifies
five unambiguous implementation/protocol errors plus cue-onset inclusion as
the sixth scientific correction. The latter reduces an acknowledged confound;
it is not proof that all remaining features are motor imagery. We do not count
the intentionally empty frontal group (Task 3) as a silent planted defect.

### Fix 1 — actual movement loaded as imagery (`data.py`)

Wrong: runs 5/9/13 contain executed bilateral movements, while the task asks
for imagined movements (6/10/14).

Effect: the original number describes a different task and can be inflated by
actual movement, muscle activity and stronger movement-related signals; the
direction of the measured accuracy change is not guaranteed. Fix: select
6/10/14 without changing the T1/T2 hands/feet label mapping.

### Fix 2 — cue-onset activity included in the decoding window (`data.py`)

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

### Sources and interpretation

### Fix 5 — preprocessing sees validation and test (`train.py`)

Wrong: channel means and standard deviations were fitted to all trials before
the split, leaking the distribution of held-out people into preprocessing.

Effect: evaluation is contaminated and may be optimistic through adaptation to
the held-out distribution, although unsupervised normalization leakage can
move a particular accuracy either way. Fix: split first, fit statistics using
training trials only, then apply those frozen statistics to every partition.

- [MNE's EEGBCI reference example](https://mne.tools/stable/auto_examples/decoding/decoding_csp_eeg.html)
  identifies runs 6, 10 and 14 as hands-versus-feet imagery and explains excluding
  the first second to avoid classifying cue-evoked responses.
- [PhysioNet dataset documentation](https://physionet.org/content/eegmmidb/1.0.0/)
  describes actual versus imagined tasks and class-dependent top/bottom cues.

Changes below have directional explanations, not invented per-fix accuracy
effects. Several corrections need not monotonically lower accuracy on one seed.
