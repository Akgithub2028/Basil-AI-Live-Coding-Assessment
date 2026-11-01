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

### Sources and interpretation

- [MNE's EEGBCI reference example](https://mne.tools/stable/auto_examples/decoding/decoding_csp_eeg.html)
  identifies runs 6, 10 and 14 as hands-versus-feet imagery and explains excluding
  the first second to avoid classifying cue-evoked responses.
- [PhysioNet dataset documentation](https://physionet.org/content/eegmmidb/1.0.0/)
  describes actual versus imagined tasks and class-dependent top/bottom cues.

Changes below have directional explanations, not invented per-fix accuracy
effects. Several corrections need not monotonically lower accuracy on one seed.
