# EEG assessment answers

## Part 1 — Task 1: diagnose and correct

Part 1 Tasks 1–4 are complete and were run on CPU, using all ten prescribed
subjects. Part 2 is unchanged. All three training runs used seed 0, 30 epochs
and two CPU threads; the observed scores are reported without further tuning.

### Original baseline

The untouched starter is preserved in commit `85b84b8`. Intended baseline:
`python train.py --channels all --seed 0 --epochs 30`, from `part1_diagnose/`.
The initial attempt failed because PyTorch was missing and terminal DNS was
unavailable; that failure log is retained in `artifacts/original_attempt.log`.
After the CPU dependencies and EEG files became available, the full original
and corrected experiments completed successfully. Nine offline checks pass.
Earlier failed records in `results.jsonl` have `status: "failed"` and null
accuracy; the four records from the successful attempt have `status: "ok"`.

<!-- PART1_RESULTS_START -->
Run: seed=0, epochs=30, CPU threads=2.

| Experiment | Accuracy / output |
| --- | --- |
| Original starter (invalid protocol) | 66.67% |
| Corrected all channels | 53.33% |
| Corrected frontal only | 48.89% |
| Mu ERD | `erd_mu.png` generated |

Full records: `results.jsonl`; per-run logs: `artifacts/attempts/`.
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
Evidence: `artifacts/epoch_boundary_audit.json`.

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
After a shared early post-cue power increase, hand imagery shows stronger sustained mu ERD than feet at C3 (−13.2% versus +1.0%) and C4 (−7.8% versus −3.9%) over 1–4 s. That bilateral central pattern broadly matches the lateral hand areas of motor cortex and this both-hands task, rather than a left-versus-right hand contrast. Cz shows small, similar decreases for hands (−3.1%) and feet (−3.6%), so this plot does not show a strong foot-specific vertex effect. The medial foot representation lies in the interhemispheric fissure, where source depth/orientation can reduce scalp sensitivity, while individual anatomy and subject averaging can further weaken the Cz contrast; these descriptive curves alone do not establish significance.
<!-- PART1_ERD_END -->

## Part 1 — Task 3: frontal-channel control

The prespecified subset is `Fp1, Fpz, Fp2, AF3, AFz, AF4, F7, F3, Fz, F4, F8`.
Both configurations used the same seed, subject partitions, normalization
procedure, epoch window, training budget and validation criterion.
Spatial kernel size and consequently parameter count change with channel count;
the result is a control experiment, not a perfectly capacity-matched comparison.

<!-- PART1_FRONTAL_START -->
On the same 90 held-out trials from subjects 9–10, all-channel accuracy was 53.33% (48/90) and frontal-only accuracy was 48.89% (44/90), a 4.44 percentage-point decrease. Both are near the held-out majority-class baseline of 51.11% (46/90), so this run provides no convincing evidence of useful cross-subject decoding or successful frontal cheating. A plausible non-brain shortcut remains vertical EOG from looking toward the top target for fists and bottom target for feet, which produces strong frontal potentials. This negative frontal control neither demonstrates EOG exploitation nor proves that the all-channel model is artifact-free, and removing the first second cannot guarantee removal of sustained gaze effects.
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
3. **Prevent the suspected cue shortcut:** use an identical centrally presented visual cue
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

## Experiment record and limits

- Successful attempt: `artifacts/attempts/20261101T071451704509Z/`;
  complete records are also in `artifacts/latest_run.json`.
- Original: 420 trials total, 84 mixed-subject test trials, 56/84 correct,
  test-selected epoch 24 (zero-based). This is the flawed before score.
- Corrected all-channel: 450 trials; train subjects 1/2/3/4/5/7 (270 trials),
  validation 6/8 (90), test 9/10 (90); validation-selected epoch 3,
  validation accuracy 57.78%, test accuracy 53.33%.
- Frontal: identical splits and budget; validation-selected epoch 12,
  validation accuracy 57.78%, test accuracy 48.89%.
- Test class counts: 46 hands and 44 feet. A single seed on two held-out
  people is a limited estimate, not evidence of population-wide performance.
- ERD uses 45 imagery trials per subject and equal subject weights.
- `artifacts/environment.txt` records the installed CPU dependencies; source
  SHA-256 hashes and run settings are in the result records. Commands in those
  historical records retain the paths used before the directory reorganization;
  their `log` fields point to the organized current layout.

Separate bug-fix commits (history is unsquashed): wrong runs `13bfc5f`,
unused epoch tail `0cb352f`, ERD reference `e7e8745`, subject split `119e5ab`,
normalization `ce51b59`, checkpoint selection `0a78619`. Additional cue-onset
control: `1ac0d25`. The aggregate before/after difference does not isolate
individual bug effects because the data, splits, retained trials and window
change together; we did not tune the corrected pipeline against the test score.

The EEGNet reference configuration and measured result are packaged in
`artifacts/baseline.json` for Part 2. The reference run did not save weights;
weight export is now implemented for future runs, but no extra training was
performed solely to produce a checkpoint.

A later layout-verification attempt hit MNE's interactive cache-path prompt;
its failure is retained in `artifacts/latest_attempt.json`. The runner now
synchronizes its project-local MNE configuration, a no-download regression
check covers stale paths, and failed attempts preserve completed reports.
That filesystem failure does not invalidate the completed experiment records
in `artifacts/latest_run.json`; no training rerun was performed after the
user requested preserving the correct results.
