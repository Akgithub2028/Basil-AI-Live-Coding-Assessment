# EEG assessment answers

## Part 1 — Diagnose & Trust

### Task 1 — defects, fixes and measured results

| Defect | Effect on the reported result | Fix |
| --- | --- | --- |
| Execution runs 5/9/13 loaded as imagery | Measures the wrong task; actual movement/muscle activity may inflate scores, but direction is not guaranteed | Load imagery runs 6/10/14 |
| Epoch requires an unused fifth post-cue second | Rejects valid shorter crops near run ends; changes sample size/balance and can move accuracy either way | End epoch at +4 s; retain the required pre-cue interval |
| ERD baseline is post-cue 0–0.5 s | Task-contaminated reference can attenuate or reverse ERD; classifier accuracy is unaffected | Use −1.5 to −0.5 s pre-cue power |
| Trial-level split ignores subjects | Usually optimistic for unseen-person generalization through subject-specific information | Split whole subjects; test 9–10, validation drawn from 1–8 |
| Normalization fitted before splitting | Leaks held-out distributions; can change accuracy either way | Fit channel statistics on training trials only |
| Best checkpoint selected by test accuracy | Selecting a noisy test maximum is optimistically biased | Select on validation, then evaluate test once |

Additional confound control: decode 1–4 s to exclude cue onset, rather than 0–4 s.
This may reduce cue-evoked information but does not guarantee removal of sustained
gaze/EOG. The empty frontal group is completed for Task 3, not counted as a silent bug.

CPU, seed 0, 30 epochs: **original 66.67% (56/84); corrected all-channel 53.33%
(48/90); frontal-only 48.89% (44/90)**. Corrected training uses six subjects,
validation two, and test subjects 9–10. The original and corrected protocols
estimate different things; their difference is not an isolated effect of any one fix.

Each demonstrated defect has its own existing unsquashed commit. Full reasoning,
source references, commit IDs and successful/failed attempt records are in
[Part 1 answers](part1_diagnose/ANSWERS.md) and
[results.jsonl](part1_diagnose/results.jsonl).

### Task 2 — mu ERD

After a shared early post-cue power increase, hand imagery shows stronger sustained mu ERD than feet at C3 (−13.2% versus +1.0%) and C4 (−7.8% versus −3.9%) over 1–4 s. That bilateral central pattern broadly matches the lateral hand areas of motor cortex and this both-hands task, rather than a left-versus-right hand contrast. Cz shows small, similar decreases for hands (−3.1%) and feet (−3.6%), so this plot does not show a strong foot-specific vertex effect. The medial foot representation lies in the interhemispheric fissure, where source depth/orientation can reduce scalp sensitivity, while individual anatomy and subject averaging can further weaken the Cz contrast; these descriptive curves alone do not establish significance.

Figure: [erd_mu.png](part1_diagnose/erd_mu.png).

### Task 3 — what signal is the model using?

Frontal subset: Fp1, Fpz, Fp2, AF3, AFz, AF4, F7, F3, Fz, F4, F8.

On the same 90 held-out trials from subjects 9–10, all-channel accuracy was 53.33% (48/90) and frontal-only accuracy was 48.89% (44/90), a 4.44 percentage-point decrease. Both are near the held-out majority-class baseline of 51.11% (46/90), so this run provides no convincing evidence of useful cross-subject decoding or successful frontal cheating. A plausible non-brain shortcut remains vertical EOG from looking toward the top target for fists and bottom target for feet, which produces strong frontal potentials. This negative frontal control neither demonstrates EOG exploitation nor proves that the all-channel model is artifact-free, and removing the first second cannot guarantee removal of sustained gaze effects.

### Task 4 — short answers

1. **58% of 45 trials:** approximately 26/45 gives one-sided exact binomial
   p=0.18565 against 0.5, so this is not persuasive evidence above chance at 5%.
   That calculation assumes independent trials and the correct null; with imbalance,
   dependence or model selection, use an appropriate restricted permutation test
   that repeats the entire analysis.
2. **EEGNet depthwise layer:** for each temporal filter, it learns D weighted
   combinations across electrodes—frequency-specific spatial filters, analogous
   in purpose to CSP. Discriminative scalp weights are not direct cortical locations.
3. **Stop the cue shortcut:** use an identical central visual cue with the imagery
   instruction given through a counterbalanced auditory mapping, and monitor fixation
   with EOG/eye tracking. This removes the class-dependent top/bottom target and
   verifies rather than assumes that gaze is not class dependent.

## Part 2 — Build & Defend

**Hypothesis, written before implementation:** Shared local temporal filters
before attention should reduce computational cost and improve held-out-subject
balanced accuracy and seed stability compared with per-sample tokens. The original
[hypothesis and protocol](part2_transformer/EXPERIMENT_PLAN.md) are preserved and
hashed in the Colab study metadata.

**What changed and why:** `YourModel` applies eight learned 32-sample (200 ms)
filters shared across electrodes, stride 16 (100 ms), followed by a learned
spatial projection and the unchanged attention backbone. This supplies reusable
local EEG features before attention. There are 29 windows plus CLS, versus 482
vanilla tokens: 900 versus 232,324 attention entries per head/layer. Both models
have 135,746 parameters. A parameter/token-matched scattered-time control tests
ordered contiguous grouping. All electrode identities remain available; the
bilateral fists/feet task does not motivate a left/right-only prior.

**What happened:** Fixed 40 epochs, five seeds (0–4), train subjects 1–8, test
9–10. Scaling fits training subjects only; no test-based checkpoint selection.
EEGNet uses the validated Part 1 architecture (kernel 80, dropout 0.5), freshly
trained under Part 2's common 7–30 Hz, 0–3 s preprocessing.

| Model | Ordinary accuracy best / worst / mean | Balanced accuracy mean ± seed SD |
| --- | ---: | ---: |
| EEGNet, Part 1 architecture | 58.89% / 50.00% / 54.67% | 54.95% ± 3.15 pp |
| Unchanged vanilla transformer | 55.56% / 45.56% / 50.44% | 50.65% ± 3.06 pp |
| Our shared local temporal model | **62.22% / 47.78% / 54.00%** | **54.02% ± 5.32 pp** |
| Scattered-time control | 55.56% / 47.78% / 51.33% | 51.32% ± 3.01 pp |

**Interpretation:** The result is mixed: mean balanced accuracy improved by
3.37 pp over vanilla and 2.70 pp over the control, but final seed variability
increased and training accuracy reached 99.89%. The development-stage stability
improvement did not transfer to the final people. The prior did not solve
overfitting; its stability prediction is unsupported. It did reduce training
time (28.31→18.32 s) and peak allocated GPU memory (649.5→128.9 MiB). Exact
repeated-seed checks passed for every model in both stages. These are exploratory
observations on two people, not a significance result or proof that motor signals
rather than cue/ocular confounds caused the predictions.

**Figure:** [figure.png](part2_transformer/figure.png) compares training losses,
held-out balanced accuracy, train/evaluation gaps and runtime, alongside learned
temporal filter responses; it is not an attention map or cortical localization.

**Reproduce:** From `part2_transformer/`, run `python main.py`, or use the
executed controlled-experiment Colab notebook. All seeds, both stages, original
ZIPs and source provenance are retained in the
[experiment history](part2_transformer/experiments/README.md). More detail is in
[Part 2 answers](part2_transformer/ANSWERS.md).
