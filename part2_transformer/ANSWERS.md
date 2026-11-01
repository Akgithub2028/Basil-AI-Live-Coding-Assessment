# Part 2 — shared local temporal tokenization

**Hypothesis (written before implementation):** Replacing per-sample tokens with
overlapping tokens built from short temporal filters shared across electrodes
will reduce attention cost and encourage local EEG features, improving held-out-
subject balanced accuracy and seed stability without changing the attention
backbone or total parameter count. The original [plan](EXPERIMENT_PLAN.md) is retained.

**What changed and why:** Eight learned 32-sample filters, shared across all 64
electrodes at stride 16, produce 29 windows plus CLS. A learned spatial projection
feeds the unchanged three-layer attention backbone. This supplies a local temporal
basis before attention while retaining all electrode identities for this bilateral
task. Proposed model and vanilla both have 135,746 parameters. An equally sized
scattered-time control tests ordered contiguous grouping. `models.py` is unchanged.

**Measured final results:** Five seeds (0–4), 40 fixed epochs; train subjects 1–8,
test 9–10; normalization fitted on training subjects only. Our model's ordinary
accuracy was **best 62.22%, worst 47.78%, mean 54.00%**. Mean balanced accuracy was
**54.02% ± 5.32 pp seed SD**, versus vanilla **50.65% ± 3.06 pp**, EEGNet with Part 1
settings **54.95% ± 3.15 pp**, and scattered control **51.32% ± 3.01 pp**.

**Interpretation:** This is a mixed result. The local model improved mean balanced
accuracy by 3.37 pp over vanilla and 2.70 pp over the control, but seed SD increased
and training accuracy reached 99.89%. The stability prediction was unsupported on
the final subjects, despite a development-stage improvement. The prior did not
prevent overfitting. It reduced mean training time from 28.31 s to 18.32 s and peak
allocated GPU memory from 649.5 to 128.9 MiB. Attention-score entries fell 258×;
actual end-to-end speedup was much smaller. Exact repeats passed for all models.
These are exploratory observations on two test people, not evidence of statistical
significance, physiological localization, or removal of the cue/ocular confound.

**Figure:** [figure.png](experiments/01_shared_temporal_tokens/results/figure.png)
compares losses, held-out balanced accuracy, train/evaluation gaps and training
time, and shows first-seed learned temporal filter responses.

**Reproduce:** `python main.py` or the controlled-experiment Colab notebook.
All seeds, development/final results, original ZIP and source provenance are in
[the experiment record](experiments/01_shared_temporal_tokens/README.md).
