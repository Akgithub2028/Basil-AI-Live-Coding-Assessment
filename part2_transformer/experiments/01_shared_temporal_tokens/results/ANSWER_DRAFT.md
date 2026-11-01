# Part 2 measured write-up draft

**Hypothesis (written before implementation):** Shared local temporal filters should reduce attention cost and improve held-out-subject balanced accuracy and seed stability relative to per-sample attention.

**Change and mechanism:** Eight learned 200 ms filters shared across all electrodes, stride 100 ms, produce 29 overlapping temporal tokens plus CLS. A learned spatial projection and the unchanged three-layer attention backbone follow. The vanilla and local models have the same 135,746 parameters. An equally sized scattered-time control tests ordered local grouping.

**Results:** Accuracy best / worst / mean = 62.22% / 47.78% / 54.00%. Balanced accuracy mean ± seed SD = 54.02% ± 5.32 pp. Full prespecified descriptive support: False. Inspect the report's individual criteria and controls before writing the interpretation; do not imply a significance test.

**Figure:** `figure.png` compares training losses, held-out scores, train/evaluation gaps and runtime, and shows the local model's first-seed learned temporal filter responses.

This was an exploratory follow-up after seeing baseline test results. Repeat checks, all seeds and subject-level results are retained. Add a short, evidence-based interpretation before submission.
