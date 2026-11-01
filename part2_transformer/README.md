# Part 2 — Build & Defend

Part 2 now contains a completed baseline audit and a controlled temporal-tokenization
experiment. The latter produced mixed evidence: lower cost and modestly higher mean
balanced accuracy than vanilla, but no final-stage stability improvement.

## Read the measured results

- [Experiment history](experiments/README.md): both original ZIPs, readable results,
  source snapshots and checksums.
- [Experiment 00: baseline audit](experiments/00_baseline_audit/README.md).
- [Experiment 01: shared local temporal tokens](experiments/01_shared_temporal_tokens/README.md).
- [Part 2 answers](ANSWERS.md): hypothesis, implementation, measured outcome and figure.
- [Hypothesis and locked protocol](EXPERIMENT_PLAN.md), written before implementation.

The final five-seed comparison trains subjects 1–8 and tests 9–10. Local-token mean
accuracy is 54.00% (best 62.22%, worst 47.78%), versus 50.44% for vanilla and 54.67%
for EEGNet with Part 1 settings. Our model is still strongly overfit. See the full
record for balanced accuracy, seed variability, controls and limitations.

## Reproduce the completed study

Open [Part2_Controlled_Experiment.ipynb](Part2_Controlled_Experiment.ipynb) in Colab,
select T4 GPU and set the direct Part 2 Drive folder path. It reproduces the original
fixed study, including its development split, final split, five seeds and exact
repeat checks. Full instructions: [CONTROLLED_EXPERIMENT.md](CONTROLLED_EXPERIMENT.md).
The baseline-only notebook is [Part2_Colab_Audit.ipynb](Part2_Colab_Audit.ipynb).

From this directory with dependencies installed:

```bash
python main.py
```

This prepares/checks data and runs the completed study using CUDA when available,
otherwise CPU. It prints every seed and saves results under timestamped
`artifacts/study_*/` folders. No test score selects an epoch or a winning seed.
The full study is much slower on a laptop CPU; measured Colab T4 times are in
[Experiment 01](experiments/01_shared_temporal_tokens/README.md).

For a synthetic pipeline check with the existing repository environment:

```bash
../.venv/bin/python -B main.py --device cpu --smoke
```

Synthetic outputs are not real EEG measurements. The baseline definitions in
`models.py` have not been modified. `main.py` exports `YourModel` from
`experiment_models.py` and delegates the measured protocol to `experiment.py`.

## Files and data

| Location | Purpose |
| --- | --- |
| `models.py` | Original EEGNet and vanilla transformer |
| `experiment_models.py` | Shared temporal stem and matched scattered-time control |
| `experiment.py`, `experiment_report.py` | Locked training, evaluation and reports |
| `data.py`, `prepare_data.py` | Loader and checksum-verified data preparation |
| `colab_audit.py`, `colab_report.py` | Original two-model GPU audit |
| `dataset.zip` (optional) | Generate with `bundle_data.py`; otherwise MNE downloads the 30 recordings |
| `experiments/` | Original immutable results, readable extracts and provenance |
| `artifacts/` | Data manifests and working outputs |
| `.cache/` | Downloaded data, arrays and local caches; ignored by Git |

Upload this folder's Python files, notebooks, Markdown, requirements and
an optional `dataset.zip` to Drive; without it the notebook downloads the data. Colab keeps its existing CUDA-enabled PyTorch; do not
upload the laptop `.venv/`. Training uses local runtime storage and backs results
up to Drive. Strict determinism and actual repeated-seed checks are documented
in [COLAB.md](COLAB.md).

## Interpretation and status

This is bilateral fists versus feet, so no left/right-only spatial prior is
imposed. Accuracy is measured and interpreted, but the assessment does not score
peak accuracy. The temporal-prior study is exploratory because earlier baseline
test results had already been inspected; two test subjects and five seeds do not
support population-level claims. Filtering and subject separation do not prove
that visual/ocular confounding has been removed.

The submission is finalized with these two completed experiments. No additional
model or test-set-driven search is planned. The readable submission copies are
[figure.png](figure.png), [results.jsonl](results.jsonl) and [summary.json](summary.json);
the originals and all development-stage records remain in `experiments/`.
Reproduction creates a fresh timestamped output directory and does not silently
replace these recorded submission copies.

The original assessment README is preserved in [docs/STARTER_README.md](docs/STARTER_README.md).
Earlier architectural notes remain in [BASELINE_AUDIT.md](BASELINE_AUDIT.md) and
[AUDIT_INTERPRETATION.md](AUDIT_INTERPRETATION.md).
