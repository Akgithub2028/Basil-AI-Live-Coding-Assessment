# Run the single-hypothesis follow-up

Open **Part2_Controlled_Experiment.ipynb**, select a Colab T4 runtime, and set
`DRIVE_PART2` to your uploaded Part 2 folder. Its initial value matches the
folder path in your completed audit. Upload the updated `.py` and `.md` files,
the new notebook and `dataset.zip`; the old Colab audit ZIP is not required to
execute the study. Run all cells in order.

Read `EXPERIMENT_PLAN.md` first. It was written before coding/training this
model and is copied and hashed into every result directory. There are no
new real-data custom-model results yet; local tests use synthetic inputs.

The new study compares **the two requested baselines** with our proposed model
and one deliberately scattered-time control. This is one hypothesis with an
ablation, not four competing new ideas. Both new variants have exactly the
vanilla parameter count. `models.py` remains unchanged.

There are two fixed stages (development and final), five seeds per configuration,
40 epochs, and an exact repeated-seed check for each model in each stage. All
models use matched minibatch order. The final stage runs unchanged even if
development results are negative; there is no automatic tuning or selection.
Training, development diagnostic time and final inference time are separated.

The old audit's test performance has already been inspected. Final results
here are exploratory evidence, not a completely untouched confirmatory test.
Five seeds characterize initialization/optimization variation; they do not
increase the number of independent held-out people.

Results save to `artifacts/controlled_experiment/study_TIMESTAMP/` on Drive:

- `EXPERIMENT_REPORT.md`: all evidence, support checks and interpretation limits.
- `dev/` and `final/`: stage reports, every seed, paired differences, loss
  histories, predictions, checkpoints and actual determinism checks.
- `figure.png`: final-stage comparison and learned filter responses.
- `ANSWER_DRAFT.md`: measured best/worst/mean and hypothesis; add your own
  interpretation of the actual result before finalizing `ANSWERS.md`.
- `metadata.json` and `EXPERIMENT_PLAN.md`: exact settings, source hashes,
  packages, hardware and the original hypothesis.

Completed seeds are backed up to Drive. The notebook creates an export ZIP.
No GPU timing for this new model is claimed before the Colab run.

## Reproduce without the notebook

From Part 2, `python main.py` prepares data and runs the full study, automatically
using a GPU when available. The implementation of `YourModel` lives in
`experiment_models.py` and is exported by `main.py` for review.

For Colab after preparing the recordings:

```bash
python -B prepare_data.py
python -B -u experiment.py --device cuda
```

A CPU-only synthetic end-to-end validation (not EEG results):

```bash
../.venv/bin/python -B main.py --device cpu --smoke
```

Main entry-point behavior now reflects this locked experiment. The original
two-model notebook `Part2_Colab_Audit.ipynb` remains usable for the baseline
audit; it calls `colab_audit.py` and is independent of the new entry point.
