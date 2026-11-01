# Hands vs Feet: corrected Part 1

| File | What it does |
| --- | --- |
| `data.py` | Downloads runs via MNE, filters, epochs, returns trials in µV, labels, subject ids |
| `erd.py` | Mu/beta ERD time courses at C3, Cz, C4 → `erd_<band>.png` |
| `model.py` | EEGNet |
| `train.py` | Trains EEGNet, reports held-out accuracy, appends to `results.jsonl` |
| `evaluation.py` | Subject-disjoint partitions and training-only normalization |
| `test_pipeline.py` | Offline checks; the training check requires PyTorch |

From the **repository root**, set up the project environment on CPU:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --cache-dir .cache/pip torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install --cache-dir .cache/pip -r part1_diagnose/requirements.txt
.venv/bin/python run_part1.py
```

`run_part1.py` runs, in order: the byte-identical original starter from commit
`85b84b8`, corrected all-channel training, mu ERD, and frontal-only training.
It uses seed 0, 30 epochs, two CPU threads and the original optimizer defaults
for all three training runs. It stores downloads and caches in `.cache/` within
the repository, not in the home directory. A GPU is neither selected nor needed.

Outputs at the repository root:

- `results.jsonl`: append-only stage records, including failed attempts with
  `test_acc: null`; compare only records with `status: "ok"`.
- `erd_mu.png` and `erd_mu_summary.json`: generated only from real EEG data.
- `ANSWERS.md`: bug explanations, analytical answers, and results populated by
  the runner. Plot interpretation must be checked against the actual curves.
- `artifacts/part1/latest_run.json`: records from the latest full attempt.
- `artifacts/part1/attempts/<timestamp>/`: full logs and individual raw outputs.

The preserved original source archive is verified against SHA-256 checksums.
The runner works without Git metadata, so a copied directory remains runnable.
The first download includes both the original execution runs and corrected
imagery runs; expect more data than the starter's single-dataset estimate.

For the default ten subjects, the corrected pipeline tests on 9–10, validates
on two seeded subjects from 1–8, and trains on the other six. Normalization
uses only training trials; checkpoint selection uses only validation accuracy.
Frontal and all-channel runs use the same partitions. Their parameter counts
differ because EEGNet's spatial filter spans the selected channel count.

Run offline verification from the repository root:

```bash
MPLCONFIGDIR="$PWD/.cache/matplotlib" PYTHONDONTWRITEBYTECODE=1 \
  .venv/bin/python -m unittest discover -s part1_diagnose -p 'test_*.py' -v
```

Individual commands also work from `part1_diagnose/` with dependencies
installed (`python train.py --channels all`, `python erd.py --band mu`, and
`python train.py --channels frontal`). Use the root runner when all caches,
logs and results must stay in the project directory.

## Current verification status

The first local CPU execution attempt could not train because PyTorch was
unavailable. The CPU dependencies have since been installed and all seven
offline signal/evaluation/checkpoint checks pass; PhysioNet downloads remain
blocked by terminal DNS resolution. No measured accuracy or real ERD figure
is claimed until a full run succeeds. See `ANSWERS.md` for the distinction
between five unambiguous errors and the additional cue-onset correction.
