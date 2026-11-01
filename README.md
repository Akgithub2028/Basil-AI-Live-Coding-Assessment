# EEG × Deep Learning — Live Exam (3 hours)

Two parts, 90 minutes each, with a 5-minute break between them. Read the
problem-statement PDF for the full brief; this file is the map.

```
part1_diagnose/     Part 1 — a buggy motor-imagery pipeline you must fix and interpret
part2_transformer/  Part 2 — give self-attention a fair shot on tiny EEG data
```

## Part 1 CPU workflow

The corrected Part 1 code, offline checks and CPU setup instructions are in
[`part1_diagnose/README.md`](part1_diagnose/README.md). From the repository root,
after installing its requirements, run `python part1_diagnose/run_part1.py` to reproduce the
original starter baseline, corrected baseline, mu ERD and frontal-channel
control. The runner keeps all downloaded data, caches and outputs in this
`part1_diagnose/` directory and updates its `ANSWERS.md` with measured results. No notebook or GPU
is required. Part 2 now includes a completed baseline audit and a controlled
temporal-tokenization study with mixed results. Its [README](part2_transformer/README.md)
links the Colab notebooks, measured answers and preserved experiment history.

## Before the clock starts

Set up the environment once (conda covers both parts):

```bash
conda env create -f environment.yml
conda activate eeg-dl-exam
```

Or use pip per part (each folder has its own `requirements.txt`, they overlap):

```bash
pip install -r part1_diagnose/requirements.txt
```

Then kick off the data download at the start of each part and read code while
it runs:

```bash
cd part1_diagnose       && python train.py
# ...later...
cd ../part2_transformer && python main.py
```

Both parts use the PhysioNet EEGBCI dataset (subjects 1–10, 64 channels,
160 Hz), downloaded and cached automatically by MNE. Part 1 supports CPU;
the completed Part 2 study used Colab T4. See its README for measured runtimes.

## Final submission

- [All answers](ANSWERS.md): links to each part's complete write-up.
- [Part 1](part1_diagnose/README.md): corrected code, before/after results,
  frontal control, ERD figure and original unsquashed bug-fix history.
- [Part 2](part2_transformer/README.md): implemented model, executed Colab
  notebooks, measured mixed result and original experiment archives.
- [Part 2 hand-in map and defence notes](part2_transformer/SUBMISSION.md).

Each part's code, data artifacts, figures and analyses live inside its own
directory. The Part 2 submission keeps both positive and negative findings;
it does not claim that the prior solved overfitting or improved seed stability.

## The arc

- **Part 1 — Diagnose & Trust.** The pipeline runs cleanly but the accuracy it
  prints is wrong. Find the planted defects, fix them, interpret the ERD, and
  work out what signal the model is really using. You end with an EEGNet
  baseline whose number you can *trust*.
- **Part 2 — Build & Defend.** Carry that trusted baseline forward. Inject one
  inductive bias into an attention-based decoder and show, honestly, whether it
  helps — and whether you understand why.

Part 1 tests rigor; Part 2 tests ideas. They're weighted equally. Accuracy is
not scored — your reasoning is.

## What you hand in

One GitHub repo (commit as you work, one commit per bug fix, no squashing) with
both parts' code, an `ANSWERS.md`, and the generated artifacts
(`results.jsonl`, `erd_mu.png`, the Part 2 figure). Then a live code
walkthrough with Q&A. Details in the PDF.
