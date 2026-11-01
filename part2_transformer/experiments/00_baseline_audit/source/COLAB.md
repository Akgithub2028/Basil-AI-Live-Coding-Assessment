# Run the Part 2 audit on Colab T4

Upload `eeg_dl_exam_3hr/` to Google Drive. The Part 2 folder now includes a
visible `dataset.zip` (43.65 MB) with the 30 verified EDF recordings, so hidden
folders are not required. The local `.venv/`, `.git/` and old `.cache/` folders
are not needed in Colab. Keep `dataset.zip` with the Python files and notebook.

1. Open `part2_transformer/Part2_Colab_Audit.ipynb` in Google Colab.
2. Select **Runtime → Change runtime type → T4 GPU**.
3. In the notebook set `DRIVE_PROJECT` to the uploaded `eeg_dl_exam_3hr` folder.
4. Run the cells in order, or use **Run all**.

The notebook installs scientific dependencies while retaining Colab's existing
CUDA PyTorch. It copies only Part 2 code/data to `/content/`, extracts and checks
the dataset, preprocesses locally, trains, and saves outputs back to
`eeg_dl_exam_3hr/part2_transformer/artifacts/colab_gpu/` on Drive. The data can be
downloaded from PhysioNet if the archive is absent. Training never reads
batches directly from Drive.

## Fixed experiment

Exactly two baseline configurations are trained:

- **EEGNet, Part 1 architecture:** F1=8, D=2, F2=16, kernel=80, dropout=0.5.
- **VanillaEEGTransformer:** unchanged `models.py` implementation.

Both use Part 2's 7–30 Hz, 0–3 s data, train subjects 1–8 / test subjects 9–10,
40 epochs, seeds 0/1/2, AdamW, learning rate 0.001 and effective batch 64.
The final epoch is specified in advance; no test-based checkpoint selection.
This is a fresh comparison using the Part 1 architecture, not a reproduction
of Part 1's preprocessing or score. `YourModel` is not trained or edited.

FP16 AMP and gradient scaling accelerate T4 training. Transformer microbatches
of 16 accumulate into an effective batch of 64; EEGNet uses full batches to
preserve batch-normalization statistics. The small dataset stays in GPU memory.
No compilation or extra attention package is required.

## Determinism

Strict deterministic algorithms are enabled by default. Python, NumPy and
PyTorch are seeded; cuDNN benchmarking and TF32 are disabled; cuBLAS workspace
configuration is set before torch import. Strict mode uses math SDPA so it
does not depend on T4 FlashAttention support. Unsupported nondeterministic
operations raise an error instead of silently weakening the guarantee.

The first seed of **each** model is repeated from initialization. Checkpoint
tensors, prediction CSVs and training loss histories must match exactly, or
the audit fails. These two verification runs are excluded from seed summaries:
six scientific runs plus two reproducibility runs at the default settings.
Exactness applies to a fixed GPU/software stack. Different GPU models or
PyTorch/CUDA releases can give different results; package versions, source
hashes and hardware are recorded. FP16 is deterministic under this protocol
when the exact repeat checks pass; it is not bitwise equivalent to CPU FP32.

## Generated audit

Each timestamped run saves:

- `AUDIT_REPORT.md` and `.html`: architecture, protocol, measured metrics,
  timing, limitations and actual repeat-check results.
- `audit_dashboard.png`: losses, train/test scores, per-subject scores,
  confusion matrices and training time.
- `metadata.json`, checksum manifests and source hashes.
- `results.jsonl`, `summary.json`, per-seed histories and prediction CSVs.
- Final model checkpoints and exact reproducibility verification records.

Completed seeds are backed up to Drive as they finish. Exceptions and normal
interrupts also trigger backup; a forcibly disconnected runtime can still
lose the current unfinished seed. Each rerun has a separate timestamped
directory. The notebook also creates an export ZIP at the end.

GPU runtime is measured in the notebook; no unmeasured T4 timing is promised.
Training and test inference have separate synchronized wall-clock fields.

## Command line and local checks

In Colab's local Part 2 working copy:

```bash
python -B prepare_data.py
python -B -u colab_audit.py --device cuda --precision fp16 --epochs 40 --seeds 0 1 2 --batch 64 --microbatch 16
```

For a small synthetic plumbing check on a CPU, with no real test-set results:

```bash
../.venv/bin/python -B colab_audit.py --device cpu --smoke --batch 8 --microbatch 4
```

Use `colab_report.py RUN_DIRECTORY` to regenerate reports from existing results.
The original `audit_baselines.py` remains the record of the earlier CPU audit;
the notebook uses `colab_audit.py`, which selects only the two requested models.

Implementation references: [PyTorch mixed precision](https://docs.pytorch.org/docs/2.15/amp.html),
[PyTorch attention backends](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html),
and [Colab runtime and storage FAQ](https://research.google.com/colaboratory/faq.html).
