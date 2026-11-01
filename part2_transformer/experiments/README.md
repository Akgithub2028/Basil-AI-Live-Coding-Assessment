# Experiment record

Keep every completed experiment, including unsupported hypotheses. Source ZIPs
are preserved byte-for-byte with their original names. `PROVENANCE.json` records
archive hashes and distinguishes exact recorded source files from supporting
files whose hashes were not recorded during the original run. Readable results
are extracted beside each archive; complete checkpoints remain in the ZIP.

| Experiment | Question | Outcome |
| --- | --- | --- |
| [00 — GPU baseline audit](00_baseline_audit/README.md) | How do the two baselines fit and generalize? | Vanilla fits almost perfectly but generalizes near chance; EEGNet is more restrained. Exact repeats pass. |
| [01 — Shared temporal tokens](01_shared_temporal_tokens/README.md) | Does shared local temporal tokenization improve generalization, stability and cost? | Mixed: better mean balanced accuracy and lower measured cost; the final-stage stability prediction is unsupported. |

The second experiment's development and final stages are both retained.
The final support flag is false because the local model's seed SD increased;
that flag is not a judgement that the whole research exercise is invalid.
Do not pool the initial three-seed baseline audit with the later five-seed
matched comparison, which used a different minibatch-order protocol.

Current reproducible source and notebooks remain in the Part 2 root. Each
experiment also contains a source snapshot and readable results. Synthetic
and incomplete local CPU checks are kept locally in `artifacts/local_checks/`
and excluded from Git to avoid presenting them as measured GPU evidence.

The baseline test results had already been seen when Experiment 01 was designed.
Its final comparison is exploratory. No independent subject cohort is created
by repeating seeds. The submission is finalized with these completed experiments; no subsequent
strategy or test-set-driven search is part of this submission.
