# Working artifacts

- `dataset_manifest.json`: SHA-256 and metadata for all 30 EEG recordings.
- `data_summary.json`: preprocessing shapes, class counts and scale checks.
- `prepare_data.log`: preparation output.
- `local_checks/`: retained local synthetic/CPU diagnostics, ignored by Git.
  These are not the measured Colab experiment results.

The completed GPU evidence, original ZIPs, readable results and source snapshots
are organized under [experiments/](../experiments/README.md). New runs create
timestamped directories here; inspect their metadata to distinguish real runs
from synthetic smoke checks.
