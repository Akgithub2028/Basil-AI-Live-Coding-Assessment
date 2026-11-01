"""Render completed baseline audit results without re-running training."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def summarize(directory):
    directory = directory.resolve()
    if not directory.is_relative_to(ROOT / "artifacts"):
        raise ValueError("Audit results must be inside Part 2 artifacts")
    meta = json.loads((directory / "metadata.json").read_text())
    summary = json.loads((directory / "summary.json").read_text())
    rows = [json.loads(line) for line in (directory / "results.jsonl").read_text().splitlines()]
    lines = ["# Measured baseline audit results", "",
             f"Source audit: `{directory.name}`.", "",
             f"CPU, {meta['arguments']['threads']} threads; {meta['arguments']['epochs']} epochs; "
             f"seeds {meta['arguments']['seeds']}; batch {meta['arguments']['batch']}.", "",
             "Checkpoint rule: prespecified final epoch; no peak-test selection.", "",
             "All models use the Part 2 7–30 Hz, 0–3 s data and subjects 1–8 / 9–10 split.", "",
             "| Configuration | Parameters | Test best | Test worst | Test mean ± SD | Mean training time |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for name, stats in summary.items():
        params = meta['architectures'][name]['parameters']
        lines.append(f"| {name} | {params:,} | {stats['best']:.2%} | {stats['worst']:.2%} | "
                     f"{stats['mean']:.2%} ± {100*stats['std']:.2f} pp | {stats['mean_train_seconds']:.1f} s |")
    lines += ["", "SD is population SD across three optimization seeds, not uncertainty across people.", "",
              "| Configuration | Seed | Train | Test | Balanced test | Subject 9 | Subject 10 | Train time |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in rows:
        lines.append(f"| {r['model']} | {r['seed']} | {r['train_accuracy']:.2%} | {r['test_accuracy']:.2%} | "
                     f"{r['balanced_accuracy']:.2%} | {r['per_subject']['9']:.2%} | {r['per_subject']['10']:.2%} | "
                     f"{r['train_seconds']:.1f} s |")
    lines += ["", "## Prediction diagnostics", "",
              "Confusion matrices have true classes as rows and predicted classes as columns; order is fists, feet.", ""]
    for r in rows:
        lines.append(f"- {r['model']}, seed {r['seed']}: confusion `{r['confusion_matrix']}`; "
                     f"predicted class counts `{r['predicted_class_counts']}`.")
    total = sum(r['total_seconds'] for r in rows)
    lines += ["", f"Total measured training + final evaluation time: {total/60:.2f} minutes "
              "(excludes setup, data preparation, architecture probes and plotting).", "",
              "See `BASELINE_AUDIT.md` for mechanisms and interpretation limits. Full histories, "
              "predictions, architecture traces, source hashes and the figure are in the source audit directory.", ""]
    (ROOT / "MEASURED_RESULTS.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    summarize(parser.parse_args().directory)
