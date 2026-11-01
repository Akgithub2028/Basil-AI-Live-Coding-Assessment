"""Run all Part 1 experiments without editing the preserved original starter.

From the repository root: python part1_diagnose/run_part1.py
Requires part1_diagnose/requirements.txt and internet for the first EEG download.
All generated files, caches and temporary files stay within this directory.
"""
import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version, PackageNotFoundError
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"


def local_environment(threads):
    env = os.environ.copy()
    paths = {
        "MNE_DATA": ".cache/mne", "MNE_DATASETS_EEGBCI_PATH": ".cache/mne",
        "_MNE_FAKE_HOME_DIR": ".cache/mne-home", "MPLCONFIGDIR": ".cache/matplotlib",
        "XDG_CACHE_HOME": ".cache", "TMPDIR": ".cache/tmp",
    }
    for key, value in paths.items():
        path = ROOT / value
        path.mkdir(parents=True, exist_ok=True)
        env[key] = str(path)
    env.update(OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads),
               OPENBLAS_NUM_THREADS=str(threads), PYTHONDONTWRITEBYTECODE="1",
               PYTHONHASHSEED="0", MNE_LOGGING_LEVEL="WARNING")
    return env


def original_source():
    """Verify the byte-identical snapshot; also works in Drive without .git."""
    destination = ARTIFACTS / "original_source"
    destination.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((ARTIFACTS / "starter_manifest.json").read_text())
    with zipfile.ZipFile(ARTIFACTS / "starter_85b84b8.zip") as archive:
        for name, expected in manifest["sha256"].items():
            content = archive.read(name)
            if hashlib.sha256(content).hexdigest() != expected:
                raise ValueError(f"Original snapshot checksum mismatch: {name}")
            (destination / name).write_bytes(content)
    return destination


def source_hashes(source):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(source.glob("*.py")) if not p.name.startswith("test_")}


def run_stage(stage, script, arguments, attempt, env, args):
    work = attempt / stage
    work.mkdir()
    # CPU for both originals and fixes; identical thread settings and budget.
    setup = ("import sys,runpy; sys.path.insert(0,sys.argv[2]); "
             "script=sys.argv.pop(1); sys.argv.pop(1); "
             "runpy.run_path(script,run_name='__main__')")
    command = [sys.executable, "-u", "-c", setup, str(script), str(script.parent), *arguments]
    record = dict(stage=stage, status="running", seed=args.seed, epochs=args.epochs,
                  threads=args.threads, device="cpu", timestamp_utc=datetime.now(timezone.utc).isoformat(),
                  command=command, log=str((work / "run.log").relative_to(ROOT)),
                  source_sha256=source_hashes(script.parent), test_acc=None)
    record["versions"] = {}
    for package in ["numpy", "scipy", "scikit-learn", "mne", "matplotlib", "torch"]:
        try:
            record["versions"][package] = version(package)
        except PackageNotFoundError:
            record["versions"][package] = None
    print(f"\n=== {stage} ===", flush=True)
    with (work / "run.log").open("w") as log:
        process = subprocess.Popen(command, cwd=work, env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
        record["returncode"] = process.wait()
    record["status"] = "ok" if record["returncode"] == 0 else "failed"
    if record["status"] == "ok" and stage != "erd_mu":
        raw = json.loads((work / "results.jsonl").read_text().splitlines()[-1])
        record.update(raw)
    elif record["status"] == "ok":
        for name in ["erd_mu.png", "erd_mu_summary.json"]:
            shutil.copyfile(work / name, ROOT / name)
        record["summary"] = json.loads((ROOT / "erd_mu_summary.json").read_text())
    with (ROOT / "results.jsonl").open("a") as f:
        f.write(json.dumps(record) + "\n")
    return record


def update_answers(records, args):
    by_stage = {r["stage"]: r for r in records}
    original, all_ch, frontal, erd = [by_stage[s] for s in ["original_all", "corrected_all", "corrected_frontal", "erd_mu"]]

    def accuracy(record):
        return f"{record['test_acc']:.2%}" if record["status"] == "ok" else "not measured (run failed)"

    table = [f"Run: seed={args.seed}, epochs={args.epochs}, CPU threads={args.threads}.",
             "", "| Experiment | Accuracy / output |", "| --- | --- |",
             f"| Original starter (invalid protocol) | {accuracy(original)} |",
             f"| Corrected all channels | {accuracy(all_ch)} |",
             f"| Corrected frontal only | {accuracy(frontal)} |",
             f"| Mu ERD | {'`erd_mu.png` generated' if erd['status'] == 'ok' else 'not generated (run failed)'} |",
             "", "Full records: `results.jsonl`; per-run logs: `artifacts/attempts/`." ]
    replacements = {"RESULTS": "\n".join(table)}
    if erd["status"] == "ok":
        values = erd["summary"]["mean_erd_pct"]
        observed = "; ".join(f"{ch}: hands {v['hands']:+.1f}%, feet {v['feet']:+.1f}%" for ch, v in values.items())
        replacements["ERD"] = (
            f"The mu ERD plot shows these mean changes over 1–4 s relative to pre-cue rest: {observed} (negative means desynchronization). "
            "The anatomical comparison is bilateral hand-related ERD near C3/C4 versus foot-related ERD nearer Cz, rather than left/right hand lateralization; the measured signs and class ordering above must be checked against the full curves before claiming agreement. "
            "A weaker Cz effect can arise because the medial foot sources lie in the interhemispheric fissure and scalp sensitivity depends on source depth/orientation, with between-subject variation further reducing the group effect. "
            "These are descriptive group averages, and neither class separation nor a weak vertex response alone establishes significance or freedom from artifacts."
        )
    if all_ch["status"] == frontal["status"] == "ok":
        difference = 100 * (frontal["test_acc"] - all_ch["test_acc"])
        replacements["FRONTAL"] = (
            f"With identical subject splits and training budgets, all-channel accuracy was {accuracy(all_ch)} and frontal-only accuracy was {accuracy(frontal)} (frontal minus all: {difference:+.2f} percentage points). "
            "Any retained frontal decoding is consistent with a cue-correlated shortcut, but the magnitude must be judged against chance and sampling uncertainty rather than treated as proof. "
            "A plausible non-brain signal is vertical EOG from gaze shifts toward the top target for fists and bottom target for feet, which produces strong frontal potentials. "
            "Frontal neural activity and volume conduction remain alternatives, and weaker frontal decoding would not establish that the full model is artifact-free."
        )
    path = ROOT / "ANSWERS.md"
    text = path.read_text()
    for section, content in replacements.items():
        start, end = f"<!-- PART1_{section}_START -->", f"<!-- PART1_{section}_END -->"
        before, rest = text.split(start, 1)
        _, after = rest.split(end, 1)
        text = before + start + "\n" + content + "\n" + end + after
    path.write_text(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.epochs < 1 or args.threads < 1:
        parser.error("epochs and threads must be positive")
    env = local_environment(args.threads)
    attempt = ARTIFACTS / "attempts" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    attempt.mkdir(parents=True)
    original = original_source()
    fixed = ROOT
    common = ["--seed", str(args.seed), "--epochs", str(args.epochs)]
    experiments = [
        ("original_all", original / "train.py", [*common, "--channels", "all"]),
        ("corrected_all", fixed / "train.py", [*common, "--channels", "all"]),
        ("erd_mu", fixed / "erd.py", ["--band", "mu"]),
        ("corrected_frontal", fixed / "train.py", [*common, "--channels", "frontal"]),
    ]
    records = [run_stage(stage, script, arguments, attempt, env, args)
               for stage, script, arguments in experiments]
    (ARTIFACTS / "latest_run.json").write_text(json.dumps(records, indent=2) + "\n")
    update_answers(records, args)
    if any(r["status"] != "ok" for r in records):
        print("\nPart 1 experiments are INCOMPLETE; see recorded failures. No missing accuracy was invented.")
        return 1
    print("\nExperiments finished. Review erd_mu.png and the data-derived answers in ANSWERS.md.")
    print("The baseline comparison changes both the task data and the evaluation protocol.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
