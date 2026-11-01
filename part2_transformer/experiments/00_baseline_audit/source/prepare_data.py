"""Populate and verify the Part 2 cache; never writes outside this directory."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for key, relative in {"MPLCONFIGDIR": ".cache/matplotlib", "XDG_CACHE_HOME": ".cache",
                      "TMPDIR": ".cache/tmp", "_MNE_FAKE_HOME_DIR": ".cache/mne-home"}.items():
    folder = ROOT / relative
    folder.mkdir(parents=True, exist_ok=True)
    os.environ[key] = str(folder)
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

import hashlib
import json
import shutil
import time
import zipfile
from importlib.resources import files

import mne
import numpy as np
from mne.datasets import eegbci
from data import DATA_DIR, RUNS, load_data, get_channel_positions


def prepare():
    start = time.perf_counter()
    artifacts = ROOT / "artifacts"
    artifacts.mkdir(exist_ok=True)
    registry = {}
    for line in files("mne").joinpath("data", "eegbci_checksums.txt").read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            name, digest = line.split()[:2]
            registry[name] = digest.removeprefix("sha256:")
    records = []
    # A visible archive survives Drive folder uploads that omit hidden folders.
    bundle = ROOT / "dataset.zip"
    if bundle.exists():
        with zipfile.ZipFile(bundle) as archive:
            for member in archive.infolist():
                relative = Path(member.filename)
                if relative.parts[:2] != (".cache", "mne") or relative.suffix != ".edf":
                    continue
                target = (ROOT / relative).resolve()
                if not target.is_relative_to(DATA_DIR.resolve()):
                    raise ValueError(f"Unsafe dataset archive member: {member.filename}")
                if not target.exists():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with archive.open(member) as source, target.open("wb") as destination:
                        shutil.copyfileobj(source, destination)
    for subject in range(1, 11):
        for run in RUNS:
            relative = Path("MNE-eegbci-data/files/eegmmidb/1.0.0") / f"S{subject:03d}" / f"S{subject:03d}R{run:02d}.edf"
            dest = DATA_DIR / relative
            source = ROOT.parent / "part1_diagnose/.cache/mne" / relative
            origin = "existing Part 2 cache"
            if not dest.exists() and source.exists():
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, dest)
                origin = "copied from Part 1 cache"
            if not dest.exists():
                eegbci.load_data(subject, [run], path=str(DATA_DIR), update_path=False)
                origin = "downloaded from PhysioNet"
            digest = hashlib.sha256(dest.read_bytes()).hexdigest()
            key = f"S{subject:03d}/S{subject:03d}R{run:02d}.edf"
            if digest != registry[key]:
                raise ValueError(f"Checksum mismatch: {dest}")
            raw = mne.io.read_raw_edf(dest, preload=False, verbose="ERROR")
            descriptions, counts = np.unique(raw.annotations.description, return_counts=True)
            assert raw.info["sfreq"] == 160 and len(raw.ch_names) == 64
            records.append(dict(subject=subject, run=run, path=str(dest.relative_to(ROOT)),
                                bytes=dest.stat().st_size, sha256=digest, origin=origin,
                                samples=int(raw.n_times), sfreq=float(raw.info["sfreq"]),
                                annotations=dict(zip(descriptions.tolist(), counts.tolist()))))
            print(f"verified {key}", flush=True)
    manifest = dict(files=records, total_bytes=sum(r["bytes"] for r in records),
                    checksum_registry="installed MNE/data/eegbci_checksums.txt",
                    upstream="https://physionet.org/files/eegmmidb/1.0.0/")
    (artifacts / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2))
    Xtr, ytr, Xte, yte = load_data()
    pos = get_channel_positions()
    assert np.isfinite(pos).all() and pos.shape == (64, 3)
    assert Xtr.shape[1:] == Xte.shape[1:] == (64, 481)
    assert np.isfinite(Xtr).all() and np.isfinite(Xte).all()
    assert np.max(np.abs(Xtr.mean(axis=(0, 2)))) < 1e-4
    normalized_std = Xtr.std(axis=(0, 2))
    assert (normalized_std > 0).all()
    # Counts are also verified independently from the source annotations.
    counts = {s: sum(r["annotations"].get("T1", 0) + r["annotations"].get("T2", 0)
                     for r in records if r["subject"] == s) for s in range(1, 11)}
    groups_tr = np.concatenate([np.full(counts[s], s) for s in range(1, 9)])
    groups_te = np.concatenate([np.full(counts[s], s) for s in [9, 10]])
    assert len(groups_tr) == len(ytr) and len(groups_te) == len(yte)
    np.savez(ROOT / ".cache/preprocessed.npz", Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte,
             groups_tr=groups_tr, groups_te=groups_te, positions=pos)
    summary = dict(train_shape=list(Xtr.shape), test_shape=list(Xte.shape),
                   train_class_counts=np.bincount(ytr).tolist(), test_class_counts=np.bincount(yte).tolist(),
                   per_subject_trials=counts, array_bytes=Xtr.nbytes + Xte.nbytes,
                   normalized_train_channel_std_range=[float(normalized_std.min()), float(normalized_std.max())],
                   normalization_note="Starter adds 1e-6 to std in volts, so channel std is below one; preserved for this audit.",
                   raw_bytes=manifest["total_bytes"], seconds=time.perf_counter() - start)
    (artifacts / "data_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    prepare()
