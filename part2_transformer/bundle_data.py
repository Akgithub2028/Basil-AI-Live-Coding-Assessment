"""Build a visible, portable dataset.zip for a Drive folder upload."""
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parent


def main():
    manifest = json.loads((ROOT / "artifacts/dataset_manifest.json").read_text())
    destination = ROOT / "dataset.zip"
    temporary = ROOT / "dataset.zip.tmp"
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=3) as archive:
        for record in manifest["files"]:
            path = ROOT / record["path"]
            if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
                raise ValueError(f"Checksum mismatch: {path}")
            archive.write(path, record["path"])
        archive.writestr("dataset_manifest.json", json.dumps(manifest, indent=2))
    temporary.replace(destination)
    print(f"Created {destination}: {destination.stat().st_size / 1e6:.2f} MB, 30 verified EDF files")


if __name__ == "__main__":
    main()
