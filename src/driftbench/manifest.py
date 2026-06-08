"""Dataset manifest: the bridge between the private Drive data and the public repo.

The repo never ships the raw CIC CSVs (size + licence). Instead it ships a
manifest of sha256 checksums, canonical per-file row counts, and official
source URLs. A third party downloads the data themselves from UNB, runs
``verify_manifest``, and is guaranteed byte-identical inputs.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from typing import Dict

# Official source pages (cited in the manuscript's Data Availability section).
SOURCE_URLS: Dict[str, str] = {
    "CSE-CIC-IDS2018": "https://www.unb.ca/cic/datasets/ids-2018.html",
    "CIC-DDoS2019": "https://www.unb.ca/cic/datasets/ddos-2019.html",
}


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    """Streaming sha256 so large CSVs don't blow up memory."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def count_rows(path: Path) -> int:
    """Newline count minus header. Approximate if quoted newlines exist
    (CIC flow CSVs do not use them, so this is exact in practice)."""
    with open(path, "rb") as f:
        n = sum(1 for _ in f)
    return max(n - 1, 0)


def build_manifest(data_dir: Path) -> dict:
    """Walk ``data_dir`` for *.csv and record size, sha256, and row count."""
    data_dir = Path(data_dir)
    entries = {}
    for ds, url in SOURCE_URLS.items():
        ds_dir = data_dir / ds
        files = {}
        if ds_dir.exists():
            for csv in sorted(ds_dir.rglob("*.csv")):
                rel = str(csv.relative_to(ds_dir))
                files[rel] = {
                    "bytes": csv.stat().st_size,
                    "sha256": sha256_file(csv),
                    "rows": count_rows(csv),
                }
        entries[ds] = {"source_url": url, "files": files}
    return {"schema": "driftbench/manifest/v1", "datasets": entries}


def save_manifest(manifest: dict, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True))


def load_manifest(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def verify_manifest(data_dir: Path, manifest: dict) -> Dict[str, list]:
    """Return {'missing': [...], 'mismatch': [...], 'ok': [...]}."""
    data_dir = Path(data_dir)
    report = {"missing": [], "mismatch": [], "ok": []}
    for ds, info in manifest["datasets"].items():
        for rel, meta in info["files"].items():
            path = data_dir / ds / rel
            tag = f"{ds}/{rel}"
            if not path.exists():
                report["missing"].append(tag)
            elif sha256_file(path) != meta["sha256"]:
                report["mismatch"].append(tag)
            else:
                report["ok"].append(tag)
    return report
