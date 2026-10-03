"""STOP wrap only: hash notes and source provenance; no scientific computation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    source_names = [line for line in (HERE / "SOURCE_READS.txt").read_text().splitlines()
                    if line and not line.startswith("Read-only")]
    provenance = {"source_sha256": {name: sha(ROOT / name) for name in source_names},
                  "scope": "metadata-only STOP wrap; new task NOT_EXECUTED",
                  "new_scientific_tests": 0, "model_forwards": 0, "GPU_calls": 0}
    (HERE / "PROVENANCE.json").write_text(json.dumps(provenance, indent=2) + "\n")
    files = sorted(p for p in HERE.iterdir() if p.is_file() and p.name != "HASHES.json")
    (HERE / "HASHES.json").write_text(json.dumps({p.name: sha(p) for p in files}, indent=2) + "\n")
    print("Metadata wrap frozen: PAUSED / INCOMPLETE / NOT_EXECUTED. No scientific computation.")


if __name__ == "__main__":
    main()
