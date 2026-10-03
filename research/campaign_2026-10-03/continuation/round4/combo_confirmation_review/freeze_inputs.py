"""Freeze authorized confirmation inputs without reading bulk prediction/model data."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]


def main() -> None:
    inputs = {
        "combo_fold23_prereg_report.json": "research/campaign_2026-10-03/scores/combo_fold23_prereg_report.json",
        "FOLD23_CONFIRMATION.txt": "research/campaign_2026-10-03/continuation/FOLD23_CONFIRMATION.txt",
        "combo_fold23_prereg.json": "research/campaign_2026-10-03/scores/combo_fold23_prereg.json",
    }
    manifest = {}
    for name, source in inputs.items():
        destination = HERE / name
        if destination.exists():
            raise RuntimeError("inputs already frozen")
        data = (ROOT / source).read_bytes()
        destination.write_bytes(data)
        manifest[name] = {
            "source": source,
            "sha256": hashlib.sha256(data).hexdigest(),
            "bytes": len(data),
        }
    (HERE / "INPUTS.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
