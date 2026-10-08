#!/usr/bin/env python3
"""Run the fixed 20-sequence callback fixture and bind its timing receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    """Hash the exact ZIP/data bytes used by the replay."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--zip", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    root = args.root.resolve()
    zip_path = args.zip.resolve()
    data_path = args.data.resolve()
    predictions_path = args.predictions.resolve()
    output_path = args.output.resolve()
    if not zip_path.is_file() or not data_path.is_file():
        raise FileNotFoundError("candidate ZIP or callback fixture is missing")
    if output_path.exists() or predictions_path.exists():
        raise FileExistsError("refusing to overwrite a replay receipt or prediction output")

    sys.path.insert(0, str(root))
    from scripts.verify_submission import verify_submission  # noqa: E402

    measured: dict[str, Any] = verify_submission(zip_path, data_path, predictions_path)
    if measured.get("rows") != 400_000 or measured.get("sequences") != 20:
        raise ValueError("timed replay did not cover exactly twenty complete streams")
    if measured.get("metric_id") != "global_wp_clip2_v1":
        raise ValueError("fixture callback diagnostic did not use the official metric")

    report = {
        "schema": "wnn.candidate-package-timed-replay.v1",
        "status": "passed",
        "zip": str(zip_path),
        "zip_sha256": sha256_file(zip_path),
        "data": str(data_path),
        "data_sha256": sha256_file(data_path),
        "predictions": str(predictions_path),
        "predictions_sha256": sha256_file(predictions_path),
        "rows": measured["rows"],
        "sequences": measured["sequences"],
        "us_per_row": measured["us_per_row"],
        "load_s": measured["load_s"],
        "metric_id": measured["metric_id"],
        "exposed_fixture_diagnostic": {
            key: measured[key] for key in ("global_wp", "global_t0", "global_t1", "selected_rows")
        },
        "scientific_interpretation": (
            "The 20-sequence fixture is exposed to these all-data candidates; these diagnostics "
            "are not held-out quality evidence and are not used for promotion."
        ),
        "scope": "complete 400000-row callback replay; timing only for matched local compatibility calibration",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "us_per_row": report["us_per_row"]}))


if __name__ == "__main__":
    main()
