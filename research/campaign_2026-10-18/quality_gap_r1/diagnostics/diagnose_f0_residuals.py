#!/usr/bin/env python3
"""Run the bounded, source-bound F0 residual diagnostic."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import cast

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT / "src"))
from input_bindings import (  # noqa: E402
    MAX_BUNDLE_BYTES,
    MAX_PREDICTION_BYTES,
    MAX_ROWS,
    SCORE_TOLERANCE,
    _load_inputs,
    _resolve,
    sha256_file,
)
from residual_analysis import analyze_arrays, write_csv  # noqa: E402
from residual_moments import CHUNK_ROWS  # noqa: E402


def _verify_endpoint_scores(report: dict[str, object], receipts: dict[str, object]) -> None:
    """Compare recomputed endpoints with their frozen F0 receipts."""
    models = cast(dict[str, dict[str, object]], report["models"])
    selection = cast(dict[str, object], receipts["BLEND_SELECTION"])
    grade = cast(dict[str, object], receipts["INDEPENDENT_NATIVE_GRADE"])
    endpoints = cast(dict[str, dict[str, object]], selection["endpoints"])
    comparisons = {
        "wide": endpoints["wide"]["recomputed_score"],
        "deep": endpoints["deep"]["recomputed_score"],
    }
    comparisons["fp32_mix"] = next(
        candidate["score"]
        for candidate in cast(list[dict[str, object]], selection["candidates"])
        if candidate["selected_prediction_rows_sha256"]
        == cast(dict[str, object], report["inputs"])["model_hashes"]["fp32_mix"]
    )
    comparisons["q8"] = grade["scores"]
    for name, expected in comparisons.items():
        actual = cast(dict[str, object], models[name])["score"]
        expected_score = cast(dict[str, object], expected)
        for key in ("t0", "t1", "wp"):
            if (
                abs(float(cast(dict[str, object], actual)[key]) - float(expected_score[key]))
                > SCORE_TOLERANCE
            ):
                raise ValueError(f"{name} score does not match its frozen receipt for {key}")


def main() -> int:
    """Run the diagnostic and create a JSON plus CSV evidence pair."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, default=Path(__file__).with_name("INPUTS.json"))
    parser.add_argument("--output", type=Path, required=True, help="create-only JSON report path")
    args = parser.parse_args()
    manifest_path = args.inputs.resolve(strict=True)
    manifest, arrays, predictions, loaded_receipts = _load_inputs(manifest_path)
    source = cast(dict[str, object], manifest["source"])
    report = analyze_arrays(
        arrays["target"],
        predictions,
        arrays["row_group"],
        arrays["component"],
        arrays.get("position"),
        cast(list[dict[str, str]], manifest["comparisons"]),
    )
    bundle = cast(dict[str, object], manifest["bundle"])
    models = cast(dict[str, dict[str, object]], manifest["models"])
    report["inputs"] = {
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "bundle_path": str(_resolve(manifest_path.parent, str(bundle["path"]))),
        "bundle_sha256": bundle["sha256"],
        "source": source,
        "receipt_hashes": {
            name: item["sha256"]
            for name, item in cast(dict[str, dict[str, object]], manifest["receipts"]).items()
        },
        "model_hashes": {
            name: item.get("array_sha256", item.get("sha256")) for name, item in models.items()
        },
        "script_hashes": {
            "entrypoint": sha256_file(Path(__file__).resolve()),
            "input_validation": sha256_file(
                Path(__file__).with_name("input_bindings.py").resolve()
            ),
            "receipt_validation": sha256_file(
                Path(__file__).with_name("receipt_validation.py").resolve()
            ),
            "analysis": sha256_file(Path(__file__).with_name("residual_analysis.py").resolve()),
            "moments": sha256_file(Path(__file__).with_name("residual_moments.py").resolve()),
        },
        "execution_limits": {
            "max_rows": MAX_ROWS,
            "max_bundle_bytes": MAX_BUNDLE_BYTES,
            "max_external_prediction_bytes": MAX_PREDICTION_BYTES,
            "chunk_rows": CHUNK_ROWS,
        },
    }
    _verify_endpoint_scores(report, loaded_receipts)
    output_path = args.output.resolve()
    csv_path = output_path.with_suffix(".csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if csv_path.exists():
        raise FileExistsError(csv_path)
    with output_path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    write_csv(csv_path, report)
    print(
        json.dumps(
            {
                "status": "complete",
                "json": str(output_path),
                "csv": str(csv_path),
                "selected_rows": report["selected_rows"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
