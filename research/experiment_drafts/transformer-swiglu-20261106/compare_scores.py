"""Compare two immutable endpoints on the same exposed fold and resample components."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from wnn.evaluation import paired_component_bootstrap
from wnn.splits import file_sha256

METRIC_ID = "global_wp_clip2_v1"
BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_SEED = 271828


def _read(path: Path) -> dict[str, Any]:
    """Read one bounded score receipt and require a JSON object."""
    if path.stat().st_size > 32_000_000:
        raise ValueError(f"score receipt is oversized: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("score receipt must be an object")
    return value


def _components(receipt: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Return sorted component IDs and centered target/prediction moments."""
    values = receipt.get("component_centered_moments")
    if not isinstance(values, dict) or not values:
        raise ValueError("score receipt is missing component-centered moments")
    identifiers = np.asarray(sorted(int(key) for key in values), dtype=np.int64)
    moments = np.asarray([values[str(key)] for key in identifiers], dtype=np.float64)
    if moments.shape != (len(identifiers), 2, 6) or not np.isfinite(moments).all():
        raise ValueError("component moments are not finite aligned (n, 2, 6) values")
    return identifiers, moments


def compare_scores(candidate_path: Path, reference_path: Path, output: Path) -> None:
    """Compute one fold's official paired delta and conditional component interval."""
    candidate, reference = _read(candidate_path), _read(reference_path)
    if candidate.get("split_sha256") != reference.get("split_sha256"):
        raise ValueError("paired scores must use the same source-qualified split")
    candidate_score, reference_score = candidate.get("score"), reference.get("score")
    if not isinstance(candidate_score, dict) or not isinstance(reference_score, dict):
        raise ValueError("paired score receipt is missing the official metric")
    if (
        candidate_score.get("metric_id") != METRIC_ID
        or reference_score.get("metric_id") != METRIC_ID
    ):
        raise ValueError("paired endpoints must use the frozen official metric")
    candidate_ids, candidate_moments = _components(candidate)
    reference_ids, reference_moments = _components(reference)
    if not np.array_equal(candidate_ids, reference_ids):
        raise ValueError("paired score receipts have different evaluation components")
    bootstrap = paired_component_bootstrap(
        candidate_moments,
        reference_moments,
        candidate_ids,
        n_boot=BOOTSTRAP_DRAWS,
        seed=BOOTSTRAP_SEED,
        centered=True,
    )
    if not np.isclose(bootstrap["candidate_wp"], candidate_score["wp"], rtol=0, atol=1e-10):
        raise ValueError("candidate component moments do not reproduce its pooled WP")
    if not np.isclose(bootstrap["reference_wp"], reference_score["wp"], rtol=0, atol=1e-10):
        raise ValueError("reference component moments do not reproduce its pooled WP")
    result = {
        "schema": "wnn.transformer-swiglu-paired-score.v1",
        "status": "paired_exposed_fold_screen",
        "metric_id": METRIC_ID,
        "split_sha256": candidate["split_sha256"],
        "candidate_score_sha256": file_sha256(candidate_path),
        "reference_score_sha256": file_sha256(reference_path),
        "candidate_predictions_manifest_sha256": candidate["predictions_manifest_sha256"],
        "reference_predictions_manifest_sha256": reference["predictions_manifest_sha256"],
        "candidate_producer": candidate["producer"],
        "reference_producer": reference["producer"],
        "paired_component_bootstrap": bootstrap,
        "uncertainty": "Conditional on the two fixed fits and exposed evaluation components; excludes training variation and transfer.",
    }
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")


def main() -> None:
    """Dispatch the fixed paired score comparison without loading a model."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    compare_scores(args.candidate, args.reference, args.output)


if __name__ == "__main__":
    main()
