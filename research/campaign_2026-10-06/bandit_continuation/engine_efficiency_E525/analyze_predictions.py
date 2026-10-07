"""Evaluate bounded engine precision diagnostics on the exact same two streams."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from src.wnn.gmetric import global_wp


def main() -> None:
    folder = Path(__file__).resolve().parent
    root = folder.parents[3]
    reference_path = folder.parent / "submission_C1_E524/torch_cpu_reference_first2.npz"
    label_path = folder.parent / "submission_C1_E524/s3_20seq_native.npz"
    with np.load(reference_path, allow_pickle=False) as archive:
        reference = archive["pred"]
    with np.load(label_path, allow_pickle=False) as archive:
        labels, mask = archive["y"][:2], archive["mask"][:2]
    records = []
    for members in (1, 2):
        anchor = reference[:members].mean(axis=0)
        score = global_wp(labels, anchor, mask)
        for version, path in (
            ("v1", folder / f"matched_v1_predictions_members{members}.npz"),
            ("v2", folder / f"predictions_members{members}.npz"),
        ):
            with np.load(path, allow_pickle=False) as archive:
                for case in archive.files:
                    prediction = archive[case]
                    candidate = global_wp(labels, prediction, mask)
                    records.append(
                        {
                            "members": members,
                            "version": version,
                            "case": case,
                            "wp": candidate["wp"],
                            "delta_vs_cpu_float32": candidate["wp"] - score["wp"],
                            "selected_rows": candidate["selected_rows"],
                            "prediction_container_sha256": hashlib.sha256(
                                path.read_bytes()
                            ).hexdigest(),
                            "prediction_array_sha256": hashlib.sha256(
                                np.ascontiguousarray(prediction).tobytes()
                            ).hexdigest(),
                            "prediction_dtype": str(prediction.dtype),
                            "prediction_shape": list(prediction.shape),
                        }
                    )
    report = {
        "scope": "Two exposed sequences only: precision diagnostic, no fold/public-quality forecast",
        "mask_source": str(label_path.relative_to(root)),
        "reference_sha256": hashlib.sha256(reference_path.read_bytes()).hexdigest(),
        "label_source_sha256": hashlib.sha256(label_path.read_bytes()).hexdigest(),
        "results": records,
    }
    report["supersedes_identity_label"] = (
        "V1 named the NPZ-container digest prediction_sha256; V2 distinguishes the container and each case array. Metric values unchanged."
    )
    (folder / "PRECISION_QUALITY_DIAGNOSTIC_V2.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
