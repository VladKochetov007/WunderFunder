"""Separate WMSE/official-WP alignment from their mask-support mismatch."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from target_geometry import (
    CACHE,
    N_SAMPLE,
    N_VALID,
    SEQ_LEN,
    load_sample_predictions,
    metric_gradient,
    sha256_file,
)

OUT = Path(__file__).resolve().parent


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    """Return a finite vector cosine."""
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    return float(np.sum(left * right) / denominator)


def main() -> None:
    """Compute scored-coordinate WMSE alignment on the original fixed sample."""
    target_path = CACHE / "valid_targets.npz"
    index_path = CACHE / "valid_index.npz"
    with np.load(target_path, allow_pickle=False) as archive:
        all_y = archive["y"]
        all_mask = archive["mask"]
        all_groups = archive["row_group"].astype(np.int64)
    if all_y.shape != (N_VALID, SEQ_LEN, 2) or all_mask.shape != all_y.shape[:2]:
        raise RuntimeError("target/mask layout differs from the frozen panel")
    with np.load(index_path, allow_pickle=False) as archive:
        sequence_ids = archive["seq_ix"]
        if not np.array_equal(all_groups, archive["row_group"]):
            raise RuntimeError("target/index row-group identity mismatch")

    positions = np.linspace(0, N_VALID - 1, N_SAMPLE, dtype=np.int64)
    sample_groups = all_groups[positions]
    y = all_y[positions].astype(np.float64)
    mask = all_mask[positions].copy()
    sample_sequence_ids = sequence_ids[positions]
    student, student_meta, student_hashes = load_sample_predictions(
        "l14_kd90_s3", all_groups, sample_groups
    )
    teacher, teacher_meta, teacher_hashes = load_sample_predictions(
        "tt29rep_fbc1k_s5", all_groups, sample_groups
    )
    if student.shape != teacher.shape or teacher.shape != y.shape:
        raise RuntimeError("sample tensor layout mismatch")

    prior_path = OUT / "RESULTS_R1.json"
    prior = json.loads(prior_path.read_text())
    if prior["sample"]["valid_positions"] != [int(value) for value in positions]:
        raise RuntimeError("sample positions differ from the original result")
    if prior["sample"]["row_groups"] != [int(value) for value in sample_groups]:
        raise RuntimeError("sample row groups differ from the original result")
    if prior["sample"]["sequence_ids"] != [int(value) for value in sample_sequence_ids]:
        raise RuntimeError("sample sequence IDs differ from the original result")

    y_clipped = np.clip(y, -2.0, 2.0)
    weights = np.abs(y_clipped)
    wmse_gradient = 2.0 * weights * (student - y_clipped) / weights.sum()
    official_score, wp_gradient = metric_gradient(y, student, mask)
    scored_coordinates = np.broadcast_to(mask[..., None], student.shape)
    scored_wmse_gradient = wmse_gradient[scored_coordinates]
    scored_metric_gradient = -wp_gradient[scored_coordinates]
    prior_cosine = float(
        prior["metric_gradient"]["directions"]["label_wmse"]["metric_loss_gradient_cosine"]
    )
    recomputed_all_cosine = cosine(wmse_gradient, -wp_gradient)
    if abs(prior_cosine - recomputed_all_cosine) > 1e-12:
        raise RuntimeError("recomputed all-row WMSE cosine differs from frozen result")

    score_weights = weights * mask[..., None]
    masked_wmse_gradient = 2.0 * score_weights * (student - y_clipped) / score_weights.sum()
    masked_gradient = masked_wmse_gradient[scored_coordinates]
    target_cosines = {}
    for target in range(2):
        target_mask = mask
        target_cosines[f"t{target}"] = cosine(
            masked_wmse_gradient[..., target][target_mask],
            -wp_gradient[..., target][target_mask],
        )

    result = {
        "schema": "masked_wmse_diagnostic_r1",
        "parent_result_sha256": sha256_file(prior_path),
        "metric_id": "global_wp_clip2_v1",
        "sample_positions": [int(value) for value in positions],
        "sample_row_groups": [int(value) for value in sample_groups],
        "sample_sequence_ids": [int(value) for value in sample_sequence_ids],
        "sample_scored_rows": int(mask.sum()),
        "sample_mask_fraction": float(mask.mean()),
        "student_sample_wp": official_score,
        "student_wp_reference": float(prior["sample_scores"]["reference_wp"]),
        "all_row_label_wmse_cosine_preserved": prior_cosine,
        "all_row_label_wmse_cosine_recomputed": recomputed_all_cosine,
        "scored_coordinates_only_cosine": cosine(scored_wmse_gradient, scored_metric_gradient),
        "scored_only_wmse_objective_cosine": cosine(masked_gradient, scored_metric_gradient),
        "wmse_gradient_l2_fraction_on_scored_coordinates": float(
            np.linalg.norm(scored_wmse_gradient) / np.linalg.norm(wmse_gradient)
        ),
        "per_target_scored_only_cosine": target_cosines,
        "definition": "Cosine of the original abs(clip2(y))-weighted label-WMSE gradient and gradient of -official WP, restricted to official scored coordinates; also recomputed WMSE with the same weights and an official-mask multiplier. The two scored-only cosines differ only by a scalar normalization and should match.",
        "same_predictions": {
            "student": {"run": "l14_kd90_s3", "sidecar": student_meta, **student_hashes},
            "teacher": {"run": "tt29rep_fbc1k_s5", "sidecar": teacher_meta, **teacher_hashes},
        },
        "interpretation": "Supplemental alignment diagnostic on the original fixed sample; no coefficient search, no candidate perturbation grade, and no model-fit evidence.",
    }
    path = OUT / "MASKED_WMSE_DIAGNOSTIC_R1.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "output": str(path),
                "all_row_cosine": recomputed_all_cosine,
                "scored_coordinates_only_cosine": result["scored_coordinates_only_cosine"],
                "scored_only_wmse_objective_cosine": result["scored_only_wmse_objective_cosine"],
                "per_target": target_cosines,
                "scored_gradient_norm_fraction": result[
                    "wmse_gradient_l2_fraction_on_scored_coordinates"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
