"""Probe fixed valid-prediction directions against official pooled WP."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from wnn.gmetric import global_wp, reference_global_wp

REPO = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
CACHE = Path("/home/vlad/development/wunder/research_tmp/chrono_root/cache")
N_SAMPLE = 200
N_VALID = 1873
SEQ_LEN = 20_000
CLIP = 2.0
ALPHA = 0.9


def require(condition: bool, message: str) -> None:
    """Fail closed on prediction alignment or analytical-check errors."""
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    """Hash an input file in bounded memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def score_wp(y: np.ndarray, prediction: np.ndarray, mask: np.ndarray) -> float:
    """Return the fixed official pooled score."""
    return float(global_wp(y, prediction, mask)["wp"])


def metric_gradient(
    targets: np.ndarray, prediction: np.ndarray, mask: np.ndarray
) -> tuple[float, np.ndarray]:
    """Compute official WP and its prediction-space gradient through clip2."""
    y = np.clip(np.asarray(targets, dtype=np.float64), -CLIP, CLIP)
    p = np.asarray(prediction, dtype=np.float64)
    q = np.clip(p, -CLIP, CLIP)
    selected = np.asarray(mask, dtype=bool)
    grad = np.zeros_like(p)
    for target in range(2):
        w = np.abs(y[..., target]) * selected
        total = w.sum()
        mean_y = np.sum(w * y[..., target]) / total
        mean_p = np.sum(w * q[..., target]) / total
        centered_y = y[..., target] - mean_y
        centered_p = q[..., target] - mean_p
        var_y = np.sum(w * centered_y**2)
        var_p = np.sum(w * centered_p**2)
        scale = np.sqrt(var_y * var_p)
        rho = np.sum(w * centered_y * centered_p) / scale
        grad[..., target] = (
            0.5
            * w
            * (centered_y / scale - rho * centered_p / var_p)
            * ((p[..., target] > -CLIP) & (p[..., target] < CLIP))
        )
    return score_wp(targets, prediction, mask), grad


def increment_loss_gradient(
    prediction: np.ndarray, teacher: np.ndarray
) -> tuple[float, np.ndarray]:
    """Return uniform reset-per-sequence adjacent-error loss and gradient."""
    error = np.asarray(prediction, dtype=np.float64) - np.asarray(teacher, dtype=np.float64)
    difference = error[:, 1:] - error[:, :-1]
    denominator = error.shape[0] * (error.shape[1] - 1) * error.shape[2]
    gradient = np.empty_like(error)
    gradient[:, 0] = -2.0 * difference[:, 0] / denominator
    gradient[:, 1:-1] = 2.0 * (difference[:, :-1] - difference[:, 1:]) / denominator
    gradient[:, -1] = 2.0 * difference[:, -1] / denominator
    return float(np.sum(difference**2) / denominator), gradient


def weighted_mse_gradient(
    prediction: np.ndarray, target: np.ndarray, weight: np.ndarray
) -> tuple[float, np.ndarray]:
    """Return weighted-mean squared loss and gradient over output entries."""
    denominator = float(np.sum(weight))
    residual = prediction - target
    return (
        float(np.sum(weight * residual**2) / denominator),
        2.0 * weight * residual / denominator,
    )


def finite_difference_fixture() -> dict[str, object]:
    """Check analytic WP and increment gradients on deterministic CPU fixtures."""
    rng = np.random.default_rng(20261114)
    y = rng.uniform(-1.2, 1.2, size=(3, 9, 2)).astype(np.float32)
    p = rng.uniform(-0.8, 0.8, size=y.shape).astype(np.float32)
    mask = rng.uniform(size=y.shape[:2]) > 0.2
    direction = rng.normal(size=y.shape)
    direction /= np.linalg.norm(direction)
    _, wp_grad = metric_gradient(y, p, mask)
    wp_analytic = float(np.sum(wp_grad * direction))
    wp_checks = []
    for step in (2e-3, 1e-3):
        finite = (
            score_wp(y, p + step * direction, mask) - score_wp(y, p - step * direction, mask)
        ) / (2.0 * step)
        wp_checks.append(
            {
                "step": step,
                "analytic": wp_analytic,
                "finite_difference": finite,
                "abs_error": abs(finite - wp_analytic),
            }
        )
    student = rng.normal(scale=0.4, size=(3, 9, 2))
    teacher = rng.normal(scale=0.3, size=student.shape)
    inc_loss, inc_grad = increment_loss_gradient(student, teacher)
    inc_direction = rng.normal(size=student.shape)
    inc_direction /= np.linalg.norm(inc_direction)
    inc_analytic = float(np.sum(inc_grad * inc_direction))
    inc_checks = []
    for step in (1e-5, 2e-6):
        plus = increment_loss_gradient(student + step * inc_direction, teacher)[0]
        minus = increment_loss_gradient(student - step * inc_direction, teacher)[0]
        finite = (plus - minus) / (2.0 * step)
        inc_checks.append(
            {
                "step": step,
                "analytic": inc_analytic,
                "finite_difference": finite,
                "abs_error": abs(finite - inc_analytic),
            }
        )
    require(
        max(item["abs_error"] for item in wp_checks) < 2e-5, "WP finite-difference check failed"
    )
    require(
        max(item["abs_error"] for item in inc_checks) < 2e-9,
        "increment finite-difference check failed",
    )
    return {
        "wp_directional_derivative": wp_checks,
        "increment_directional_derivative": inc_checks,
        "increment_fixture_loss": inc_loss,
    }


def load_sample_predictions(
    run: str, all_groups: np.ndarray, sample_groups: np.ndarray
) -> tuple[np.ndarray, dict[str, object], dict[str, str]]:
    """Load full sequences by the sidecar's source row-group identity."""
    sidecar_path = CACHE / f"pred_{run}_valid.json"
    prediction_path = CACHE / f"pred_{run}_valid.npy"
    sidecar = json.loads(sidecar_path.read_text())
    row_groups = [int(value) for value in sidecar["row_groups"]]
    require(len(row_groups) == len(set(row_groups)), f"duplicate row groups for {run}")
    require(set(row_groups) == set(map(int, all_groups)), f"valid coverage mismatch for {run}")
    array = np.load(prediction_path, mmap_mode="r", allow_pickle=False)
    require(
        array.shape == (N_VALID, SEQ_LEN, 2), f"prediction layout mismatch for {run}: {array.shape}"
    )
    require(array.dtype == np.float32, f"prediction dtype mismatch for {run}")
    positions = {group: index for index, group in enumerate(row_groups)}
    sample = np.stack([array[positions[int(group)]] for group in sample_groups]).astype(np.float64)
    require(np.isfinite(sample).all(), f"nonfinite sampled predictions for {run}")
    metadata = {
        "run": sidecar.get("run"),
        "campaign_split_manifest_sha256": sidecar.get("campaign_split_manifest_sha256"),
        "evaluation_membership_sha256": sidecar.get("evaluation_membership_sha256"),
        "campaign_export_scope": sidecar.get("campaign_export_scope"),
        "feature_only": sidecar.get("feature_only"),
    }
    hashes = {
        "predictions_sha256": sha256_file(prediction_path),
        "sidecar_sha256": sha256_file(sidecar_path),
    }
    return sample, metadata, hashes


def direction_report(
    name: str, loss: float, loss_gradient: np.ndarray, wp_gradient: np.ndarray
) -> dict[str, float | str]:
    """Summarize normalized downhill WP derivative and gradient alignment."""
    loss_norm = float(np.linalg.norm(loss_gradient))
    wp_norm = float(np.linalg.norm(wp_gradient))
    cosine = float(np.sum(loss_gradient * -wp_gradient) / (loss_norm * wp_norm))
    return {
        "loss_name": name,
        "loss_value": loss,
        "loss_gradient_l2_norm": loss_norm,
        "metric_loss_gradient_cosine": cosine,
        "wp_derivative_along_unit_l2_downhill": wp_norm * cosine,
        "normalized_wp_directional_derivative_cosine": cosine,
    }


def main() -> None:
    """Run the fixed sample geometry probe and write one JSON receipt."""
    target_path = CACHE / "valid_targets.npz"
    index_path = CACHE / "valid_index.npz"
    with np.load(target_path, allow_pickle=False) as archive:
        all_y = archive["y"]
        all_mask = archive["mask"]
        all_groups = archive["row_group"].astype(np.int64)
    require(all_y.shape == (N_VALID, SEQ_LEN, 2), f"target shape mismatch: {all_y.shape}")
    require(all_mask.shape == all_y.shape[:2], "valid mask shape mismatch")
    with np.load(index_path, allow_pickle=False) as archive:
        require(np.array_equal(all_groups, archive["row_group"]), "target/index group mismatch")
        all_seq_ix = archive["seq_ix"]
    positions = np.linspace(0, N_VALID - 1, N_SAMPLE, dtype=np.int64)
    require(len(np.unique(positions)) == N_SAMPLE, "uniform sequence positions are not unique")
    sample_groups = all_groups[positions]
    sample_seq_ix = all_seq_ix[positions]
    y = all_y[positions].astype(np.float64)
    mask = all_mask[positions].copy()
    del all_y, all_mask

    student, student_meta, student_hashes = load_sample_predictions(
        "l14_kd90_s3", all_groups, sample_groups
    )
    teacher, teacher_meta, teacher_hashes = load_sample_predictions(
        "tt29rep_fbc1k_s5", all_groups, sample_groups
    )
    require(student.shape == teacher.shape == y.shape, "sample target/prediction shape mismatch")
    base_score, wp_gradient = metric_gradient(y, student, mask)
    reference_score = reference_global_wp(y, student, mask)
    official_detail = global_wp(y, student, mask)
    require(abs(base_score - reference_score) < 1e-10, "sample score/reference mismatch")

    clipped_y = np.clip(y, -CLIP, CLIP)
    label_weight = np.abs(clipped_y)
    clipped_teacher = np.clip(teacher, -CLIP, CLIP)
    recipe_target = (1.0 - ALPHA) * clipped_y + ALPHA * clipped_teacher
    inc_loss, inc_gradient = increment_loss_gradient(student, teacher)
    point_loss, point_gradient = weighted_mse_gradient(student, teacher, np.ones_like(student))
    recipe_loss, recipe_gradient = weighted_mse_gradient(student, recipe_target, label_weight)
    wmse_loss, wmse_gradient = weighted_mse_gradient(student, clipped_y, label_weight)
    directions = {
        "increment_kd_uniform": direction_report(
            "uniform D(p-teacher) squared", inc_loss, inc_gradient, wp_gradient
        ),
        "pointwise_teacher_mse_uniform": direction_report(
            "uniform (p-teacher) squared", point_loss, point_gradient, wp_gradient
        ),
        "recipe_pointwise_kd_alpha09": direction_report(
            "weighted (p-(0.1*clip2(y)+0.9*clip2(teacher))) squared",
            recipe_loss,
            recipe_gradient,
            wp_gradient,
        ),
        "label_wmse": direction_report(
            "weighted (p-clip2(y)) squared", wmse_loss, wmse_gradient, wp_gradient
        ),
    }
    result = {
        "schema": "target_geometry_r1",
        "metric_id": "global_wp_clip2_v1",
        "scope": "exposed-valid output-space geometry; no network-parameter derivatives or fit evidence",
        "sample": {
            "selection": "integer linspace positions 0..1872 inclusive",
            "sequence_count": N_SAMPLE,
            "valid_sequence_count": N_VALID,
            "valid_positions": [int(value) for value in positions],
            "row_groups": [int(value) for value in sample_groups],
            "sequence_ids": [int(value) for value in sample_seq_ix],
            "scored_rows": int(mask.sum()),
            "official_mask_fraction": float(mask.mean()),
        },
        "source": {
            "student": {
                "run": "l14_kd90_s3",
                "training": "train-only KD90, seed 3; local frozen same-count split recipe",
                "sidecar": student_meta,
                **student_hashes,
            },
            "teacher": {
                "run": "tt29rep_fbc1k_s5",
                "training": "local train-only PR49 recipe reproduction; same-count membership, byte-level equivalence not asserted",
                "sidecar": teacher_meta,
                **teacher_hashes,
            },
            "valid_targets_sha256": sha256_file(target_path),
            "valid_index_sha256": sha256_file(index_path),
            "ideas_cards_sha256": sha256_file(
                REPO / "research/campaign_2026-11-14/quality_r2/IDEAS_R1.md"
            ),
            "official_metric_sha256": sha256_file(REPO / "src/wnn/gmetric.py"),
            "trainer_losses_sha256": sha256_file(REPO / "src/wnn/training/losses.py"),
            "weighted_mse_sha256": sha256_file(REPO / "src/wnn/nn.py"),
        },
        "definitions": {
            "increment_loss": "sum over sequences, adjacent rows t=1..19999, and two targets of ((p_t-teacher_t)-(p_(t-1)-teacher_(t-1)))^2, divided by 200*19999*2; raw saved outputs; no label weights or score mask; no cross-sequence difference",
            "uniform_pointwise_kd": "mean over all sampled row/target entries of (student-teacher)^2, raw outputs",
            "recipe_pointwise_kd": "trainer alpha=.9 target 0.1*clip2(y)+0.9*clip2(teacher), squared error weighted by abs(clip2(y)) over all non-padding valid rows; normalized by total weight; since-based row weights omitted",
            "label_wmse": "squared error to clip2(y), weighted by abs(clip2(y)) over all non-padding valid rows; normalized by total weight",
            "official_wp_gradient": "official pooled two-target WP on supplied scored mask, clip2 predictions and labels, abs(clipped-label) weights; prediction clip derivative is zero outside (-2,2)",
            "direction_normalization": "loss downhill vector has unit L2 norm; reported WP slope is dWP/d(step) for an L2-unit output perturbation; normalized slope is its cosine with the WP ascent gradient",
            "supervised_metric_loss": "negative official WP",
            "since_weighting": "omitted",
        },
        "sample_scores": {
            "student_wp": base_score,
            "student_t0_wp": float(official_detail["t0"]),
            "student_t1_wp": float(official_detail["t1"]),
            "reference_wp": reference_score,
            "abs_error": abs(base_score - reference_score),
        },
        "clip_diagnostics": {
            "student_outside_clip_fraction_on_scored_outputs": float(
                np.mean(np.abs(student[mask]) > CLIP)
            ),
            "teacher_outside_clip_fraction_all_outputs": float(np.mean(np.abs(teacher) > CLIP)),
            "labels_outside_clip_fraction_all_outputs": float(np.mean(np.abs(y) > CLIP)),
        },
        "metric_gradient": {
            "l2_norm": float(np.linalg.norm(wp_gradient)),
            "directions": directions,
        },
        "pointwise_kd_vs_supervised_metric_gradient_cosine": {
            "uniform_teacher_mse": directions["pointwise_teacher_mse_uniform"][
                "metric_loss_gradient_cosine"
            ],
            "current_alpha09_recipe": directions["recipe_pointwise_kd_alpha09"][
                "metric_loss_gradient_cosine"
            ],
            "sign_convention": "cos(gradient of pointwise loss, gradient of -official WP); positive means local downhill directions agree",
        },
        "gradient_fixture": finite_difference_fixture(),
        "limitations": [
            "Uniformly sampled, exposed valid sequences only; no confidence claim or model selection follows.",
            "Prediction-space directions do not account for the GRU Jacobian, optimizer, minibatch normalization, or training trajectory.",
            "Increment weighting is a deliberately simple uniform descriptive choice; no weight or coefficient was searched.",
            "Since-based training weights are omitted from all probe losses.",
        ],
    }
    output_path = OUT / "RESULTS_R1.json"
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    compact = {
        "output": str(output_path),
        "sample_scores": result["sample_scores"],
        "directions": directions,
        "pointwise_kd_vs_supervised_metric_gradient_cosine": result[
            "pointwise_kd_vs_supervised_metric_gradient_cosine"
        ],
        "gradient_fixture": result["gradient_fixture"],
    }
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
