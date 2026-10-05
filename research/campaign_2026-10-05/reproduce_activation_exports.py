"""Reproduce E477/E480 using shifted raw sums, without campaign scorer imports."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TOLERANCE = 1e-10


def file_hash(path: Path) -> str:
    """Hash the bounded scientific export."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def component_statistics(
    targets: np.ndarray, predictions: np.ndarray, counts: np.ndarray
) -> np.ndarray:
    """Build additive statistics around fixed anchors, independently of graders."""
    if not np.isfinite(targets).all() or not np.isfinite(predictions).all():
        raise ValueError("nonfinite raw export")
    y = np.clip(targets.astype(np.float32), -2, 2).astype(np.float64)
    p = np.clip(predictions.astype(np.float32), -2, 2).astype(np.float64)
    if y.shape != p.shape or y.shape != (int(counts.sum()), 2):
        raise ValueError("export shape/count mismatch")
    weight = np.abs(y)
    # Shared anchors preserve additivity under component resampling. This is
    # distinct from graders' within-component centering and pairwise merging.
    dy, dp = y - y[0], p - p[0]
    rows = np.stack(
        (weight, weight * dy, weight * dp, weight * dy**2, weight * dp**2, weight * dy * dp),
        axis=-1,
    )
    boundaries = np.cumsum(np.r_[0, counts])
    return np.stack(
        [
            rows[start:stop].sum(axis=0)
            for start, stop in zip(boundaries[:-1], boundaries[1:], strict=True)
        ]
    )


def scores(statistics: np.ndarray, multiplicities: np.ndarray) -> np.ndarray:
    """Compute both target correlations from additive resampled statistics."""
    sums = np.einsum("bc,ctm->btm", multiplicities, statistics)
    total, sy, sp, syy, spp, syp = np.moveaxis(sums, -1, 0)
    safe = np.where(total > 1e-8, total, 1)
    vy = np.maximum(syy - sy * sy / safe, 0)
    vp = np.maximum(spp - sp * sp / safe, 0)
    covariance = syp - sy * sp / safe
    denominator = np.sqrt(vy * vp)
    valid = (total > 1e-8) & (np.sqrt(vy / safe) > 1e-8)
    valid &= np.sqrt(vp / safe) > 1e-8
    result = np.zeros_like(covariance)
    np.divide(covariance, denominator, out=result, where=valid)
    return np.clip(result, -1, 1)


def draws(seed: int, components: int) -> np.ndarray:
    """Regenerate registered draws without trusting exported multiplicities."""
    rng = np.random.default_rng(seed)
    return np.stack(
        [
            np.bincount(rng.integers(0, components, components), minlength=components)
            for _ in range(2000)
        ]
    )


def assert_near(actual: np.ndarray | float, expected: np.ndarray | float) -> float:
    """Reject any disagreement beyond numerical summation noise."""
    error = float(np.max(np.abs(np.asarray(actual) - np.asarray(expected))))
    if error > TOLERANCE:
        raise ValueError(f"reproduction discrepancy {error:.3g}")
    return error


def verify_study(study: str) -> dict:
    """Verify every score and every global paired/interacting contrast."""
    is_schedule = study == "E480"
    directory = HERE / (
        "activation_schedule_replication/eda/twentyfour_v1"
        if is_schedule
        else "activation_head_mixture/eda/twelve_v6b"
    )
    result_path = directory / ("scores.json" if is_schedule else "RESULT.json")
    export_path = directory / (
        "selected_exports.npz" if is_schedule else "RESULT_paired_components.npz"
    )
    result = json.loads(result_path.read_text())
    error = 0.0
    points, boot = {}, {}
    with np.load(export_path, allow_pickle=False) as export:
        names = export["arm_names" if is_schedule else "arm_keys"].tolist()
        targets = export["targets_raw" if is_schedule else "raw_targets"]
        predictions = export["predictions_raw" if is_schedule else "raw_predictions"]
        counts = export["selected_row_counts" if is_schedule else "selected_counts"]
        ids = export["component_ids"]
        if len(np.unique(ids)) != 16 or len(names) != (24 if is_schedule else 12):
            raise ValueError("incomplete arm/component export")
        masks = export["selected_mask" if is_schedule else "full_masks"]
        if not np.array_equal(masks.sum(axis=1), counts):
            raise ValueError("raw selection count mismatch")
        order = np.argsort(ids, kind="stable")
        common_draws = draws(271828, len(ids)) if is_schedule else None
        if is_schedule and not np.array_equal(
            common_draws, export["component_bootstrap_multiplicities"]
        ):
            raise ValueError("exported component draws differ from regenerated draws")
        for index, name in enumerate(names):
            seed = int(name.split("_")[0].removeprefix("seed"))
            statistics = component_statistics(targets, predictions[index], counts)[order]
            target_scores = scores(statistics, np.ones((1, len(ids))))[0]
            points[name] = float(target_scores.mean())
            boot[name] = scores(
                statistics, common_draws if is_schedule else draws(seed, len(ids))
            ).mean(axis=1)
            arm = result["arm_scores" if is_schedule else "arms"][name]
            expected = (
                [arm["t0"], arm["t1"]]
                if is_schedule
                else [arm["target_wp"]["t0"], arm["target_wp"]["t1"]]
            )
            error = max(error, assert_near(target_scores, expected))
            error = max(error, assert_near(points[name], arm["wp" if is_schedule else "global_wp"]))
    contrasts = []
    if is_schedule:
        for seed, records in result["pair_contrasts_by_seed"].items():
            for record in records.values():
                terms = [
                    (f"seed{seed}__{record['candidate']}", 1),
                    (f"seed{seed}__{record['reference']}", -1),
                ]
                contrasts.append((record, terms))
        for seed, records in result["joint_interactions_by_seed"].items():
            for record in records.values():
                optimizer = record["optimizer"]
                terms = [
                    (f"seed{seed}__{activation}__{optimizer}__{schedule}", sign)
                    for activation, schedule, sign in (
                        ("sine_embed", "warmcos_47_427", 1),
                        ("base", "warmcos_47_427", -1),
                        ("sine_embed", "constant_mean_warmcos", -1),
                        ("base", "constant_mean_warmcos", 1),
                    )
                ]
                contrasts.append((record, terms))
    else:
        for name, record in result["contrasts"].items():
            prefix = name.split(":")[0]
            if "candidate" in record:
                terms = [
                    (f"{prefix}_{record['candidate']}", 1),
                    (f"{prefix}_{record['reference']}", -1),
                ]
            else:
                activation = "same_channel_mix" if "mixture_x" in name else "channel_split"
                terms = [
                    (f"{prefix}_{activation}_p01", 1),
                    (f"{prefix}_{activation}_p0", -1),
                    (f"{prefix}_gelu_p01", -1),
                    (f"{prefix}_gelu_p0", 1),
                ]
            contrasts.append((record, terms))
    for record, terms in contrasts:
        delta = sum(sign * points[name] for name, sign in terms)
        sampled = sum(sign * boot[name] for name, sign in terms)
        error = max(error, assert_near(delta, record["mean_diff"]))
        error = max(error, assert_near(np.quantile(sampled, [0.025, 0.975]), record["ci95"]))
        error = max(error, assert_near(sampled.std(ddof=1), record["bootstrap_sd"]))
        error = max(error, assert_near(float((sampled <= 0).mean()), record["p_le_0"]))
    return {
        "study": study,
        "status": "all_checked_values_match",
        "arms": len(points),
        "global_contrasts": len(contrasts),
        "maximum_absolute_error": error,
        "tolerance": TOLERANCE,
        "source_sha256": file_hash(Path(__file__)),
        "result_sha256": file_hash(result_path),
        "export_sha256": file_hash(export_path),
        "scope": "independent numerical method on existing exports; no model refit, inference or public confirmation",
    }


def main() -> None:
    """Run both complete export checks and preserve the reproduction receipt."""
    checks = [verify_study(study) for study in ("E477", "E480")]
    destination = HERE / "checks/activation_export_reproduction.json"
    destination.write_text(json.dumps(checks, indent=2) + "\n")
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
