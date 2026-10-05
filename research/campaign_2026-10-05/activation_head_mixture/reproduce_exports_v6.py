"""Independently reproduce E477 v4 scores and contrasts from paired exports."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
RUN = HERE / "eda/twelve_v6b"
RESULT = RUN / "RESULT.json"
EXPORT = RUN / "RESULT_paired_components.npz"
ATTEMPT = ROOT / "research/campaign_2026-10-04/runner/attempts/e477-head-mixture-cpu-test-grade-v6b.json"
RESULT_SEEDS = (271828, 271829)
ACTIVATIONS = ("gelu", "channel_split", "same_channel_mix")
DROPOUTS = (0.0, 0.1)
ARM_KEYS = tuple(
    f"seed{seed}_{activation}_p{'0' if probability == 0 else '01'}"
    for seed in RESULT_SEEDS
    for activation in ACTIVATIONS
    for probability in DROPOUTS
)
PAIR_CONTRASTS = (
    ("split_p0 - gelu_p0", "channel_split_p0", "gelu_p0"),
    ("mix_p0 - gelu_p0", "same_channel_mix_p0", "gelu_p0"),
    ("split_p0 - mix_p0", "channel_split_p0", "same_channel_mix_p0"),
    ("split_p01 - gelu_p01", "channel_split_p01", "gelu_p01"),
    ("mix_p01 - gelu_p01", "same_channel_mix_p01", "gelu_p01"),
    ("split_p01 - mix_p01", "channel_split_p01", "same_channel_mix_p01"),
)
N_BOOT = 2_000
CLIP = 2.0


def sha256_file(path: Path) -> str:
    """Hash a file with bounded memory."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    """Fail closed on missing, mismatched, or nonfinite exports."""
    if not condition:
        raise ValueError(message)


def centered_moments(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Build stable absolute-target-weighted centered moments for one component."""
    y64 = np.clip(np.asarray(y, dtype=np.float32), -CLIP, CLIP).astype(np.float64)
    p64 = np.clip(np.asarray(p, dtype=np.float32), -CLIP, CLIP).astype(np.float64)
    weight = np.abs(y64)
    active = weight > 0
    if not active.any():
        return np.zeros(6, dtype=np.float64)
    y64, p64, weight = y64[active], p64[active], weight[active]
    total = float(weight.sum())
    dy, dp = y64 - y64[0], p64 - p64[0]
    mean_y = float(np.sum(weight * dy) / total)
    mean_p = float(np.sum(weight * dp) / total)
    centered_y, centered_p = dy - mean_y, dp - mean_p
    return np.asarray(
        [
            total,
            y64[0] + mean_y,
            p64[0] + mean_p,
            np.sum(weight * centered_y**2),
            np.sum(weight * centered_p**2),
            np.sum(weight * centered_y * centered_p),
        ],
        dtype=np.float64,
    )


def merge_moments(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Merge two centered moments without raw-square subtraction."""
    if left[0] == 0:
        return right.copy()
    if right[0] == 0:
        return left.copy()
    total = left[0] + right[0]
    dy, dp = right[1] - left[1], right[2] - left[2]
    cross = left[0] * right[0] / total
    return np.asarray(
        [
            total,
            left[1] + dy * right[0] / total,
            left[2] + dp * right[0] / total,
            left[3] + right[3] + dy**2 * cross,
            left[4] + right[4] + dp**2 * cross,
            left[5] + right[5] + dy * dp * cross,
        ],
        dtype=np.float64,
    )


def score_target(moments: np.ndarray, counts: np.ndarray, target: int) -> float:
    """Recompute one pooled target WP from component moments."""
    weighted = counts * moments[:, target, 0]
    total = weighted.sum()
    if total <= 0:
        return 0.0
    active = np.flatnonzero(weighted > 0)
    anchor_y, anchor_p = moments[active[0], target, 1:3]
    dy = moments[:, target, 1] - anchor_y
    dp = moments[:, target, 2] - anchor_p
    mean_y, mean_p = np.sum(weighted * dy) / total, np.sum(weighted * dp) / total
    variance_y = np.sum(counts * moments[:, target, 3] + weighted * (dy - mean_y) ** 2)
    variance_p = np.sum(counts * moments[:, target, 4] + weighted * (dp - mean_p) ** 2)
    covariance = np.sum(counts * moments[:, target, 5] + weighted * (dy - mean_y) * (dp - mean_p))
    if np.sqrt(variance_y / total) <= 1e-8 or np.sqrt(variance_p / total) <= 1e-8:
        return 0.0
    return float(np.clip(covariance / np.sqrt(variance_y * variance_p), -1.0, 1.0))


def score_wp(moments: np.ndarray, counts: np.ndarray, target: int | None = None) -> float:
    """Evaluate the official equal mean of target-specific weighted Pearson values."""
    if target is not None:
        return score_target(moments, counts, target)
    return float(np.mean([score_target(moments, counts, index) for index in range(2)]))


def summarize(point: float, draws: np.ndarray, seed: int) -> dict:
    """Build the scorer's descriptive bootstrap fields."""
    return {
        "mean_diff": float(point),
        "bootstrap_sd": float(draws.std(ddof=1)),
        "ci95": np.percentile(draws, [2.5, 97.5]).tolist(),
        "p_le_0": float(np.mean(draws <= 0)),
        "n_components": 16,
        "n_boot": N_BOOT,
        "seed": seed,
    }


def compare_summary(recomputed: dict, recorded: dict, label: str) -> None:
    """Assert the independently recomputed point and interval match the result."""
    for field in ("mean_diff", "bootstrap_sd", "p_le_0"):
        require(np.isclose(recomputed[field], recorded[field], atol=2e-9, rtol=0),
                f"{label} {field}: {recomputed[field]} != {recorded[field]}")
    require(np.allclose(recomputed["ci95"], recorded["ci95"], atol=2e-9, rtol=0),
            f"{label} CI mismatch")
    require(recorded.get("n_boot") == N_BOOT and recorded.get("n_components") == 16,
            f"{label} bootstrap design mismatch")
    require(recorded.get("seed") == recomputed["seed"], f"{label} bootstrap seed mismatch")


def read_and_validate_exports(result_path: Path, export_path: Path) -> tuple[dict, dict]:
    """Load paired result/export and verify array alignment before scoring."""
    result = json.loads(result_path.read_text())
    require(result.get("status") == "scored_exposed_components_descriptive_only",
            "result is not a completed descriptive E477 grade")
    export_receipt = result.get("paired_export", {})
    require(export_receipt.get("sha256") == sha256_file(export_path),
            "paired export hash differs from JSON receipt")
    require(export_receipt.get("path") == str(export_path.relative_to(ROOT)),
            "paired export path differs from JSON receipt")
    require(result.get("metric_id") == "global_wp_clip2_v1", "official metric id mismatch")
    scorer_path = HERE / "score_head_mixture_v6.py"
    require(result.get("scorer", {}).get("source_sha256") == sha256_file(scorer_path),
            "result scorer hash differs from reviewed v6 source")
    expected_attempt = json.loads(ATTEMPT.read_text())
    require(expected_attempt.get("status") == "finished" and expected_attempt.get("exit_code") == 0,
            "grading attempt did not finish successfully")
    require(expected_attempt.get("failure") is None and expected_attempt.get("teardown_verified") is True,
            "grading attempt failure or teardown is unresolved")
    unit = expected_attempt.get("unit_properties", {})
    require(unit.get("Result") == "success" and str(unit.get("ExecMainStatus")) == "0",
            "grading systemd unit did not report success")
    artifact_map = {row.get("path"): row for row in expected_attempt.get("artifacts", [])}
    result_rel = str(result_path.relative_to(ROOT))
    export_rel = str(export_path.relative_to(ROOT))
    require(set(artifact_map) == {result_rel, export_rel}, "runner artifact inventory mismatch")
    for path, rel in ((result_path, result_rel), (export_path, export_rel)):
        row = artifact_map[rel]
        require(path.stat().st_size == row.get("size") and sha256_file(path) == row.get("sha256"),
                f"runner artifact hash/size mismatch: {rel}")
    with np.load(export_path, allow_pickle=False) as archive:
        expected_arrays = {
            "arm_keys", "component_ids", "row_groups", "selected_counts", "full_masks",
            "raw_targets", "raw_predictions", "selected_row_indices", "selected_sequence_indices",
        }
        require(set(archive.files) == expected_arrays, "paired export array inventory mismatch")
        arrays = {name: archive[name].copy() for name in expected_arrays}
    require(tuple(arrays["arm_keys"].tolist()) == ARM_KEYS, "arm-key order mismatch")
    require(arrays["component_ids"].shape == (16,) and arrays["row_groups"].shape == (16,),
            "component identity shape mismatch")
    require(arrays["selected_counts"].shape == (16,) and np.all(arrays["selected_counts"] > 0),
            "selected count vector invalid")
    counts = arrays["selected_counts"].astype(np.int64)
    rows = int(counts.sum())
    require(arrays["full_masks"].shape == (16, 20_000), "full mask shape mismatch")
    require(np.array_equal(arrays["full_masks"].sum(axis=1), counts), "mask/count mismatch")
    require(arrays["raw_targets"].shape == (rows, 2), "raw target shape mismatch")
    require(arrays["raw_predictions"].shape == (12, rows, 2), "raw prediction shape mismatch")
    require(arrays["selected_row_indices"].shape == (rows,), "row index shape mismatch")
    require(arrays["selected_sequence_indices"].shape == (rows,), "sequence index shape mismatch")
    expected_rows = np.concatenate([np.flatnonzero(mask) for mask in arrays["full_masks"]])
    expected_sequences = np.repeat(np.arange(16), counts)
    require(np.array_equal(arrays["selected_row_indices"], expected_rows),
            "selected row indices do not follow full masks")
    require(np.array_equal(arrays["selected_sequence_indices"], expected_sequences),
            "selected sequence index alignment mismatch")
    require(np.array_equal(arrays["component_ids"], np.asarray(result["data"]["component_ids"])),
            "component IDs differ from result metadata")
    require(np.array_equal(arrays["row_groups"], np.asarray(result["data"]["sequence_row_groups"])),
            "row groups differ from result metadata")
    require(np.array_equal(counts, np.asarray(result["data"]["selected_rows_by_component"])),
            "selected counts differ from result metadata")
    require(np.isfinite(arrays["raw_targets"]).all(), "raw selected labels contain nonfinite values")
    require(np.isfinite(arrays["raw_predictions"]).all(), "raw predictions contain nonfinite values")
    return result, arrays


def reproduce(result_path: Path, export_path: Path) -> dict:
    """Recompute all twelve scores and all eighteen registered contrasts."""
    result, data = read_and_validate_exports(result_path, export_path)
    targets = data["raw_targets"].astype(np.float32)
    predictions = data["raw_predictions"].astype(np.float32)
    component_ids = data["component_ids"].astype(np.int64)
    selected = int(len(targets))
    clipped_targets = np.clip(targets, -CLIP, CLIP).astype(np.float64)
    clipped_predictions = np.clip(predictions, -CLIP, CLIP).astype(np.float64)
    direct_scores: dict[str, dict] = {}
    moments_by_arm: dict[str, np.ndarray] = {}
    offsets = np.concatenate(([0], np.cumsum(data["selected_counts"], dtype=np.int64)))
    for arm_index, arm_key in enumerate(ARM_KEYS):
        moments = np.zeros((16, 2, 6), dtype=np.float64)
        for component in range(16):
            start, end = int(offsets[component]), int(offsets[component + 1])
            for target in range(2):
                moments[component, target] = centered_moments(
                    clipped_targets[start:end, target],
                    clipped_predictions[arm_index, start:end, target],
                )
        point = score_wp(moments, np.ones(16, dtype=np.int64))
        target_scores = [score_wp(moments, np.ones(16, dtype=np.int64), k) for k in range(2)]
        record = result["arms"].get(arm_key)
        require(record is not None, f"missing scored arm {arm_key}")
        require(record.get("selected_rows") == selected, f"selected row count mismatch: {arm_key}")
        require(record.get("component_ids") == component_ids.tolist(), f"arm component order mismatch: {arm_key}")
        require(np.isclose(point, record["global_wp"], atol=2e-7, rtol=0),
                f"global WP differs for {arm_key}")
        for target, label in enumerate(("t0", "t1")):
            require(np.isclose(target_scores[target], record["target_wp"][label], atol=2e-7, rtol=0),
                    f"{label} WP differs for {arm_key}")
        direct_scores[arm_key] = {"global_wp": point, "t0": target_scores[0], "t1": target_scores[1]}
        moments_by_arm[arm_key] = moments

    order = np.argsort(component_ids, kind="stable")
    sorted_moments = {key: value[order] for key, value in moments_by_arm.items()}
    all_contrasts: dict[str, dict] = {}
    for seed in RESULT_SEEDS:
        prefix = f"seed{seed}:"
        arms = {name: f"seed{seed}_{name}" for name in (
            "gelu_p0", "gelu_p01", "channel_split_p0", "channel_split_p01",
            "same_channel_mix_p0", "same_channel_mix_p01",
        )}
        for name, candidate_short, reference_short in PAIR_CONTRASTS:
            candidate, reference = arms[candidate_short], arms[reference_short]
            rng = np.random.default_rng(seed)
            draws = np.empty(N_BOOT, dtype=np.float64)
            ones = np.ones(16, dtype=np.int64)
            point = (score_wp(sorted_moments[candidate], ones)
                     - score_wp(sorted_moments[reference], ones))
            target_draws = {k: np.empty(N_BOOT, dtype=np.float64) for k in range(2)}
            target_points = {}
            for target in range(2):
                target_points[target] = (
                    score_wp(sorted_moments[candidate], ones, target)
                    - score_wp(sorted_moments[reference], ones, target)
                )
            for draw_index in range(N_BOOT):
                multiplicities = np.bincount(rng.integers(0, 16, 16), minlength=16)
                draws[draw_index] = (
                    score_wp(sorted_moments[candidate], multiplicities)
                    - score_wp(sorted_moments[reference], multiplicities)
                )
                for target in range(2):
                    target_draws[target][draw_index] = (
                        score_wp(sorted_moments[candidate], multiplicities, target)
                        - score_wp(sorted_moments[reference], multiplicities, target)
                    )
            summary = summarize(point, draws, seed)
            summary.update({"candidate": candidate_short, "reference": reference_short})
            summary["target_wp"] = {
                ("t0" if target == 0 else "t1"): summarize(target_points[target], target_draws[target], seed)
                for target in range(2)
            }
            recorded = result["contrasts"].get(prefix + name)
            require(recorded is not None, f"missing registered contrast {prefix + name}")
            require(recorded.get("family") == "preregistered_primary"
                    and recorded.get("candidate") == candidate_short
                    and recorded.get("reference") == reference_short,
                    f"primary contrast identity differs: {prefix + name}")
            compare_summary(summary, recorded, prefix + name)
            for target_index, target_label in enumerate(("t0", "t1")):
                compare_summary(summary["target_wp"][target_label],
                                recorded["target_wp"][str(target_index)],
                                prefix + name + "/" + target_label)
            all_contrasts[prefix + name] = summary

        secondary_pairs = (
            ("gelu_dropout_effect_secondary", arms["gelu_p01"], arms["gelu_p0"]),
        )
        for name, candidate, reference in secondary_pairs:
            # Use the same numeric component order and seed-specific draws as primary pairs.
            rng = np.random.default_rng(seed)
            draws = np.empty(N_BOOT, dtype=np.float64)
            ones = np.ones(16, dtype=np.int64)
            point = score_wp(sorted_moments[candidate], ones) - score_wp(sorted_moments[reference], ones)
            target_points = {k: score_wp(sorted_moments[candidate], ones, k)
                             - score_wp(sorted_moments[reference], ones, k) for k in range(2)}
            target_draws = {k: np.empty(N_BOOT, dtype=np.float64) for k in range(2)}
            for draw_index in range(N_BOOT):
                multiplicities = np.bincount(rng.integers(0, 16, 16), minlength=16)
                draws[draw_index] = score_wp(sorted_moments[candidate], multiplicities) - score_wp(
                    sorted_moments[reference], multiplicities
                )
                for target in range(2):
                    target_draws[target][draw_index] = score_wp(sorted_moments[candidate], multiplicities, target) - score_wp(
                        sorted_moments[reference], multiplicities, target
                    )
            summary = summarize(point, draws, seed)
            summary.update({"candidate": "gelu_p01", "reference": "gelu_p0"})
            summary["target_wp"] = {
                ("t0" if target == 0 else "t1"): summarize(target_points[target], target_draws[target], seed)
                for target in range(2)
            }
            recorded = result["contrasts"].get(prefix + name)
            require(recorded is not None and recorded.get("family") == "secondary_descriptive",
                    f"missing/misclassified secondary result {prefix + name}")
            compare_summary(summary, recorded, prefix + name)
            for target_index, target_label in enumerate(("t0", "t1")):
                compare_summary(summary["target_wp"][target_label],
                                recorded["target_wp"][str(target_index)],
                                prefix + name + "/" + target_label)
            all_contrasts[prefix + name] = summary

        interaction_defs = (
            ("mixture_x_dropout", arms["same_channel_mix_p01"], arms["same_channel_mix_p0"],
             arms["gelu_p01"], arms["gelu_p0"], "preregistered_primary"),
            ("split_x_dropout_secondary", arms["channel_split_p01"], arms["channel_split_p0"],
             arms["gelu_p01"], arms["gelu_p0"], "secondary_descriptive"),
        )
        for name, high_a, low_a, high_b, low_b, family in interaction_defs:
            rng = np.random.default_rng(seed)
            ones = np.ones(16, dtype=np.int64)
            def interaction(
                counts: np.ndarray,
                target: int | None = None,
                high_a: str = high_a,
                low_a: str = low_a,
                high_b: str = high_b,
                low_b: str = low_b,
            ) -> float:
                return ((score_wp(sorted_moments[high_a], counts, target)
                         - score_wp(sorted_moments[low_a], counts, target))
                        - (score_wp(sorted_moments[high_b], counts, target)
                           - score_wp(sorted_moments[low_b], counts, target)))
            point = interaction(ones)
            draws = np.empty(N_BOOT, dtype=np.float64)
            target_draws = {k: np.empty(N_BOOT, dtype=np.float64) for k in range(2)}
            target_points = {k: interaction(ones, k) for k in range(2)}
            for draw_index in range(N_BOOT):
                multiplicities = np.bincount(rng.integers(0, 16, 16), minlength=16)
                draws[draw_index] = interaction(multiplicities)
                for target in range(2):
                    target_draws[target][draw_index] = interaction(multiplicities, target)
            summary = summarize(point, draws, seed)
            summary["family"] = family
            summary["target_wp"] = {
                ("t0" if target == 0 else "t1"): summarize(target_points[target], target_draws[target], seed)
                for target in range(2)
            }
            recorded = result["contrasts"].get(prefix + name)
            require(recorded is not None and recorded.get("family") == family,
                    f"missing/misclassified interaction {prefix + name}")
            compare_summary(summary, recorded, prefix + name)
            for target_index, target_label in enumerate(("t0", "t1")):
                compare_summary(summary["target_wp"][target_label],
                                recorded["target_wp"][str(target_index)],
                                prefix + name + "/" + target_label)
            all_contrasts[prefix + name] = summary

    require(len(all_contrasts) == 18, f"expected 18 fixed paired results, found {len(all_contrasts)}")
    primary_count = sum(v.get("family") == "preregistered_primary" for v in result["contrasts"].values())
    secondary_count = sum(v.get("family") == "secondary_descriptive" for v in result["contrasts"].values())
    require(primary_count == 14 and secondary_count == 4, "primary/secondary contrast inventory mismatch")
    return {
        "status": "independent_e477_v6b_export_reproduction_passed",
        "result_sha256": sha256_file(result_path),
        "paired_export_sha256": sha256_file(export_path),
        "fit_attempt": result["seal"]["fit_attempt"],
        "metric_id": result["metric_id"],
        "arm_count": len(direct_scores),
        "arm_scores": direct_scores,
        "arm_score_max_abs_error_against_result": "<=2e-7",
        "contrast_count": len(all_contrasts),
        "primary_results": 14,
        "secondary_results": 4,
        "contrasts": all_contrasts,
        "bootstrap": {"n_boot": N_BOOT, "seed_per_fit_seed": True,
                      "components_sorted_numeric_for_shared_draw_mapping": True},
        "scope": "export-only reproduction for previously exposed components; no public inference or winner selection",
    }


def main() -> None:
    """Write a new receipt without overwriting any prior result."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, default=RESULT)
    parser.add_argument("--export", type=Path, default=EXPORT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite reproduction receipt: {args.output}")
    if not args.result.is_file() or not args.export.is_file():
        raise FileNotFoundError("wait for completed RESULT.json and paired NPZ before reproducing")
    receipt = reproduce(args.result, args.export)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": receipt["status"], "arms": receipt["arm_count"],
                      "contrasts": receipt["contrast_count"], "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
