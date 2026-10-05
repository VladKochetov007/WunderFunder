"""Independently reproduce E476 exported arm scores and selected contrasts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
RUN = HERE / "eda/twelve_v1"
SCORES = RUN / "scores.json"
EXPORT = RUN / "selected_exports.npz"
ATTEMPT = ROOT / "research/campaign_2026-10-04/runner/attempts/e476-activation-cpu-test-grade-v1.json"
COHORT = ROOT / "research/campaign_2026-10-05/target_mechanics/cohort.json"
DATA = ROOT / "datasets/valid.parquet"

import sys

sys.path.insert(0, str(ROOT / "src"))
from wnn.gmetric import METRIC_CLIP, METRIC_ID, global_wp  # noqa: E402

SEED = 271_828
N_BOOT = 2_000
ARM_NAMES = tuple(
    f"{activation}__{optimizer}__{schedule}"
    for activation in ("base", "sine_embed")
    for optimizer in ("adamw", "muon_adamw")
    for schedule in ("constant_peak", "warmcos_47_427", "constant_mean_warmcos")
)
PAIR_KEYS = (
    "base__adamw__warmcos_47_427_minus_base__adamw__constant_mean_warmcos",
    "sine_embed__adamw__warmcos_47_427_minus_sine_embed__adamw__constant_mean_warmcos",
)
INTERACTION_KEYS = (
    "sine_activation_x_warmcos_shape__adamw",
    "sine_activation_x_warmcos_shape__muon_adamw",
)


def sha256_file(path: Path) -> str:
    """Compute bounded-memory SHA-256."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    """Fail closed on export, cohort, or receipt mismatch."""
    if not condition:
        raise ValueError(message)


def _moments(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Compute centered weighted moments for one selected component/target."""
    weight = np.abs(y.astype(np.float64))
    active = weight > 0
    if not active.any():
        return np.zeros(6, dtype=np.float64)
    y64, p64, weight = y[active].astype(np.float64), p[active].astype(np.float64), weight[active]
    total = weight.sum()
    shifted_y, shifted_p = y64 - y64[0], p64 - p64[0]
    mean_y_shift = np.sum(weight * shifted_y) / total
    mean_p_shift = np.sum(weight * shifted_p) / total
    centered_y, centered_p = shifted_y - mean_y_shift, shifted_p - mean_p_shift
    return np.asarray(
        [
            total,
            y64[0] + mean_y_shift,
            p64[0] + mean_p_shift,
            np.sum(weight * centered_y * centered_y),
            np.sum(weight * centered_p * centered_p),
            np.sum(weight * centered_y * centered_p),
        ],
        dtype=np.float64,
    )


def _merge(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Merge two weighted centered-moment rows without raw-square subtraction."""
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
            left[1] + dy * (right[0] / total),
            left[2] + dp * (right[0] / total),
            left[3] + right[3] + dy * dy * cross,
            left[4] + right[4] + dp * dp * cross,
            left[5] + right[5] + dy * dp * cross,
        ],
        dtype=np.float64,
    )


def _wp(moment: np.ndarray) -> float:
    """Evaluate one target's centered weighted Pearson coefficient."""
    if moment[0] < 1e-8:
        return 0.0
    var_y, var_p = max(moment[3] / moment[0], 0.0), max(moment[4] / moment[0], 0.0)
    if np.sqrt(var_y) <= 1e-8 or np.sqrt(var_p) <= 1e-8:
        return 0.0
    return float(np.clip(moment[5] / moment[0] / np.sqrt(var_y * var_p), -1.0, 1.0))


def _pool(moments: np.ndarray, counts: np.ndarray) -> float:
    """Pool per-component two-target moments under a component count vector."""
    pooled_scores = []
    for target in range(2):
        current = np.zeros(6, dtype=np.float64)
        for component, multiplicity in enumerate(counts):
            if multiplicity == 0:
                continue
            repeated = moments[component, target].copy()
            repeated[[0, 3, 4, 5]] *= int(multiplicity)
            current = _merge(current, repeated)
        pooled_scores.append(_wp(current))
    return float(np.mean(pooled_scores))


def _arm_moments(targets: np.ndarray, predictions: np.ndarray, row_counts: np.ndarray) -> np.ndarray:
    """Build one model's 16-component moments from selected flat exports."""
    result = np.zeros((len(row_counts), 2, 6), dtype=np.float64)
    offset = 0
    for component, count in enumerate(row_counts):
        end = offset + int(count)
        y = np.clip(targets[offset:end], -METRIC_CLIP, METRIC_CLIP)
        p = np.clip(predictions[offset:end], -METRIC_CLIP, METRIC_CLIP)
        for target in range(2):
            result[component, target] = _moments(y[:, target], p[:, target])
        offset = end
    require(offset == len(targets) == len(predictions), "selected rows do not segment by component")
    return result


def _summarize_draws(point: float, draws: np.ndarray) -> dict:
    """Return the same descriptive bootstrap summaries as the sealed grader."""
    return {
        "mean_diff": float(point),
        "bootstrap_sd": float(draws.std(ddof=1)),
        "ci95": np.percentile(draws, [2.5, 97.5]).tolist(),
        "p_le_0": float(np.mean(draws <= 0)),
        "n_components": 16,
        "n_boot": N_BOOT,
        "seed": SEED,
    }


def verify_source_rows(
    components: np.ndarray,
    row_groups: np.ndarray,
    counts: np.ndarray,
    exported_targets: np.ndarray,
) -> dict:
    """Verify exported component order, test roles, row counts, masks and labels."""
    cohort_doc = json.loads(COHORT.read_text())
    source_nodes = cohort_doc["cohort"]
    fit_nodes = [node for node in source_nodes if node["role"] == "fit"]
    test_nodes = [node for node in source_nodes if node["role"] == "test"]
    require(len(fit_nodes) == 24 and len(test_nodes) == 16, "source cohort role counts differ")
    require(len({node["component"] for node in fit_nodes}) == 24, "duplicate fit components")
    expected_components = np.asarray([str(node["component"]) for node in test_nodes])
    expected_groups = np.asarray([node["row_group"] for node in test_nodes], dtype=np.int64)
    require(np.array_equal(components, expected_components), "export component order differs from cohort")
    require(np.array_equal(row_groups, expected_groups), "export row-group order differs from cohort")
    require(not ({node["component"] for node in fit_nodes} &
                 {node["component"] for node in test_nodes}), "fit/test component leakage")
    require(np.all(counts > 0) and int(counts.sum()) == len(exported_targets),
            "exported selected row counts do not cover target array")

    parquet = pq.ParquetFile(DATA)
    offset = 0
    for index, node in enumerate(test_nodes):
        table = parquet.read_row_group(
            int(node["row_group"]),
            columns=["t0", "t1", "need_prediction", "is_scored"],
            use_threads=False,
        )
        raw = np.column_stack((table["t0"].to_numpy(), table["t1"].to_numpy())).astype(np.float64)
        mask = table["need_prediction"].to_numpy().astype(bool)
        mask &= table["is_scored"].to_numpy().astype(bool)
        require(mask.any() and np.isfinite(raw[mask]).all(),
                f"selected source labels are empty/nonfinite in group {node['row_group']}")
        selected = np.clip(raw[mask], -METRIC_CLIP, METRIC_CLIP).astype(np.float32)
        require(len(selected) == int(counts[index]), f"selected count differs for group {node['row_group']}")
        end = offset + len(selected)
        require(np.array_equal(selected, exported_targets[offset:end]),
                f"raw parquet labels/mask do not reproduce export in group {node['row_group']}")
        offset = end
    require(offset == len(exported_targets), "source selected target rows do not cover export")
    return {"fit_components": len(fit_nodes), "test_components": len(test_nodes),
            "component_ids": components.tolist(), "row_groups": row_groups.tolist(),
            "selected_rows_per_component": counts.tolist(), "selected_rows": int(counts.sum())}


def reproduce(output: Path) -> dict:
    """Recompute all arm metrics and four selected paired contrasts."""
    scores = json.loads(SCORES.read_text())
    attempt = json.loads(ATTEMPT.read_text())
    require(attempt.get("id") == "e476-activation-cpu-test-grade-v1"
            and attempt.get("kind") == "cpu" and attempt.get("status") == "finished"
            and attempt.get("exit_code") == 0 and attempt.get("teardown_verified") is True,
            "scoring runner attempt did not finish cleanly")
    require(attempt.get("signals") == [] and attempt.get("failure") is None,
            "scoring runner attempt has a failure or signal")
    artifact_map = {item["path"]: item for item in attempt.get("artifacts", [])}
    result_rel = str(SCORES.relative_to(ROOT))
    export_rel = str(EXPORT.relative_to(ROOT))
    require(set(artifact_map) == {result_rel, export_rel}, "runner artifact inventory differs")
    require(SCORES.stat().st_size == artifact_map[result_rel]["size"]
            and sha256_file(SCORES) == artifact_map[result_rel]["sha256"],
            "scores.json hash/size differs from grading runner receipt")
    export_hash = sha256_file(EXPORT)
    require(EXPORT.stat().st_size == artifact_map[export_rel]["size"]
            and export_hash == artifact_map[export_rel]["sha256"]
            and export_hash == scores.get("selected_export_sha256"),
            "selected export hash/size differs from result or runner receipt")

    with np.load(EXPORT, allow_pickle=False) as data:
        require(set(data.files) == {"arm_names", "targets", "predictions", "row_counts",
                                   "component_ids", "row_groups"}, "export array inventory differs")
        arms = data["arm_names"].copy()
        targets = data["targets"].copy()
        predictions = data["predictions"].copy()
        counts = data["row_counts"].copy()
        components = data["component_ids"].copy()
        row_groups = data["row_groups"].copy()

    require(np.array_equal(arms, np.asarray(ARM_NAMES)), "registered arm names/order mismatch")
    require(targets.shape == (26990, 2) and predictions.shape == (12, 26990, 2),
            "selected target/prediction export shapes differ")
    require(counts.shape == (16,) and components.shape == (16,) and row_groups.shape == (16,),
            "component metadata shape differs")
    require(np.isfinite(targets).all() and np.isfinite(predictions).all(),
            "selected exported target or prediction contains nonfinite raw values")
    require(np.array_equal(counts, np.asarray(scores["sequence_row_counts"], dtype=np.int64)),
            "export row counts differ from score result")
    source_check = verify_source_rows(components, row_groups, counts, targets)
    require(scores.get("component_ids") == components.astype(np.int64).tolist()
            and scores.get("row_groups") == row_groups.tolist()
            and scores.get("selected_rows") == int(counts.sum()) == targets.shape[0],
            "result component indices or row counts do not match export")

    direct_arm_results = {}
    moments_by_arm = {}
    arm_score_errors = {}
    for index, name in enumerate(arms.tolist()):
        p = predictions[index]
        official = global_wp(targets[None], p[None], np.ones((1, len(targets)), dtype=bool))
        recorded = scores["arm_scores"].get(name)
        require(recorded is not None, f"missing score result for {name}")
        fields = {"wp": "wp", "t0": "t0", "t1": "t1"}
        errors = {}
        for output_field, official_field in fields.items():
            delta = float(official[official_field]) - float(recorded[output_field])
            errors[output_field] = delta
            require(abs(delta) <= 1e-7, f"official {output_field} mismatch for {name}: {delta}")
        require(official["metric_id"] == METRIC_ID and official["selected_rows"] == len(targets),
                f"official metric identity/count mismatch for {name}")
        parsed = tuple(name.split("__"))
        require(recorded.get("activation") == parsed[0]
                and recorded.get("optimizer") == parsed[1]
                and recorded.get("schedule") == parsed[2]
                and recorded.get("sequence_row_counts") == counts.tolist(),
                f"arm metadata mismatch for {name}")
        direct_arm_results[name] = {"wp": float(official["wp"]),
                                    "t0": float(official["t0"]),
                                    "t1": float(official["t1"]),
                                    "selected_rows": int(official["selected_rows"])}
        arm_score_errors[name] = errors
        moments_by_arm[name] = _arm_moments(targets, p, counts)

    # Match the grader's sorted-component indexing and fixed 2,000 common draws.
    numeric_components = components.astype(np.int64)
    sort_order = np.argsort(numeric_components, kind="stable")
    sorted_components = numeric_components[sort_order]
    sorted_moments = {key: value[sort_order] for key, value in moments_by_arm.items()}
    rng = np.random.default_rng(SEED)
    draws = [rng.integers(0, len(sorted_components), len(sorted_components)) for _ in range(N_BOOT)]
    counts_draws = [np.bincount(sample, minlength=len(sorted_components)) for sample in draws]

    paired_results = {}
    for key in PAIR_KEYS:
        record = scores["pair_contrasts"].get(key)
        require(record is not None, f"missing registered schedule contrast {key}")
        candidate, reference = record["candidate"], record["reference"]
        require(candidate.endswith("warmcos_47_427") and reference.endswith("constant_mean_warmcos"),
                f"contrast arm direction is unexpected: {key}")
        point = direct_arm_results[candidate]["wp"] - direct_arm_results[reference]["wp"]
        estimates = np.empty(N_BOOT, dtype=np.float64)
        for i, multiplicities in enumerate(counts_draws):
            estimates[i] = (_pool(sorted_moments[candidate], multiplicities)
                            - _pool(sorted_moments[reference], multiplicities))
        manual = _summarize_draws(point, estimates)
        _compare_bootstrap(manual, record, key)
        paired_results[key] = manual

    interaction_results = {}
    for key in INTERACTION_KEYS:
        record = scores["interactions"].get(key)
        require(record is not None, f"missing registered interaction {key}")
        optimizer = record["optimizer"]
        arms4 = (
            f"sine_embed__{optimizer}__warmcos_47_427",
            f"base__{optimizer}__warmcos_47_427",
            f"sine_embed__{optimizer}__constant_mean_warmcos",
            f"base__{optimizer}__constant_mean_warmcos",
        )
        point_scores = {arm: direct_arm_results[arm]["wp"] for arm in arms4}
        point = (point_scores[arms4[0]] - point_scores[arms4[1]]) - (
            point_scores[arms4[2]] - point_scores[arms4[3]])
        estimates = np.empty(N_BOOT, dtype=np.float64)
        for i, multiplicities in enumerate(counts_draws):
            pooled = {arm: _pool(sorted_moments[arm], multiplicities) for arm in arms4}
            estimates[i] = (pooled[arms4[0]] - pooled[arms4[1]]) - (
                pooled[arms4[2]] - pooled[arms4[3]])
        manual = _summarize_draws(point, estimates)
        manual["joint_component_resampling"] = True
        _compare_bootstrap(manual, record, key)
        interaction_results[key] = manual

    return {
        "status": "independent_export_reproduction_passed",
        "official_metric": METRIC_ID,
        "source_export_sha256": export_hash,
        "scores_json_sha256": sha256_file(SCORES),
        "grading_attempt": attempt["id"],
        "source_rows": source_check,
        "arm_count": len(direct_arm_results),
        "arm_score_max_abs_error": {
            metric: max(abs(values[metric]) for values in arm_score_errors.values())
            for metric in ("wp", "t0", "t1")
        },
        "reproduced_schedule_pairs": paired_results,
        "reproduced_interactions": interaction_results,
        "bootstrap": {"component_draws": N_BOOT, "seed": SEED,
                      "common_draws_reused_across_four_reproductions": True,
                      "pooling": "independently recomputed centered moments then official WP formula"},
        "scope": "all 12 exported arms and four requested contrasts; no model selection or public inference",
    }


def _compare_bootstrap(manual: dict, recorded: dict, key: str) -> None:
    """Require manual point, uncertainty, and sign summaries to match scorer output."""
    for field in ("mean_diff", "bootstrap_sd", "p_le_0"):
        require(np.isclose(manual[field], recorded[field], atol=2e-9, rtol=0),
                f"manual bootstrap {field} differs for {key}: {manual[field]} vs {recorded[field]}")
    require(np.allclose(manual["ci95"], recorded["ci95"], atol=2e-9, rtol=0),
            f"manual bootstrap interval differs for {key}")
    require(recorded.get("n_boot") == N_BOOT and recorded.get("seed") == SEED
            and recorded.get("n_components") == 16,
            f"bootstrap metadata differs for {key}")


def main() -> None:
    """Write a fresh independent receipt without changing scorer artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite independent receipt: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    receipt = reproduce(args.output)
    with args.output.open("x") as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": receipt["status"], "output": str(args.output),
                      "arms": receipt["arm_count"],
                      "contrasts": len(receipt["reproduced_schedule_pairs"])
                                   + len(receipt["reproduced_interactions"]),
                      "max_wp_abs_error": receipt["arm_score_max_abs_error"]["wp"]}, indent=2))


if __name__ == "__main__":
    main()
