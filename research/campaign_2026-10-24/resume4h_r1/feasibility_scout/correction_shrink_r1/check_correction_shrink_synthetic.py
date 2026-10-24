"""Check shrink-grade invariants on generated arrays only; no campaign payloads."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[5]
HERE = Path(__file__).resolve().parent
SCOUT_DIR = HERE.parent
ONLINE_DIR = ROOT / "research/experiment_drafts/online-relationships-20261022"
for path in (ROOT / "src", SCOUT_DIR, ONLINE_DIR, HERE):
    sys.path.insert(0, str(path))


def sha256_file(path: Path) -> str:
    """Hash one small synthetic or source file with bounded memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def synthetic_prepared_type() -> None:
    """Stub only the F0 dataclass so imports do not invoke its preflight."""
    module = ModuleType("f0_grade_preflight")
    module.PreparedF0Grade = type("PreparedF0Grade", (), {})  # type: ignore[attr-defined]
    module.prepare_f0_grade = lambda *_args, **_kwargs: None  # type: ignore[attr-defined]
    module.write_json_exclusive = lambda *_args, **_kwargs: None  # type: ignore[attr-defined]
    sys.modules["f0_grade_preflight"] = module


def main() -> int:
    """Exercise the scorer on toy arrays with campaign-file loads blocked."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--recipe",
        type=Path,
        default=HERE / "CORRECTION_SHRINK_RECIPE_R3.json",
    )
    args = parser.parse_args()
    recipe_path = args.recipe.resolve()
    recipe: dict[str, Any] = json.loads(recipe_path.read_text(encoding="utf-8"))
    assert recipe["revision"] == 3
    assert recipe["new_fit_count"] == 0

    synthetic_prepared_type()
    import correction_shrink_scoring as scorer
    import grade_correction_shrink_r1 as grade_runner
    from f0_grade_scoring import ROWS, TIME_BINS

    from wnn.gmetric import GlobalAccumulator, reference_global_wp

    assert recipe["shrink"]["alpha"] == 0.1
    assert recipe["shrink"]["coefficient_search"] is False
    scorer.validate_alpha(recipe["shrink"]["alpha"])
    for invalid_alpha in (float(scorer.ALPHA), 0.10000001, True):
        try:
            scorer.validate_alpha(invalid_alpha)
        except ValueError:
            continue
        raise AssertionError(f"noncanonical alpha was accepted: {invalid_alpha!r}")
    parent_values = np.asarray([1.25, -0.5], dtype=np.float32)
    raw_values = np.asarray([1.75, -1.5], dtype=np.float32)
    exact_shrink = parent_values + scorer.ALPHA * (raw_values - parent_values)
    expected_shrink = np.asarray([1.3, -0.6], dtype=np.float32)
    assert scorer.ALPHA.dtype == np.dtype(np.float32)
    assert exact_shrink.dtype == np.dtype(np.float32)
    assert np.array_equal(exact_shrink, expected_shrink)

    clipped_parent = np.float32(1.9)
    clipped_raw = np.float32(4.0)
    shrink_then_metric_clip = np.clip(
        clipped_parent + scorer.ALPHA * (clipped_raw - clipped_parent), -2.0, 2.0
    )
    metric_clip_before_shrink = clipped_parent + scorer.ALPHA * (
        np.clip(clipped_raw, -2.0, 2.0) - clipped_parent
    )
    assert shrink_then_metric_clip == np.float32(2.0)
    assert metric_clip_before_shrink == np.float32(1.91)
    assert shrink_then_metric_clip != metric_clip_before_shrink

    primary, secondary = scorer.expected_contrasts()
    assert recipe["primary_contrasts"] == primary
    assert recipe["secondary_contrasts"] == secondary
    assert len(primary) == 5 and len(secondary) == 13
    assert {item["name"] for item in primary} == {
        f"shrink_{cell}-minus-parent" for cell in scorer.CELL_NAMES
    }
    assert {item["name"] for item in secondary if "-minus-raw_" in item["name"]} == {
        f"shrink_{cell}-minus-raw_{cell}" for cell in scorer.CELL_NAMES
    }

    scope = recipe["scope"]
    target_pin = recipe["target_cache"]
    synthetic_preflight: dict[str, Any] = {
        "status": "passed_label_free_preflight",
        "recipe_sha256": recipe["base_recipe"]["sha256"],
        "source_manifest_sha256": recipe["base_source_manifest"]["sha256"],
        "evaluation_membership_sha256": scope["evaluation_membership_sha256"],
        "target_cache": {
            "path": target_pin["path"],
            "expected_sha256": target_pin["sha256"],
            "file_bytes": target_pin["file_bytes"],
            "labels_read": False,
            "content_hash_checked": False,
        },
        "artifacts": [],
    }
    parent_pin = recipe["parent_prediction"]
    synthetic_preflight["artifacts"].append(
        {
            "endpoint": "parent",
            "path": parent_pin["path"],
            "sha256": parent_pin["sha256"],
            "file_bytes": parent_pin["file_bytes"],
            "shape": parent_pin["shape"],
            "dtype": parent_pin["dtype"],
            "evaluation_membership_sha256": parent_pin["evaluation_membership_sha256"],
        }
    )
    for pin in recipe["candidate_predictions"].values():
        synthetic_preflight["artifacts"].append(
            {
                "run": pin["run"],
                "prediction_path": pin["prediction_path"],
                "prediction_sha256": pin["prediction_sha256"],
                "prediction_sidecar_sha256": pin["sidecar_sha256"],
                "provenance_sha256": pin["provenance_sha256"],
                "checkpoint_sha256": pin["checkpoint_sha256"],
                "file_bytes": pin["prediction_bytes"],
                "shape": pin["shape"],
                "dtype": pin["dtype"],
                "evaluation_membership_sha256": pin["evaluation_membership_sha256"],
            }
        )
    duplicate_pin = recipe["omitted_duplicate"]
    synthetic_preflight["artifacts"].append(
        {
            "run": duplicate_pin["run"],
            "prediction_path": duplicate_pin["prediction_path"],
            "prediction_sha256": duplicate_pin["prediction_sha256"],
            "prediction_sidecar_sha256": duplicate_pin["sidecar_sha256"],
            "provenance_sha256": duplicate_pin["provenance_sha256"],
            "checkpoint_sha256": duplicate_pin["checkpoint_sha256"],
            "file_bytes": duplicate_pin["prediction_bytes"],
            "shape": duplicate_pin["shape"],
            "dtype": duplicate_pin["dtype"],
            "evaluation_membership_sha256": duplicate_pin["evaluation_membership_sha256"],
        }
    )
    synthetic_metadata_fixture = SimpleNamespace(
        report=synthetic_preflight,
        candidate_prediction_paths={
            f"{arm}_lag{lag}": Path("synthetic.npy") for lag in (1, 16) for arm in ("A", "B", "C")
        },
        evaluation_groups=list(range(scope["evaluation_groups"])),
        components=np.asarray(
            [index % scope["evaluation_components"] for index in range(scope["evaluation_groups"])],
            dtype=np.int64,
        ),
    )
    synthetic_baseline = {
        "schema": "wnn.online-relationships-f0-grade.v1",
        "status": "graded_exploratory_f0_only",
        "preflight": synthetic_preflight,
    }
    grade_runner.verify_label_free_inputs(
        synthetic_metadata_fixture, synthetic_preflight, synthetic_baseline, recipe
    )
    invalid_metadata_fixture = SimpleNamespace(**vars(synthetic_metadata_fixture))
    invalid_metadata_fixture.candidate_prediction_paths = {"wrong": Path("synthetic.npy")}
    try:
        grade_runner.verify_label_free_inputs(
            invalid_metadata_fixture, synthetic_preflight, synthetic_baseline, recipe
        )
    except ValueError:
        candidate_key_mismatch_rejected = True
    else:
        raise AssertionError("candidate map with wrong keys was accepted")
    synthetic_admission = {
        "schema": "wnn.correction-shrink-f0-grade-admission.v1",
        "status": "admitted",
        "recipe_sha256": "synthetic-recipe-sha256",
        "source_manifest_sha256": recipe["source_manifest"]["sha256"],
        "base_recipe_sha256": recipe["base_recipe"]["sha256"],
        "base_source_manifest_sha256": recipe["base_source_manifest"]["sha256"],
        "baseline_grade_sha256": "synthetic-baseline-sha256",
        "preflight_receipt_sha256": "synthetic-preflight-sha256",
        "metric": "global_wp_clip2_v1",
        "alpha": 0.1,
        "new_fit_count": 0,
        "target_quality_reads_authorized": True,
        "output_path": recipe["output"]["path"],
        "resources": {
            "memory_max_bytes": 400000000,
            "cpu_cores": 1,
            "max_wall_seconds": 300,
            "swap_bytes": 0,
            "scratch": "/home/vlad/development/wunder/research_tmp",
        },
    }
    validated_resources = grade_runner.validate_admission(
        synthetic_admission,
        recipe,
        "synthetic-recipe-sha256",
        "synthetic-preflight-sha256",
        "synthetic-baseline-sha256",
    )
    assert validated_resources == synthetic_admission["resources"]
    rejected_admission = dict(synthetic_admission, alpha=0.2)
    try:
        grade_runner.validate_admission(
            rejected_admission,
            recipe,
            "synthetic-recipe-sha256",
            "synthetic-preflight-sha256",
            "synthetic-baseline-sha256",
        )
    except ValueError:
        admission_violation_rejected = True
    else:
        raise AssertionError("admission with a different alpha was accepted")

    phase = np.arange(2 * ROWS * 2, dtype=np.float32).reshape(2, ROWS, 2)
    targets = np.sin(phase * np.float32(0.0017)) * np.float32(2.8)
    targets[1] = np.cos(phase[1] * np.float32(0.0013)) * np.float32(2.4)
    base_parent = np.sin(phase * np.float32(0.0009)) * np.float32(2.7)
    parent = (base_parent + targets * np.float32(0.18)).astype(np.float32)
    masks = np.ones((2, ROWS), dtype=bool)
    masks[:, ::23] = False
    raw_arrays: dict[str, np.ndarray] = {}
    for index, cell in enumerate(scorer.CELL_NAMES):
        correction = np.cos(phase * np.float32(0.0007 + index * 0.00003))
        raw_arrays[cell] = (parent + correction * np.float32(0.22 + index * 0.01)).astype(
            np.float32
        )

    with tempfile.TemporaryDirectory(
        prefix="correction-shrink-synthetic-", dir="/home/vlad/development/wunder/research_tmp"
    ) as temporary:
        scratch = Path(temporary)
        target_marker = scratch / "synthetic-target-marker.bin"
        target_marker.write_bytes(b"synthetic-only-no-label-payload")
        parent_path = scratch / "parent.npy"
        np.save(parent_path, parent, allow_pickle=False)
        candidate_paths: dict[str, Path] = {}
        allowed_prediction_paths = {parent_path.resolve()}
        for index, (cell, prediction) in enumerate(raw_arrays.items()):
            path = scratch / f"candidate-{index}.npy"
            np.save(path, prediction, allow_pickle=False)
            candidate_paths[cell] = path
            allowed_prediction_paths.add(path.resolve())

        prepared = SimpleNamespace(
            parent_recipe={
                "target_cache": {"sha256": sha256_file(target_marker)},
                "resource_contract": {"scratch": "/home/vlad/development/wunder/research_tmp"},
            },
            target_cache_path=target_marker,
            parent_prediction_path=parent_path,
            candidate_prediction_paths=candidate_paths,
            evaluation_groups=[101, 205],
            components=np.asarray([3, 8], dtype=np.int64),
            synthetic_targets=targets,
            synthetic_masks=masks,
        )

        def synthetic_target_maps(
            value: SimpleNamespace, _scratch: Path
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[int, int], list[Path]]:
            groups = np.asarray(value.evaluation_groups, dtype=np.int64)
            return (
                value.synthetic_targets,
                value.synthetic_masks,
                groups,
                {int(group): index for index, group in enumerate(groups)},
                [],
            )

        scorer._materialize_target_maps = synthetic_target_maps
        original_load = np.load

        def guarded_load(file: Any, *args: Any, **kwargs: Any) -> np.ndarray:
            path = Path(file).resolve()
            if path not in allowed_prediction_paths:
                raise AssertionError(f"campaign or non-synthetic array access blocked: {path}")
            return original_load(file, *args, **kwargs)

        np.load = guarded_load
        try:
            scored = scorer.score_endpoints(prepared)
        finally:
            np.load = original_load

        rows_index = np.arange(ROWS)
        for cell, raw in raw_arrays.items():
            shrunk = parent + scorer.ALPHA * (raw - parent)
            assert shrunk.dtype == np.dtype(np.float32)
            for band, (start, stop) in TIME_BINS.items():
                selected = masks & (rows_index >= start) & (rows_index < stop)
                reference_raw = reference_global_wp(targets, raw, selected)
                reference_shrunk = reference_global_wp(targets, shrunk, selected)
                assert (
                    abs(scored["endpoint_scores"][f"raw_{cell}"][band]["wp"] - reference_raw)
                    < 1e-12
                )
                assert (
                    abs(scored["endpoint_scores"][f"shrink_{cell}"][band]["wp"] - reference_shrunk)
                    < 1e-12
                )
        baseline = {"endpoint_scores": {}}
        for cell in scorer.CELL_NAMES:
            baseline["endpoint_scores"][cell] = {}
            for band, (start, stop) in TIME_BINS.items():
                selected = masks & (rows_index >= start) & (rows_index < stop)
                accumulator = GlobalAccumulator()
                accumulator.add(targets, raw_arrays[cell], selected)
                baseline["endpoint_scores"][cell][band] = accumulator.result()
                reference = reference_global_wp(targets, raw_arrays[cell], selected)
                assert abs(accumulator.result()["wp"] - reference) < 1e-12
        scorer.compare_raw_scores(scored["endpoint_scores"], baseline)
        broken_baseline = json.loads(json.dumps(baseline))
        broken_baseline["endpoint_scores"][scorer.CELL_NAMES[0]]["rows_99_500"]["wp"] += 1e-5
        try:
            scorer.compare_raw_scores(scored["endpoint_scores"], broken_baseline)
        except ValueError:
            parity_mismatch_rejected = True
        else:
            raise AssertionError("raw baseline mismatch was not rejected")

        admission_path = scratch / "synthetic-admission.json"
        preflight_path = scratch / "synthetic-preflight.json"
        admission_path.write_text("{}\n", encoding="utf-8")
        preflight_path.write_text("{}\n", encoding="utf-8")
        result = scorer.build_result(
            recipe,
            "synthetic-recipe-sha256",
            prepared,
            scored,
            {"target_quality_reads_authorized": True},
            admission_path,
            preflight_path,
        )
        assert len(result["contrasts"]) == 18
        assert all(
            item["paired_component_bootstrap"]["n_components"] == 2 for item in result["contrasts"]
        )

    receipt = {
        "schema": "wnn.correction-shrink-synthetic-validation.v1",
        "status": "passed_synthetic_invariants",
        "recipe_id": recipe["recipe_id"],
        "recipe_revision": recipe["revision"],
        "campaign_prediction_or_target_payloads_read": False,
        "synthetic_arrays_only": True,
        "synthetic_shape": [2, ROWS, 2],
        "checks": {
            "fixed_alpha_float32_formula": True,
            "recipe_alpha_matches_exact_decimal_and_fp32_contract": True,
            "metric_clips_after_shrink_counterexample": True,
            "five_primary_and_thirteen_secondary_recipe_pairs": True,
            "raw_and_shrunken_wp_match_canonical_reference_in_every_time_bin": True,
            "raw_baseline_parity_accepts_exact_scores_and_rejects_a_mismatch": parity_mismatch_rejected,
            "all_18_paired_component_bootstrap_contrasts_build": True,
            "synthetic_label_free_metadata_fixture_passes": True,
            "candidate_map_key_mismatch_rejected_before_scoring": candidate_key_mismatch_rejected,
            "admission_gate_accepts_valid_contract_and_rejects_alpha_mismatch": admission_violation_rejected,
            "campaign_prediction_loads_blocked": True,
        },
        "source_sha256": {
            "checker": sha256_file(Path(__file__).resolve()),
            "scorer": sha256_file(HERE / "correction_shrink_scoring.py"),
            "official_metric": sha256_file(ROOT / "src/wnn/gmetric.py"),
            "component_bootstrap": sha256_file(ROOT / "src/wnn/evaluation.py"),
        },
    }
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    import numpy as np

    raise SystemExit(main())
