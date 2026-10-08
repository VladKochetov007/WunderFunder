from __future__ import annotations

import argparse
import hashlib
import json
import resource
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
ROOT_FROM_SCRIPT = HERE.parents[3]

EXPERIMENT_ID = "LUNA-STACK-R2-20261009"
EXPECTED_SPLIT_SHA256 = "09263ad4fe2b18e2b2f3b79f4562678dce801b067b868134ddc79279f9316a3d"
TARGETS = ("t0", "t1")
SEQUENCE_ROWS = 20_000
BOOTSTRAP_SEED = 20261009
BOOTSTRAP_REPLICATES = 2_000
WARMUP = 99


@dataclass
class WeightedMoments:
    """Stable weighted centered moments for small meta-regression vectors."""

    weight: float
    mean: np.ndarray
    centered: np.ndarray

    @classmethod
    def empty(cls, dimensions: int) -> WeightedMoments:
        """Create an empty accumulator with a fixed vector width."""
        return cls(
            0.0,
            np.zeros(dimensions, dtype=np.float64),
            np.zeros((dimensions, dimensions), dtype=np.float64),
        )

    def add(self, values: np.ndarray, weights: np.ndarray) -> None:
        """Merge one batch into the centered weighted moments."""
        x = np.asarray(values, dtype=np.float64)
        w = np.asarray(weights, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != len(self.mean) or w.shape != (len(x),):
            raise ValueError("weighted moment batch has an invalid shape")
        batch_weight = float(w.sum(dtype=np.float64))
        if batch_weight <= 0:
            return
        batch_mean = np.sum(w[:, None] * x, axis=0, dtype=np.float64) / batch_weight
        centered = x - batch_mean
        batch_moment = (centered * w[:, None]).T @ centered
        self.merge(batch_weight, batch_mean, batch_moment)

    def merge(self, weight: float, mean: np.ndarray, centered: np.ndarray) -> None:
        """Merge another weighted moment state using a Chan update."""
        if weight <= 0:
            return
        if self.weight == 0:
            self.weight = weight
            self.mean = np.asarray(mean, dtype=np.float64).copy()
            self.centered = np.asarray(centered, dtype=np.float64).copy()
            return
        total = self.weight + weight
        delta = np.asarray(mean, dtype=np.float64) - self.mean
        cross = self.weight * weight / total
        self.centered += centered + np.outer(delta, delta) * cross
        self.mean += delta * (weight / total)
        self.weight = total


def _parse_args() -> argparse.Namespace:
    """Parse the root data path and root-owned admission receipt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--admission", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    """Hash a file in bounded chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    """Read a UTF-8 JSON object and reject unexpected top-level values."""
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _verify_authority(root: Path, admission_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Require a root receipt bound to the exact recipe and component assignment."""
    config_path = HERE / "LUNA_STACKING_META_CROSSFIT_R2.json"
    assignment_path = HERE / "LUNA_STACKING_COMPONENT_FOLDS_R2.json"
    config = _read_json(config_path)
    assignment = _read_json(assignment_path)
    admission = _read_json(admission_path)
    config_sha = _sha256(config_path)
    assignment_sha = assignment.get("assignment_sha256")
    if (
        config.get("experiment_id") != EXPERIMENT_ID
        or config.get("status") != "ready_for_root_admission"
    ):
        raise ValueError("R2 config is not the frozen ready candidate")
    if config.get("source_commit_base") != "8cba3f4d87765c9d9cf8145f9063cd2439aea25d":
        raise ValueError("unexpected R2 source base")
    if assignment_sha != config["outer_component_assignment"].get("assignment_sha256"):
        raise ValueError("component assignment hash differs from R2 config")
    if _sha256(assignment_path) != config["outer_component_assignment"].get(
        "membership_file_sha256"
    ):
        raise ValueError("component membership bytes differ from R2 config")
    if admission.get("experiment_id") != EXPERIMENT_ID or admission.get("status") != "admitted":
        raise ValueError("a root admission receipt for this exact experiment is required")
    if (
        admission.get("config_sha256") != config_sha
        or admission.get("assignment_sha256") != assignment_sha
    ):
        raise ValueError("root admission does not bind the frozen config and assignment")
    if admission.get("resource_caps") != {
        "cpu_cores": 1,
        "memory_max_bytes": 1_000_000_000,
        "runtime_max_seconds": 600,
        "gpu": False,
        "provider_or_kernel": False,
        "paid_spend": False,
    }:
        raise ValueError("root admission resource caps differ from the proposal")
    if root.resolve() == ROOT_FROM_SCRIPT.resolve():
        raise ValueError("--root must point to the shared main data workspace")
    return config, assignment


def _verify_grade_evidence(root: Path, config: dict[str, Any]) -> None:
    """Bind all eight eligible members to the two independent endpoint reports."""
    expected_members = {
        f"{member['run']}@ep{member['epoch']}": (
            member["checkpoint_sha256"],
            member["prediction_sha256"],
        )
        for endpoint in config["base_prediction_sources"]
        for member in endpoint["members"]
    }
    observed_members: dict[str, tuple[str, str]] = {}
    for evidence in config["ancestry_evidence"]["reports"]:
        report_path = root / evidence["path"]
        if _sha256(report_path) != evidence["sha256"]:
            raise ValueError(f"grade report hash mismatch: {evidence['path']}")
        report = _read_json(report_path)
        report_metric_id = report.get("metric_id", report.get("metric", {}).get("id"))
        if report_metric_id != "global_wp_clip2_v1":
            raise ValueError(f"grade report metric mismatch: {evidence['path']}")
        if (
            report.get(
                "split_manifest_sha256", report.get("metric", {}).get("split_manifest_sha256")
            )
            != EXPECTED_SPLIT_SHA256
        ):
            raise ValueError(f"grade report split mismatch: {evidence['path']}")
        if (
            report.get("evaluation_groups", report.get("metric", {}).get("evaluation_groups"))
            != 468
        ):
            raise ValueError(f"grade report population mismatch: {evidence['path']}")
        if report.get(
            "evaluation_components",
            report.get("components", report.get("metric", {}).get("components")),
        ) not in (141, None):
            raise ValueError(f"grade report component count mismatch: {evidence['path']}")
        no_public_forecast = report.get("claim_limits", {}).get("public_forecast") is False
        no_public_forecast |= "not a public-score forecast" in report.get("exposure", "").lower()
        if not no_public_forecast:
            raise ValueError(
                f"grade report does not rule out a public forecast: {evidence['path']}"
            )
        if evidence["kind"] == "h230" and report.get("status") != "graded_exposed_discovery_panel":
            raise ValueError("H230 evidence report has an unexpected status")
        if (
            evidence["kind"] == "z3"
            and report.get("status") != "comparison_valid_exploratory_no_improvement_observed"
        ):
            raise ValueError("exact-Z3 evidence report has an unexpected status")
        for section_name in (
            "candidate",
            "control",
            "candidate_endpoint",
            "matched_control_endpoint",
        ):
            section = report.get(section_name)
            if not isinstance(section, dict):
                continue
            bindings = section.get("member_bindings")
            if bindings is None:
                bindings = section.get("members", [])
            for binding in bindings:
                if isinstance(binding, str):
                    # Some frozen reports list names separately; their exact hash bindings
                    # live under the sibling member_bindings field.
                    continue
                run_at = binding.get("run_at")
                if run_at is None:
                    run_at = f"{binding.get('run')}@ep{binding.get('epoch')}"
                observed_members[run_at] = (
                    binding["checkpoint_sha256"],
                    binding["prediction_sha256"],
                )
        if (
            evidence["kind"] == "h230"
            and report.get("claim_limits", {}).get("heldout_claim") is not False
        ):
            raise ValueError("H230 report does not explicitly deny a heldout claim")
        if evidence["kind"] == "h230":
            for section_name in ("candidate", "control"):
                section = report[section_name]
                if any(
                    row.get("fit_membership_status") != "passed"
                    for row in section["member_bindings"]
                ):
                    raise ValueError("H230 member fit-membership guard did not pass")
        if evidence["kind"] == "z3":
            checks = report.get("recipe_and_provenance_checks", {})
            membership = checks.get("hft_fit_membership", {})
            if membership.get("evaluation_group_overlap") != 0:
                raise ValueError("Z3 HFT fit ancestry overlaps evaluated groups")
            if (
                checks.get("prediction_validation", {}).get(
                    "score_script_fit_membership_guard_passed"
                )
                is not True
            ):
                raise ValueError("Z3 prediction report fit-membership guard did not pass")
    if observed_members != expected_members:
        raise ValueError("frozen endpoint report members do not exactly match config pins")


def _verify_split(
    root: Path, config: dict[str, Any], assignment: dict[str, Any]
) -> tuple[dict[int, int], list[int], dict[int, int]]:
    """Bind evaluation groups and components to the hashed split manifest."""
    split_path = root / config["population"]["split_path"]
    split_bytes = split_path.read_bytes()
    split_sha = hashlib.sha256(split_bytes).hexdigest()
    if split_sha != EXPECTED_SPLIT_SHA256:
        raise ValueError(f"split manifest hash mismatch: {split_sha}")
    split = json.loads(split_bytes)
    if split.get("component_method") != config["population"]["component_method"]:
        raise ValueError("feature-only component construction differs from R2 config")
    groups = list(map(int, split["evaluation"]["valid"]))
    component_by_group = {
        int(row["row_group"]): int(row["component"]) for row in split["sequences"]["valid"]
    }
    if len(groups) != 468 or len(set(groups)) != 468:
        raise ValueError("expected 468 unique fold-0 evaluation row groups")
    assignment_by_component = {
        int(row["component_id"]): int(row["outer_fold"])
        for row in assignment["assignment"]["components"]
    }
    if assignment.get("source_split_sha256") != EXPECTED_SPLIT_SHA256:
        raise ValueError("component assignment was built from a different split manifest")
    assigned_components = assignment["assignment"]["components"]
    groups_by_component: dict[int, list[int]] = {}
    for group_id in groups:
        groups_by_component.setdefault(component_by_group[group_id], []).append(group_id)
    for component_row in assigned_components:
        component_id = int(component_row["component_id"])
        expected_groups = sorted(groups_by_component.get(component_id, []))
        if component_row.get("evaluation_groups") != expected_groups or not expected_groups:
            raise ValueError(f"component assignment row-group membership mismatch: {component_id}")
    fold_by_group: dict[int, int] = {}
    for group_id in groups:
        component_id = component_by_group.get(group_id)
        if component_id not in assignment_by_component:
            raise ValueError(f"evaluation group lacks frozen component assignment: {group_id}")
        fold_by_group[group_id] = assignment_by_component[component_id]
    if (
        len(assignment_by_component) != 141
        or sorted(assignment_by_component.values()).count(0) != 47
    ):
        raise ValueError("outer assignment is not the frozen 3x47 component design")
    if any(list(assignment_by_component.values()).count(fold) != 47 for fold in range(3)):
        raise ValueError("outer assignment has an incorrect component count")
    return component_by_group, groups, fold_by_group


def _verify_member_sources(
    root: Path,
    config: dict[str, Any],
    groups: list[int],
    deadline: float,
) -> tuple[dict[str, np.memmap], dict[str, dict[str, str]]]:
    """Verify exact sidecars, checkpoints, NPY bytes, and direct fit exclusion."""
    predictions: dict[str, np.memmap] = {}
    observed_hashes: dict[str, dict[str, str]] = {}
    eval_set = set(groups)
    full_row_groups = list(range(1_873))
    for endpoint in config["base_prediction_sources"]:
        for member in endpoint["members"]:
            name = member["run"]
            prediction_path = root / member["prediction_path"]
            sidecar_path = prediction_path.with_suffix(".json")
            checkpoint_path = root / "artifacts/runs" / name / f"model_ep{member['epoch']}.pt"
            provenance_path = root / "artifacts/runs" / name / "provenance.json"
            if (
                not prediction_path.is_file()
                or not sidecar_path.is_file()
                or not checkpoint_path.is_file()
                or not provenance_path.is_file()
            ):
                raise FileNotFoundError(f"missing frozen input for {name}")
            if time.monotonic() > deadline:
                raise TimeoutError(
                    "600-second admitted runtime budget exhausted during input verification"
                )
            sidecar = _read_json(sidecar_path)
            expected_prediction_sha = member["prediction_sha256"]
            actual_prediction_sha = _sha256(prediction_path)
            if actual_prediction_sha != expected_prediction_sha:
                raise ValueError(
                    f"prediction bytes differ from frozen pin: {name}; expected={expected_prediction_sha}; observed={actual_prediction_sha}"
                )
            actual_checkpoint_sha = _sha256(checkpoint_path)
            if actual_checkpoint_sha != member["checkpoint_sha256"]:
                raise ValueError(
                    f"checkpoint bytes differ from frozen pin: {name}; expected={member['checkpoint_sha256']}; observed={actual_checkpoint_sha}"
                )
            actual_sidecar_sha = _sha256(sidecar_path)
            actual_provenance_sha = _sha256(provenance_path)
            for identity, expected, actual in (
                ("sidecar", member["sidecar_sha256"], actual_sidecar_sha),
                ("provenance", member["provenance_sha256"], actual_provenance_sha),
            ):
                if actual != expected:
                    raise ValueError(
                        f"{identity} bytes differ from frozen pin: {name}; expected={expected}; observed={actual}"
                    )
            if sidecar.get("prediction_sha256") != expected_prediction_sha:
                raise ValueError(f"prediction sidecar hash differs from pin: {name}")
            if (
                sidecar.get("checkpoint_sha256") != member["checkpoint_sha256"]
                or sidecar.get("epoch") != member["epoch"]
            ):
                raise ValueError(f"sidecar checkpoint binding differs from pin: {name}")
            if sidecar.get("row_groups") != full_row_groups:
                raise ValueError(f"prediction row order is not canonical: {name}")
            if sidecar.get("split_manifest_sha256") != EXPECTED_SPLIT_SHA256:
                raise ValueError(f"sidecar split differs from frozen split: {name}")
            array = np.load(prediction_path, mmap_mode="r", allow_pickle=False)
            if array.shape != (1_873, SEQUENCE_ROWS, 2) or array.dtype != np.float32:
                raise ValueError(f"prediction NPY shape/dtype mismatch: {name}")
            provenance = _read_json(provenance_path)
            if provenance.get("split_manifest_sha256") != EXPECTED_SPLIT_SHA256:
                raise ValueError(f"provenance split differs from frozen split: {name}")
            fit_groups = provenance.get("fit_row_groups", {}).get("valid")
            if not isinstance(fit_groups, list) or eval_set.intersection(map(int, fit_groups)):
                raise ValueError(
                    f"direct fold-0 evaluation fit overlap or missing membership: {name}"
                )
            predictions[f"{name}@ep{member['epoch']}"] = array
            observed_hashes[f"{name}@ep{member['epoch']}"] = {
                "prediction_sha256": actual_prediction_sha,
                "checkpoint_sha256": actual_checkpoint_sha,
                "sidecar_sha256": actual_sidecar_sha,
                "provenance_sha256": actual_provenance_sha,
            }
    return predictions, observed_hashes


def _load_targets(
    root: Path,
    groups: list[int],
    component_by_group: dict[int, int],
    deadline: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read only the selected target/mask columns for frozen evaluation row groups."""
    source = _read_json(HERE / "LUNA_STACKING_META_CROSSFIT_R2.json")["target_source"]
    parquet_path = root / source["parquet_path"]
    file_size = parquet_path.stat().st_size
    footer_bytes = int(source["parquet_footer_size_bytes"])
    with parquet_path.open("rb") as stream:
        stream.seek(file_size - footer_bytes)
        footer = stream.read(footer_bytes)
    observed_footer_sha = hashlib.sha256(footer).hexdigest()
    if observed_footer_sha != source["parquet_footer_sha256"]:
        raise ValueError(
            f"valid parquet footer identity differs from the frozen split; expected={source['parquet_footer_sha256']}; observed={observed_footer_sha}"
        )
    parquet = pq.ParquetFile(parquet_path)
    if parquet.num_row_groups != 1_873:
        raise ValueError("valid parquet row-group count changed")
    targets = np.zeros((len(groups), SEQUENCE_ROWS, 2), dtype=np.float32)
    is_scored = np.zeros((len(groups), SEQUENCE_ROWS), dtype=bool)
    sequence_ids = np.zeros(len(groups), dtype=np.int64)
    split = json.loads(
        (root / "research/campaign_2026-09-29/artifacts/splits/fold0.json").read_text()
    )
    seq_by_group = {
        int(row["row_group"]): int(row["seq_ix"]) for row in split["sequences"]["valid"]
    }
    for index, group_id in enumerate(groups):
        if time.monotonic() > deadline:
            raise TimeoutError(
                "600-second admitted runtime budget exhausted while reading selected targets"
            )
        table = parquet.read_row_group(
            group_id, columns=["seq_ix", "t0", "t1", "is_scored"], use_threads=False
        )
        if table.num_rows != SEQUENCE_ROWS:
            raise ValueError(f"valid group has an unexpected row count: {group_id}")
        seq = table.column("seq_ix").to_numpy(zero_copy_only=False)
        if not np.all(seq == seq_by_group[group_id]):
            raise ValueError(f"source-qualified sequence identity mismatch: {group_id}")
        targets[index, :, 0] = (
            table.column("t0").to_numpy(zero_copy_only=False).astype(np.float32, copy=False)
        )
        targets[index, :, 1] = (
            table.column("t1").to_numpy(zero_copy_only=False).astype(np.float32, copy=False)
        )
        is_scored[index] = (
            table.column("is_scored").to_numpy(zero_copy_only=False).astype(bool, copy=False)
        )
        sequence_ids[index] = seq_by_group[group_id]
    if not np.isfinite(targets).all():
        raise ValueError("selected targets contain nonfinite values")
    need_prediction = np.arange(SEQUENCE_ROWS) >= WARMUP
    mask = is_scored & need_prediction[None, :]
    return targets, mask, sequence_ids


def _endpoint_group(
    predictions: dict[str, np.memmap], endpoint: dict[str, Any], group_id: int
) -> np.ndarray:
    """Recreate one endpoint by its frozen raw-member average."""
    member_predictions = []
    for member in endpoint["members"]:
        key = f"{member['run']}@ep{member['epoch']}"
        member_predictions.append(np.asarray(predictions[key][group_id], dtype=np.float64))
    return (0.5 * (member_predictions[0] + member_predictions[1])).astype(np.float32)


def _fit_fold(
    train_indices: list[int],
    groups: list[int],
    endpoint_specs: list[dict[str, Any]],
    predictions: dict[str, np.memmap],
    targets: np.ndarray,
    mask: np.ndarray,
) -> tuple[list[np.ndarray], list[np.ndarray], list[float], list[list[list[float]]]]:
    """Fit the fixed two-target ridge correction on outer-training components."""
    stats = [WeightedMoments.empty(4) for _ in TARGETS]
    for index in train_indices:
        group_id = groups[index]
        p3, h3, p4, z4 = [_endpoint_group(predictions, spec, group_id) for spec in endpoint_specs]
        d = np.stack((p3 - p4, h3 - p4, z4 - p4), axis=2)
        selected = mask[index]
        for target_index in range(2):
            y = np.clip(targets[index, selected, target_index], -2.0, 2.0).astype(np.float64)
            weight = np.abs(y)
            residual = y - p4[selected, target_index]
            values = np.column_stack((d[selected, target_index, :], residual))
            stats[target_index].add(values, weight)
    coefficients = []
    feature_means = []
    lambdas = []
    covariance_output = []
    for target_stats in stats:
        if target_stats.weight <= 0:
            raise ValueError("outer-training components have no weighted selected labels")
        covariance = target_stats.centered[:3, :3] / target_stats.weight
        cross_covariance = target_stats.centered[:3, 3] / target_stats.weight
        regularizer = 0.5 * float(np.trace(covariance)) / 3.0 + 1e-12
        theta = np.linalg.solve(covariance + regularizer * np.eye(3), cross_covariance)
        coefficients.append(theta)
        feature_means.append(target_stats.mean[:3].copy())
        lambdas.append(regularizer)
        covariance_output.append(covariance.tolist())
    return coefficients, feature_means, lambdas, covariance_output


def _metric_moments(y: np.ndarray, p: np.ndarray, mask: np.ndarray) -> WeightedMoments:
    """Build stable weighted centered moments for one target/component."""
    y_clip = np.clip(np.asarray(y, dtype=np.float64), -2.0, 2.0)[mask]
    p_clip = np.clip(np.asarray(p, dtype=np.float64), -2.0, 2.0)[mask]
    weights = np.abs(y_clip)
    selected = weights > 0
    values = np.column_stack((y_clip[selected], p_clip[selected]))
    moments = WeightedMoments.empty(2)
    moments.add(values, weights[selected])
    return moments


def _rho(moments: WeightedMoments) -> float:
    """Compute one target's weighted Pearson from centered moments."""
    if moments.weight <= 1e-8:
        return 0.0
    sy = np.sqrt(max(float(moments.centered[0, 0]) / moments.weight, 0.0))
    sp = np.sqrt(max(float(moments.centered[1, 1]) / moments.weight, 0.0))
    if sy <= 1e-8 or sp <= 1e-8:
        return 0.0
    return float(np.clip(moments.centered[0, 1] / moments.weight / (sy * sp), -1.0, 1.0))


def _merge_metric_moments(left: WeightedMoments, right: WeightedMoments) -> WeightedMoments:
    """Return a merged copy for paired component bootstrap resampling."""
    merged = WeightedMoments.empty(2)
    merged.merge(left.weight, left.mean, left.centered)
    merged.merge(right.weight, right.mean, right.centered)
    return merged


def _component_bootstrap(
    component_moments: dict[int, dict[str, list[WeightedMoments]]],
    folds_by_component: dict[int, int],
) -> dict[str, float | list[float] | int]:
    """Bootstrap paired ridge-anchor scores within each frozen outer fold."""
    components_by_fold = {
        fold: [component for component, current in folds_by_component.items() if current == fold]
        for fold in range(3)
    }
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    deltas = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    for replicate in range(BOOTSTRAP_REPLICATES):
        sampled = [
            int(x)
            for fold in range(3)
            for x in rng.choice(
                components_by_fold[fold], size=len(components_by_fold[fold]), replace=True
            )
        ]
        merged = {
            name: [WeightedMoments.empty(2), WeightedMoments.empty(2)] for name in ("ridge", "P4")
        }
        for component in sampled:
            for name in ("ridge", "P4"):
                for target_index in range(2):
                    merged[name][target_index] = _merge_metric_moments(
                        merged[name][target_index],
                        component_moments[component][name][target_index],
                    )
        score = {
            name: 0.5 * sum(_rho(m) for m in target_moments)
            for name, target_moments in merged.items()
        }
        deltas[replicate] = score["ridge"] - score["P4"]
    return {
        "replicates": BOOTSTRAP_REPLICATES,
        "seed": BOOTSTRAP_SEED,
        "delta_mean": float(deltas.mean()),
        "ci95_percentile": [float(x) for x in np.quantile(deltas, [0.025, 0.975])],
        "probability_delta_positive": float(np.mean(deltas > 0)),
    }


def main() -> int:
    """Run the root-admitted bounded meta-crossfit and preserve exact outputs."""
    args = _parse_args()
    root = args.root.resolve()
    sys.path.insert(0, str(root / "src"))
    from wnn import gmetric  # noqa: PLC0415

    if (
        Path(gmetric.__file__).resolve() != (root / "src/wnn/gmetric.py").resolve()
        or gmetric.METRIC_ID != "global_wp_clip2_v1"
        or gmetric.WARMUP != WARMUP
    ):
        raise ValueError("main-workspace metric implementation or warmup differs from frozen R2")
    config, assignment = _verify_authority(root, args.admission.resolve())
    if _sha256(root / "src/wnn/gmetric.py") != config["metric"]["implementation_sha256"]:
        raise ValueError("official metric source hash differs from the admitted config")
    output_root = args.out.resolve()
    allowed_root = (root / "artifacts/analysis/luna_stacking_meta_r2").resolve()
    if not output_root.is_relative_to(allowed_root) or output_root.exists():
        raise ValueError("output path must be a new child of the frozen R2 artifact directory")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=False, exist_ok=False)
    started = time.monotonic()
    deadline = started + 600.0
    try:
        _verify_grade_evidence(root, config)
        component_by_group, groups, fold_by_group = _verify_split(root, config, assignment)
        endpoints = config["base_prediction_sources"]
        predictions, observed_input_hashes = _verify_member_sources(root, config, groups, deadline)
        targets, mask, sequence_ids = _load_targets(root, groups, component_by_group, deadline)
        if int(mask.sum()) != 808_863:
            raise ValueError(f"selected row count changed: {int(mask.sum())}")

        oof_path = output_root / "meta_oof_ridge.npy"
        oof = np.lib.format.open_memmap(
            oof_path,
            mode="w+",
            dtype=np.float32,
            shape=(len(groups), SEQUENCE_ROWS, 2),
        )
        metric_names = ["ridge", "equal_four_way", "P3", "H3", "P4", "Z4"]
        accumulators = {name: gmetric.GlobalAccumulator() for name in metric_names}
        coefficient_report = []
        component_moments: dict[int, dict[str, list[WeightedMoments]]] = {}
        folds_by_component: dict[int, int] = {}

        for outer_fold in range(3):
            if time.monotonic() > deadline:
                raise TimeoutError("600-second admitted runtime budget exhausted before outer fold")
            test_indices = [
                i for i, group in enumerate(groups) if fold_by_group[group] == outer_fold
            ]
            train_indices = [
                i for i, group in enumerate(groups) if fold_by_group[group] != outer_fold
            ]
            train_components = {component_by_group[groups[i]] for i in train_indices}
            test_components = {component_by_group[groups[i]] for i in test_indices}
            if (
                len(train_components) != 94
                or len(test_components) != 47
                or train_components & test_components
            ):
                raise ValueError("component outer fold counts or separation are invalid")
            theta, means, lambdas, covariance = _fit_fold(
                train_indices,
                groups,
                endpoints,
                predictions,
                targets,
                mask,
            )
            coefficient_report.append(
                {
                    "outer_fold": outer_fold,
                    "training_components": len(train_components),
                    "evaluation_components": len(test_components),
                    "theta_t0": theta[0].tolist(),
                    "theta_t1": theta[1].tolist(),
                    "feature_mean_t0": means[0].tolist(),
                    "feature_mean_t1": means[1].tolist(),
                    "lambda_t0": lambdas[0],
                    "lambda_t1": lambdas[1],
                    "training_covariance_t0": covariance[0],
                    "training_covariance_t1": covariance[1],
                }
            )
            for index in test_indices:
                group_id = groups[index]
                component_id = component_by_group[group_id]
                folds_by_component[component_id] = outer_fold
                p3, h3, p4, z4 = [
                    _endpoint_group(predictions, spec, group_id) for spec in endpoints
                ]
                delta = np.stack((p3 - p4, h3 - p4, z4 - p4), axis=2).astype(np.float64)
                ridge = np.empty((SEQUENCE_ROWS, 2), dtype=np.float32)
                for target_index in range(2):
                    ridge[:, target_index] = (
                        p4[:, target_index].astype(np.float64)
                        + (delta[:, target_index, :] - means[target_index][None, :])
                        @ theta[target_index]
                    ).astype(np.float32)
                oof[index] = ridge
                equal = ((p3.astype(np.float64) + h3 + p4 + z4) * 0.25).astype(np.float32)
                current = {
                    "ridge": ridge,
                    "equal_four_way": equal,
                    "P3": p3,
                    "H3": h3,
                    "P4": p4,
                    "Z4": z4,
                }
                component_moments.setdefault(
                    component_id,
                    {
                        name: [WeightedMoments.empty(2), WeightedMoments.empty(2)]
                        for name in metric_names
                    },
                )
                for name, prediction in current.items():
                    accumulators[name].add(
                        targets[index], prediction, mask[index], group=component_id
                    )
                    for target_index in range(2):
                        group_moment = _metric_moments(
                            targets[index, :, target_index],
                            prediction[:, target_index],
                            mask[index],
                        )
                        stored = component_moments[component_id][name][target_index]
                        stored.merge(group_moment.weight, group_moment.mean, group_moment.centered)

        oof.flush()
        del oof
        if set(folds_by_component) != set(component_by_group[group] for group in groups):
            raise ValueError("OOF component population is incomplete")
        metrics = {name: accumulator.result() for name, accumulator in accumulators.items()}
        for name in metric_names:
            for target_index, target_name in enumerate(("t0", "t1")):
                merged = WeightedMoments.empty(2)
                for component in component_moments:
                    moment = component_moments[component][name][target_index]
                    merged.merge(moment.weight, moment.mean, moment.centered)
                observed_rho = _rho(merged)
                if abs(observed_rho - metrics[name][target_name]) > 1e-12:
                    raise ValueError(
                        f"component sufficient-statistic mismatch for {name}/{target_name}; "
                        f"official={metrics[name][target_name]}; components={observed_rho}"
                    )
        expected_scores = config["source_endpoint_scores"]
        for endpoint_name in ("P3", "H3", "P4", "Z4"):
            observed = metrics[endpoint_name]["wp"]
            expected = expected_scores[endpoint_name]["wp"]
            if abs(observed - expected) > 1e-6:
                raise ValueError(
                    f"published {endpoint_name} endpoint score mismatch; expected={expected}; observed={observed}"
                )
        bootstrap = _component_bootstrap(component_moments, folds_by_component)
        report = {
            "schema": "wnn.luna-meta-crossfit-result.v1",
            "experiment_id": EXPERIMENT_ID,
            "status": "completed_exposed_meta_crossfit",
            "metric_id": "global_wp_clip2_v1",
            "config_sha256": _sha256(HERE / "LUNA_STACKING_META_CROSSFIT_R2.json"),
            "assignment_sha256": assignment["assignment_sha256"],
            "split_manifest_sha256": EXPECTED_SPLIT_SHA256,
            "evaluation_groups": len(groups),
            "evaluation_components": len(folds_by_component),
            "selected_rows": int(mask.sum()),
            "sequence_id_sha256": hashlib.sha256(sequence_ids.astype("<i8").tobytes()).hexdigest(),
            "evaluation_group_order_sha256": hashlib.sha256(
                np.asarray(groups, dtype="<i8").tobytes()
            ).hexdigest(),
            "evaluation_row_groups_in_prediction_order": groups,
            "metrics": metrics,
            "ridge_vs_P4_component_bootstrap": bootstrap,
            "outer_coefficients": coefficient_report,
            "component_moments": {
                str(component): {
                    name: [
                        {
                            "weight": moment.weight,
                            "mean_y": float(moment.mean[0]),
                            "mean_p": float(moment.mean[1]),
                            "m2_y": float(moment.centered[0, 0]),
                            "m2_p": float(moment.centered[1, 1]),
                            "m2_yp": float(moment.centered[0, 1]),
                        }
                        for moment in target_moments
                    ]
                    for name, target_moments in model_moments.items()
                }
                for component, model_moments in component_moments.items()
            },
            "oof_prediction": {
                "path": str(oof_path),
                "shape": [len(groups), SEQUENCE_ROWS, 2],
                "dtype": "float32",
                "sha256": _sha256(oof_path),
            },
            "observed_input_hashes": observed_input_hashes,
            "ancestry_evidence_hashes": {
                row["kind"]: row["sha256"] for row in config["ancestry_evidence"]["reports"]
            },
            "resource_observation": {
                "peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
                "elapsed_seconds": time.monotonic() - started,
            },
            "elapsed_seconds": time.monotonic() - started,
            "exposure": "repeatedly inspected fold0 development panel; not fresh confirmation or public forecast",
            "limitations": [
                "bootstrap conditions on the frozen base models and crossfitted meta procedure",
                "no training-seed or public-transfer uncertainty is estimated",
                "all-data seed99 predictions were excluded",
            ],
        }
        result_path = output_root / "LUNA_META_CROSSFIT_R2_RESULT.json"
        result_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        return 0
    except BaseException as error:
        failure = {
            "schema": "wnn.luna-meta-crossfit-failure.v1",
            "experiment_id": EXPERIMENT_ID,
            "status": "failed_or_interrupted_preserve_partial",
            "error_type": type(error).__name__,
            "error": str(error),
            "config_sha256": _sha256(HERE / "LUNA_STACKING_META_CROSSFIT_R2.json"),
            "assignment_sha256": assignment.get("assignment_sha256"),
            "elapsed_seconds": time.monotonic() - started,
        }
        (output_root / "failure.json").write_text(
            json.dumps(failure, indent=2, sort_keys=True) + "\n"
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
