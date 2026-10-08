from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import shlex
import struct
import sys
import time
import zipfile
from datetime import UTC, datetime
from pathlib import Path

# Any is needed for nested JSON records whose schemas come from frozen receipts.
from typing import Any

import numpy as np

METRIC_ID = "global_wp_clip2_v1"
WARMUP = 99
N_BOOT = 2000
BOOTSTRAP_SEED = 271828
PARENT_SHA = "0203b109a1f25f3c82794ad3684b83fe4a4bc815f4b5c1bc4bc1ea450dba0a8d"
PARENT_PRED_SHA = "f1a3b07072a62dce620dc0c90e9568093367d0aed1afa4e9ec550866f94df12b"
CONTROL_PRED_SHA = "5b36f059112b1d88a0c6dab1c7ea6486d197e98e5ca31a3dcec96a3ae45a837a"
TARGET_SHA = "dbb7a1a48dd0af9fa524a80c2694008a56cb91d5dae80e8d0d4984c0c5622955"
SPLIT_SHA = "09263ad4fe2b18e2b2f3b79f4562678dce801b067b868134ddc79279f9316a3d"
CONTROL_RECIPE_SHA = "ebd86d9e5c9be09456e3e97277dfc30e415901f44c863adf8a731e2c1a7f63c8"
CONTROL_CHECKPOINT_SHA = "e6fd249022be91bda51d72376ece38289d8a46fa0d558d8c2e5b280849d0915a"
CONTROL_PROVENANCE_SHA = "4f02e119d2322210d9417085d97284ac1eeb6d9da65ffc67d19ba001a1e4b096"
CONTROL_SIDECAR_SHA = "19ce469de9421bd0fd44ddb408c56905d0848217f6857daa4faa615a2aaa5fc7"
CONTROL_MEMBERSHIP_SHA = "355a59e25251b2af2324535dd3b4929bbc2faedd5cdf6415b64565c888bc88b2"
CONTROL_EXPORT_SHA = "a634fdc24dae97aa18f30c4b8b8a856d02880d5f99476e0042868afe8b3f37f9"
H272_PREREG_SHA = "c72cfa84e7646771114966b2b238cb9cacadc9971a6b40483fbe2b3b7069a80b"
H272_EXPORT_SHA = "87c8f112c92134001743d4cc4737624904bac1536670b8761b10eae250ff877b"
READOUT_CHECKPOINT_SHA = "6c57427ab603803229b5afa7024d492bb304aff7d9518b22fd3aa776ada3c1e7"
READOUT_PROVENANCE_SHA = "489848891071700bb3a2c4c24095e3b59ed58eda160f94f12b71497d1e9828ce"
METRIC_SOURCE_SHA = "2cf10e740a831379f62b7d1269456053f5c1f9efa2dbab03156e366972bed5aa"
H272_ARRAY_SHA = {
    "parent": PARENT_PRED_SHA,
    "hidden_only": "c5fbb9e8c3da118bdb9f2bc3de039056ab2698429e119806142ccb4111dcbe97",
    "multiplicative": "b2dc11aeff3768ab6dcf824426c1cf0b85cacee504fea7c34782e7d4f3e84fb3",
}


def sha256_file(path: Path) -> str:
    """Hash a file in bounded chunks so large prediction files stay out of RAM."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_int64(values: list[int] | np.ndarray) -> str:
    """Return the canonical little-endian group-list digest."""
    array = np.asarray(values, dtype="<i8")
    return hashlib.sha256(array.tobytes()).hexdigest()


def record_artifact(
    root: Path,
    records: dict[str, dict[str, object]],
    name: str,
    path: Path,
    expected_sha256: str | None = None,
) -> Path:
    """Resolve and hash an input artifact before it contributes to the grade."""
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"{name} is not a file: {resolved}")
    observed = sha256_file(resolved)
    if expected_sha256 is not None and observed != expected_sha256:
        raise ValueError(f"{name} hash mismatch: expected {expected_sha256}, observed {observed}")
    try:
        shown_path = str(resolved.relative_to(root))
    except ValueError:
        shown_path = str(resolved)
    records[name] = {
        "path": shown_path,
        "sha256": observed,
        "bytes": resolved.stat().st_size,
    }
    return resolved


def mmap_stored_npz(path: Path) -> dict[str, np.memmap]:
    """Map stored NPY members directly from the frozen target archive."""
    arrays: dict[str, np.memmap] = {}
    with zipfile.ZipFile(path) as archive, path.open("rb") as raw:
        for info in archive.infolist():
            if info.compress_type != zipfile.ZIP_STORED or not info.filename.endswith(".npy"):
                raise ValueError("target cache must contain stored NPY members")
            with archive.open(info) as member:
                version = np.lib.format.read_magic(member)
                if version == (1, 0):
                    shape, fortran, dtype = np.lib.format.read_array_header_1_0(member)
                elif version == (2, 0):
                    shape, fortran, dtype = np.lib.format.read_array_header_2_0(member)
                else:
                    raise ValueError("unsupported target-cache NPY version")
                header_bytes = member.tell()
            if dtype.hasobject:
                raise ValueError("object arrays are not permitted in the target cache")
            raw.seek(info.header_offset)
            local_header = raw.read(30)
            if local_header[:4] != b"PK\x03\x04":
                raise ValueError("invalid target-cache ZIP member header")
            name_bytes, extra_bytes = struct.unpack("<HH", local_header[26:30])
            offset = info.header_offset + 30 + name_bytes + extra_bytes + header_bytes
            arrays[info.filename.removesuffix(".npy")] = np.memmap(
                path,
                dtype=dtype,
                shape=shape,
                offset=offset,
                mode="r",
                order="F" if fortran else "C",
            )
    required = {"y", "mask", "row_group"}
    if not required.issubset(arrays):
        raise ValueError(f"target cache is missing {sorted(required - arrays.keys())}")
    return arrays


def fit_groups(provenance: dict[str, Any]) -> dict[str, list[int]]:
    """Read source-qualified group lists from a completed producer provenance."""
    raw = provenance.get("fit_row_groups")
    if not isinstance(raw, dict):
        raise ValueError("fit provenance lacks source-qualified fit_row_groups")
    return {source: [int(group) for group in raw.get(source, [])] for source in ("train", "valid")}


def validate_membership(
    name: str,
    provenance: dict[str, Any],
    component_of: dict[tuple[str, int], int],
    expected: dict[str, set[int]],
    evaluation: list[int],
    split_sha256: str,
) -> dict[str, object]:
    """Check fit sets, source namespaces, component separation and fingerprints."""
    if provenance.get("split_manifest_sha256") != split_sha256:
        raise ValueError(f"{name} is bound to another split manifest")
    actual_lists = fit_groups(provenance)
    actual: dict[str, set[int]] = {}
    order_hashes: dict[str, str] = {}
    sorted_hashes: dict[str, str] = {}
    for source in ("train", "valid"):
        values = actual_lists[source]
        if len(values) != len(set(values)):
            raise ValueError(f"{name} has duplicate {source} fit groups")
        actual[source] = set(values)
        if actual[source] != expected[source]:
            raise ValueError(f"{name} {source} fit membership differs from the frozen split")
        order_hashes[source] = sha256_int64(values)
        sorted_hashes[source] = sha256_int64(sorted(values))

    declared_hashes = provenance.get("fit_membership_sha256")
    if isinstance(declared_hashes, dict):
        for source, declared in declared_hashes.items():
            if source in sorted_hashes and declared not in (
                order_hashes[source],
                sorted_hashes[source],
            ):
                raise ValueError(f"{name} {source} membership digest does not match its groups")
    identity = provenance.get("fit_membership_identity")
    if isinstance(identity, dict):
        for source, record in identity.items():
            if source in order_hashes and isinstance(record, dict):
                if record.get("count") != len(actual[source]):
                    raise ValueError(f"{name} {source} membership count differs")
                declared = record.get("ordered_int64_le_sha256")
                if declared not in (order_hashes[source], sorted_hashes[source]):
                    raise ValueError(f"{name} {source} ordered membership digest differs")

    fit_keys = {(source, group) for source in actual for group in actual[source]}
    eval_keys = {("valid", int(group)) for group in evaluation}
    fit_components = {component_of[key] for key in fit_keys}
    eval_components = {component_of[key] for key in eval_keys}
    if fit_keys & eval_keys or fit_components & eval_components:
        raise ValueError(f"{name} overlaps evaluation groups or components")
    return {
        "fit_groups_by_source": {source: len(actual[source]) for source in actual},
        "fit_order_sha256": order_hashes,
        "fit_canonical_sorted_sha256": sorted_hashes,
        "evaluation_source_qualified_group_overlap": 0,
        "fit_component_count": len(fit_components),
        "evaluation_component_count": len(eval_components),
        "fit_evaluation_component_overlap": 0,
    }


def weighted_centered_moments(target: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    """Compute clip-2 weighted centered moments for one target and one sequence."""
    raw_y = np.asarray(target, dtype=np.float32)
    raw_p = np.asarray(prediction, dtype=np.float32)
    if raw_y.shape != raw_p.shape:
        raise ValueError("target and prediction sequence shapes differ")
    if not np.isfinite(raw_y).all() or not np.isfinite(raw_p).all():
        raise ValueError("selected target or prediction contains a nonfinite value")
    y = np.clip(raw_y, -2.0, 2.0).astype(np.float64)
    p = np.clip(raw_p, -2.0, 2.0).astype(np.float64)
    weight = np.abs(y)
    active = weight > 0
    if not active.any():
        return np.zeros(6, dtype=np.float64)
    y, p, weight = y[active], p[active], weight[active]
    total = float(weight.sum())
    dy = y - y[0]
    dp = p - p[0]
    mean_y = float(np.dot(weight, dy) / total)
    mean_p = float(np.dot(weight, dp) / total)
    dy -= mean_y
    dp -= mean_p
    return np.array(
        [
            total,
            float(y[0] + mean_y),
            float(p[0] + mean_p),
            float(np.dot(weight, dy * dy)),
            float(np.dot(weight, dp * dp)),
            float(np.dot(weight, dy * dp)),
        ],
        dtype=np.float64,
    )


def merge_moments(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Merge two centered weighted-moment records without subtracting raw squares."""
    if left[0] == 0:
        return right.copy()
    if right[0] == 0:
        return left.copy()
    total = left[0] + right[0]
    delta_y = right[1] - left[1]
    delta_p = right[2] - left[2]
    fraction = right[0] / total
    cross_weight = left[0] * fraction
    return np.array(
        [
            total,
            left[1] + delta_y * fraction,
            left[2] + delta_p * fraction,
            left[3] + right[3] + delta_y * delta_y * cross_weight,
            left[4] + right[4] + delta_p * delta_p * cross_weight,
            left[5] + right[5] + delta_y * delta_p * cross_weight,
        ],
        dtype=np.float64,
    )


def score_from_moments(moments: np.ndarray) -> dict[str, float]:
    """Compute global weighted Pearson scores from pooled centered moments."""
    values = np.asarray(moments, dtype=np.float64)
    scores: list[float] = []
    for target_index in range(2):
        weight, _, _, target_m2, prediction_m2, cross_m2 = values[target_index]
        if weight <= 1e-8 or target_m2 <= 0 or prediction_m2 <= 0:
            scores.append(0.0)
            continue
        score = cross_m2 / np.sqrt(target_m2 * prediction_m2)
        scores.append(float(np.clip(score, -1.0, 1.0)))
    return {"t0": scores[0], "t1": scores[1], "wp": 0.5 * (scores[0] + scores[1])}


def component_score(moments: np.ndarray, multiplicity: np.ndarray) -> float:
    """Pool a component resample and return the official two-target average."""
    target_scores: list[float] = []
    for target_index in range(2):
        per_component = moments[:, target_index, :]
        weights = multiplicity * per_component[:, 0]
        total = float(weights.sum())
        if total <= 1e-8:
            target_scores.append(0.0)
            continue
        active = np.flatnonzero(weights > 0)
        anchor_y, anchor_p = per_component[active[0], 1:3]
        delta_y = per_component[:, 1] - anchor_y
        delta_p = per_component[:, 2] - anchor_p
        mean_y = float(np.dot(weights, delta_y) / total)
        mean_p = float(np.dot(weights, delta_p) / total)
        target_m2 = float(
            np.dot(multiplicity, per_component[:, 3]) + np.dot(weights, (delta_y - mean_y) ** 2)
        )
        prediction_m2 = float(
            np.dot(multiplicity, per_component[:, 4]) + np.dot(weights, (delta_p - mean_p) ** 2)
        )
        cross_m2 = float(
            np.dot(multiplicity, per_component[:, 5])
            + np.dot(weights, (delta_y - mean_y) * (delta_p - mean_p))
        )
        if target_m2 > 0 and prediction_m2 > 0:
            target_scores.append(
                float(np.clip(cross_m2 / np.sqrt(target_m2 * prediction_m2), -1.0, 1.0))
            )
        else:
            target_scores.append(0.0)
    return 0.5 * (target_scores[0] + target_scores[1])


def bootstrap_comparison(
    candidate: np.ndarray,
    reference: np.ndarray,
    draws: np.ndarray,
    delta_point: float,
) -> dict[str, object]:
    """Return the paired component bootstrap for a fixed prediction pair."""
    differences = np.empty(len(draws), dtype=np.float64)
    for index, draw in enumerate(draws):
        counts = np.bincount(draw, minlength=candidate.shape[0]).astype(np.float64)
        differences[index] = component_score(candidate, counts) - component_score(reference, counts)
    lower, upper = np.quantile(differences, [0.025, 0.975])
    return {
        "n_components": int(candidate.shape[0]),
        "n_boot": int(len(draws)),
        "seed": BOOTSTRAP_SEED,
        "paired": True,
        "conditional_on_fixed_predictions": True,
        "mean_delta": float(differences.mean()),
        "bootstrap_sd": float(differences.std(ddof=0)),
        "percentile_ci95": [float(lower), float(upper)],
        "bootstrap_fraction_delta_le_zero": float(np.mean(differences <= 0)),
        "bootstrap_mean_minus_point_delta": float(differences.mean() - delta_point),
    }


def cgroup_snapshot() -> dict[str, object]:
    """Capture this worker's cgroup limits and memory accounting."""
    result: dict[str, object] = {}
    try:
        lines = Path("/proc/self/cgroup").read_text(encoding="utf-8").splitlines()
        relative = next(line.split("::", 1)[1] for line in lines if line.startswith("0::"))
        child = Path("/sys/fs/cgroup") / relative.lstrip("/")
        parent = child.parent
        for label, path in (("worker", child), ("campaign_parent", parent)):
            values: dict[str, str | None] = {}
            for filename in (
                "cpu.max",
                "memory.max",
                "memory.current",
                "memory.peak",
                "memory.events",
            ):
                file_path = path / filename
                values[filename] = (
                    file_path.read_text(encoding="utf-8").strip() if file_path.exists() else None
                )
            result[label] = values
    except (OSError, StopIteration) as exc:
        result["read_error"] = str(exc)
    return result


def fmt(value: float) -> str:
    """Format a point score or delta consistently in the Markdown handoff."""
    return f"{value:.10f}"


def main() -> None:
    """Validate delivered artifacts and independently grade fixed S4 endpoints."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--control-worktree", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()

    root = args.project_root.resolve(strict=True)
    control_root = args.control_worktree.resolve(strict=True)
    output_path = args.output.resolve()
    markdown_path = args.markdown.resolve()
    if output_path.exists() or markdown_path.exists():
        raise FileExistsError("Preserve prior grade artifacts; choose a new output identity")

    started = time.monotonic()
    started_utc = datetime.now(UTC).isoformat()
    cgroup_before = cgroup_snapshot()
    artifacts: dict[str, dict[str, object]] = {}
    e529 = root / "research/campaign_2026-10-06/bandit_continuation/continuous_execution_E529"

    control_recipe_path = e529 / "recipes/E529_S4_SERVED_HEAD_CLEANFOLD_CONTROL_V1/recipe.json"
    control_recipe = json.loads(control_recipe_path.read_text(encoding="utf-8"))
    control_export_path = e529 / "H272_S4_CONTROL_EXPORT_TERMINAL_OPERATOR_RECEIPT_20261008_R1.json"
    control_export = json.loads(control_export_path.read_text(encoding="utf-8"))
    control_membership_path = e529 / "H272_S4_CONTROL_FIT_MEMBERSHIP_VERIFICATION_20261008_R1.json"
    control_membership_receipt = json.loads(control_membership_path.read_text(encoding="utf-8"))
    prereg_path = e529 / "H272_S4_CLEANFOLD_V2_GRADE_PREREG_20261008_R1.json"
    prereg = json.loads(prereg_path.read_text(encoding="utf-8"))
    prior_result_path = e529 / "H272_S4_CLEANFOLD_V2_GRADE_RESULT_20261008_R1.json"
    prior_result = json.loads(prior_result_path.read_text(encoding="utf-8"))
    if (
        control_membership_receipt.get("result") != "PASS"
        or control_membership_receipt.get("membership", {}).get(
            "fit_evaluation_same_source_overlap"
        )
        != 0
        or control_membership_receipt.get("membership", {}).get(
            "fit_evaluation_component_overlap", 0
        )
        not in (None, 0)
        or control_export.get("execution", {}).get("exit_status") != 0
        or control_export.get("execution", {}).get("status") != "finished"
        or control_export.get("prediction", {}).get("file_sha256") != CONTROL_PRED_SHA
        or prereg.get("status") != "frozen_before_prediction_scoring"
        or prereg.get("comparison", {}).get("metric") != METRIC_ID
        or prior_result.get("status") != "evaluated_exposed_discovery"
        or prior_result.get("metric") != METRIC_ID
    ):
        raise ValueError("Frozen fit/export/preregistration evidence does not pass")

    export_rel = prereg["export"]["path"].split(
        "research/campaign_2026-10-06/bandit_continuation/continuous_execution_E529/",
        1,
    )[-1]
    h272_export_dir = e529 / export_rel

    split_path = root / "research/campaign_2026-09-29/artifacts/splits/fold0.json"
    target_path = root / "cache/valid_targets.npz"
    parent_prediction_path = root / "cache/pred_c20261003_combo_s4_f0_valid.npy"
    parent_sidecar_path = root / "cache/pred_c20261003_combo_s4_f0_valid.json"
    control_prediction_path = (
        control_root / "cache/pred_e529_s4_served_head_f0_control_v1_fp32_valid.npy"
    )
    control_sidecar_path = (
        control_root / "cache/pred_e529_s4_served_head_f0_control_v1_fp32_valid.json"
    )
    control_run_dir = control_root / "artifacts/runs/e529_s4_served_head_f0_control_v1"
    parent_run_dir = control_root / "artifacts/runs/c20261003_combo_s4_f0"
    readout_run_dir = root / "artifacts/runs/h272s4_parent_readout_v2_cleanscreen"

    record_artifact(root, artifacts, "control_recipe", control_recipe_path, CONTROL_RECIPE_SHA)
    source_lock_path = control_recipe_path.parent / "source_lock.json"
    source_audit_path = control_recipe_path.parent / "source_audit.json"
    record_artifact(root, artifacts, "control_source_lock", source_lock_path)
    record_artifact(root, artifacts, "control_source_audit", source_audit_path)
    record_artifact(
        root,
        artifacts,
        "control_membership_receipt",
        control_membership_path,
        CONTROL_MEMBERSHIP_SHA,
    )
    record_artifact(
        root, artifacts, "control_export_terminal_receipt", control_export_path, CONTROL_EXPORT_SHA
    )
    record_artifact(
        root,
        artifacts,
        "control_fit_terminal_receipt",
        e529 / "H272_S4_CONTROL_FIT_TERMINAL_OPERATOR_RECEIPT_20261008_R1.json",
        "3d8f1af31a7f5b5bc06e01df2f332c50761a9c95581ad81c42efa56483edd110",
    )
    record_artifact(root, artifacts, "h272_grade_preregistration", prereg_path, H272_PREREG_SHA)
    record_artifact(root, artifacts, "h272_prior_grade_result", prior_result_path)
    record_artifact(
        root,
        artifacts,
        "h272_export_receipt",
        h272_export_dir / "export_receipt.json",
        H272_EXPORT_SHA,
    )
    record_artifact(root, artifacts, "split_manifest", split_path, SPLIT_SHA)
    record_artifact(root, artifacts, "target_cache", target_path, TARGET_SHA)
    record_artifact(
        root, artifacts, "official_metric_source", root / "src/wnn/gmetric.py", METRIC_SOURCE_SHA
    )

    parent_checkpoint_path = parent_run_dir / "model_ep3.pt"
    parent_provenance_path = parent_run_dir / "provenance.json"
    control_checkpoint_path = control_run_dir / "model_ep0.pt"
    control_provenance_path = control_run_dir / "provenance.json"
    readout_provenance_path = readout_run_dir / "provenance.json"
    record_artifact(root, artifacts, "parent_checkpoint", parent_checkpoint_path, PARENT_SHA)
    record_artifact(
        root,
        artifacts,
        "parent_provenance",
        parent_provenance_path,
        control_recipe["parent"]["provenance_sha256"],
    )
    record_artifact(
        root, artifacts, "control_checkpoint", control_checkpoint_path, CONTROL_CHECKPOINT_SHA
    )
    record_artifact(
        root, artifacts, "control_provenance", control_provenance_path, CONTROL_PROVENANCE_SHA
    )
    record_artifact(
        root, artifacts, "control_prediction", control_prediction_path, CONTROL_PRED_SHA
    )
    record_artifact(
        root, artifacts, "control_prediction_sidecar", control_sidecar_path, CONTROL_SIDECAR_SHA
    )
    record_artifact(root, artifacts, "parent_prediction", parent_prediction_path, PARENT_PRED_SHA)
    record_artifact(root, artifacts, "parent_prediction_sidecar", parent_sidecar_path)
    record_artifact(
        root,
        artifacts,
        "h272_readout_provenance",
        readout_provenance_path,
        READOUT_PROVENANCE_SHA,
    )
    record_artifact(
        root,
        artifacts,
        "h272_readout_checkpoint",
        readout_run_dir / "model_ep1.pt",
        READOUT_CHECKPOINT_SHA,
    )
    record_artifact(
        root,
        artifacts,
        "h272_readout_recipe",
        root / prereg["recipe"]["path"],
        prereg["recipe"]["sha256"],
    )
    record_artifact(
        root,
        artifacts,
        "h272_readout_source_lock",
        root / prereg["recipe"]["source_lock_path"],
        prereg["recipe"]["source_lock_sha256"],
    )

    h272_recipe = json.loads((root / prereg["recipe"]["path"]).read_text(encoding="utf-8"))
    propensity_spec = h272_recipe["propensity"]
    propensity_run_dir = root / "artifacts/runs/c20260929_prop_f0"
    propensity_provenance_path = propensity_run_dir / "provenance.json"
    record_artifact(
        root,
        artifacts,
        "h272_propensity_provenance",
        propensity_provenance_path,
        propensity_spec["provenance_sha256"],
    )
    record_artifact(
        root,
        artifacts,
        "h272_propensity_checkpoint",
        root / propensity_spec["checkpoint"],
        propensity_spec["checkpoint_sha256"],
    )
    propensity_cache_records: dict[str, dict[str, object]] = {}
    for source, cache_spec in propensity_spec["caches"].items():
        cache_path = root / cache_spec["path"]
        sidecar_path = cache_path.with_suffix(".json")
        record_artifact(
            root, artifacts, f"h272_propensity_{source}_cache", cache_path, cache_spec["sha256"]
        )
        record_artifact(
            root,
            artifacts,
            f"h272_propensity_{source}_cache_sidecar",
            sidecar_path,
            cache_spec["sidecar_sha256"],
        )
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        expected_shape = cache_spec["shape"]
        if (
            sidecar.get("model") != propensity_spec["run"]
            or sidecar.get("model_sha256") != propensity_spec["checkpoint_sha256"]
            or sidecar.get("split") != source
            or sidecar.get("rows") != expected_shape[0]
            or sidecar.get("n_filled") != cache_spec["n_filled"]
        ):
            raise ValueError(f"{source} propensity cache sidecar differs from the readout recipe")
        propensity_cache_records[source] = {
            "split": source,
            "shape": expected_shape,
            "n_filled": cache_spec["n_filled"],
            "sha256": artifacts[f"h272_propensity_{source}_cache"]["sha256"],
            "sidecar_sha256": artifacts[f"h272_propensity_{source}_cache_sidecar"]["sha256"],
        }

    for name, expected in H272_ARRAY_SHA.items():
        record_artifact(
            root,
            artifacts,
            f"h272_prediction_{name}",
            h272_export_dir / f"{name}.npy",
            expected,
        )
    export_hashes = prereg["export"]
    mask_paths = {
        "row_groups": "row_groups.npy",
        "evaluation_group_mask": "evaluation_group_mask.npy",
        "evaluation_selected_mask": "evaluation_selected_mask.npy",
        "scored_mask": "scored_mask.npy",
    }
    mask_hashes = {
        "row_groups": export_hashes["row_groups_sha256"],
        "evaluation_group_mask": export_hashes["evaluation_group_mask_sha256"],
        "evaluation_selected_mask": export_hashes["evaluation_selected_mask_sha256"],
        "scored_mask": export_hashes["scored_mask_sha256"],
    }
    for name, filename in mask_paths.items():
        record_artifact(
            root,
            artifacts,
            f"h272_{name}",
            h272_export_dir / filename,
            mask_hashes[name],
        )

    split = json.loads(split_path.read_text(encoding="utf-8"))
    evaluation = [int(group) for group in split["evaluation"]["valid"]]
    if len(evaluation) != 468 or len(set(evaluation)) != 468:
        raise ValueError("Fold-0 evaluation must contain 468 unique valid-source groups")
    evaluation_hash = sha256_int64(evaluation)
    if evaluation_hash != prereg["export"]["evaluation_group_ids_sha256"]:
        raise ValueError("Frozen valid evaluation group identity differs from export")
    component_of = {
        (source, int(node["row_group"])): int(node["component"])
        for source, nodes in split["sequences"].items()
        for node in nodes
    }
    if len(component_of) != sum(len(nodes) for nodes in split["sequences"].values()):
        raise ValueError("Split contains duplicate source-qualified component entries")

    expected_fit = {source: set(map(int, split["fit"][source])) for source in ("train", "valid")}
    expected_control_fit = {"train": set(), "valid": expected_fit["valid"]}
    parent_provenance = json.loads(parent_provenance_path.read_text(encoding="utf-8"))
    control_provenance = json.loads(control_provenance_path.read_text(encoding="utf-8"))
    readout_provenance = json.loads(readout_provenance_path.read_text(encoding="utf-8"))
    propensity_provenance = json.loads(propensity_provenance_path.read_text(encoding="utf-8"))
    membership = {
        "parent": validate_membership(
            "parent", parent_provenance, component_of, expected_fit, evaluation, SPLIT_SHA
        ),
        "ordinary_gcorr_head_control": validate_membership(
            "ordinary GCorr head control",
            control_provenance,
            component_of,
            expected_control_fit,
            evaluation,
            SPLIT_SHA,
        ),
        "h272_readout": validate_membership(
            "H272 readout",
            readout_provenance,
            component_of,
            expected_fit,
            evaluation,
            SPLIT_SHA,
        ),
        "h272_propensity_ancestor": validate_membership(
            "H272 propensity ancestor",
            propensity_provenance,
            component_of,
            expected_control_fit,
            evaluation,
            SPLIT_SHA,
        ),
    }
    if control_provenance.get("initial_checkpoint_sha256") != PARENT_SHA:
        raise ValueError("GCorr control was not initialized from the declared cleanfold parent")
    if readout_provenance.get("parent_checkpoint_sha256") != PARENT_SHA:
        raise ValueError("H272 readout does not bind the same parent checkpoint")
    if parent_provenance.get("checkpoint_sha256", {}).get("model_ep3.pt") != PARENT_SHA:
        raise ValueError("Parent provenance checkpoint does not match the frozen parent")
    if (
        readout_provenance.get("propensity_checkpoint_sha256")
        != propensity_spec["checkpoint_sha256"]
    ):
        raise ValueError("H272 readout propensity checkpoint binding differs from its recipe")
    if readout_provenance.get("propensity_cache_sha256") != {
        source: spec["sha256"] for source, spec in propensity_spec["caches"].items()
    }:
        raise ValueError("H272 readout propensity cache binding differs from its recipe")
    if (
        readout_provenance.get("selected_epoch") != 1
        or readout_provenance.get("status") != "complete"
    ):
        raise ValueError("H272 readout is not the fixed completed epoch-1 artifact")
    if readout_provenance.get("recipe_sha256") != prereg["recipe"]["sha256"]:
        raise ValueError("H272 readout recipe hash differs from preregistration")
    if readout_provenance.get("source_lock_sha256") != prereg["recipe"]["source_lock_sha256"]:
        raise ValueError("H272 readout source lock differs from preregistration")

    control_sidecar = json.loads(control_sidecar_path.read_text(encoding="utf-8"))
    parent_sidecar = json.loads(parent_sidecar_path.read_text(encoding="utf-8"))
    if (
        control_sidecar.get("prediction_sha256") != CONTROL_PRED_SHA
        or control_sidecar.get("checkpoint_sha256") != CONTROL_CHECKPOINT_SHA
        or control_sidecar.get("split_manifest_sha256") != SPLIT_SHA
        or control_sidecar.get("row_groups") != list(range(1873))
    ):
        raise ValueError("GCorr control prediction sidecar identity or row order differs")
    if parent_sidecar.get("prediction_sha256") != PARENT_PRED_SHA or parent_sidecar.get(
        "row_groups"
    ) != list(range(1873)):
        raise ValueError("Parent prediction sidecar identity or row order differs")

    targets_cache = mmap_stored_npz(target_path)
    targets = targets_cache["y"]
    scored_mask = targets_cache["mask"]
    target_groups = [int(value) for value in targets_cache["row_group"]]
    if (
        targets.shape != (1873, 20_000, 2)
        or targets.dtype != np.float32
        or scored_mask.shape != (1873, 20_000)
        or scored_mask.dtype != np.bool_
        or target_groups != list(range(1873))
    ):
        raise ValueError("Target cache shape, dtype or valid-source row order differs")

    h272_arrays = {
        name: np.load(h272_export_dir / f"{name}.npy", mmap_mode="r", allow_pickle=False)
        for name in H272_ARRAY_SHA
    }
    row_groups = np.load(h272_export_dir / "row_groups.npy", mmap_mode="r", allow_pickle=False)
    group_mask = np.load(
        h272_export_dir / "evaluation_group_mask.npy", mmap_mode="r", allow_pickle=False
    )
    exported_selected = np.load(
        h272_export_dir / "evaluation_selected_mask.npy", mmap_mode="r", allow_pickle=False
    )
    exported_scored = np.load(
        h272_export_dir / "scored_mask.npy", mmap_mode="r", allow_pickle=False
    )
    if (
        row_groups.dtype != np.int64
        or row_groups.shape != (1873,)
        or group_mask.dtype != np.bool_
        or group_mask.shape != (1873,)
        or exported_selected.dtype != np.bool_
        or exported_selected.shape != (1873, 20_000)
        or exported_scored.dtype != np.bool_
        or exported_scored.shape != (1873, 20_000)
    ):
        raise ValueError("Export row or mask arrays have unexpected shapes/dtypes")
    if list(map(int, row_groups)) != target_groups:
        raise ValueError("H272 export row groups differ from target-cache order")
    expected_group_mask = np.isin(row_groups, np.asarray(evaluation, dtype=np.int64))
    if not np.array_equal(group_mask, expected_group_mask):
        raise ValueError("H272 export group mask differs from frozen fold-0 membership")
    if not np.array_equal(exported_scored, scored_mask):
        raise ValueError("H272 export scored mask differs from official target mask")
    step_mask = np.arange(20_000) >= WARMUP
    expected_selected = scored_mask & expected_group_mask[:, None] & step_mask[None, :]
    if not np.array_equal(exported_selected, expected_selected):
        raise ValueError("H272 selected mask differs from need_prediction AND is_scored")
    selected_rows = int(exported_selected.sum())
    if selected_rows != 808_863:
        raise ValueError(f"Official selected row count changed: {selected_rows}")
    evaluation_components = sorted({component_of[("valid", group)] for group in evaluation})
    if len(evaluation_components) != 141:
        raise ValueError("Frozen fold-0 evaluation must contain 141 components")

    control_prediction = np.load(control_prediction_path, mmap_mode="r", allow_pickle=False)
    parent_prediction = np.load(parent_prediction_path, mmap_mode="r", allow_pickle=False)
    prediction_shape = (1873, 20_000, 2)
    all_predictions = {
        "control": control_prediction,
        "root_parent": parent_prediction,
        **h272_arrays,
    }
    for name, prediction in all_predictions.items():
        if prediction.shape != prediction_shape or prediction.dtype != np.float32:
            raise ValueError(f"{name} prediction shape/dtype differs from the frozen export")
    if (
        sha256_int64(control_sidecar["row_groups"])
        != control_export["prediction"]["row_groups_sha256_int64_le"]
    ):
        raise ValueError("Control row-group identity differs from terminal receipt")

    component_index = {component: index for index, component in enumerate(evaluation_components)}
    component_groups: dict[int, list[int]] = {component: [] for component in evaluation_components}
    for group in evaluation:
        component_groups[component_of[("valid", group)]].append(group)
    endpoint_names = (
        "parent",
        "control_halfblend",
        "h272_hidden_only_halfblend",
        "h272_multiplicative_halfblend",
    )
    endpoint_moments = np.zeros(
        (len(endpoint_names), len(evaluation_components), 2, 6), dtype=np.float64
    )
    component_selected_rows = np.zeros(len(evaluation_components), dtype=np.int64)
    prediction_sources = {
        "parent": h272_arrays["parent"],
        "control": control_prediction,
        "hidden_only": h272_arrays["hidden_only"],
        "multiplicative": h272_arrays["multiplicative"],
    }
    for row, group in enumerate(row_groups):
        group_id = int(group)
        if group_id not in evaluation:
            continue
        component_slot = component_index[component_of[("valid", group_id)]]
        selected = np.asarray(exported_selected[row], dtype=bool)
        component_selected_rows[component_slot] += int(selected.sum())
        if not selected.any():
            continue
        labels = np.asarray(targets[row, selected], dtype=np.float32)
        parent_values = np.asarray(prediction_sources["parent"][row, selected], dtype=np.float32)
        control_values = np.asarray(prediction_sources["control"][row, selected], dtype=np.float32)
        hidden_values = np.asarray(
            prediction_sources["hidden_only"][row, selected], dtype=np.float32
        )
        contextual_values = np.asarray(
            prediction_sources["multiplicative"][row, selected], dtype=np.float32
        )
        half = np.float32(0.5)
        raw_endpoints = (
            parent_values,
            np.add(np.multiply(parent_values, half), np.multiply(control_values, half)),
            np.add(np.multiply(parent_values, half), np.multiply(hidden_values, half)),
            np.add(np.multiply(parent_values, half), np.multiply(contextual_values, half)),
        )
        for endpoint_slot, raw_values in enumerate(raw_endpoints):
            for target_index in range(2):
                sequence_moments = weighted_centered_moments(
                    labels[:, target_index], raw_values[:, target_index]
                )
                endpoint_moments[endpoint_slot, component_slot, target_index] = merge_moments(
                    endpoint_moments[endpoint_slot, component_slot, target_index],
                    sequence_moments,
                )

    endpoint_scores: dict[str, dict[str, float]] = {}
    for endpoint_slot, endpoint_name in enumerate(endpoint_names):
        pooled = np.zeros((2, 6), dtype=np.float64)
        for component_slot in range(len(evaluation_components)):
            for target_index in range(2):
                pooled[target_index] = merge_moments(
                    pooled[target_index],
                    endpoint_moments[endpoint_slot, component_slot, target_index],
                )
        endpoint_scores[endpoint_name] = score_from_moments(pooled)

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = rng.integers(0, len(evaluation_components), size=(N_BOOT, len(evaluation_components)))
    comparison_specs = (
        ("control_halfblend_vs_parent", "control_halfblend", "parent"),
        ("h272_hidden_only_halfblend_vs_parent", "h272_hidden_only_halfblend", "parent"),
        ("h272_multiplicative_halfblend_vs_parent", "h272_multiplicative_halfblend", "parent"),
        (
            "h272_hidden_only_halfblend_vs_control",
            "h272_hidden_only_halfblend",
            "control_halfblend",
        ),
        (
            "h272_multiplicative_halfblend_vs_control",
            "h272_multiplicative_halfblend",
            "control_halfblend",
        ),
    )
    comparisons: dict[str, dict[str, object]] = {}
    for name, candidate_name, reference_name in comparison_specs:
        candidate = endpoint_moments[endpoint_names.index(candidate_name)]
        reference = endpoint_moments[endpoint_names.index(reference_name)]
        candidate_score = endpoint_scores[candidate_name]
        reference_score = endpoint_scores[reference_name]
        delta = candidate_score["wp"] - reference_score["wp"]
        comparisons[name] = {
            "candidate": candidate_name,
            "reference": reference_name,
            "candidate_score": candidate_score,
            "reference_score": reference_score,
            "delta_wp": delta,
            "paired_component_bootstrap": bootstrap_comparison(candidate, reference, draws, delta),
        }

    prior_checks: dict[str, dict[str, float]] = {}
    for mode, endpoint_name in (
        ("hidden_only", "h272_hidden_only_halfblend"),
        ("multiplicative", "h272_multiplicative_halfblend"),
    ):
        prior_entry = next(
            item
            for item in prior_result["fixed_comparisons_all_reported"][mode]
            if item["alpha_on_readout"] == 0.5
        )
        new_score = endpoint_scores[endpoint_name]
        prior_score = prior_entry["score"]
        errors = {key: abs(new_score[key] - float(prior_score[key])) for key in ("t0", "t1", "wp")}
        if max(errors.values()) > 1e-8:
            raise ValueError(f"Independent {mode} point grade disagrees with delivered H272 grade")
        prior_checks[mode] = errors

    source_audit = json.loads(source_audit_path.read_text(encoding="utf-8"))
    teacher_ancestry = source_audit["cleanfold_parent"]["teacher_ancestry"]
    if not teacher_ancestry or any(
        entry.get("fold0_valid_overlap_groups") != 0 or entry.get("fold0_component_overlap") != 0
        for entry in teacher_ancestry
    ):
        raise ValueError("Parent teacher ancestry has a recorded fold-0 overlap")
    if prereg.get("exposure", {}).get("previously_exposed") is not True:
        raise ValueError("Exposure status is not explicitly recorded")
    if prior_result.get("public_forecast") is not None:
        raise ValueError("Prior readout grade contains a public forecast")

    per_component: list[dict[str, object]] = []
    moment_fields = (
        "weight",
        "target_mean",
        "prediction_mean",
        "target_M2",
        "prediction_M2",
        "target_prediction_cross_M2",
    )
    for component_slot, component in enumerate(evaluation_components):
        endpoint_records: dict[str, dict[str, dict[str, float]]] = {}
        for endpoint_slot, endpoint_name in enumerate(endpoint_names):
            target_records: dict[str, dict[str, float]] = {}
            for target_index in range(2):
                target_records[f"t{target_index}"] = {
                    field: float(value)
                    for field, value in zip(
                        moment_fields,
                        endpoint_moments[endpoint_slot, component_slot, target_index],
                        strict=True,
                    )
                }
            endpoint_records[endpoint_name] = target_records
        per_component.append(
            {
                "component_id": component,
                "valid_source_row_groups": sorted(component_groups[component]),
                "sequence_count": len(component_groups[component]),
                "official_selected_row_count": int(component_selected_rows[component_slot]),
                "moments_by_endpoint_and_target": endpoint_records,
            }
        )

    elapsed = time.monotonic() - started
    peak_rss_bytes = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
    report: dict[str, object] = {
        "schema": "h272-s4-served-head-control-independent-grade-v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "evaluated_exposed_discovery",
        "metric": {
            "id": METRIC_ID,
            "source": "src/wnn/gmetric.py",
            "source_sha256": METRIC_SOURCE_SHA,
            "independent_formula": (
                "Clip target and raw prediction to [-2,2]; weight each selected row by "
                "absolute clipped target; merge centered weighted moments across complete "
                "sequences; compute weighted Pearson for each target and average t0/t1."
            ),
            "warmup_rows_excluded": WARMUP,
        },
        "exposure": {
            "fold": 0,
            "source": "valid",
            "evaluation_groups": len(evaluation),
            "evaluation_components": len(evaluation_components),
            "previously_exposed": True,
            "claim_scope": "fixed-prediction local discovery comparison",
            "public_forecast": None,
        },
        "membership_and_ancestry": {
            "split_manifest_sha256": SPLIT_SHA,
            "evaluation_group_ids_sha256_int64_le": evaluation_hash,
            "evaluation_groups": len(evaluation),
            "evaluation_components": len(evaluation_components),
            "source_qualified_fit_checks": membership,
            "control_fit_checkpoint_sha256": CONTROL_CHECKPOINT_SHA,
            "control_initial_parent_checkpoint_sha256": PARENT_SHA,
            "readout_checkpoint_sha256": READOUT_CHECKPOINT_SHA,
            "readout_parent_checkpoint_sha256": PARENT_SHA,
            "propensity_ancestor_run": propensity_spec["run"],
            "propensity_ancestor_checkpoint_sha256": propensity_spec["checkpoint_sha256"],
            "propensity_cache_artifacts": propensity_cache_records,
            "teacher_ancestry": teacher_ancestry,
            "all_fit_evaluation_component_overlaps": 0,
        },
        "population": {
            "valid_row_groups": 1873,
            "evaluation_group_count": len(evaluation),
            "evaluation_component_count": len(evaluation_components),
            "official_selected_row_count": selected_rows,
            "selected_mask_sha256": artifacts["h272_evaluation_selected_mask"]["sha256"],
        },
        "comparison": {
            "blend_weight": 0.5,
            "blend_formula": "float32(0.5 * parent) + float32(0.5 * candidate), before metric clip",
            "weight_selection": False,
            "training_or_checkpoint_selection": False,
            "readout_fit_group_counts": membership["h272_readout"]["fit_groups_by_source"],
            "ordinary_gcorr_control_fit_group_counts": membership["ordinary_gcorr_head_control"][
                "fit_groups_by_source"
            ],
            "endpoint_scores": endpoint_scores,
            "comparisons": comparisons,
            "prior_h272_result_point_check_abs_errors": prior_checks,
            "component_moment_fields": list(moment_fields),
            "per_component_moments": per_component,
            "bootstrap": {
                "unit": "141 frozen connected components, paired across endpoints",
                "replicates": N_BOOT,
                "seed": BOOTSTRAP_SEED,
                "interval": "two-sided 2.5th and 97.5th percentiles",
                "conditional_on_fixed_predictions": True,
            },
        },
        "interpretation": (
            "Compare each fixed H272 half-blend directly with the ordinary GCorr "
            "head half-blend as well as with the parent. The readout and control fit "
            "different training memberships, so these endpoint contrasts do not isolate "
            "a pure method effect. Fold 0 is exposed; bootstrap uncertainty is conditional "
            "on fixed predictions and does not estimate training variation or public transfer."
        ),
        "artifact_sha256": artifacts,
        "commands": {
            "python_argv": [sys.executable, *sys.argv],
            "resource_wrapper": {
                "slice": os.environ.get("WNN_SLICE"),
                "cpu_quota": os.environ.get("WNN_CPU_QUOTA"),
                "threads": os.environ.get("WNN_THREADS"),
                "memory_max_bytes": os.environ.get("WNN_MEM_MAX"),
                "runtime_max": os.environ.get("WNN_RUNTIME_MAX"),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            },
        },
        "execution": {
            "started_utc": started_utc,
            "finished_utc": datetime.now(UTC).isoformat(),
            "elapsed_seconds": elapsed,
            "peak_worker_rss_bytes": peak_rss_bytes,
            "cgroup_before": cgroup_before,
            "cgroup_after": cgroup_snapshot(),
            "numpy_version": np.__version__,
            "gpu_used": False,
            "fit_or_export_performed": False,
            "source_or_metric_modified": False,
            "output_package_or_submission_created": False,
        },
    }

    primary = comparisons["control_halfblend_vs_parent"]
    hidden_vs_control = comparisons["h272_hidden_only_halfblend_vs_control"]
    contextual_vs_control = comparisons["h272_multiplicative_halfblend_vs_control"]
    report["interpretation"] += (
        f" Observed fixed endpoint deltas: ordinary GCorr control vs parent "
        f"{primary['delta_wp']:+.10f}; H272 hidden-only vs GCorr control "
        f"{hidden_vs_control['delta_wp']:+.10f}; H272 multiplicative/contextual "
        f"vs GCorr control {contextual_vs_control['delta_wp']:+.10f}."
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    bootstrap_rows = []
    for name, comparison in comparisons.items():
        interval = comparison["paired_component_bootstrap"]["percentile_ci95"]
        bootstrap_rows.append(
            f"| {name} | {fmt(comparison['candidate_score']['wp'])} | "
            f"{fmt(comparison['reference_score']['wp'])} | {fmt(comparison['delta_wp'])} | "
            f"[{fmt(interval[0])}, {fmt(interval[1])}] | "
            f"{comparison['paired_component_bootstrap']['bootstrap_fraction_delta_le_zero']:.4f} |"
        )
    point_rows = [
        f"| {name} | {fmt(score['wp'])} | {fmt(score['t0'])} | {fmt(score['t1'])} |"
        for name, score in endpoint_scores.items()
    ]
    command = (
        "WNN_SLICE=wnn-campaign.slice WNN_CPU_QUOTA=100% WNN_THREADS=1 "
        "WNN_MEM_MAX=4000000000 WNN_RUNTIME_MAX=1200s CUDA_VISIBLE_DEVICES= "
        f"{root / 'scripts/limited'} {shlex.join([sys.executable, *sys.argv])}"
    )
    markdown = f"""# H272 S4 served-head control: independent grade

Recorded: {report["created_utc"]}

This grade compares fixed raw 50/50 parent blends on the 468 valid-source fold-0 evaluation groups and 141 connected components. Fold 0 was previously exposed. The result is development evidence and has no public-score forecast.

## Point grades

| Endpoint | WP | t0 | t1 |
|---|---:|---:|---:|
{chr(10).join(point_rows)}

| Paired comparison | Candidate WP | Reference WP | Delta WP | Component bootstrap 95% interval | Bootstrap fraction delta <= 0 |
|---|---:|---:|---:|---:|---:|
{chr(10).join(bootstrap_rows)}

The primary endpoint is the ordinary GCorr head fixed 50/50 raw blend against its parent. The two H272 readout comparisons use the same parent and fixed 50/50 weights. The direct H272-versus-GCorr comparisons answer whether the readout adds beyond the ordinary head endpoint. No blend weights were selected here.

H272 hidden-only versus ordinary GCorr control: {fmt(hidden_vs_control["delta_wp"])} WP with conditional component-bootstrap interval [{fmt(hidden_vs_control["paired_component_bootstrap"]["percentile_ci95"][0])}, {fmt(hidden_vs_control["paired_component_bootstrap"]["percentile_ci95"][1])}]. H272 multiplicative/contextual versus ordinary GCorr control: {fmt(contextual_vs_control["delta_wp"])} WP with interval [{fmt(contextual_vs_control["paired_component_bootstrap"]["percentile_ci95"][0])}, {fmt(contextual_vs_control["paired_component_bootstrap"]["percentile_ci95"][1])}].

## Membership and limits

The control fits on 1,405 valid groups; H272's recorded readout fit uses 10,535 train plus 1,405 valid groups. Each excludes the 468 evaluation groups and their 141 components. Both bind the same parent checkpoint. H272's propensity ancestor is also checked against the source-qualified split. These different fit memberships mean the endpoint contrast does not isolate a pure method effect.

All comparisons use the same frozen official selected mask: need_prediction warmup exclusion (first {WARMUP} rows) AND is_scored, restricted to the 468 evaluation groups, totaling {selected_rows:,} scored rows. Prediction, target, split, recipe, checkpoint, provenance, sidecar, and receipt hashes are recorded in the JSON. H272's two fixed half-blend scores independently reproduce its delivered result within 1e-8.

The 2,000-resample paired component bootstrap uses seed {BOOTSTRAP_SEED}. Its intervals condition on these fixed predictions. They do not capture fit-seed variation, post-selection uncertainty, or public transfer. No training, GPU, export, package, submission, metric change, or campaign-state mutation was performed.

## Reproduction

{command}

The JSON includes Python argv, resource-wrapper settings, cgroup snapshots, per-component centered moments for each endpoint and target, and complete input SHA-256 identities.
"""
    markdown_path.write_text(markdown, encoding="utf-8")


if __name__ == "__main__":
    main()
