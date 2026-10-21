"""Verify exported sidecars and open fixed read-only prediction mappings."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from first_cell_score_core import (
    ARM_MODES,
    EXPECTED_BINDING_SHA256,
    EXPECTED_SPLIT_SHA256,
    MAX_PREDICTION_BYTES,
    MAX_SMALL_JSON_BYTES,
    N_GROUPS,
    PREDICTION_DATA_BYTES,
    PREDICTION_SHAPE,
    _pin_file,
    _read_json,
    _safe_child,
    _stable_identity,
    membership_sha256,
    require_ordered_groups,
)


def _open_predictions(
    export_root: Path, binding: dict[str, Any], manifest: dict[str, Any]
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """Validate all five files, then open their read-only NPY mappings."""
    manifest_groups = manifest.get("ordered_groups")
    if not isinstance(manifest_groups, list) or len(manifest_groups) != N_GROUPS:
        raise ValueError("export manifest does not contain the fixed 468 group order")
    if (
        any(type(group) is not int for group in manifest_groups)
        or len(set(manifest_groups)) != N_GROUPS
    ):
        raise ValueError("export manifest group IDs are not strict unique integers")
    if any(group < 0 or group >= 20_000 for group in manifest_groups):
        raise ValueError("export manifest contains an out-of-range validation row-group ID")
    expected_membership_sha256 = membership_sha256(manifest_groups)
    if manifest.get("membership_sha256") != expected_membership_sha256:
        raise ValueError("export manifest ordered-membership digest differs")
    expected_modes = {arm["arm"]: arm for arm in binding.get("arms", [])}
    if set(expected_modes) != set(ARM_MODES) or any(
        expected_modes[name].get("representation_mode") != mode for name, mode in ARM_MODES.items()
    ):
        raise ValueError("export binding does not register the five fixed arms/modes")
    manifest_arms = manifest.get("arms")
    if (
        not isinstance(manifest_arms, list)
        or [row.get("arm") for row in manifest_arms if isinstance(row, dict)] != list(ARM_MODES)
        or len(manifest_arms) != len(ARM_MODES)
    ):
        raise ValueError("export manifest arm inventory differs")
    rows = {row.get("arm"): row for row in manifest_arms if isinstance(row, dict)}
    if set(rows) != set(ARM_MODES):
        raise ValueError("export manifest arms are missing, duplicated, or unexpected")
    arrays: dict[str, np.ndarray] = {}
    pins: dict[str, dict[str, Any]] = {}
    identities: list[dict[str, Any]] = []
    mapping_inputs: dict[str, tuple[Path, int, tuple[int, int, int, int]]] = {}
    for name, mode in ARM_MODES.items():
        arm = expected_modes[name]
        row = rows[name]
        sidecar_path = _safe_child(export_root, f"{name}/PREDICTION_SIDECAR.json")
        sidecar, sidecar_pin, sidecar_identity = _read_json(sidecar_path, MAX_SMALL_JSON_BYTES)
        prediction_path = _safe_child(export_root, f"{name}/predictions.npy")
        prediction_pin, prediction_identity = _pin_file(prediction_path, MAX_PREDICTION_BYTES)
        if (row.get("directory"), row.get("sidecar_bytes"), row.get("sidecar_sha256")) != (
            name,
            sidecar_pin["bytes"],
            sidecar_pin["sha256"],
        ):
            raise ValueError(f"{name} sidecar does not join the completion manifest")
        if (row.get("prediction_bytes"), row.get("prediction_sha256")) != (
            prediction_pin["bytes"],
            prediction_pin["sha256"],
        ):
            raise ValueError(f"{name} prediction bytes do not join the completion manifest")
        expected_checkpoint = {
            **arm["model"],
            "loaded_sha256": arm["model"]["sha256"],
        }
        expected_row = {
            "arm": name,
            "directory": name,
            "run_name": arm["run_name"],
            "unit": arm["unit"],
            "invocation_id": arm["invocation_id"],
            "representation_mode": mode,
            "selected_epoch": 5,
            "successful_updates": 44760,
            "checkpoint": expected_checkpoint,
            "membership_sha256": expected_membership_sha256,
            "prediction_path": f"{name}/predictions.npy",
            "sidecar_path": f"{name}/PREDICTION_SIDECAR.json",
            "prediction_bytes": prediction_pin["bytes"],
            "prediction_sha256": prediction_pin["sha256"],
            "sidecar_bytes": sidecar_pin["bytes"],
            "sidecar_sha256": sidecar_pin["sha256"],
        }
        if any(row.get(key) != value for key, value in expected_row.items()):
            raise ValueError(f"{name} repeated arm identity or file pins differ in the manifest")
        if set(row) != set(expected_row):
            raise ValueError(f"{name} manifest row has unexpected or missing fields")
        expected_sidecar_fields = {
            "schema",
            "status",
            "binding_sha256",
            "arm",
            "run_name",
            "unit",
            "invocation_id",
            "selected_epoch",
            "successful_updates",
            "terminal_acceptance",
            "exact_stop",
            "checkpoint",
            "representation_mode",
            "source",
            "split",
            "ordered_groups",
            "membership_encoding",
            "membership_sha256",
            "feature_only",
            "representation",
            "inference",
            "prediction",
            "scoring_performed",
        }
        if set(sidecar) != expected_sidecar_fields:
            raise ValueError(f"{name} sidecar has unexpected or missing fields")
        if (
            sidecar.get("schema") != "fg19.first-cell-feature-only-prediction.v1"
            or sidecar.get("status") != "PREDICTION_EXPORT_ONLY_NOT_SCORED"
            or sidecar.get("binding_sha256") != EXPECTED_BINDING_SHA256
            or sidecar.get("arm") != name
            or sidecar.get("run_name") != arm.get("run_name")
            or sidecar.get("unit") != arm.get("unit")
            or sidecar.get("invocation_id") != arm.get("invocation_id")
            or sidecar.get("selected_epoch") != 5
            or sidecar.get("successful_updates") != 44760
            or sidecar.get("terminal_acceptance") != arm.get("terminal_acceptance")
            or sidecar.get("exact_stop") != arm.get("exact_stop")
            or sidecar.get("source") != binding.get("source")
        ):
            raise ValueError(f"{name} sidecar identity differs from the registered arm")
        checkpoint = sidecar.get("checkpoint")
        if (
            not isinstance(checkpoint, dict)
            or any(
                checkpoint.get(key) != arm.get("model", {}).get(key)
                for key in ("bytes", "sha256", "relative_path")
            )
            or checkpoint.get("loaded_sha256") != arm.get("model", {}).get("sha256")
        ):
            raise ValueError(f"{name} checkpoint pin differs")
        if sidecar.get("split", {}).get("sha256") != EXPECTED_SPLIT_SHA256:
            raise ValueError(f"{name} split pin differs")
        if sidecar.get("split") != binding.get("split") | {
            "role": "fit-excluded-exposed-development-evaluation"
        }:
            raise ValueError(f"{name} split contract differs")
        if sidecar.get("representation", {}).get("mode") != mode:
            raise ValueError(f"{name} representation mode differs")
        representation = binding.get("representation")
        if not isinstance(representation, dict) or sidecar.get("representation") != {
            "mode": mode,
            "artifact_sha256": representation.get("artifact_sha256"),
            "state_sha256": representation.get("state_sha256"),
            "normalization": representation.get("normalization"),
        }:
            raise ValueError(f"{name} representation artifact or normalization differs")
        require_ordered_groups(manifest_groups, sidecar.get("ordered_groups"))
        if (
            sidecar.get("membership_encoding")
            != "ASCII decimal row-group IDs, one per line, final LF"
            or sidecar.get("membership_sha256") != expected_membership_sha256
        ):
            raise ValueError(f"{name} ordered-membership encoding or digest differs")
        if sidecar.get("feature_only") != {"targets_read": False, "scored_mask_read": False}:
            raise ValueError(f"{name} prediction pass was not feature-only")
        if sidecar.get("inference") != {
            "precision": "strict_fp32",
            "autocast": False,
            "tf32": False,
            "chunk_rows": 2000,
            "sequence_batch_size": 1,
        }:
            raise ValueError(f"{name} inference precision or batch contract differs")
        prediction_meta = sidecar.get("prediction")
        if not isinstance(prediction_meta, dict):
            raise ValueError(f"{name} prediction metadata is malformed")
        header_bytes = prediction_meta.get("npy_header_bytes")
        if type(header_bytes) is not int or not 0 < header_bytes <= 4096:
            raise ValueError(f"{name} NPY header length is outside the fixed bound")
        if (
            prediction_meta
            != {
                "path": "predictions.npy",
                "bytes": prediction_pin["bytes"],
                "npy_header_bytes": header_bytes,
                "data_bytes": PREDICTION_DATA_BYTES,
                "sha256": prediction_pin["sha256"],
                "dtype": "float32-little-endian",
                "shape": list(PREDICTION_SHAPE),
            }
            or prediction_pin["bytes"] != PREDICTION_DATA_BYTES + header_bytes
        ):
            raise ValueError(f"{name} NPY metadata differs")
        if sidecar.get("scoring_performed") is not False:
            raise ValueError(f"{name} export packet already claims scoring")
        mapping_inputs[name] = (prediction_path, header_bytes, prediction_identity)
        pins[name] = {"prediction": prediction_pin, "sidecar": sidecar_pin}
        identities.extend(
            [
                {"arm": name, "kind": "prediction", "identity": prediction_identity},
                {"arm": name, "kind": "sidecar", "identity": sidecar_identity},
            ]
        )
    for name, (prediction_path, header_bytes, prediction_identity) in mapping_inputs.items():
        array = np.load(prediction_path, mmap_mode="r", allow_pickle=False)
        if (
            not isinstance(array, np.memmap)
            or array.shape != PREDICTION_SHAPE
            or array.dtype.str != "<f4"
            or array.flags.writeable
            or not array.flags.c_contiguous
            or array.offset != header_bytes
        ):
            raise ValueError(f"{name} NPY array shape, dtype, or read-only mode differs")
        linked = prediction_path.lstat()
        if _stable_identity(linked, linked) != prediction_identity:
            raise ValueError(f"{name} prediction path changed before mapping")
        arrays[name] = array
    return arrays, pins, identities
