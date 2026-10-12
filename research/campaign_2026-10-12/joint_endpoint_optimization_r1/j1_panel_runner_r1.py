"""Run the root-admitted, fixed-prediction J1 component-panel comparison.

This runner reads no data payload until an exact root admission binds its
manifest, source pins, phases, and one-CPU resource ceiling. It never loads a
model or invokes a predictor. Calibration and evaluation are separate calls so
calibration arrays can be released before the untouched evaluation components
are read.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import mmap
import os
import struct
import sys
import zipfile
from contextlib import ExitStack
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "src"))
MANIFEST_PATH = HERE / "J1_COMPONENT_PANEL_MANIFEST_R2.json"
MAX_NPY_HEADER_BYTES = 65_536
ROW_COUNT = 20_000
WARMUP_ROWS = 99
POLICY_WEIGHTS: dict[str, tuple[float, float, float]] = {
    "seed3_single": (1.0, 0.0, 0.0),
    "seed4_single": (0.0, 1.0, 0.0),
    "seed5_single": (0.0, 0.0, 1.0),
    "seed3_seed4_equal": (0.5, 0.5, 0.0),
    "seed3_seed5_equal": (0.5, 0.0, 0.5),
    "seed4_seed5_equal": (0.0, 0.5, 0.5),
    "all_three_equal": (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
}
RECEIPT_NAMES = {
    "actual_prediction_grade",
    "actual_full_prediction_validation",
    "actual_source_preflight",
    "raw_valid_content_review",
}


def sha256_bytes(raw: bytes) -> str:
    """Return a lowercase SHA-256 digest."""
    return hashlib.sha256(raw).hexdigest()


def sha256_path(path: Path) -> str:
    """Hash one file using a small bounded buffer."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_stream(stream: BinaryIO) -> str:
    """Hash an already-open file without allocating the payload in memory."""
    digest = hashlib.sha256()
    stream.seek(0)
    for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
        digest.update(chunk)
    stream.seek(0)
    return digest.hexdigest()


def load_json(path: Path) -> tuple[dict[str, Any], bytes]:
    """Load a small JSON receipt and retain the exact bytes for hash checks."""
    raw = path.read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value, raw


def _read_npy_header(stream: BinaryIO, file_size: int, *, base_offset: int = 0) -> dict[str, Any]:
    """Read a bounded NPY header from the stream's current position."""
    prefix = stream.read(8)
    if len(prefix) != 8 or prefix[:6] != b"\x93NUMPY":
        raise ValueError("invalid NPY magic")
    major, minor = prefix[6], prefix[7]
    if major == 1:
        length_bytes = stream.read(2)
        encoding = "latin1"
        length_format = "<H"
    elif major in (2, 3):
        length_bytes = stream.read(4)
        encoding = "utf-8" if major == 3 else "latin1"
        length_format = "<I"
    else:
        raise ValueError(f"unsupported NPY version {major}.{minor}")
    if len(length_bytes) != struct.calcsize(length_format):
        raise ValueError("truncated NPY header length")
    header_length = struct.unpack(length_format, length_bytes)[0]
    if not 0 < header_length <= MAX_NPY_HEADER_BYTES:
        raise ValueError(f"NPY header length outside bound: {header_length}")
    header_bytes = stream.read(header_length)
    if len(header_bytes) != header_length:
        raise ValueError("truncated NPY header")
    header = ast.literal_eval(header_bytes.decode(encoding).strip())
    if not isinstance(header, dict):
        raise ValueError("NPY header is not a dictionary")
    shape = tuple(int(value) for value in header["shape"])
    dtype = np.dtype(header["descr"])
    data_offset = stream.tell() - base_offset
    expected_size = data_offset + math.prod(shape) * dtype.itemsize
    if expected_size != file_size:
        raise ValueError("NPY header and file/member size disagree")
    return {
        "version": f"{major}.{minor}",
        "shape": shape,
        "dtype": dtype,
        "fortran_order": bool(header["fortran_order"]),
        "data_offset": data_offset,
    }


class MappedNpy:
    """Open, hash, and memory-map a pinned NPY through the same descriptor."""

    def __init__(
        self,
        path: Path,
        *,
        expected_sha256: str,
        expected_stat_identity: dict[str, Any],
        expected_shape: tuple[int, ...],
        expected_dtype: str,
    ) -> None:
        self.path = path
        self.expected_sha256 = expected_sha256
        self.expected_stat_identity = expected_stat_identity
        self.expected_shape = expected_shape
        self.expected_dtype = np.dtype(expected_dtype)
        self.stream: BinaryIO | None = None
        self.mapping: mmap.mmap | None = None
        self.array: np.ndarray | None = None
        self.header: dict[str, Any] | None = None

    def __enter__(self) -> MappedNpy:
        self.stream = self.path.open("rb")
        before = os.fstat(self.stream.fileno())
        if (
            before.st_ino != int(self.expected_stat_identity["inode"])
            or before.st_size != int(self.expected_stat_identity["bytes"])
            or before.st_mtime_ns != int(self.expected_stat_identity["mtime_ns"])
        ):
            raise ValueError(f"prediction NPY stat identity changed: {self.path}")
        self.header = _read_npy_header(self.stream, before.st_size)
        if (
            self.header["shape"] != self.expected_shape
            or self.header["dtype"] != self.expected_dtype
            or self.header["fortran_order"]
        ):
            raise ValueError(f"prediction NPY layout changed: {self.path}")
        if sha256_stream(self.stream) != self.expected_sha256:
            raise ValueError(f"prediction NPY full SHA-256 changed: {self.path}")
        _discard_file_cache(self.stream.fileno(), before.st_size)
        after = os.fstat(self.stream.fileno())
        if (before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise ValueError(f"prediction NPY changed while it was being verified: {self.path}")
        self.mapping = mmap.mmap(self.stream.fileno(), 0, access=mmap.ACCESS_READ)
        self.array = np.ndarray(
            shape=self.header["shape"],
            dtype=self.header["dtype"],
            buffer=self.mapping,
            offset=self.header["data_offset"],
            order="C",
        )
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.array = None
        if self.mapping is not None:
            self.mapping.close()
            self.mapping = None
        if self.stream is not None:
            self.stream.close()
            self.stream = None


class StoredTargetArchive:
    """Hash and memory-map uncompressed NPY members in the official target NPZ."""

    def __init__(self, path: Path, identity: dict[str, Any], expected_sha256: str) -> None:
        self.path = path
        self.identity = identity
        self.expected_sha256 = expected_sha256
        self.stream: BinaryIO | None = None
        self.archive: zipfile.ZipFile | None = None
        self.mapping: mmap.mmap | None = None
        self.arrays: dict[str, np.ndarray] = {}

    def __enter__(self) -> StoredTargetArchive:
        self.stream = self.path.open("rb")
        before = os.fstat(self.stream.fileno())
        if (
            before.st_ino != int(self.identity["inode"])
            or before.st_size != int(self.identity["bytes"])
            or before.st_mtime_ns != int(self.identity["mtime_ns"])
        ):
            raise ValueError("valid target cache stat identity changed")
        if sha256_stream(self.stream) != self.expected_sha256:
            raise ValueError("valid target cache full SHA-256 changed")
        _discard_file_cache(self.stream.fileno(), before.st_size)
        after = os.fstat(self.stream.fileno())
        if (before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise ValueError("valid target cache changed while it was being verified")
        self.archive = zipfile.ZipFile(self.stream, mode="r")
        self.mapping = mmap.mmap(self.stream.fileno(), 0, access=mmap.ACCESS_READ)
        return self

    def array(self, member_name: str, expected_shape: tuple[int, ...]) -> np.ndarray:
        """Map one stored member after checking its local and NPY headers."""
        if self.archive is None or self.stream is None or self.mapping is None:
            raise RuntimeError("target archive is not open")
        if member_name not in {"y.npy", "mask.npy", "row_group.npy", "seq_ix.npy"}:
            raise ValueError(f"unapproved target cache member: {member_name}")
        try:
            info = self.archive.getinfo(member_name)
        except KeyError as error:
            raise ValueError(f"target cache member is absent: {member_name}") from error
        if info.compress_type != zipfile.ZIP_STORED:
            raise ValueError(f"target cache member must be stored, not compressed: {member_name}")
        self.stream.seek(info.header_offset)
        fixed = self.stream.read(30)
        if len(fixed) != 30 or fixed[:4] != b"PK\x03\x04":
            raise ValueError(f"invalid local ZIP header for {member_name}")
        name_bytes = struct.unpack_from("<H", fixed, 26)[0]
        extra_bytes = struct.unpack_from("<H", fixed, 28)[0]
        local_name = self.stream.read(name_bytes).decode("utf-8")
        if local_name != member_name:
            raise ValueError(f"local ZIP filename differs for {member_name}")
        data_offset = info.header_offset + 30 + name_bytes + extra_bytes
        self.stream.seek(data_offset)
        header = _read_npy_header(self.stream, info.file_size, base_offset=data_offset)
        if header["shape"] != expected_shape or header["fortran_order"]:
            raise ValueError(f"target cache member shape/order changed: {member_name}")
        array = np.ndarray(
            shape=header["shape"],
            dtype=header["dtype"],
            buffer=self.mapping,
            offset=data_offset + header["data_offset"],
            order="C",
        )
        self.arrays[member_name] = array
        return array

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.arrays.clear()
        if self.mapping is not None:
            self.mapping.close()
            self.mapping = None
        if self.archive is not None:
            self.archive.close()
            self.archive = None
        if self.stream is not None:
            self.stream.close()
            self.stream = None


def _discard_file_cache(fd: int, length: int) -> None:
    """Ask Linux to reclaim bytes just streamed for hashing when supported."""
    advise = getattr(os, "posix_fadvise", None)
    dontneed = getattr(os, "POSIX_FADV_DONTNEED", None)
    if advise is not None and dontneed is not None:
        advise(fd, 0, length, dontneed)


def load_release(
    release_path: Path, manifest: dict[str, Any], manifest_sha256: str, runner_sha256: str
) -> tuple[dict[str, Any], str]:
    """Require a fresh root release before opening any prediction or target payload."""
    release, raw = load_json(release_path)
    digest = sha256_bytes(raw)
    future = manifest["future_execution_contract"]
    limits = future["conditional_future_resource_profile"]
    expected = {
        "schema": "wnn.j1-root-admission.v1",
        "status": "admitted_j1_exposed_f0_component_blend",
        "manifest_sha256": manifest_sha256,
        "runner_sha256": runner_sha256,
        "allow_blend_weight_optimization": True,
        "allow_model_fit_or_prediction_generation": False,
        "allow_provider_or_gpu": False,
        "phases": ["calibration", "evaluation"],
        "max_unique_weight_vectors": manifest["frozen_comparisons"]["max_unique_weight_vectors"],
        "resource_limits": {
            "memory_max_bytes": limits["memory_max_bytes"],
            "runtime_max_seconds": limits["runtime_max_seconds"],
            "cpu_quota": limits["cpu_quota"],
            "swap_bytes": limits["swap_bytes"],
            "threads": future["threads"],
        },
    }
    for key, expected_value in expected.items():
        if release.get(key) != expected_value:
            raise ValueError(f"root admission field {key!r} is absent or differs")
    return release, digest


def validate_static_sources(root: Path, manifest: dict[str, Any]) -> list[dict[str, Any]]:
    """Recheck small source receipts and exact predictor file-stat bindings."""
    source = manifest["source_identity"]
    for key in RECEIPT_NAMES:
        pin = source[key]
        path = root / pin["path"]
        if sha256_path(path) != pin["sha256"]:
            raise ValueError(f"source receipt bytes changed: {path}")
    split_path = root / source["split_manifest"]
    if sha256_path(split_path) != source["split_manifest_sha256"]:
        raise ValueError("fold0 split manifest changed")
    for relative, pin in source["component_graph_sources"].items():
        path = root / relative
        if path.stat().st_size != pin["bytes"] or sha256_path(path) != pin["sha256"]:
            raise ValueError(f"component graph source changed: {path}")
    scoring = source["scoring_sources"]
    for path_key, hash_key in (
        ("metric_source", "metric_source_sha256"),
        ("blend_primitive", "blend_primitive_sha256"),
        ("fixed_mask_source", "fixed_mask_source_sha256"),
    ):
        if sha256_path(root / scoring[path_key]) != scoring[hash_key]:
            raise ValueError(f"scoring source changed: {scoring[path_key]}")
    grade_pin = source["actual_prediction_grade"]
    if sha256_path(root / grade_pin["grade_driver_path"]) != grade_pin["grade_driver_sha256"]:
        raise ValueError("official grade driver source changed")
    propensity = manifest["shared_propensity_ancestry"]
    if sha256_path(root / propensity["provenance_path"]) != propensity["provenance_sha256"]:
        raise ValueError("shared propensity provenance changed")
    runner_pin = source["runner"]
    runner_path = root / runner_pin["path"]
    if sha256_path(runner_path) != runner_pin["sha256"]:
        raise ValueError("J1 runner source changed")
    cache_validation = source["actual_full_prediction_validation"]
    cache_validation_path = root / cache_validation["cache_consumer_validation_path"]
    if sha256_path(cache_validation_path) != cache_validation["cache_consumer_validation_sha256"]:
        raise ValueError("actual cache-consumer validation receipt changed")
    input_validation_path = root / source["actual_full_prediction_validation"]["path"]
    input_validation, _ = load_json(input_validation_path)
    records = {item["provenance"]["run"]: item for item in input_validation["teacher_records"]}
    grade_path = root / source["actual_prediction_grade"]["path"]
    grade, _ = load_json(grade_path)
    current: list[dict[str, Any]] = []
    for member in manifest["members"]:
        run = member["run_name"]
        prediction = member["prediction"]
        prediction_path = root / prediction["path"]
        identity = prediction["full_payload_validation_stat"]
        stat = prediction_path.stat()
        if (stat.st_size, stat.st_mtime_ns, stat.st_ino) != (
            identity["bytes"],
            identity["mtime_ns"],
            identity["inode"],
        ):
            raise ValueError(f"current prediction NPY stat changed: {prediction_path}")
        sidecar_path = root / prediction["producer_sidecar_path"]
        if sha256_path(sidecar_path) != prediction["producer_sidecar_sha256"]:
            raise ValueError(f"prediction sidecar changed: {sidecar_path}")
        provenance_path = root / prediction["provenance_path"]
        if sha256_path(provenance_path) != prediction["provenance_sha256"]:
            raise ValueError(f"model provenance changed: {provenance_path}")
        if run not in records:
            raise ValueError(f"actual input receipt lacks {run}")
        endpoint = grade["endpoints"]["teachers"][f"s{member['seed']}"]
        if (
            endpoint["checkpoint_epoch"] != member["training_recipe"]["selected_epoch"]
            or endpoint["checkpoint_sha256"] != member["checkpoint"]["sha256_from_provenance"]
            or endpoint["prediction_sha256"] != prediction["producer_prediction_sha256"]
            or endpoint["sidecar_sha256"] != prediction["producer_sidecar_sha256"]
            or endpoint["provenance_sha256"] != prediction["provenance_sha256"]
        ):
            raise ValueError(f"actual grade no longer binds {run}")
        current.append(
            {
                "run": run,
                "prediction_path": prediction["path"],
                "prediction_stat_matches_prior_full_sha_receipt": True,
                "prediction_sha256_expected": prediction["producer_prediction_sha256"],
                "full_sha_recheck_at_point_of_use": True,
                "selected_mask_finite_recheck_at_point_of_use": True,
            }
        )
    target = source["valid_target_cache"]
    target_stat = (root / target["path"]).stat()
    if (target_stat.st_size, target_stat.st_mtime_ns, target_stat.st_ino) != (
        target["bytes"],
        target["mtime_ns"],
        target["inode"],
    ):
        raise ValueError("valid target cache stat identity changed")
    return current


def _stage_panel(
    groups: list[dict[str, Any]],
    target_archive: StoredTargetArchive,
    prediction_maps: list[MappedNpy],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Copy only the frozen group rows into bounded float32 stage arrays."""
    group_count = len(groups)
    if group_count < 1:
        raise ValueError("panel partition has no groups")
    y = target_archive.array("y.npy", (1873, ROW_COUNT, 2))
    scored = target_archive.array("mask.npy", (1873, ROW_COUNT))
    row_group = target_archive.array("row_group.npy", (1873,))
    seq_ix = target_archive.array("seq_ix.npy", (1873,))
    if y.dtype != np.dtype("<f4") and y.dtype != np.dtype("float32"):
        raise ValueError("target cache dtype is not float32")
    if scored.dtype != np.dtype("bool"):
        raise ValueError("target cache score mask is not boolean")
    if not np.array_equal(row_group, np.arange(1873, dtype=row_group.dtype)):
        raise ValueError("target cache row-group map changed")
    for mapped in prediction_maps:
        if mapped.array is None or mapped.array.shape != (1873, ROW_COUNT, 2):
            raise ValueError("a bound prediction map is not ready")

    rows = group_count * ROW_COUNT
    targets = np.empty((rows, 2), dtype=np.float32)
    predictions = np.empty((rows, 3, 2), dtype=np.float32)
    mask = np.empty(rows, dtype=bool)
    warmup_mask = np.arange(ROW_COUNT) >= WARMUP_ROWS
    for index, group in enumerate(groups):
        row_group_id = int(group["row_group"])
        if row_group_id < 0 or row_group_id >= 1873:
            raise ValueError("panel row group is outside valid source")
        if int(seq_ix[row_group_id]) != int(group["seq_ix"]):
            raise ValueError("target-cache sequence identity differs from panel manifest")
        start, end = index * ROW_COUNT, (index + 1) * ROW_COUNT
        targets[start:end] = y[row_group_id]
        mask[start:end] = scored[row_group_id] & warmup_mask
        for model_index, mapped in enumerate(prediction_maps):
            if mapped.array is None:
                raise ValueError("prediction mapping closed before staging")
            predictions[start:end, model_index] = mapped.array[row_group_id]
    selected_targets = targets[mask]
    selected_predictions = predictions[mask]
    if selected_targets.size == 0:
        raise ValueError("component panel has no official scored rows")
    if not np.isfinite(selected_targets).all() or not np.isfinite(selected_predictions).all():
        raise ValueError("selected component panel contains nonfinite metric values")
    return targets, predictions, mask


def _unique_candidates(fit: Any, fixed_scores: dict[str, float]) -> list[dict[str, Any]]:
    """Collect at most eleven distinct official-scored calibration vectors."""
    candidates: dict[bytes, dict[str, Any]] = {}

    def add(name: str, weights: np.ndarray, official_score: float) -> None:
        values = np.asarray(weights, dtype=np.float64)
        if values.shape != (3,) or not np.isfinite(values).all():
            raise ValueError("solver returned invalid candidate weights")
        key = values.tobytes()
        candidates.setdefault(key, {"names": [], "weights": values.tolist()})
        candidates[key]["official_wp"] = float(official_score)
        if name not in candidates[key]["names"]:
            candidates[key]["names"].append(name)

    for name, values in POLICY_WEIGHTS.items():
        add(name, np.asarray(values, dtype=np.float64), fixed_scores[name])
    for attempt in fit.attempts:
        add(f"{attempt.start_name}:start", attempt.start_weights, attempt.start_official_score)
        add(f"{attempt.start_name}:slsqp", attempt.weights, attempt.official_score)
    add("selected_solver_policy", fit.weights, fit.official_score)
    if len(candidates) > 11:
        raise ValueError("solver exceeded the frozen 11 unique calibration vectors")
    return list(candidates.values())


def _official_score_table(
    targets: np.ndarray, predictions: np.ndarray, mask: np.ndarray
) -> dict[str, float]:
    """Score the seven fixed policy vectors through the unchanged official metric."""
    from wnn.blend_optimization import official_clip2_wp

    return {
        name: official_clip2_wp(targets, predictions, mask, np.asarray(weights))
        for name, weights in POLICY_WEIGHTS.items()
    }


def run_phase(
    root: Path,
    manifest: dict[str, Any],
    admission_sha256: str,
    phase: str,
    output_path: Path,
    calibration_receipt_path: Path | None,
) -> dict[str, Any]:
    """Run one admitted phase and return a JSON-safe immutable receipt value."""
    if phase not in {"calibration", "evaluation"}:
        raise ValueError("phase must be calibration or evaluation")
    if phase == "evaluation" and calibration_receipt_path is None:
        raise ValueError("evaluation requires a frozen calibration receipt")
    if phase == "calibration" and calibration_receipt_path is not None:
        raise ValueError("calibration cannot consume a prior weight receipt")
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite output receipt: {output_path}")

    calibration_receipt: dict[str, Any] | None = None
    calibration_sha256: str | None = None
    if calibration_receipt_path is not None:
        calibration_receipt, raw = load_json(calibration_receipt_path)
        calibration_sha256 = sha256_bytes(raw)
        if (
            calibration_receipt.get("schema") != "wnn.j1-calibration-result.r1"
            or calibration_receipt.get("status") != "PASS_CALIBRATION_ONLY"
            or calibration_receipt.get("manifest_sha256") != manifest["_manifest_sha256"]
            or calibration_receipt.get("root_admission_sha256") != admission_sha256
            or calibration_receipt.get("phase") != "calibration"
        ):
            raise ValueError("calibration receipt does not bind this manifest/admission")
        weights = np.asarray(calibration_receipt.get("selected_weights"), dtype=np.float64)
        if (
            weights.shape != (3,)
            or not np.isfinite(weights).all()
            or np.any(weights < 0.0)
            or not np.isclose(weights.sum(), 1.0, rtol=0.0, atol=1e-10)
        ):
            raise ValueError("calibration receipt weights are not a valid simplex")

    source_rows = validate_static_sources(root, manifest)
    source = manifest["source_identity"]
    target_pin = source["valid_target_cache"]
    target_path = root / target_pin["path"]
    target_sha256 = target_pin["sha256_from_official_grade"]
    selection = manifest["panel"]["selection"]
    groups = selection[f"{phase}_groups"]
    members = manifest["members"]
    prediction_maps: list[MappedNpy] = []
    with ExitStack() as stack:
        target_archive = stack.enter_context(
            StoredTargetArchive(target_path, target_pin, target_sha256)
        )
        for member in members:
            prediction = member["prediction"]
            prediction_maps.append(
                stack.enter_context(
                    MappedNpy(
                        root / prediction["path"],
                        expected_sha256=prediction["producer_prediction_sha256"],
                        expected_stat_identity=prediction["full_payload_validation_stat"],
                        expected_shape=(1873, ROW_COUNT, 2),
                        expected_dtype="<f4",
                    )
                )
            )
        targets, predictions, mask = _stage_panel(groups, target_archive, prediction_maps)

        base_scores = _official_score_table(targets, predictions, mask)
        selected_policy: dict[str, Any] | None = None
        solver_receipt: dict[str, Any] | None = None
        if phase == "calibration":
            from wnn.blend_optimization import fit_shared_clip2_weights

            fit = fit_shared_clip2_weights(
                targets,
                predictions,
                mask,
                np.asarray(POLICY_WEIGHTS["seed3_seed4_equal"], dtype=np.float64),
                ridge=0.1,
                shrink_to_equal=0.0,
            )
            unique = _unique_candidates(fit, base_scores)
            if len(unique) > manifest["frozen_comparisons"]["max_unique_weight_vectors"]:
                raise ValueError("calibration candidate count exceeded the frozen cap")
            selected_policy = {
                "name": "calibration_selected_local_solver_policy",
                "weights": fit.weights.tolist(),
                "official_wp": float(fit.official_score),
                "surrogate_score": float(fit.surrogate_score),
                "global_optimality_claim": False,
            }
            solver_receipt = {
                "selected_candidate": fit.selected_candidate,
                "ridge": 0.1,
                "shrink_to_equal": 0.0,
                "starts": [
                    {
                        "name": attempt.start_name,
                        "start_weights": attempt.start_weights.tolist(),
                        "start_official_wp": float(attempt.start_official_score),
                        "slsqp_weights": attempt.weights.tolist(),
                        "slsqp_official_wp": float(attempt.official_score),
                        "slsqp_surrogate_wp": float(attempt.surrogate_score),
                        "iterations": attempt.iterations,
                        "success": attempt.success,
                        "status": attempt.status,
                        "message": attempt.message,
                    }
                    for attempt in fit.attempts
                ],
                "unique_candidate_count": len(unique),
                "unique_candidates": unique,
                "gradient_interpretation": (
                    "continuous float64 clip-2 surrogate gradient; official candidate "
                    "scores use float32 synthesis and unchanged global_wp_clip2_v1"
                ),
            }
        else:
            assert calibration_receipt is not None
            selected_weights = np.asarray(calibration_receipt["selected_weights"], dtype=np.float64)
            from wnn.blend_optimization import official_clip2_wp

            selected_score = float(official_clip2_wp(targets, predictions, mask, selected_weights))
            selected_policy = {
                "name": "calibration_selected_local_solver_policy",
                "weights": selected_weights.tolist(),
                "official_wp": selected_score,
                "calibration_receipt_sha256": calibration_sha256,
                "weights_frozen_before_evaluation_read": True,
            }

        result = {
            "schema": (
                "wnn.j1-calibration-result.r1"
                if phase == "calibration"
                else "wnn.j1-evaluation-result.r1"
            ),
            "status": (
                "PASS_CALIBRATION_ONLY" if phase == "calibration" else "PASS_EVALUATION_ONLY"
            ),
            "experiment_id": manifest["experiment"]["id"],
            "phase": phase,
            "manifest_sha256": manifest["_manifest_sha256"],
            "runner_sha256": source["runner"]["sha256"],
            "root_admission_sha256": admission_sha256,
            "source_checks": source_rows,
            "panel": {
                "components": selection[f"{phase}_component_ids"],
                "groups": groups,
                "groups_sha256_sorted_int64le": selection[f"{phase}_groups_sha256_sorted_int64le"],
                "official_scored_row_count": int(mask.sum()),
                "mask": "valid_targets.npz:is_scored AND row_index >= 99",
                "mask_finite_validation": True,
            },
            "fixed_policy_official_wp": base_scores,
            "selected_policy": selected_policy,
            "solver": solver_receipt,
            "limitations": [
                "F0 is an exposed development fold; this fixed component split is not a pristine holdout.",
                "No model fit, prediction generation, GPU or provider access occurred.",
                "The local SLSQP endpoint is selected among three starts and is not a global optimum.",
                "The solver gradient is a continuous surrogate; reported candidate scores use the unchanged official float32 metric path.",
                "No cross-fold or public score claim follows from this bounded panel.",
            ],
        }
        del targets, predictions, mask
    if phase == "calibration":
        result["selected_weights"] = selected_policy["weights"] if selected_policy else None
    else:
        result["calibration_receipt_path"] = str(calibration_receipt_path)
        result["calibration_receipt_sha256"] = calibration_sha256
    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    with output_path.open("xb") as stream:
        stream.write(raw)
    return result


def main() -> None:
    """Require root admission, then run exactly one frozen panel phase."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("calibration", "evaluation"), required=True)
    parser.add_argument("--root-admission", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calibration-receipt", type=Path)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    args = parser.parse_args()

    manifest, manifest_raw = load_json(args.manifest)
    manifest_sha256 = sha256_bytes(manifest_raw)
    manifest["_manifest_sha256"] = manifest_sha256
    runner_path = Path(__file__).resolve()
    runner_sha256 = sha256_path(runner_path)
    release, admission_sha256 = load_release(
        args.root_admission, manifest, manifest_sha256, runner_sha256
    )
    del release
    # No source array is opened before the complete frozen root release passes.
    result = run_phase(
        ROOT,
        manifest,
        admission_sha256,
        args.phase,
        args.output.resolve(),
        args.calibration_receipt.resolve() if args.calibration_receipt else None,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "phase": result["phase"],
                "output": str(args.output.resolve()),
                "output_sha256": sha256_path(args.output.resolve()),
                "manifest_sha256": manifest_sha256,
                "root_admission_sha256": admission_sha256,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
