#!/usr/bin/env python3
"""Export frozen E540 V8/V9 checkpoints on the fixed 468 feature-only groups."""

from __future__ import annotations

import ast
import contextlib
import hashlib
import io
import json
import math
import os
import shutil
import struct
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import TextIO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from e540_cli_preflight import normalize_entry_argv, parse_project_cli  # noqa: E402
from e540_export_validation import (  # noqa: E402
    CHECKPOINT_WRITE_ALLOWANCE_SECONDS,
    HISTORY_ROUNDING_TOLERANCE_SECONDS,
    canonical_json_sha256,
    validate_fit_artifact_metadata,
    validate_pair_metadata,
    validate_snapshot_provenance,
)

PACKET_ID = "E540_FP32_V10_FIXED_CHECKPOINT_EXPORT_ONLY"
SPEC_NAME = "e540-v10-fixed-checkpoint-export-20261008"
AUTHORIZATION_PATH = Path(
    "research/campaign_2026-10-06/bandit_continuation/numerical_embedding_E540/"
    "v10_export_only/E540_V10_ROOT_EXPORT_AUTHORIZATION.json"
)
RECIPE_PATH = Path(
    "research/campaign_2026-10-06/bandit_continuation/numerical_embedding_E540/"
    "v10_export_only/RECIPE_V10_EXPORT_ONLY.json"
)
BINDING_PATH = Path(
    "research/campaign_2026-10-06/bandit_continuation/numerical_embedding_E540/"
    "v10_export_only/E540_V10_ARTIFACT_BINDING.json"
)
SUBSET_PATH = Path(
    "research/campaign_2026-10-06/bandit_continuation/numerical_embedding_E540/E540_FIT_SUBSET_V5.json"
)
SPLIT_PATH = Path("research/campaign_2026-09-29/artifacts/splits/fold0.json")
PORTABLE_INPUTS_PATH = Path("research/campaign_2026-10-04/portable_inputs.json")
RUN_NAMES = {
    "linear": "e540v8-linear-s540-fp32",
    "ple": "e540v9-ple-s540-fp32",
}
RUNTIME = ROOT / "e540_v10_export"
MAX_WORKER_SECONDS = 5_200
FINALIZATION_RESERVE_SECONDS = 300


def write_json(path: Path, value: object) -> None:
    """Atomically persist a finite phase receipt."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    """Hash large inputs and outputs with bounded memory."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


class Tee(io.TextIOBase):
    """Flush prediction subprocess-style logs to the provider output stream."""

    def __init__(self, target: TextIO, capture: io.StringIO) -> None:
        self.target = target
        self.capture = capture

    def write(self, value: str) -> int:
        self.capture.write(value)
        written = self.target.write(value)
        self.target.flush()
        return written

    def flush(self) -> None:
        self.target.flush()
        self.capture.flush()


def prediction_argv(mode: str) -> list[str]:
    """Return the exact feature-only fixed-epoch prediction command."""
    return [
        "scripts/predict_run.py",
        RUN_NAMES[mode],
        "valid",
        "--epoch",
        "0",
        "--inference-precision",
        "fp32",
        "--output-name",
        f"{RUN_NAMES[mode]}-eval",
        "--fit-subset-manifest",
        SUBSET_PATH.as_posix(),
    ]


def _feature_reader_requests_no_labels(method: ast.FunctionDef | None) -> bool:
    if method is None:
        return False
    for node in ast.walk(method):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "read":
            continue
        keywords = {keyword.arg: keyword.value for keyword in node.keywords}
        if all(
            isinstance(keywords.get(name), ast.Constant) and keywords[name].value is False
            for name in ("targets", "mask")
        ):
            return True
    return False


def validate_prediction_clis() -> list[dict[str, object]]:
    """Parse both actual predictor CLIs without importing Torch or data."""
    source = ROOT / "scripts/predict_run.py"
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    feature_class = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "FeatureOnlySequences"
        ),
        None,
    )
    feature_getitem = (
        next(
            (
                node
                for node in feature_class.body
                if isinstance(node, ast.FunctionDef) and node.name == "__getitem__"
            ),
            None,
        )
        if feature_class is not None
        else None
    )
    if not _feature_reader_requests_no_labels(feature_getitem):
        raise ValueError("predict_run no longer has the frozen targets=False/mask=False reader")
    checks = []
    for mode in ("linear", "ple"):
        logged = prediction_argv(mode)
        cli = normalize_entry_argv("predict_run.py", logged)
        parsed = parse_project_cli(source, cli, parser_name="parser")
        if (
            parsed.name != RUN_NAMES[mode]
            or parsed.split != "valid"
            or parsed.epoch != 0
            or parsed.inference_precision != "fp32"
            or parsed.output_name != f"{RUN_NAMES[mode]}-eval"
            or parsed.fit_subset_manifest != SUBSET_PATH
            or parsed.gate_mode != "soft"
        ):
            raise ValueError(f"E540 {mode} export CLI differs from the frozen recipe")
        checks.append(
            {
                "mode": mode,
                "entry_point": "scripts/predict_run.py",
                "status": "parsed",
                "feature_only_subset": True,
                "reader_requests_no_targets_or_mask": True,
                "script_token_removed": logged[1:] == cli,
            }
        )
    return checks


def validate_authorization(path: Path, expected: Mapping[str, object]) -> dict[str, object]:
    """Require root's hash-bound export authorization before mounted-data hashing."""
    receipt = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema": "wnn.e540-export-only-authorization.v1",
        "owner": "root",
        "status": "approved",
        "authorized": True,
        "export_only": True,
        "fit_allowed": False,
        "provider_launch_authorized": True,
    }
    for key, value in required.items():
        if receipt.get(key) != value:
            raise ValueError(f"root export authorization is missing valid {key}")
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise ValueError(f"root export authorization does not bind {key}")
    authorization_id = receipt.get("authorization_id")
    if not isinstance(authorization_id, str) or not authorization_id.strip():
        raise ValueError("root export authorization requires an authorization_id")
    return receipt


def worker_budget_seconds() -> int:
    """Bound work to the kernel deadline while reserving runner finalization time."""
    raw = os.environ.get("WNN_KERNEL_DEADLINE_MONOTONIC")
    if raw is None:
        return MAX_WORKER_SECONDS
    deadline = float(raw)
    if not math.isfinite(deadline):
        raise ValueError("kernel deadline is not finite")
    remaining = math.floor(deadline - time.monotonic() - FINALIZATION_RESERVE_SECONDS)
    budget = min(MAX_WORKER_SECONDS, remaining)
    if budget < 1:
        raise TimeoutError("kernel deadline leaves no export time after runner reserve")
    return budget


def _expected_authorization(
    binding_sha: str, recipe_sha: str, subset_sha: str
) -> dict[str, object]:
    return {
        "packet_id": PACKET_ID,
        "code_manifest_sha256": os.environ["WNN_E540_CODE_MANIFEST_SHA256"],
        "artifact_binding_sha256": binding_sha,
        "recipe_sha256": recipe_sha,
        "subset_manifest_sha256": subset_sha,
        "kernel_process_wall_seconds_max": 6_000,
        "worker_wall_seconds_max": MAX_WORKER_SECONDS,
        "linear_epoch0_sha256": os.environ["WNN_E540_LINEAR_EPOCH0_SHA256"],
        "ple_epoch0_sha256": os.environ["WNN_E540_PLE_EPOCH0_SHA256"],
        "linear_provenance_sha256": os.environ["WNN_E540_LINEAR_PROVENANCE_SHA256"],
        "ple_provenance_sha256": os.environ["WNN_E540_PLE_PROVENANCE_SHA256"],
        "linear_history_sha256": os.environ["WNN_E540_LINEAR_HISTORY_SHA256"],
        "ple_history_sha256": os.environ["WNN_E540_PLE_HISTORY_SHA256"],
        "v9_terminal_receipt_sha256": os.environ["WNN_E540_V9_TERMINAL_RECEIPT_SHA256"],
    }


def _asset_path(binding: Mapping[str, object], key: str) -> Path:
    assets = binding.get("assets")
    if not isinstance(assets, Mapping) or key not in assets:
        raise ValueError(f"artifact binding omits {key}")
    entry = assets[key]
    if not isinstance(entry, Mapping) or not isinstance(entry.get("logical_path"), str):
        raise ValueError(f"artifact binding has malformed {key}")
    return ROOT / entry["logical_path"]


def _verify_asset_files(binding: Mapping[str, object]) -> dict[str, dict[str, object]]:
    assets = binding.get("assets")
    if not isinstance(assets, Mapping) or set(assets) != {
        "linear_model",
        "linear_epoch0",
        "linear_provenance",
        "linear_history",
        "linear_control_manifest",
        "ple_model",
        "ple_epoch0",
        "ple_provenance",
        "ple_history",
    }:
        raise ValueError("export packet asset set is not the exact frozen V8/V9 pair")
    observations = {}
    for name, entry in assets.items():
        if not isinstance(entry, Mapping):
            raise ValueError(f"asset descriptor is malformed: {name}")
        path = _asset_path(binding, name)
        observed_size = path.stat().st_size if path.is_file() else None
        observed_sha = sha256_file(path) if observed_size is not None else None
        if observed_size != entry.get("size") or observed_sha != entry.get("sha256"):
            raise ValueError(f"artifact asset identity mismatch: {name}")
        observations[name] = {
            "path": entry["logical_path"],
            "size": observed_size,
            "sha256": observed_sha,
        }
    return observations


def _validate_upstream(binding: Mapping[str, object], recipe: Mapping[str, object]) -> None:
    if (
        binding.get("schema") != "wnn.e540-v10-artifact-binding.v1"
        or binding.get("packet_id") != PACKET_ID
        or binding.get("subset_manifest_sha256") != sha256_file(ROOT / SUBSET_PATH)
        or binding.get("split_manifest_sha256") != sha256_file(ROOT / SPLIT_PATH)
        or recipe.get("packet_id") != PACKET_ID
        or recipe.get("kind") != "export_only"
        or recipe.get("fit_arms") != []
        or recipe.get("profile_arms") != []
        or recipe.get("prediction_arms") != [RUN_NAMES["linear"], RUN_NAMES["ple"]]
    ):
        raise ValueError("V10 artifact binding, recipe, subset or split identity changed")
    if os.environ.get("WNN_E540_PACKET_ID") != PACKET_ID:
        raise ValueError("V10 runtime packet identifier differs from the worker")
    if os.environ.get("WNN_E540_RECIPE_SHA256") != sha256_file(ROOT / RECIPE_PATH):
        raise ValueError("V10 recipe identity differs from the reviewed runtime spec")
    if os.environ.get("WNN_E540_SUBSET_SHA256") != sha256_file(ROOT / SUBSET_PATH):
        raise ValueError("V10 subset identity differs from the reviewed runtime spec")
    portable = json.loads((ROOT / PORTABLE_INPUTS_PATH).read_text(encoding="utf-8"))
    registry = portable.get("sources", {}).get("datasets/valid.parquet", {})
    original = registry.get("original_identity", {})
    if registry.get("sha256") != binding.get("valid_source", {}).get("sha256") or original.get(
        "bytes"
    ) != binding.get("valid_source", {}).get("size"):
        raise ValueError("V10 mounted valid.parquet identity differs from portable input registry")
    input_data = recipe.get("input_data")
    valid_descriptor = binding.get("valid_source")
    if not isinstance(input_data, Mapping) or not isinstance(valid_descriptor, Mapping):
        raise ValueError("V10 valid-only input identity is missing")
    if (
        input_data.get("path") != "datasets/valid.parquet"
        or input_data.get("dataset") != "vladkochetov07/wunder-funder"
        or input_data.get("sha256") != valid_descriptor.get("sha256")
        or input_data.get("size_bytes") != valid_descriptor.get("size")
    ):
        raise ValueError("V10 valid-only source identity differs from the frozen registry")
    if (
        binding.get("selected_component_count") != 512
        or binding.get("evaluation_group_count") != 468
    ):
        raise ValueError("V10 source binding differs from the frozen 512/468 membership")


def _verify_valid_input(binding: Mapping[str, object]) -> dict[str, object]:
    """Hash only valid.parquet; no train labels or target/mask NPZ files are mounted."""
    descriptor = binding["valid_source"]
    path = ROOT / "datasets/valid.parquet"
    expected = {
        "path": "datasets/valid.parquet",
        "size": descriptor["size"],
        "sha256": descriptor["sha256"],
    }
    observed_size = path.stat().st_size if path.is_file() else None
    observed_sha = sha256_file(path) if observed_size is not None else None
    receipt = {
        "status": "verified"
        if observed_size == expected["size"] and observed_sha == expected["sha256"]
        else "failed",
        "expected": expected,
        "observed": {"size": observed_size, "sha256": observed_sha},
        "read_policy": "byte identity check only; predictor reads selected features with targets=False, mask=False",
    }
    write_json(RUNTIME / "valid_input_identity.json", receipt)
    if receipt["status"] != "verified":
        raise ValueError("mounted valid.parquet identity differs from the frozen input")
    return receipt


def _validate_training_sources(
    binding: Mapping[str, object], provenances: Mapping[str, Mapping[str, object]]
) -> None:
    source_hashes = binding.get("training_source_hashes")
    if not isinstance(source_hashes, Mapping):
        raise ValueError("V10 binding omits actual training source hashes")
    expected_sources = {
        "scripts/train_rnn.py": "trainer_source_sha256",
        "src/wnn/nn.py": "model_source_sha256",
        "src/wnn/run_identity.py": "run_identity_source_sha256",
        "src/wnn/numeric_ple.py": "module_source_sha256",
    }
    for source_path, provenance_key in expected_sources.items():
        expected = source_hashes.get(source_path)
        if not isinstance(expected, str) or sha256_file(ROOT / source_path) != expected:
            raise ValueError(f"V10 shipped training source identity mismatch: {source_path}")
        for mode, provenance in provenances.items():
            numeric = provenance.get("numeric_preprocessing")
            observed = (
                numeric.get(provenance_key)
                if provenance_key == "module_source_sha256" and isinstance(numeric, Mapping)
                else provenance.get(provenance_key)
            )
            if observed != expected:
                raise ValueError(f"{mode} fit provenance source identity mismatch: {source_path}")


def _validate_checkpoints(
    binding: Mapping[str, object],
    subset: dict[str, object],
    artifact_observations: Mapping[str, Mapping[str, object]],
    checkpoint_write_allowance_seconds: float,
) -> dict[str, object]:
    import numpy as np
    import torch

    receipts: dict[str, object] = {}
    provenances: dict[str, Mapping[str, object]] = {}
    for mode in ("linear", "ple"):
        prefix = "linear" if mode == "linear" else "ple"
        run_name = RUN_NAMES[mode]
        history_path = _asset_path(binding, f"{prefix}_history")
        provenance_path = _asset_path(binding, f"{prefix}_provenance")
        history = json.loads(history_path.read_text(encoding="utf-8"))
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        if not isinstance(provenance, Mapping):
            raise ValueError(f"{mode} sibling provenance is malformed")
        model_sha = artifact_observations[f"{prefix}_model"]["sha256"]
        epoch0_sha = artifact_observations[f"{prefix}_epoch0"]["sha256"]
        metadata_receipt = validate_fit_artifact_metadata(
            mode,
            history,
            provenance,
            subset,
            model_sha256=model_sha,
            epoch0_sha256=epoch0_sha,
            expected_run_name=run_name,
            checkpoint_write_allowance_seconds=checkpoint_write_allowance_seconds,
        )
        model_path = _asset_path(binding, f"{prefix}_model")
        epoch0_path = _asset_path(binding, f"{prefix}_epoch0")
        checkpoint = torch.load(epoch0_path, map_location="cpu", weights_only=False)
        selected = torch.load(model_path, map_location="cpu", weights_only=False)
        validate_snapshot_provenance(checkpoint, provenance)
        validate_snapshot_provenance(selected, provenance)
        if (
            checkpoint.get("epoch") != 0
            or selected.get("epoch") != 0
            or checkpoint.get("n_in") != 211
            or selected.get("n_in") != 211
        ):
            raise ValueError(f"{mode} snapshots are not the fixed raw-211 epoch-0 checkpoints")
        if checkpoint.get("args") != selected.get("args"):
            raise ValueError(f"{mode} model.pt and model_ep0.pt trainer arguments differ")
        state = checkpoint.get("model")
        selected_state = selected.get("model")
        if (
            not isinstance(state, Mapping)
            or not isinstance(selected_state, Mapping)
            or state.keys() != selected_state.keys()
        ):
            raise ValueError(f"{mode} model snapshots have different state dictionaries")
        for name, tensor in state.items():
            if not torch.is_tensor(tensor) or not bool(torch.isfinite(tensor).all().item()):
                raise ValueError(f"{mode} checkpoint has a nonfinite or malformed tensor: {name}")
            if not torch.equal(tensor, selected_state[name]):
                raise ValueError(f"{mode} model.pt differs from fixed epoch-0 tensor {name}")
        readout = checkpoint.get("readout")
        selected_readout = selected.get("readout")
        if (readout is None) != (selected_readout is None):
            raise ValueError(f"{mode} readout differs across model snapshots")
        if readout is not None:
            if (
                not isinstance(readout, (list, tuple))
                or not isinstance(selected_readout, (list, tuple))
                or len(readout) != len(selected_readout)
            ):
                raise ValueError(f"{mode} readout snapshots are malformed")
            if any(
                not torch.equal(left, right)
                for left, right in zip(readout, selected_readout, strict=True)
            ):
                raise ValueError(f"{mode} readout differs across model snapshots")
            if any(not bool(torch.isfinite(tensor).all().item()) for tensor in readout):
                raise ValueError(f"{mode} readout contains a nonfinite tensor")
        numeric = provenance.get("numeric_preprocessing")
        if not isinstance(numeric, Mapping) or numeric.get("fit_members") != subset.get(
            "selected_groups"
        ):
            raise ValueError(f"{mode} numeric preprocessing differs from the frozen fit members")
        if mode == "ple":
            knots = state.get("numeric_feature_map.knots")
            counts = state.get("numeric_feature_map.knot_counts")
            fitted = state.get("numeric_feature_map.knots_fitted")
            if knots is None or counts is None or fitted is None:
                raise ValueError("PLE checkpoint is missing its installed knot buffers")
            knot_array = knots.detach().cpu().numpy().astype("<f4", copy=False)
            count_array = counts.detach().cpu().numpy().astype("<i8", copy=False)
            if (
                knot_array.shape != (112, 9)
                or count_array.shape != (112,)
                or not np.isfinite(knot_array).all()
                or np.any(count_array < 1)
                or np.any(count_array > 9)
                or not bool(fitted.detach().cpu().item())
            ):
                raise ValueError("PLE checkpoint knot buffers are invalid")
            knot_sha = hashlib.sha256(knot_array.tobytes() + count_array.tobytes()).hexdigest()
            if numeric.get("knot_sha256") != knot_sha:
                raise ValueError("PLE knot buffers differ from original fitted preprocessing")
        run_dir = ROOT / "artifacts/runs" / run_name
        if run_dir.exists():
            raise FileExistsError(f"refusing to overwrite existing run: {run_name}")
        run_dir.mkdir(parents=True)
        for source, filename in (
            (model_path, "model.pt"),
            (epoch0_path, "model_ep0.pt"),
            (provenance_path, "provenance.json"),
            (history_path, "history.json"),
        ):
            shutil.copyfile(source, run_dir / filename)
        receipts[mode] = {
            "status": "checkpoint_and_fit_metadata_pass",
            "run_name": run_name,
            "metadata": metadata_receipt,
            "checkpoint_sha256": epoch0_sha,
            "model_sha256": model_sha,
            "parameter_finiteness": {"status": "finite", "tensor_count": len(state)},
            "history_sha256": artifact_observations[f"{prefix}_history"]["sha256"],
            "provenance_sha256": artifact_observations[f"{prefix}_provenance"]["sha256"],
            "numeric_preprocessing_sha256": canonical_json_sha256(numeric),
        }
        provenances[mode] = provenance

    pair_identity = validate_pair_metadata(provenances["linear"], provenances["ple"])
    receipts["paired_fit_identity"] = pair_identity
    receipts["paired_fit_identity"]["status"] = "matched_sampler_normalization_and_recipe"
    return receipts


def call_predictor(module: object, argv: list[str], log_path: Path) -> str:
    """Run one exact predict_run CLI in-process and keep a flushed phase log."""
    capture = io.StringIO()
    old_argv = sys.argv
    try:
        cli = normalize_entry_argv("predict_run.py", argv)
        sys.argv = ["predict_run.py", *cli]
        with (
            log_path.open("w", encoding="utf-8") as log,
            contextlib.redirect_stdout(Tee(log, capture)),
        ):
            module.main()
    finally:
        sys.argv = old_argv
    return capture.getvalue()


def validate_prediction(mode: str, subset: Mapping[str, object]) -> dict[str, object]:
    import numpy as np

    prediction = ROOT / "cache" / f"pred_{RUN_NAMES[mode]}-eval_valid.npy"
    sidecar = prediction.with_suffix(".json")
    metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    groups = subset["evaluation_groups"]
    digest = hashlib.sha256(struct.pack(f"<{len(groups)}q", *groups)).hexdigest()
    observed_prediction_sha = sha256_file(prediction)
    checkpoint_sha = sha256_file(ROOT / "artifacts/runs" / RUN_NAMES[mode] / "model_ep0.pt")
    observed_array = np.load(prediction, mmap_mode="r", allow_pickle=False)
    if observed_array.shape != (468, 20_000, 2) or observed_array.dtype != np.dtype(np.float32):
        raise ValueError(f"{mode} export has the wrong 468-group FP32 shape")
    for start in range(0, 468, 8):
        if not np.isfinite(observed_array[start : start + 8]).all():
            raise ValueError(f"{mode} export has nonfinite predictions")
    if (
        metadata.get("run") != RUN_NAMES[mode]
        or metadata.get("split") != "valid"
        or metadata.get("epoch") != 0
        or metadata.get("checkpoint_sha256") != checkpoint_sha
        or metadata.get("prediction_sha256") != observed_prediction_sha
        or metadata.get("row_groups") != groups
        or metadata.get("evaluation_membership_sha256") != digest
        or metadata.get("fit_subset_manifest_sha256") != sha256_file(ROOT / SUBSET_PATH)
        or metadata.get("inference_precision") != "fp32"
    ):
        raise ValueError(
            f"{mode} prediction sidecar differs from the fixed feature-only membership"
        )
    return {
        "status": "prediction_pass",
        "path": prediction.relative_to(ROOT).as_posix(),
        "size_bytes": prediction.stat().st_size,
        "sha256": observed_prediction_sha,
        "sidecar": sidecar.relative_to(ROOT).as_posix(),
        "sidecar_sha256": sha256_file(sidecar),
        "row_group_count": 468,
        "evaluation_membership_sha256": digest,
        "checkpoint_sha256": checkpoint_sha,
        "shape": list(observed_array.shape),
        "dtype": str(observed_array.dtype),
        "finite": True,
        "target_columns_requested": False,
        "mask_column_requested": False,
        "target_npz_mounted": False,
    }


def main() -> None:
    """Validate source and fixed checkpoints, then export both frozen predictions."""
    started = time.monotonic()
    RUNTIME.mkdir(parents=True, exist_ok=False)
    result: dict[str, object] = {
        "schema": "wnn.e540-v10-export-worker-result.v1",
        "packet_id": PACKET_ID,
        "status": "starting",
        "fit_started": False,
        "profile_started": False,
        "grade_started": False,
        "prediction_arms": {"linear": {"status": "not_started"}, "ple": {"status": "not_started"}},
        "started_unix": time.time(),
    }
    write_json(RUNTIME / "result.json", result)
    try:
        worker_seconds = worker_budget_seconds()
        recipe = json.loads((ROOT / RECIPE_PATH).read_text(encoding="utf-8"))
        binding_path = ROOT / BINDING_PATH
        binding = json.loads(binding_path.read_text(encoding="utf-8"))
        if not isinstance(binding, Mapping) or not isinstance(recipe, Mapping):
            raise ValueError("V10 recipe or artifact binding is malformed")
        binding_sha = sha256_file(binding_path)
        recipe_sha = sha256_file(ROOT / RECIPE_PATH)
        subset_sha = sha256_file(ROOT / SUBSET_PATH)
        expected_auth = _expected_authorization(binding_sha, recipe_sha, subset_sha)
        auth_path = ROOT / AUTHORIZATION_PATH
        if os.environ.get("WNN_E540_AUTHORIZATION_PATH") != AUTHORIZATION_PATH.as_posix():
            raise ValueError("V10 authorization path differs from the approved packet")
        if auth_path.as_posix().split("/")[-1] != os.environ.get("WNN_E540_AUTHORIZATION_FILENAME"):
            raise ValueError("V10 authorization receipt name differs from the approved packet")
        validate_authorization(auth_path, expected_auth)
        cli_checks = validate_prediction_clis()
        write_json(
            RUNTIME / "phase.json",
            {"phase": "authorization_and_actual_cli_preflight_pass", "cli_checks": cli_checks},
        )
        _validate_upstream(binding, recipe)
        expected_binding_sha = os.environ.get("WNN_E540_ARTIFACT_BINDING_SHA256")
        if binding_sha != expected_binding_sha:
            raise ValueError("V10 artifact binding differs from the frozen packet hash")
        result.update(
            {
                "status": "source_identity_verification",
                "worker_timeout_seconds": worker_seconds,
                "artifact_binding_sha256": binding_sha,
                "recipe_sha256": recipe_sha,
                "subset_manifest_sha256": subset_sha,
            }
        )
        write_json(RUNTIME / "result.json", result)
        artifact_observations = _verify_asset_files(binding)
        result["asset_observations"] = artifact_observations
        result["status"] = "valid_input_identity"
        write_json(RUNTIME / "result.json", result)
        valid_identity = _verify_valid_input(binding)
        result["valid_input_identity"] = valid_identity
        split = json.loads((ROOT / SPLIT_PATH).read_text(encoding="utf-8"))
        subset = json.loads((ROOT / SUBSET_PATH).read_text(encoding="utf-8"))
        subset["_sha256"] = subset_sha
        if (
            subset.get("selected_component_count") != 512
            or subset.get("evaluation_group_count") != 468
            or len(subset.get("evaluation_groups", [])) != 468
            or subset.get("split_manifest_sha256") != subset_sha_of_split(split, ROOT / SPLIT_PATH)
        ):
            raise ValueError("V10 frozen E540 subset or split identity is invalid")
        clock_contract = recipe.get("producer_clock_validation")
        if (
            not isinstance(clock_contract, Mapping)
            or clock_contract.get("history_rounding_tolerance_seconds")
            != HISTORY_ROUNDING_TOLERANCE_SECONDS
            or clock_contract.get("maximum_checkpoint_write_allowance_seconds")
            != CHECKPOINT_WRITE_ALLOWANCE_SECONDS
        ):
            raise ValueError("V10 producer-clock contract is missing")
        checkpoint_write_allowance = clock_contract.get(
            "maximum_checkpoint_write_allowance_seconds"
        )
        if (
            isinstance(checkpoint_write_allowance, bool)
            or not isinstance(checkpoint_write_allowance, (int, float))
            or not math.isfinite(float(checkpoint_write_allowance))
            or float(checkpoint_write_allowance) < 0
        ):
            raise ValueError("V10 checkpoint-write allowance is invalid")
        result["membership"] = {
            "selected_component_count": 512,
            "selected_group_counts": {
                key: len(value) for key, value in subset["selected_groups"].items()
            },
            "evaluation_group_count": 468,
            "evaluation_membership_sha256": subset["evaluation_membership_sha256"],
        }
        _validate_training_sources(
            binding,
            {
                "linear": json.loads(_asset_path(binding, "linear_provenance").read_text()),
                "ple": json.loads(_asset_path(binding, "ple_provenance").read_text()),
            },
        )
        import torch

        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("E540 V10 requires exactly one visible CUDA device")
        device = torch.cuda.get_device_name(0)
        if "T4" not in device:
            raise RuntimeError(f"E540 V10 requires the reviewed Kaggle T4; observed {device}")
        hardware = {
            "device_name": device,
            "device_capability": list(torch.cuda.get_device_capability(0)),
            "device_total_bytes": torch.cuda.get_device_properties(0).total_memory,
            "torch": str(torch.__version__),
            "cuda": torch.version.cuda,
        }
        write_json(RUNTIME / "hardware.json", hardware)
        result["hardware"] = hardware
        result["status"] = "checkpoint_and_fit_metadata_validation"
        write_json(RUNTIME / "result.json", result)
        checkpoint_receipts = _validate_checkpoints(
            binding,
            subset,
            artifact_observations,
            float(checkpoint_write_allowance),
        )
        result["checkpoint_receipts"] = checkpoint_receipts
        result["status"] = "prediction_running"
        write_json(RUNTIME / "result.json", result)

        import scripts.predict_run as predict_run

        for mode in ("linear", "ple"):
            if time.monotonic() - started >= worker_seconds:
                raise TimeoutError("V10 worker deadline reached before both predictions completed")
            arm = result["prediction_arms"][mode]
            arm["status"] = "prediction_running"
            arm["argv"] = prediction_argv(mode)
            write_json(RUNTIME / "result.json", result)
            log_path = RUNTIME / f"{mode}-prediction.log"
            call_predictor(predict_run, prediction_argv(mode), log_path)
            arm.update(validate_prediction(mode, subset))
            arm["log"] = log_path.relative_to(ROOT).as_posix()
            result["status"] = "prediction_running" if mode == "linear" else "complete_ungraded"
            write_json(RUNTIME / "result.json", result)

        result.update(
            {
                "status": "complete_ungraded",
                "fit_started": False,
                "profile_started": False,
                "grade_started": False,
                "elapsed_worker_seconds": round(time.monotonic() - started, 3),
                "prediction_membership": "exact frozen 468 groups; no target or mask columns requested",
                "fit_status": "both source fits completed earlier; this worker performed no fitting",
            }
        )
        write_json(RUNTIME / "result.json", result)
    except BaseException as error:
        result.update(
            {
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
                "elapsed_worker_seconds": round(time.monotonic() - started, 3),
            }
        )
        write_json(RUNTIME / "result.json", result)
        raise


def subset_sha_of_split(split: Mapping[str, object], path: Path) -> str:
    """Return the bound source split SHA after checking its local structural record."""
    declared = split.get("sha256")
    observed = sha256_file(path)
    if declared is not None and declared != observed:
        raise ValueError("local split manifest self-identity is invalid")
    return observed


if __name__ == "__main__":
    main()
