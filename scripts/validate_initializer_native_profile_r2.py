"""Check the actual R3 initializer profile payloads through the native predictor."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import resource
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

EXPECTED_PROFILE_CONTRACT_SHA256 = (
    "08b943d63913e5ceebd845f1419feed65af36b375589304ed85fb50ed03fa3e7"
)
EXPECTED_PROFILE_RESULT_SHA256 = "42e8ad8aa838f0570d2a26ba2dabb6b8d93fb00f5c321a6c4d6f28b327e74711"
EXPECTED_PROFILE_SOURCE_SHA256 = "b2e01a312b5f7e7a785c5bd92ad3d106e6758e78432ebac0528444cad2c140c9"
EXPECTED_PROFILE_ATTEMPT_ID = "student-initializer-profile-f0s3-h256l2-b1-c2000-fp32-r3"
EXPECTED_PROFILE_ATTEMPT_SHA256 = "97d69a1da3435f5cdbb749ddd3729171143465180320fa4e6a6035a7c3548a52"
EXPECTED_PROFILE_ADMISSION_SHA256 = (
    "ae54dea5e8cdb728db29932897e58f730205036ade82831928955df826e75dc6"
)
EXPECTED_PROFILE_COMMIT = "2800d26d200c70d858a8bbe50d9fdabd90190d5f"
EXPECTED_CONTROL_SHA256 = "1baa583573089da3eec02bee9256c32e3caeae59e15f77c7475eea0717f006e6"
EXPECTED_MODES = ("constant", "first_row", "prefix")
PREDICTION_RTOL = 3e-4
PREDICTION_ATOL = 3e-5
MAX_JSON_BYTES = 8_000_000
MAX_CHECKPOINT_BYTES = 20_000_000
MAX_NATIVE_ARCHIVE_BYTES = 1_000_000
CUDA_ALLOCATOR_CAP_BYTES = 3_500_000_000


def sha256_file(path: Path) -> str:
    """Hash a file in bounded chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def bounded_json(path: Path) -> tuple[dict[str, Any], bytes]:
    """Read one small JSON object after checking its actual byte bound."""
    size = path.stat().st_size
    if size <= 0 or size > MAX_JSON_BYTES:
        raise ValueError(f"JSON input exceeds the {MAX_JSON_BYTES}-byte bound: {path}")
    raw = path.read_bytes()
    if len(raw) != size:
        raise ValueError(f"JSON input changed while being read: {path}")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value, raw


def validate_controller_receipts(
    attempt_path: Path,
    admission_path: Path,
    contract_path: Path,
    result_path: Path,
) -> tuple[str, str, dict[str, Any]]:
    """Require the exact root admission and zero-OOM terminal R3 controller record."""
    attempt, attempt_raw = bounded_json(attempt_path)
    admission, admission_raw = bounded_json(admission_path)
    attempt_sha = hashlib.sha256(attempt_raw).hexdigest()
    admission_sha = hashlib.sha256(admission_raw).hexdigest()
    if attempt_sha != EXPECTED_PROFILE_ATTEMPT_SHA256:
        raise ValueError("controller terminal receipt differs from the sealed R3 attempt")
    if admission_sha != EXPECTED_PROFILE_ADMISSION_SHA256:
        raise ValueError("root admission receipt differs from the sealed R3 admission")
    if (
        attempt.get("id") != EXPECTED_PROFILE_ATTEMPT_ID
        or attempt.get("kind") != "gpu"
        or attempt.get("status") != "finished"
        or attempt.get("phase") != "terminal"
        or attempt.get("exit_status") != 0
        or attempt.get("contract_sha256") != EXPECTED_PROFILE_CONTRACT_SHA256
    ):
        raise ValueError("R3 controller attempt is not a successful terminal profile")
    if admission.get("returncode") != 0:
        raise ValueError("R3 profile has no successful root admission receipt")
    attempt_command = attempt.get("command")
    admission_argv = admission.get("argv")
    if not isinstance(attempt_command, list) or not isinstance(admission_argv, list):
        raise ValueError("R3 controller receipts lack their exact argv records")
    if (
        str(contract_path) not in attempt_command
        or str(result_path) not in attempt_command
        or "scripts/profile_initializer_frontier_r3.py" not in attempt_command
        or "--execute" not in attempt_command
    ):
        raise ValueError("R3 controller argv differs from the frozen profile contract/result")
    properties = attempt.get("attempt_properties_at_start", {})
    reservation = attempt.get("reservation", {})
    if (
        properties.get("CPUQuotaPerSecUSec") != "1s"
        or properties.get("MemoryMax") != "5000000000"
        or properties.get("MemorySwapMax") != "0"
        or reservation.get("cpus") != 1
        or reservation.get("ram_bytes") != 5_000_000_000
    ):
        raise ValueError("R3 profile controller caps differ from the reviewed admission")
    if (
        not isinstance(attempt.get("peak_ram_bytes"), int)
        or attempt["peak_ram_bytes"] > 5_000_000_000
        or not isinstance(attempt.get("peak_observed_vram_bytes"), int)
        or attempt["peak_observed_vram_bytes"] >= 5_000_000_000
    ):
        raise ValueError("R3 terminal host/observed-VRAM peak is missing or over cap")
    for field in ("memory_event_deltas", "parent_hierarchical_memory_event_deltas"):
        events = attempt.get(field)
        if not isinstance(events, dict) or any(
            events.get(name) != 0 for name in ("oom", "oom_kill", "oom_group_kill")
        ):
            raise ValueError(f"R3 controller has a nonzero or missing {field} OOM delta")
    observers = attempt.get("observer_status", {})
    required_observers = (
        "attempt_memory_events_final",
        "attempt_memory_peak_final",
        "attempt_systemd_runtime",
        "nvidia_smi_runtime",
        "nvidia_smi_final",
        "parent_hierarchical_memory_events_final",
    )
    if any(observers.get(name) != "ok" for name in required_observers):
        raise ValueError("R3 controller terminal observers are incomplete")
    expected_admission_prefix = [
        "/home/vlad/development/wunder/wnn_connectome_starterpack/.venv/bin/python",
        "/home/vlad/development/wunder/wnn_connectome_starterpack/scripts/bandit_resources.py",
        "--root",
        "/home/vlad/development/wunder/worktrees/student-initializer-profile-r3-20261015",
        "--state",
        "/home/vlad/development/wunder/wnn_connectome_starterpack/research/campaign_2026-10-06/bandit_continuation/runtime",
        "launch",
        "--attempt",
        EXPECTED_PROFILE_ATTEMPT_ID,
    ]
    if admission_argv[: len(expected_admission_prefix)] != expected_admission_prefix:
        raise ValueError("root admission does not target the frozen R3 profile attempt")
    return attempt_sha, admission_sha, attempt


def load_pinned_module(name: str, path: Path) -> Any:
    """Load a source-pinned Python module without relying on a mutable import path."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load module source: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def state_initializer_checkpoint(
    torch_module: Any,
    checkpoint_path: Path,
    backbone: Any,
    mode: str,
    residual_bound: float,
    normalization_hashes: dict[str, str],
) -> Any:
    """Rebuild one profiled initializer around the exact frozen control backbone."""
    from wnn.state_initialization import InitializedGRU

    size = checkpoint_path.stat().st_size
    if size <= 0 or size > MAX_CHECKPOINT_BYTES:
        raise ValueError("initializer profile checkpoint exceeds its 20 MB bound")
    saved = torch_module.load(checkpoint_path, map_location="cpu", weights_only=False)
    if (
        saved.get("format") != "wnn.initializer-only-profile-checkpoint"
        or saved.get("version") != 1
        or saved.get("mode") != mode
        or saved.get("backbone_checkpoint_sha256") != EXPECTED_CONTROL_SHA256
        or saved.get("residual_bound") != residual_bound
        or saved.get("normalization_sha256") != normalization_hashes
    ):
        raise ValueError(f"{mode} profile checkpoint is not bound to the fixed H256x2 CP0")
    model = InitializedGRU(backbone, mode, residual_bound=residual_bound).eval()
    state = model.state_dict()
    learned = saved.get("initializer_state_dict")
    if not isinstance(learned, dict) or not learned:
        raise ValueError(f"{mode} profile checkpoint lacks learned initializer parameters")
    for key, value in learned.items():
        if key not in state or key.startswith("backbone."):
            raise ValueError(f"{mode} checkpoint contains an unsupported initializer key")
        state[key] = value
    model.load_state_dict(state, strict=True)
    if not all(torch_module.isfinite(value).all() for value in model.state_dict().values()):
        raise FloatingPointError(f"{mode} profile checkpoint has a nonfinite tensor")
    return model


def _run_native(native_model: Any, raw: np.ndarray, seq_ix: int) -> np.ndarray:
    """Run one raw-feature sequence through the stateful native callback."""
    outputs = []
    for row in raw:
        value = native_model.predict(
            SimpleNamespace(seq_ix=seq_ix, state=row, need_prediction=True)
        )
        if value is None:
            raise RuntimeError("native predictor omitted a requested sequence row")
        outputs.append(np.asarray(value, dtype=np.float32))
    return np.stack(outputs)


def validate_mode(
    mode: str,
    arm: dict[str, Any],
    contract: dict[str, Any],
    profile_result_root: Path,
    base_state: dict[str, Any],
    raw: np.ndarray,
    sequence_ix: int,
    trainer: Any,
    native: Any,
) -> dict[str, Any]:
    """Compare an actual one-update initializer export with the native C++ callback."""
    if arm.get("mode") != mode:
        raise ValueError(f"profile arm order/name mismatch for {mode}")
    if (
        arm.get("updates_attempted") != 1
        or arm.get("updates_completed") != 1
        or arm.get("loss_finite") is not True
        or arm.get("nonzero_initializer_gradient") is not True
        or arm.get("parameter_update_observed") is not True
        or arm.get("backbone_unchanged") is not True
        or arm.get("reset_after_prior_prefix_verified") is not True
        or not isinstance(arm.get("cuda_peak_reserved_bytes"), int)
        or arm["cuda_peak_reserved_bytes"] > CUDA_ALLOCATOR_CAP_BYTES
        or not isinstance(arm.get("observed_vram_bytes_after_arm"), int)
        or arm["observed_vram_bytes_after_arm"] >= 5_000_000_000
    ):
        raise ValueError(f"{mode} actual profile does not prove its finite initializer update")

    checkpoint_record = arm.get("checkpoint_reload")
    native_record = arm.get("native_export")
    if not isinstance(checkpoint_record, dict) or not isinstance(native_record, dict):
        raise ValueError(f"{mode} profile lacks a checkpoint or native export record")
    checkpoint_name = checkpoint_record.get("path")
    archive_name = native_record.get("archive")
    if (
        not isinstance(checkpoint_name, str)
        or Path(checkpoint_name).name != checkpoint_name
        or not isinstance(archive_name, str)
        or Path(archive_name).name != archive_name
    ):
        raise ValueError(f"{mode} profile output names are unsafe")
    checkpoint_path = profile_result_root / checkpoint_name
    archive_path = profile_result_root / archive_name
    if archive_path.stat().st_size > MAX_NATIVE_ARCHIVE_BYTES:
        raise ValueError(f"{mode} native archive exceeds its 1 MB bound")
    if sha256_file(checkpoint_path) != checkpoint_record.get("sha256") or sha256_file(
        archive_path
    ) != native_record.get("archive_sha256"):
        raise ValueError(f"{mode} checkpoint or export archive differs from the profile result")

    args = {
        "arch": "gru",
        "cell": "gru",
        "enc": "linear",
        "hidden": 256,
        "layers": 2,
        "dropout": 0.0,
        "readout_activation": "gelu",
        "numeric_map": "none",
        "ple_intervals": 8,
    }
    backbone = trainer.make_model(args, 211, 2)
    backbone.load_state_dict(base_state, strict=True)
    model = state_initializer_checkpoint(
        torch,
        checkpoint_path,
        backbone,
        mode,
        float(contract["residual_bound"]),
        contract["normalization"]["hashes"],
    )
    input_tensor = trainer.build_inputs(torch.from_numpy(raw)[None])
    if tuple(input_tensor.shape) != (1, 20_000, 211):
        raise ValueError("actual profile Parquet sequence did not produce 211 causal features")
    with torch.inference_mode():
        python_prediction, python_state = model(input_tensor, None)
    expected = python_prediction[0].cpu().numpy().astype(np.float32, copy=False)
    from wnn.state_initialization import export_native_initializer

    config = export_native_initializer(model)
    with np.load(archive_path, allow_pickle=False) as arrays:
        if set(arrays.files) != {
            key for key, value in config.items() if isinstance(value, np.ndarray)
        }:
            raise ValueError(f"{mode} archive fields differ from the actual Python export")
        for key in arrays.files:
            value = np.asarray(arrays[key])
            if (
                value.dtype != np.float32
                or not value.flags.c_contiguous
                or not np.isfinite(value).all()
            ):
                raise ValueError(
                    f"{mode} native archive field is not finite contiguous FP32: {key}"
                )
            if not np.array_equal(value, config[key]):
                raise ValueError(f"{mode} native archive field differs from its checkpoint: {key}")

    weights = {
        name: np.ascontiguousarray(tensor.detach().cpu().numpy(), dtype=np.float32)
        for name, tensor in model.backbone.state_dict().items()
    }
    native_model = native.EnsemblePredictor(
        [weights],
        [1.0],
        fast_activations=False,
        precision="f32",
        state_initializers=[config],
    )
    started = time.perf_counter()
    actual = _run_native(native_model, raw, sequence_ix)
    replay = _run_native(native_model, raw, sequence_ix + 1)
    elapsed = time.perf_counter() - started
    np.testing.assert_allclose(
        actual,
        expected,
        rtol=PREDICTION_RTOL,
        atol=PREDICTION_ATOL,
        err_msg=f"actual native callback differs from Python for {mode}",
    )
    np.testing.assert_array_equal(replay, actual)
    hidden = native_model.hidden_states[0]
    if not np.isfinite(actual).all() or not np.isfinite(hidden).all():
        raise FloatingPointError(f"{mode} native result contains a nonfinite value")
    difference = actual - expected
    return {
        "mode": mode,
        "profile_checkpoint_sha256": checkpoint_record["sha256"],
        "profile_native_archive_sha256": native_record["archive_sha256"],
        "cpp_constructor": "wnn_native.EnsemblePredictor precision=f32 fast_activations=false state_initializers=[export_native_initializer]",
        "sequence_rows": int(len(raw)),
        "native_prediction_rows": int(len(actual)),
        "python_native_atol": PREDICTION_ATOL,
        "python_native_rtol": PREDICTION_RTOL,
        "max_abs_prediction_difference": float(np.abs(difference).max()),
        "rms_prediction_difference": float(np.sqrt(np.mean(difference.astype(np.float64) ** 2))),
        "reset_replay_exact": True,
        "native_hidden_finite": True,
        "elapsed_seconds_for_two_native_passes": elapsed,
        "quality_claim": None,
    }


def validate(args: argparse.Namespace) -> dict[str, Any]:
    """Validate R3 ancestry, controller terminal status and all three native payloads."""
    profile_contract, contract_raw = bounded_json(args.profile_contract)
    contract_sha = hashlib.sha256(contract_raw).hexdigest()
    if contract_sha != EXPECTED_PROFILE_CONTRACT_SHA256:
        raise ValueError("R3 initializer profile contract hash differs from the reviewed packet")
    profile_result, result_raw = bounded_json(args.profile_result)
    result_sha = hashlib.sha256(result_raw).hexdigest()
    if result_sha != EXPECTED_PROFILE_RESULT_SHA256:
        raise ValueError("profile result differs from the sealed R3 actual-data result")
    if profile_result.get("contract_sha256") != contract_sha:
        raise ValueError("R3 profile result is not bound to its reviewed contract")
    if (
        profile_result.get("source_sha256", {}).get("scripts/profile_initializer_frontier_r3.py")
        != EXPECTED_PROFILE_SOURCE_SHA256
    ):
        raise ValueError("R3 profile result does not identify the reviewed source")
    attempt_sha, admission_sha, attempt = validate_controller_receipts(
        args.attempt_receipt,
        args.admission_receipt,
        args.profile_contract,
        args.profile_result,
    )
    sample = profile_contract["sample"]
    if (
        sample.get("source") != "valid"
        or sample.get("row_group") != 37
        or sample.get("seq_ix") != 44_346_154
        or sample.get("sequence_rows") != 20_000
        or profile_result.get("sample") != sample
    ):
        raise ValueError("actual profile sequence is not the pinned source-qualified F0 member")
    if profile_result.get("quality_claim") is not None:
        raise ValueError("the resource profile must remain non-scientific-quality evidence")
    execution_device = profile_result.get("execution_device", {})
    if (
        profile_result.get("device_policy", {}).get("requested_device") != "cuda"
        or execution_device.get("resolved") != "cuda:0"
        or execution_device.get("index") != 0
    ):
        raise ValueError("R3 profile did not use the reviewed canonical CUDA device")
    profile_scope = profile_result.get("profile_scope", {})
    if (
        profile_scope.get("kind") != "resource_only"
        or profile_scope.get("arms") != ["constant", "first_row", "prefix"]
        or profile_scope.get("backbone_frozen") is not True
        or profile_scope.get("no_full_fit") is not True
        or profile_scope.get("no_quality_claim") is not True
    ):
        raise ValueError("R3 profile scope is not the frozen resource-only initializer smoke")
    resource_bounds = profile_contract["resource_envelope"]
    observed_vram = profile_result.get("observed_vram_bytes_at_completion")
    host_cap = profile_result.get("host_memory_max_bytes_at_start")
    if (
        not isinstance(observed_vram, int)
        or observed_vram >= resource_bounds["observed_vram_ceiling_bytes"]
    ):
        raise ValueError("actual profile did not finish below the 5 GB observed-VRAM ceiling")
    if (
        not isinstance(host_cap, int)
        or host_cap > resource_bounds["host_memory_max_bytes"]
        or resource_bounds["host_memory_max_bytes"] - host_cap >= 4096
    ):
        raise ValueError("actual profile host cgroup differs from its declared cap")
    host_peak = profile_result.get("host_memory_peak_bytes")
    if not isinstance(host_peak, int) or host_peak > resource_bounds["host_memory_max_bytes"]:
        raise ValueError("actual profile host peak is absent or exceeds its admitted cap")
    expected_cache_hashes = profile_contract["cache"]["target_files"]
    verified_cache_hashes = profile_result.get("strict_cache_verified_files")
    if verified_cache_hashes != expected_cache_hashes:
        raise ValueError("actual profile did not verify both full target payloads")
    if profile_result.get("cache_ancestry") != profile_contract["cache_ancestry"]:
        raise ValueError("actual profile cache ancestry differs from its pinned contract")

    profile_root = Path(profile_contract["source_root"])
    training_root = Path(profile_contract["training_source"]["source_root"])
    for relative, expected in profile_contract["source_sha256"].items():
        if sha256_file(profile_root / relative) != expected:
            raise ValueError(f"reviewed profile source changed: {relative}")
    for relative, expected in profile_contract["training_source"]["source_sha256"].items():
        if sha256_file(training_root / relative) != expected:
            raise ValueError(f"reviewed training source changed: {relative}")

    arms = profile_result.get("arms")
    if not isinstance(arms, list) or [item.get("mode") for item in arms] != list(EXPECTED_MODES):
        raise ValueError("profile result does not contain the exact three learned modes")
    control_path = Path(profile_contract["data_root"]) / profile_contract["control"]["checkpoint"]
    if sha256_file(control_path) != EXPECTED_CONTROL_SHA256:
        raise ValueError("frozen zero-state H256×2 control checkpoint changed")
    checkpoint_size = control_path.stat().st_size
    if checkpoint_size > 50_000_000:
        raise ValueError("frozen control checkpoint exceeds the 50 MB read bound")
    checkpoint = torch.load(control_path, map_location="cpu", weights_only=False)
    if checkpoint.get("epoch") != 0 or not isinstance(checkpoint.get("model"), dict):
        raise ValueError("frozen H256×2 parent is not its fixed epoch-0 checkpoint")
    base_state = {
        name: value.detach().to(device="cpu", dtype=torch.float32).contiguous()
        for name, value in checkpoint["model"].items()
    }
    for name in ("mean", "std"):
        if name not in base_state or not torch.isfinite(base_state[name]).all():
            raise FloatingPointError(
                f"frozen control normalization buffer is missing/nonfinite: {name}"
            )
        if profile_contract["normalization"]["hashes"][name] != (
            hashlib.sha256(
                base_state[name].numpy().astype("<f4", copy=False).tobytes(order="C")
            ).hexdigest()
        ):
            raise ValueError(f"frozen control {name} hash differs from profile contract")
    del checkpoint

    normalization_source = profile_contract["normalization"]["source"]
    normalization_run = (
        Path(profile_contract["data_root"]) / "artifacts/runs" / normalization_source["run"]
    )
    normalization_checkpoint = normalization_run / normalization_source["checkpoint_file"]
    normalization_provenance = normalization_run / "provenance.json"
    if (
        sha256_file(normalization_checkpoint) != normalization_source["checkpoint_sha256"]
        or sha256_file(normalization_provenance) != normalization_source["provenance_sha256"]
        or normalization_source.get("learned_weights_loaded") is not False
        or normalization_source.get("loaded_fields_only") != ["mean", "std"]
    ):
        raise ValueError("normalization source checkpoint/provenance identity changed")

    data_root = Path(profile_contract["data_root"])
    split_path = data_root / profile_contract["split_path"]
    if sha256_file(split_path) != profile_contract["split_sha256"]:
        raise ValueError("source-qualified F0 split changed after profile")
    sys.path.insert(0, str(training_root / "src"))
    sys.path.insert(0, str(profile_root / "src"))
    trainer = load_pinned_module("train_rnn", training_root / "scripts/train_rnn.py")
    from wnn.data import SequenceFile

    sequence = SequenceFile(data_root / "datasets/valid.parquet").read(
        int(sample["row_group"]), targets=False, mask=False
    )
    if sequence.seq_ix != sample["seq_ix"] or sequence.features.shape != (20_000, 112):
        raise ValueError("actual validation Parquet identity differs from the profile contract")
    native = __import__("wnn_native")
    result_root = args.profile_result.parent
    mode_results = [
        validate_mode(
            mode,
            arm,
            profile_contract,
            result_root,
            base_state,
            sequence.features,
            sequence.seq_ix,
            trainer,
            native,
        )
        for mode, arm in zip(EXPECTED_MODES, arms, strict=True)
    ]
    extension_path = Path(native.__file__).resolve()
    native_source_paths = {
        "validator": Path(__file__).resolve(),
        "profile_state_initialization": profile_root / "src/wnn/state_initialization.py",
        "trainer": training_root / "scripts/train_rnn.py",
        "model": training_root / "src/wnn/nn.py",
        "data": training_root / "src/wnn/data.py",
        "features": training_root / "src/wnn/features.py",
    }
    receipt = {
        "format": "wnn.initializer-native-profile-gate",
        "version": 2,
        "profile_attempt_id": EXPECTED_PROFILE_ATTEMPT_ID,
        "profile_attempt_receipt_sha256": attempt_sha,
        "profile_admission_receipt_sha256": admission_sha,
        "profile_attempt_finished_utc": attempt["finished_utc"],
        "profile_contract_sha256": contract_sha,
        "profile_result_sha256": result_sha,
        "profile_source_sha256": EXPECTED_PROFILE_SOURCE_SHA256,
        "profile_source_commit": EXPECTED_PROFILE_COMMIT,
        "control_checkpoint_sha256": EXPECTED_CONTROL_SHA256,
        "split_manifest_sha256": profile_contract["split_sha256"],
        "sequence": {
            "source": "valid",
            "row_group": int(sample["row_group"]),
            "seq_ix": int(sample["seq_ix"]),
            "rows": int(len(sequence.features)),
        },
        "native_extension": {"path": str(extension_path), "sha256": sha256_file(extension_path)},
        "source_sha256": {name: sha256_file(path) for name, path in native_source_paths.items()},
        "arms": mode_results,
        "result": "PASS",
        "quality_claim": None,
        "interpretation": "C++ native consumer/resource implementation gate only; no subset fit or quality evidence",
        "max_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024,
    }
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-contract", type=Path, required=True)
    parser.add_argument("--profile-result", type=Path, required=True)
    parser.add_argument("--attempt-receipt", type=Path, required=True)
    parser.add_argument("--admission-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--attempt-id", required=True)
    args = parser.parse_args()
    if args.attempt_id != EXPECTED_PROFILE_ATTEMPT_ID:
        parser.error("R2 requires the exact root-admitted R3 B1 actual-data profile attempt")
    if args.output.exists():
        raise FileExistsError(f"preserving existing native profile gate receipt: {args.output}")
    started = time.perf_counter()
    result = validate(args)
    result["attempt_id"] = args.attempt_id
    result["elapsed_seconds"] = time.perf_counter() - started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    with args.output.open("xb") as output_file:
        output_file.write(raw)
    print(json.dumps({"result": result["result"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
