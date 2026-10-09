"""FP16 autocast successor of the reviewed D01 GPU worker.

The reviewed R4 worker owns cache validation, sampling, schedule construction,
training, prediction export, and resource monitoring. This entry point changes
only the declared compute precision, binds all shared controls to completed R5,
and adds a real-CUDA reset/reload/prefix gate before a conditional fit.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any

from native_gpu_fp16_contract import (
    AMP_INITIAL_SCALE,
    PROFILE_UPDATE_COUNT,
    R5_CONTRACT_SHA256,
    R5_REVIEW_SHA256,
    ContractError,
    compare_npz_arrays,
    create_native_trainer_overlay,
    native_prediction_argv,
    native_trainer_argv,
    sha256_file,
    summarize_update_trace,
    validate_fit_headroom,
    validate_native_profile_evidence,
    verify_fp32_master_checkpoint,
    verify_r5_control_tree,
)

WORK = Path(os.environ.get("KAGGLE_WORKING", "/kaggle/working"))
OUT = WORK / "out"


class FitNotAuthorized(RuntimeError):
    """The profile passed, but the embedded root authority permits profile only."""

    def __init__(self, message: str, state: str = "profile_passed_fit_not_authorized") -> None:
        super().__init__(message)
        self.state = state


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ContractError(f"expected JSON object: {path}")
    return value


def _validate_admission(bootstrap: dict[str, Any]) -> tuple[dict[str, Any], Path, Path]:
    admission_path = Path(str(bootstrap.get("native_root_admission_path", "")))
    controls_root = Path(str(bootstrap.get("native_controls_root", "")))
    if not admission_path.is_file() or not controls_root.is_dir():
        raise ContractError("bootstrap omitted the native root authority or R5 controls")
    admission = _read_json(admission_path)
    if (
        admission.get("schema") != "wnn.native-gpu-fp16.root-admission.v1"
        or admission.get("state") != "admitted"
        or admission.get("push_allowed") is not True
        or admission.get("profile_allowed") is not True
        or admission.get("r5_training_contract_sha256") != R5_CONTRACT_SHA256
        or admission.get("r5_root_review_sha256") != R5_REVIEW_SHA256
        or admission.get("amp_initial_scale") != AMP_INITIAL_SCALE
        or admission.get("compute_precision") != "fp16"
    ):
        raise ContractError("root admission does not bind this FP16 source and exact R5 controls")
    source_manifest_sha = bootstrap.get("native_source_manifest_sha256")
    if (
        not isinstance(source_manifest_sha, str)
        or len(source_manifest_sha) != 64
        or admission.get("source_manifest_sha256") != source_manifest_sha
    ):
        raise ContractError("root admission does not bind the extracted native source closure")
    if not isinstance(admission.get("conditional_fit_allowed"), bool):
        raise ContractError("root admission must state conditional_fit_allowed explicitly")
    verify_r5_control_tree(controls_root)
    return admission, admission_path, controls_root


def _load_base_worker() -> Any:
    repo = Path(__file__).resolve().parents[1]
    script = repo / "scripts" / "loss_pair_gpu_worker.py"
    spec = importlib.util.spec_from_file_location("wnn_reviewed_r4_gpu_worker", script)
    if spec is None or spec.loader is None:
        raise ContractError("cannot load the pinned R4 GPU worker")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _same_json(left: Path, right: Path, role: str) -> None:
    if _read_json(left) != _read_json(right):
        raise ContractError(f"native producer differs from R5 control JSON: {role}")


def _verify_prefit_against_r5(base: Any, controls_root: Path) -> None:
    producer_shared = controls_root / "shared"
    consumer_prefit = base.OUT / "profile_gate" / "prefit"
    for filename, role in (
        ("initial_state.npz", "initial_state"),
        ("normalization.npz", "normalization"),
        ("profile_inputs.npz", "profile_inputs"),
    ):
        compare_npz_arrays(
            producer_shared / filename,
            consumer_prefit / filename,
            role,
        )
    _same_json(
        producer_shared / "sample_orders.json",
        consumer_prefit / "sample_orders.json",
        "sample_orders",
    )


def _profile_headroom(base: Any, deadline: float) -> dict[str, Any]:
    sample = base._sample("native_fp16_conditional_fit_headroom")
    reason = base._guard(sample, deadline)
    if reason:
        raise ContractError(f"conditional-fit host/disk guard failed: {reason}")
    memory = sample.get("memory", {})
    if memory.get("cgroup_max_bytes") is None:
        directory = base._cgroup_directory()
        if directory is None:
            raise ContractError("conditional fit cannot verify the cgroup memory limit")
        for filename in ("memory.max", "memory.limit_in_bytes"):
            limit_path = directory / filename
            if not limit_path.is_file():
                continue
            raw = limit_path.read_text(encoding="utf-8").strip()
            if raw == "max":
                memory["cgroup_max_unbounded"] = True
                break
            value = int(raw)
            if filename == "memory.limit_in_bytes" and value >= (1 << 60):
                memory["cgroup_max_unbounded"] = True
            else:
                memory["cgroup_max_bytes"] = value
            break
    evidence = validate_fit_headroom(sample, required_gpu_indices={0, 1})
    return {"sample": sample, **evidence}


def _patch_worker(
    base: Any,
    admission: dict[str, Any],
    controls_root: Path,
    admission_path: Path,
) -> None:
    original_train_argv = base._common_train_argv
    original_export_predictions = base._export_predictions
    original_gate = base._run_device_shape_update_gate
    original_collect = base._collect_shared
    original_prefit_verify = base._verify_prefit_pair
    trainer_source = base.ROOT / "scripts" / "train_rnn.py"
    trainer_overlay = base.ROOT / "scripts" / "native_train_rnn_fp16.py"
    trainer_overlay_receipt = create_native_trainer_overlay(trainer_source, trainer_overlay)
    trainer_overlay_receipt_path = base.ROOT / "native_controls" / "trainer_precision_overlay.json"
    base._write_json(trainer_overlay_receipt_path, trainer_overlay_receipt)

    def native_train_argv_wrapper(*args: Any, **kwargs: Any) -> list[str]:
        return native_trainer_argv(
            original_train_argv(*args, **kwargs), trainer_source, trainer_overlay
        )

    def native_export_predictions(*args: Any, **kwargs: Any) -> dict[str, Any]:
        prediction = original_export_predictions(*args, **kwargs)
        prediction["command"] = native_prediction_argv(prediction["command"])
        return prediction

    def native_profile_gate(*args: Any, **kwargs: Any) -> dict[str, Any]:
        deadline = float(args[4] if len(args) > 4 else kwargs["deadline"])
        try:
            receipt = original_gate(*args, **kwargs)
            validate_native_profile_evidence(receipt["profile"]["evidence"])
        except BaseException as error:
            if trainer_overlay_receipt_path.is_file():
                base._copy_required(
                    trainer_overlay_receipt_path,
                    base.OUT / "shared" / "trainer_precision_overlay.json",
                )
            log_path = base.OUT / "logs" / "device_shape_update_gate.log"
            evidence_rows = []
            if log_path.is_file():
                for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
                    marker = "PROFILE_UPDATE_EVIDENCE "
                    if marker in line:
                        try:
                            evidence_rows.append(json.loads(line.partition(marker)[2]))
                        except json.JSONDecodeError:
                            continue
            evidence = evidence_rows[-1] if len(evidence_rows) == 1 else None
            failure_receipt = {
                "schema": "wnn.loss-pair.device-shape-update-gate.v1",
                "state": "failed",
                "profile": {
                    "precision": "float16",
                    "amp_initial_scale": AMP_INITIAL_SCALE,
                    "attempted_updates": (evidence.get("optimizer_attempts") if evidence else None),
                    "completed_updates": (
                        evidence.get("optimizer_completed_updates") if evidence else None
                    ),
                    "skipped_updates": evidence.get("amp_skipped_updates") if evidence else None,
                    "skips_count_as_throughput": False,
                    "evidence": evidence,
                },
                "failure": {"type": type(error).__name__, "message": str(error)},
                "profile_log_sha256": sha256_file(log_path) if log_path.is_file() else None,
            }
            base._write_json(base.OUT / "shared" / "device_profile_gate.json", failure_receipt)
            base._append_phase(
                "native_fp16_profile_failed",
                evidence=evidence,
                error_type=type(error).__name__,
                error=str(error),
            )
            raise
        _verify_prefit_against_r5(base, controls_root)
        lifecycle_out = base.OUT / "shared" / "native_fp16_lifecycle_gate.json"
        lifecycle_log = base.OUT / "logs" / "native_fp16_lifecycle_gate.log"
        command = [
            sys.executable,
            str(base.ROOT / "scripts" / "native_gpu_fp16_lifecycle_gate.py"),
            "--initial-state",
            str(base.OUT / "profile_gate" / "prefit" / "initial_state.npz"),
            "--profile-inputs",
            str(base.OUT / "profile_gate" / "prefit" / "profile_inputs.npz"),
            "--output",
            str(lifecycle_out),
        ]
        lifecycle = base._wait_one(
            command,
            "native_fp16_lifecycle_gate",
            lifecycle_log,
            deadline,
            base._environment(gpu="0"),
            poll_seconds=1.0,
        )
        if lifecycle.get("returncode") != 0:
            raise ContractError(f"actual CUDA FP16 lifecycle gate failed: {lifecycle}")
        lifecycle_receipt = _read_json(lifecycle_out)
        if lifecycle_receipt.get("state") != "passed":
            raise ContractError("actual CUDA lifecycle receipt did not pass")
        try:
            headroom = {"state": "passed", **_profile_headroom(base, deadline)}
        except ContractError as error:
            headroom = {"state": "deferred", "reason": str(error)}
        receipt["native_precision"] = {
            "compute_precision": "fp16 autocast",
            "master_parameter_dtype": "float32",
            "optimizer_state_dtype": "float32",
            "grad_scaler_enabled": True,
            "amp_initial_scale": AMP_INITIAL_SCALE,
            "profile_attempts": PROFILE_UPDATE_COUNT,
            "profile_completed_updates": receipt["profile"]["evidence"][
                "optimizer_completed_updates"
            ],
            "profile_skipped_updates": receipt["profile"]["evidence"]["amp_skipped_updates"],
            "profile_skips_count_as_throughput": False,
        }
        receipt["profile"]["precision"] = "float16"
        receipt["lifecycle_gate"] = lifecycle_receipt
        receipt["conditional_fit_headroom"] = headroom
        base._copy_required(
            trainer_overlay_receipt_path,
            base.OUT / "shared" / "trainer_precision_overlay.json",
        )
        base._write_json(base.OUT / "shared" / "device_profile_gate.json", receipt)
        base._append_phase(
            "native_fp16_profile_pass",
            profile_completed_updates=receipt["native_precision"]["profile_completed_updates"],
            profile_skipped_updates=receipt["native_precision"]["profile_skipped_updates"],
            lifecycle_state=lifecycle_receipt.get("state"),
            cgroup_headroom_bytes=headroom.get("cgroup_headroom_bytes"),
        )
        if admission.get("conditional_fit_allowed") is not True:
            raise FitNotAuthorized("root admission permits the standalone profile only")
        if admission.get("training_contract_sha256") != R5_CONTRACT_SHA256:
            raise ContractError("conditional fit authority is bound to another R5 contract")
        if headroom.get("state") != "passed":
            raise FitNotAuthorized(
                str(headroom.get("reason", "conditional-fit headroom was not verified")),
                state="profile_passed_fit_deferred_headroom",
            )
        return receipt

    def native_prefit_verify(arms: dict[str, Path]) -> None:
        original_prefit_verify(arms)
        producer_shared = controls_root / "shared"
        for arm_dir in arms.values():
            for filename, role in (
                ("initial_state.npz", "initial_state"),
                ("normalization.npz", "normalization"),
            ):
                compare_npz_arrays(
                    producer_shared / filename,
                    arm_dir / "prefit" / filename,
                    role,
                )
            _same_json(
                producer_shared / "sample_orders.json",
                arm_dir / "prefit" / "sample_orders.json",
                "sample_orders",
            )
            _same_json(
                producer_shared / "schedule_trace.json",
                arm_dir / "schedule_trace.json",
                "OneCycle trace",
            )

    def native_collect(*args: Any, **kwargs: Any):
        contract, profile_receipt = original_collect(*args, **kwargs)
        for arm in ("wmse", "huber_d1"):
            arm_dir = base.OUT / "arms" / arm
            summary = summarize_update_trace(arm_dir / "update_trace.jsonl", expected_attempts=220)
            if not summary["all_attempts_completed"]:
                raise ContractError(f"{arm} did not complete all 220 schedule updates: {summary}")
            precision = verify_fp32_master_checkpoint(
                arm_dir / "model" / "model.pt",
                arm_dir / "model" / "training_state.pt",
            )
            contract["arms"][arm]["amp_update_counts"] = summary
            contract["arms"][arm]["precision_state"] = precision
        contract["model"]["precision"] = "fp16_autocast"
        contract["model"]["master_parameter_dtype"] = "float32"
        contract["model"]["optimizer_accumulator_dtype"] = "float32"
        contract["fit"]["amp"] = {
            "initial_scale": AMP_INITIAL_SCALE,
            "skip_policy": "all 220 schedule attempts must complete; any skipped update invalidates the fit artifact",
            "throughput_counts_only_completed_updates": True,
        }
        contract["native_control_binding"] = {
            "producer_training_contract_sha256": R5_CONTRACT_SHA256,
            "producer_review_sha256": R5_REVIEW_SHA256,
            "trainer_precision_overlay_sha256": sha256_file(
                base.OUT / "shared" / "trainer_precision_overlay.json"
            ),
            "producer_roles_sha256": {
                role: row["sha256"]
                for role, row in verify_r5_control_tree(controls_root)["artifacts"].items()
                if role
                in {
                    "cache_manifest",
                    "cache_reader_profile",
                    "cache_validation_receipt",
                    "device_profile_gate",
                    "fold0_asset_validation_receipt",
                    "initial_state",
                    "normalization",
                    "normalization_metadata",
                    "onecycle_trace",
                    "profile_inputs",
                    "profile_inputs_receipt",
                    "sample_orders",
                    "source_stage_receipt",
                    "source_verification",
                }
            },
        }
        contract["artifacts"]["native_fp16_lifecycle_gate"] = base._artifact(
            base.OUT / "shared" / "native_fp16_lifecycle_gate.json", base.OUT
        )
        contract["artifacts"]["trainer_precision_overlay"] = base._artifact(
            base.OUT / "shared" / "trainer_precision_overlay.json", base.OUT
        )
        contract["native_admission"] = {
            "source_manifest_sha256": admission["source_manifest_sha256"],
            "root_admission_sha256": sha256_file(admission_path),
            "conditional_fit_allowed": True,
        }
        base._write_json(base.OUT / "training_contract.json", contract)
        return contract, profile_receipt

    base._common_train_argv = native_train_argv_wrapper
    base._export_predictions = native_export_predictions
    base._run_device_shape_update_gate = native_profile_gate
    base._verify_prefit_pair = native_prefit_verify
    base._collect_shared = native_collect


def main() -> int:
    bootstrap_path = OUT / "bootstrap_receipt.json"
    status_path = WORK / "final_status.json"
    try:
        bootstrap = _read_json(bootstrap_path)
        admission, admission_path, controls_root = _validate_admission(bootstrap)
        base = _load_base_worker()
        _patch_worker(base, admission, controls_root, admission_path)
        result = base.run_worker(bootstrap)
        state = result["state"]
        base._write_json(status_path, {"state": state})
        return 0 if state == "complete" else 2
    except FitNotAuthorized as error:
        value = {
            "schema": "wnn.native-gpu-fp16-profile-only.v1",
            "state": error.state,
            "reason": str(error),
            "profile_receipt_sha256": sha256_file(OUT / "shared" / "device_profile_gate.json"),
            "root_admission_sha256": sha256_file(admission_path),
        }
        _write_status(status_path, value)
        return 0
    except BaseException as error:
        failure = {
            "state": "failed",
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }
        try:
            _write_status(WORK / "failure.json", failure)
            _write_status(status_path, failure)
        except OSError:
            pass
        print(json.dumps(failure, sort_keys=True), file=sys.stderr, flush=True)
        return 2


def _write_status(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    with temporary.open("xb") as stream:
        stream.write((json.dumps(value, sort_keys=True, indent=2) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


if __name__ == "__main__":
    raise SystemExit(main())
