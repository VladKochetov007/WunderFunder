from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import json
import math
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import torch

PACKET_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = Path("/home/vlad/development/wunder/wnn_connectome_starterpack")
CONTRACT_PATH = PACKET_ROOT / "HFT_EXPORT_CONTRACT_R3.json"
R2_PACKET = PROJECT_ROOT / (
    "research/campaign_2026-10-06/bandit_continuation/"
    "diversity_stack_20261009/ops_continuation_20261011/oldz4_f1_core_hft_r2"
)
R2_RUNNER_PATH = R2_PACKET / "run_f1_core_hft_stage_r2.py"
R3_RESOURCE_PACKET = PROJECT_ROOT / (
    "research/campaign_2026-10-06/bandit_continuation/"
    "diversity_stack_20261009/ops_continuation_20261011/oldz4_f1_hft_resource_fit_r3"
)
TERMINAL_GATE_PATH = PACKET_ROOT / "ROOT_F1_HFT_TERMINAL_ADMISSION_R3.json"
EXPORT_GATE_PATH = PACKET_ROOT / "ROOT_F1_HFT_EXPORT_ADMISSION_R3.json"


class _DeviceBoundary(RuntimeError):
    """Mark the exact point after predictor parsing and before CUDA setup."""


def sha256_file(path: Path) -> str:
    """Hash a file in bounded chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_object(path: Path) -> dict[str, Any]:
    """Load a JSON object and reject unexpected top-level types."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def require(condition: bool, message: str) -> None:
    """Raise a precise error when a release condition is not met."""
    if not condition:
        raise ValueError(message)


def load_contract() -> dict[str, Any]:
    """Load the immutable R3 export-only contract."""
    contract = read_object(CONTRACT_PATH)
    require(
        contract.get("schema") == "wnn.oldz4-hft-export-contract.r3",
        "unexpected R3 export contract schema",
    )
    return contract


def load_r2_runner(contract: dict[str, Any]) -> ModuleType:
    """Import the exact frozen R2 helper used for source and membership checks."""
    pins = contract["source"]
    require(
        sha256_file(R2_RUNNER_PATH) == pins["r2_runner_sha256"],
        "frozen R2 source/membership helper changed",
    )
    if os.environ.get("PYTHONDONTWRITEBYTECODE") != "1":
        raise ValueError("set PYTHONDONTWRITEBYTECODE=1 before frozen-source imports")
    spec = importlib.util.spec_from_file_location("oldz4_hft_r2_frozen", R2_RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise ImportError("cannot import the pinned R2 runner")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def iter_numeric_leaves(value: Any, path: str = "") -> list[str]:
    """Find nonfinite floating tensors/scalars in trusted local producer state."""
    if isinstance(value, torch.Tensor):
        if value.is_floating_point() and not torch.isfinite(value).all():
            return [path]
        return []
    if isinstance(value, np.ndarray):
        if np.issubdtype(value.dtype, np.floating) and not np.isfinite(value).all():
            return [path]
        return []
    if isinstance(value, np.generic):
        return [path] if np.issubdtype(value.dtype, np.floating) and not np.isfinite(value) else []
    if isinstance(value, float):
        return [path] if not math.isfinite(value) else []
    if isinstance(value, dict):
        bad: list[str] = []
        for key, item in value.items():
            bad.extend(iter_numeric_leaves(item, f"{path}.{key}" if path else str(key)))
        return bad
    if isinstance(value, (list, tuple)):
        bad = []
        for index, item in enumerate(value):
            bad.extend(iter_numeric_leaves(item, f"{path}[{index}]"))
        return bad
    return []


def assert_state_equal(left: dict[str, Any], right: dict[str, Any], label: str) -> int:
    """Require exact tensor equality for a checkpoint model mapping."""
    require(set(left) == set(right), f"{label} model tensor keys differ")
    tensor_count = 0
    for key, left_value in left.items():
        right_value = right[key]
        require(
            isinstance(left_value, torch.Tensor) and isinstance(right_value, torch.Tensor),
            f"{label} model value is not a tensor: {key}",
        )
        require(torch.equal(left_value, right_value), f"{label} model differs at {key}")
        tensor_count += 1
    return tensor_count


def expected_output_paths(project_root: Path, contract: dict[str, Any]) -> dict[str, Path]:
    """Resolve all files the predictor may create or remove."""
    prediction = contract["prediction"]
    return {
        "prediction": project_root / prediction["prediction_relative_path"],
        "partial": project_root / prediction["partial_relative_path"],
        "sidecar": project_root / prediction["sidecar_relative_path"],
        "sequence_sums": project_root / prediction["sequence_sums_relative_path"],
        "result_receipt": project_root / prediction["export_result_relative_path"],
    }


def assert_outputs_absent(project_root: Path, contract: dict[str, Any]) -> dict[str, bool]:
    """Refuse overwrite of final, partial, sidecar, or predictor-deleted files."""
    paths = expected_output_paths(project_root, contract)
    present = {name: path.exists() for name, path in paths.items()}
    require(not any(present.values()), f"export output collision: {present}")
    return present


def _fit_counters(log_path: Path) -> dict[str, int]:
    """Extract the actual completed epoch counters from the producer log."""
    fit_line = next(
        (
            line[len("FIT ") :]
            for line in log_path.read_text().splitlines()
            if line.startswith("FIT ")
        ),
        None,
    )
    require(fit_line is not None, "fit log has no terminal FIT record")
    fit = json.loads(fit_line)
    epoch = fit["update_diagnostics"]["epoch"]
    require(fit.get("evaluation") == "withheld", "fit log does not record withheld evaluation")
    return {
        "optimizer_attempts": int(epoch["optimizer_attempts"]),
        "optimizer_completed_updates": int(epoch["optimizer_completed_updates"]),
        "amp_skipped_updates": int(epoch["amp_skipped_updates"]),
        "completed_batches": int(epoch["completed_batches"]),
    }


def _check_export_contracts(contract: dict[str, Any], r2: ModuleType) -> dict[str, Any]:
    """Bind immutable R2 recipe/source and R3 resource-only producer receipts."""
    pins = contract["source"]
    fit_contract = read_object(r2.FIT_CONTRACT_PATH)
    export_contract = read_object(r2.EXPORT_CONTRACT_PATH)
    resource_contract_path = R3_RESOURCE_PACKET / "HFT_RESOURCE_ONLY_FIT_CONTRACT_R3.json"
    resource_contract = read_object(resource_contract_path)
    hash_paths = {
        "r2_fit_contract_sha256": r2.FIT_CONTRACT_PATH,
        "r2_export_contract_sha256": r2.EXPORT_CONTRACT_PATH,
        "r2_source_input_bindings_sha256": r2.INPUT_BINDINGS_PATH,
        "r2_packet_manifest_sha256": R2_PACKET / "PACKET_FILE_MANIFEST_R2B.json",
        "r3_resource_fit_contract_sha256": resource_contract_path,
        "r3_resource_fit_manifest_sha256": R3_RESOURCE_PACKET / "PACKET_FILE_MANIFEST_R3.json",
        "r3_resource_fit_adapter_sha256": R3_RESOURCE_PACKET / "run_f1_hft_resource_fit_r3.py",
        "r3_resource_fit_admission_sha256": R3_RESOURCE_PACKET
        / "ROOT_F1_HFT_RESOURCE_ONLY_FIT_ADMISSION_R3.json",
        "r3_resource_fit_launch_sha256": R3_RESOURCE_PACKET
        / "ROOT_F1_HFT_RESOURCE_ONLY_FIT_LAUNCH_R3.json",
        "core_crop_contract_sha256": r2.CORE_CONTRACT_PATH,
        "core_crop_receipt_sha256": r2.CORE_RECEIPT_PATH,
    }
    for key, path in hash_paths.items():
        require(sha256_file(path) == pins[key], f"frozen source changed: {key}")
    trainer_argv = fit_contract.get("trainer_argv")
    require(isinstance(trainer_argv, list), "R2 fit contract has no trainer argv")

    def option_value(option: str) -> str:
        indices = [index for index, value in enumerate(trainer_argv) if value == option]
        require(len(indices) == 1 and indices[0] + 1 < len(trainer_argv), f"bad {option} arg")
        return str(trainer_argv[indices[0] + 1])

    require(
        fit_contract.get("run") == contract["fit_run"]
        and fit_contract.get("expected_checkpoint_epoch") == contract["fit_epoch"]
        and option_value("--name") == contract["fit_run"]
        and option_value("--checkpoint-policy") == "fixed"
        and option_value("--checkpoint-epoch") == str(contract["fit_epoch"])
        and option_value("--epochs") == "1"
        and "--no-eval-during-fit" in trainer_argv,
        "R2 fit contract/argv do not identify the fixed CP0 used by the R3 fit",
    )
    require(
        resource_contract.get("fit_run") == contract["fit_run"]
        and resource_contract.get("attempt") == contract["fit_attempt_id"]
        and resource_contract.get("fit", {}).get("trainer_argv_unchanged") is True,
        "R3 resource-only contract does not identify the completed fit attempt",
    )
    require(
        export_contract.get("run") == contract["fit_run"]
        and export_contract.get("epoch") == contract["fit_epoch"]
        and export_contract.get("predictor_argv") == contract["prediction"]["argv"],
        "R2 export recipe differs from the fixed R3 endpoint",
    )
    argv_hash = hashlib.sha256(
        json.dumps(fit_contract["trainer_argv"], separators=(",", ":")).encode()
    ).hexdigest()
    require(
        argv_hash == resource_contract["fit"]["trainer_argv_sha256"],
        "R3 fit adapter does not bind the unchanged R2 trainer argv",
    )
    require(
        "trainer_argv" not in export_contract and "trainer_argv" in fit_contract,
        "R2 export contract schema changed; re-audit the argument-source correction",
    )
    return {"fit_contract": fit_contract, "export_contract": export_contract}


def _verify_actual_fit(
    contract: dict[str, Any], r2: ModuleType, trainer: ModuleType
) -> dict[str, Any]:
    """Audit the actual R3 fit receipt, fixed CP0, state, ancestry, and F1 membership."""
    pins = contract["source"]
    artifacts = contract["artifacts"]
    _check_export_contracts(contract, r2)
    resource_admission = read_object(
        R3_RESOURCE_PACKET / "ROOT_F1_HFT_RESOURCE_ONLY_FIT_ADMISSION_R3.json"
    )
    resource_launch = read_object(
        R3_RESOURCE_PACKET / "ROOT_F1_HFT_RESOURCE_ONLY_FIT_LAUNCH_R3.json"
    )
    resource_contract_path = R3_RESOURCE_PACKET / "HFT_RESOURCE_ONLY_FIT_CONTRACT_R3.json"
    attempt_path = (
        PROJECT_ROOT
        / "research/campaign_2026-10-06/bandit_continuation/runtime/attempts/oldz4-r3-f1-s3-hft-fit-resource.json"
    )
    log_path = (
        PROJECT_ROOT
        / "research/campaign_2026-10-06/bandit_continuation/runtime/logs/oldz4-r3-f1-s3-hft-fit-resource.log"
    )
    attempt = read_object(attempt_path)
    require(
        sha256_file(attempt_path) == pins["r3_attempt_receipt_sha256"], "actual fit receipt changed"
    )
    require(sha256_file(log_path) == pins["r3_fit_log_sha256"], "actual fit log changed")
    require(
        attempt.get("id") == contract["fit_attempt_id"]
        and attempt.get("phase") == "terminal"
        and attempt.get("status") == "finished"
        and attempt.get("exit_status") == 0,
        "R3 resource-only fit attempt is not terminal successful",
    )
    require(
        attempt.get("contract_sha256") == pins["r3_resource_fit_contract_sha256"]
        and attempt.get("source_manifest_sha256") == pins["r3_attempt_source_manifest_sha256"],
        "R3 attempt does not bind the frozen resource contract/source manifest",
    )
    attempt_sources = attempt.get("source_sha256", {})
    require(
        attempt_sources.get("scripts/train_rnn.py") == pins["trainer_sha256"],
        "actual fit attempt used another trainer source",
    )
    resource_contract_rel = str(resource_contract_path.relative_to(PROJECT_ROOT))
    adapter_rel = str(
        (R3_RESOURCE_PACKET / "run_f1_hft_resource_fit_r3.py").relative_to(PROJECT_ROOT)
    )
    require(
        attempt_sources.get(resource_contract_rel) == pins["r3_resource_fit_contract_sha256"]
        and attempt_sources.get(adapter_rel) == pins["r3_resource_fit_adapter_sha256"],
        "actual fit attempt source map differs from the frozen R3 adapter/contract",
    )
    required_launch_sha = pins["r3_resource_fit_admission_sha256"]
    command = attempt.get("command", [])
    require(
        command[-1:] == [required_launch_sha]
        and resource_launch.get("root_release_sha256") == required_launch_sha
        and resource_admission.get("decision") == "PASS_FOR_ONE_BOUNDED_ORIGINAL_FIT",
        "actual fit does not bind the root resource-only R3 release",
    )
    event_maps = [
        attempt.get("memory_event_deltas", {}),
        attempt.get("final_memory_event_deltas", {}),
        attempt.get("final_parent_memory_event_deltas", {}),
        attempt.get("final_parent_hierarchical_memory_event_deltas", {}),
    ]
    expected_event_keys = {
        "high",
        "low",
        "max",
        "oom",
        "oom_group_kill",
        "oom_kill",
        "sock_throttled",
    }
    require(
        all(
            isinstance(events, dict)
            and set(events) == expected_event_keys
            and all(type(value) is int and value == 0 for value in events.values())
            for events in event_maps
        ),
        "actual fit has a nonzero or malformed memory event delta",
    )
    require(
        attempt.get("peak_ram_bytes", 0) <= 6_800_000_000
        and attempt.get("peak_observed_vram_bytes", 0) <= 4_300_000_000,
        "actual fit exceeded the bounded R3 RAM/VRAM envelope",
    )
    counters = _fit_counters(log_path)
    expected_counters = contract["recipe"]["training"]
    require(
        counters == {key: expected_counters[key] for key in counters},
        "actual fit log counters differ from the frozen terminal summary",
    )
    history = json.loads((PROJECT_ROOT / artifacts["history_path"]).read_text(encoding="utf-8"))
    require(len(history) == 1 and history[0].get("epoch") == 0, "history is not one fixed epoch")
    hist_counters = history[0]["update_diagnostics"]["epoch"]
    require(
        counters
        == {
            "optimizer_attempts": hist_counters["optimizer_attempts"],
            "optimizer_completed_updates": hist_counters["optimizer_completed_updates"],
            "amp_skipped_updates": hist_counters["amp_skipped_updates"],
            "completed_batches": hist_counters["completed_batches"],
        },
        "fit log and history counters disagree",
    )
    for key in (
        "checkpoint_path",
        "model_alias_path",
        "training_state_path",
        "provenance_path",
        "history_path",
        "initializer_checkpoint_path",
        "initializer_provenance_path",
    ):
        artifact_path = PROJECT_ROOT / artifacts[key]
        require(artifact_path.is_file(), f"missing fit artifact: {key}")
        require(
            sha256_file(artifact_path) == artifacts[key.replace("_path", "_sha256")],
            f"fit artifact hash changed: {key}",
        )
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        # These checkpoints/state were generated by our own frozen local trainer.
        checkpoint = torch.load(
            PROJECT_ROOT / artifacts["checkpoint_path"], map_location="cpu", weights_only=False
        )
        alias = torch.load(
            PROJECT_ROOT / artifacts["model_alias_path"], map_location="cpu", weights_only=False
        )
        state = torch.load(
            PROJECT_ROOT / artifacts["training_state_path"], map_location="cpu", weights_only=False
        )
        initializer = torch.load(
            PROJECT_ROOT / artifacts["initializer_checkpoint_path"],
            map_location="cpu",
            weights_only=False,
        )
    checkpoint_model = checkpoint.get("model")
    alias_model = alias.get("model")
    state_model = state.get("model")
    init_model = initializer.get("model")
    require(
        all(
            isinstance(value, dict)
            for value in (checkpoint_model, alias_model, state_model, init_model)
        ),
        "checkpoint/state model maps are malformed",
    )
    model_count = assert_state_equal(checkpoint_model, alias_model, "model.pt alias")
    require(
        assert_state_equal(checkpoint_model, state_model, "training_state model") == model_count,
        "state model tensor count differs",
    )
    require(
        checkpoint.get("epoch") == contract["fit_epoch"]
        and alias.get("epoch") == contract["fit_epoch"],
        "checkpoint epoch is not fixed CP0",
    )
    require(
        checkpoint.get("n_in") == 211 and alias.get("n_in") == 211,
        "checkpoint input width differs from the fixed HFT source",
    )
    require(not iter_numeric_leaves(checkpoint), "checkpoint contains nonfinite numeric values")
    require(not iter_numeric_leaves(state), "training state contains nonfinite numeric values")
    frozen_keys = [
        key
        for key in checkpoint_model
        if key in init_model
        and (key.startswith("inp.") or key.startswith("gru.") or key in {"mean", "std"})
    ]
    require(len(frozen_keys) == 12, "expected frozen trunk and normalization key set is incomplete")
    require(
        all(torch.equal(checkpoint_model[key], init_model[key]) for key in frozen_keys),
        "frozen input/GRU or normalization tensors changed from the cropped initializer",
    )
    provenance_path = PROJECT_ROOT / artifacts["provenance_path"]
    provenance = read_object(provenance_path)
    init_provenance = read_object(PROJECT_ROOT / artifacts["initializer_provenance_path"])
    args = provenance.get("args", {})
    expected_args = {
        "name": contract["fit_run"],
        "seed": contract["recipe"]["seed"],
        "valid_fold": contract["recipe"]["fold"],
        "train_source": "valid",
        "epochs": 1,
        "checkpoint_policy": "fixed",
        "checkpoint_epoch": 0,
        "no_eval_during_fit": True,
        "freeze": "inp,gru",
        "inherit_init_preprocessing": True,
        "aux_next": 0.3,
        "aux_struct": 0.3,
        "loss": "gcorr",
        "optim": "adamw",
        "sched": "onecycle",
        "precision": "fp16",
        "mask_weight": True,
        "lr": 0.0005,
        "wd": 0.01,
        "init": "oldz4_h230_core_f1_s3_r2@ep1",
    }
    require(
        all(args.get(key) == value for key, value in expected_args.items()),
        "fit provenance args differ from the frozen HFT recipe",
    )
    require(provenance.get("selected_epoch") == 0, "provenance does not select fixed epoch 0")
    require(
        provenance.get("initial_checkpoint_sha256") == artifacts["initializer_checkpoint_sha256"],
        "HFT ancestry checkpoint does not match the audited core crop",
    )
    require(
        provenance.get("normalization_identity") == init_provenance.get("normalization_identity"),
        "HFT preprocessing/normalizer differs from the exact initializer",
    )
    split_path = PROJECT_ROOT / "research/campaign_2026-09-29/artifacts/splits/fold1.json"
    split = trainer.load_campaign_split(split_path, PROJECT_ROOT)
    fit_groups = provenance.get("fit_row_groups", {})
    evaluation_groups = provenance.get("evaluation_row_groups", [])
    require(
        set(fit_groups.get("train", [])) == set(),
        "HFT provenance directly fit train.parquet groups",
    )
    require(
        set(fit_groups.get("valid", [])) == set(split.valid_groups)
        and len(split.valid_groups) == 1416,
        "HFT valid fit rows differ from the source-qualified 1,416 group membership",
    )
    require(
        set(evaluation_groups) == set(split.evaluation_groups)
        and len(split.evaluation_groups) == 457,
        "HFT evaluation exclusion differs from the source-qualified F1 457 groups",
    )
    require(
        not (set(fit_groups.get("valid", [])) & set(evaluation_groups)),
        "HFT direct fit rows overlap the F1 evaluation groups",
    )
    require(
        provenance.get("split_manifest_sha256")
        == contract["recipe"]["membership"]["split_manifest_sha256"],
        "HFT provenance split digest changed",
    )
    require(
        provenance.get("fit_membership_identity", {}).get("valid", {}).get("count") == 1416,
        "HFT provenance fit identity count differs",
    )
    return {
        "attempt_id": attempt["id"],
        "attempt_exit_status": attempt["exit_status"],
        "attempt_finished_utc": attempt["finished_utc"],
        "attempt_elapsed_seconds": attempt["elapsed_seconds"],
        "peak_ram_bytes": attempt["peak_ram_bytes"],
        "peak_observed_vram_bytes": attempt["peak_observed_vram_bytes"],
        "memory_event_deltas_all_zero": True,
        "fit_counters": counters,
        "checkpoint_epoch": checkpoint["epoch"],
        "checkpoint_sha256": artifacts["checkpoint_sha256"],
        "model_alias_sha256": artifacts["model_alias_sha256"],
        "training_state_sha256": artifacts["training_state_sha256"],
        "provenance_sha256": artifacts["provenance_sha256"],
        "history_sha256": artifacts["history_sha256"],
        "model_tensor_count": model_count,
        "all_checkpoint_and_state_numeric_values_finite": True,
        "frozen_trunk_and_normalization_equal_initializer": True,
        "frozen_key_count": len(frozen_keys),
        "normalization_identity_equal_initializer": True,
        "membership": {
            "fold": split.fold,
            "train_fit_groups": len(fit_groups["train"]),
            "valid_fit_groups": len(fit_groups["valid"]),
            "outer_eval_groups_excluded": len(evaluation_groups),
            "outer_fit_overlap": 0,
            "split_manifest_sha256": sha256_file(split_path),
        },
        "selection_scope": "Previously inspected F1 robustness evidence, not fresh holdout/public result.",
        "strict_profile_gate_passed": False,
        "profile_valid_for_throughput": False,
    }


def _verify_target_order(contract: dict[str, Any], project_root: Path) -> dict[str, Any]:
    """Bind actual prediction row order to the canonical target archive."""
    target_path = project_root / contract["prediction"]["valid_targets_path"]
    require(target_path.is_file(), "canonical full-valid target archive is missing")
    require(
        target_path.stat().st_size == contract["prediction"]["valid_targets_bytes"],
        "target archive byte size changed",
    )
    require(
        sha256_file(target_path) == contract["prediction"]["valid_targets_sha256"],
        "target archive bytes changed",
    )
    with np.load(target_path, allow_pickle=False) as archive:
        groups = np.asarray(archive["row_group"], dtype="<i8")
    expected = np.arange(contract["prediction"]["row_group_count"], dtype="<i8")
    require(
        groups.shape == expected.shape and np.array_equal(groups, expected),
        "target row groups are not canonical 0..1872 order",
    )
    group_sha = hashlib.sha256(groups.tobytes()).hexdigest()
    require(
        group_sha == contract["prediction"]["row_group_ids_sha256"],
        "canonical row-group identity changed",
    )
    return {
        "path": str(target_path),
        "bytes": target_path.stat().st_size,
        "sha256": sha256_file(target_path),
        "row_group_count": len(groups),
        "row_group_ids_sha256": group_sha,
        "first": int(groups[0]),
        "last": int(groups[-1]),
    }


def _probe_actual_predictor(
    contract: dict[str, Any], predictor: ModuleType, predictor_path: Path, project_root: Path
) -> dict[str, Any]:
    """Parse the real predictor command and stop before CUDA or any output write."""
    before = assert_outputs_absent(project_root, contract)
    old_argv = sys.argv[:]
    original_configure = predictor.configure_torch
    reached_device = False

    def stop_before_device() -> None:
        nonlocal reached_device
        reached_device = True
        raise _DeviceBoundary("stopped at predictor CUDA setup boundary")

    try:
        predictor.configure_torch = stop_before_device
        sys.argv = [str(predictor_path), *contract["prediction"]["argv"]]
        with contextlib.suppress(_DeviceBoundary):
            predictor.main()
    finally:
        predictor.configure_torch = original_configure
        sys.argv = old_argv
    require(reached_device, "actual predictor CLI did not reach the CUDA boundary")
    after = assert_outputs_absent(project_root, contract)
    require(before == after, "predictor CLI probe changed an output path")
    return {
        "status": "PASS_ACTUAL_C337_PREDICTOR_CLI_BEFORE_CUDA",
        "predictor_path": str(predictor_path),
        "predictor_root": str(predictor.ROOT),
        "trainer_root": str(predictor.train_rnn.ROOT),
        "argv": contract["prediction"]["argv"],
        "reached_configure_torch": True,
        "device_started": False,
        "output_paths_unchanged_and_absent": True,
    }


def audit_actual_producer(run_cli_probe: bool = True) -> dict[str, Any]:
    """Audit the completed producer and exact exporter before any root release."""
    contract = load_contract()
    r2 = load_r2_runner(contract)
    bindings = r2.verify_packet_bindings()
    expected_bindings = {
        "r4_input_bindings_sha256": contract["source"]["r4_input_bindings_sha256"],
        "r4_source_bindings_sha256": contract["source"]["r4_source_bindings_sha256"],
        "r4_student_fit_contract_sha256": contract["source"]["r4_student_fit_contract_sha256"],
        "r3_source_snapshot_bindings_sha256": contract["source"]["c337_snapshot_bindings_sha256"],
        "trainer_sha256": contract["source"]["trainer_sha256"],
        "predictor_sha256": contract["source"]["predictor_sha256"],
    }
    for key, expected in expected_bindings.items():
        require(bindings.get(key) == expected, f"R2 import binding differs at {key}")
    trainer, trainer_path = r2.import_trainer()
    membership = r2.validate_inputs(trainer)
    require(
        membership["source_qualified"] is True and membership["outer_evaluation_fit_overlap"] == 0,
        "actual c337 membership validation failed",
    )
    core_receipt, core_receipt_sha = r2.validate_core_receipt()
    require(
        core_receipt_sha == contract["source"]["core_crop_receipt_sha256"],
        "actual core crop receipt changed",
    )
    producer = _verify_actual_fit(contract, r2, trainer)
    target_order = _verify_target_order(contract, PROJECT_ROOT)
    outputs_absent = assert_outputs_absent(PROJECT_ROOT, contract)
    predictor, predictor_path = r2.import_predictor(trainer)
    require(
        sha256_file(predictor_path) == contract["source"]["predictor_sha256"],
        "actual predictor source changed",
    )
    require(
        Path(trainer_path).resolve() == Path(predictor.train_rnn.__file__).resolve(),
        "predictor did not use the exact trainer module",
    )
    probe = (
        _probe_actual_predictor(contract, predictor, predictor_path, PROJECT_ROOT)
        if run_cli_probe
        else None
    )
    return {
        "schema": "wnn.oldz4-hft-cp0-export-preflight.r1",
        "status": "PASS_PRODUCER_PREFLIGHT",
        "created_utc": datetime.now(UTC).isoformat(),
        "contract_sha256": sha256_file(CONTRACT_PATH),
        "contract_path": str(CONTRACT_PATH),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "producer": producer,
        "core_crop_receipt_sha256": core_receipt_sha,
        "core_crop_receipt_status": core_receipt.get("status"),
        "source_and_membership": membership,
        "target_order": target_order,
        "predictor_cli_probe": probe,
        "output_paths_absent": outputs_absent,
        "export_performed": False,
        "public_or_fresh_holdout_claim": False,
    }


def _check_export_environment(contract: dict[str, Any]) -> None:
    """Require the exact single-threaded export resource profile."""
    expected = {
        "WNN_SLICE": "wnn-campaign.slice",
        "WNN_CPU_QUOTA": contract["resource_limit"]["cpu_quota"],
        "WNN_THREADS": str(contract["resource_limit"]["threads"]),
        "WNN_MEM_MAX": str(contract["resource_limit"]["memory_reserve_bytes"]),
        "WNN_RUNTIME_MAX": f"{contract['resource_limit']['runtime_max_seconds']}s",
        "WNN_LOADER_WORKERS": str(contract["resource_limit"]["loader_workers"]),
        "WNN_GPU_MEMORY_BYTES": str(contract["resource_limit"]["vram_allocator_limit_bytes"]),
        "WNN_LOADER_RAM_GB": "6",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "PYTHONHASHSEED": "0",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    mismatches = {
        key: os.environ.get(key) for key, value in expected.items() if os.environ.get(key) != value
    }
    require(not mismatches, f"export environment differs from its bounded contract: {mismatches}")


def _check_gate(
    path: Path, provided_sha: str, required: dict[str, Any]
) -> tuple[dict[str, Any], str]:
    """Verify one root-owned gate byte hash and all required exact bindings."""
    require(path.is_file(), f"required root gate is absent: {path.name}")
    observed_sha = sha256_file(path)
    require(observed_sha == provided_sha, f"root gate SHA differs from CLI pin: {path.name}")
    gate = read_object(path)
    for key, value in required.items():
        require(gate.get(key) == value, f"root gate binding mismatch at {key}: {path.name}")
    return gate, observed_sha


def verify_root_gates(terminal_sha: str, export_sha: str) -> dict[str, str]:
    """Require independent terminal and one-export authorizations."""
    contract = load_contract()
    artifacts = contract["artifacts"]
    source = contract["source"]
    terminal_expected = {
        "schema": contract["root_gates"]["terminal_schema"],
        "decision": "PASS",
        "fit_attempt_id": contract["fit_attempt_id"],
        "fit_run": contract["fit_run"],
        "original_fit_contract_sha256": source["r2_fit_contract_sha256"],
        "resource_fit_contract_sha256": source["r3_resource_fit_contract_sha256"],
        "resource_fit_admission_sha256": source["r3_resource_fit_admission_sha256"],
        "resource_fit_launch_sha256": source["r3_resource_fit_launch_sha256"],
        "resource_attempt_receipt_sha256": source["r3_attempt_receipt_sha256"],
        "fit_log_sha256": source["r3_fit_log_sha256"],
        "core_crop_receipt_sha256": source["core_crop_receipt_sha256"],
        "checkpoint_epoch": contract["fit_epoch"],
        "checkpoint_sha256": artifacts["checkpoint_sha256"],
        "model_alias_sha256": artifacts["model_alias_sha256"],
        "training_state_sha256": artifacts["training_state_sha256"],
        "provenance_sha256": artifacts["provenance_sha256"],
        "history_sha256": artifacts["history_sha256"],
        "source_snapshot_bindings_sha256": source["c337_snapshot_bindings_sha256"],
        "split_manifest_sha256": contract["recipe"]["membership"]["split_manifest_sha256"],
        "membership_counts": {
            "train": 0,
            "valid_fit": 1416,
            "outer_eval": 457,
            "outer_fit_overlap": 0,
        },
        "optimizer_counts": {
            key: contract["recipe"]["training"][key]
            for key in (
                "optimizer_attempts",
                "optimizer_completed_updates",
                "amp_skipped_updates",
                "completed_batches",
            )
        },
        "strict_profile_gate_passed": False,
        "profile_valid_for_throughput": False,
        "resource_event_deltas": {
            "high": 0,
            "low": 0,
            "max": 0,
            "oom": 0,
            "oom_group_kill": 0,
            "oom_kill": 0,
            "sock_throttled": 0,
        },
    }
    terminal, terminal_digest = _check_gate(TERMINAL_GATE_PATH, terminal_sha, terminal_expected)
    preflight_path = PACKET_ROOT / contract["prediction"]["preflight_receipt_relative_path"]
    require(preflight_path.is_file(), "committed producer preflight receipt is missing")
    preflight = read_object(preflight_path)
    require(
        preflight.get("status") == "PASS_PRODUCER_PREFLIGHT"
        and preflight.get("contract_sha256") == sha256_file(CONTRACT_PATH)
        and preflight.get("export_performed") is False,
        "preflight receipt is not a read-only pass for this exact contract",
    )
    export_expected = {
        "schema": contract["root_gates"]["export_schema"],
        "decision": "PASS_FOR_ONE_EXPORT",
        "export_attempt_id": contract["export_attempt_id"],
        "fit_attempt_id": contract["fit_attempt_id"],
        "fit_run": contract["fit_run"],
        "export_contract_sha256": sha256_file(CONTRACT_PATH),
        "export_runner_sha256": sha256_file(Path(__file__).resolve()),
        "terminal_gate_sha256": terminal_digest,
        "predictor_source_sha256": source["predictor_sha256"],
        "checkpoint_sha256": artifacts["checkpoint_sha256"],
        "provenance_sha256": artifacts["provenance_sha256"],
        "training_state_sha256": artifacts["training_state_sha256"],
        "prediction_relative_path": contract["prediction"]["prediction_relative_path"],
        "partial_relative_path": contract["prediction"]["partial_relative_path"],
        "sidecar_relative_path": contract["prediction"]["sidecar_relative_path"],
        "sequence_sums_relative_path": contract["prediction"]["sequence_sums_relative_path"],
        "export_result_relative_path": contract["prediction"]["export_result_relative_path"],
        "resource_limits": contract["resource_limit"],
        "output_paths_absent_at_authorization": True,
        "preflight_receipt_sha256": sha256_file(preflight_path),
    }
    _, export_digest = _check_gate(EXPORT_GATE_PATH, export_sha, export_expected)
    require(terminal.get("decision") == "PASS", "terminal gate is not a terminal pass")
    return {"terminal_gate_sha256": terminal_digest, "export_gate_sha256": export_digest}


def run_export(terminal_sha: str, export_sha: str) -> dict[str, Any]:
    """Export the already-fitted CP0 once, only under both exact root gates."""
    contract = load_contract()
    _check_export_environment(contract)
    audit = audit_actual_producer(run_cli_probe=False)
    gates = verify_root_gates(terminal_sha, export_sha)
    outputs_before = assert_outputs_absent(PROJECT_ROOT, contract)
    r2 = load_r2_runner(contract)
    trainer, _ = r2.import_trainer()
    predictor, predictor_path = r2.import_predictor(trainer)
    old_argv = sys.argv[:]
    try:
        sys.argv = [str(predictor_path), *contract["prediction"]["argv"]]
        predictor.main()
    finally:
        sys.argv = old_argv
    outputs = expected_output_paths(PROJECT_ROOT, contract)
    require(
        outputs["prediction"].is_file() and outputs["sidecar"].is_file(),
        "predictor did not create both final outputs",
    )
    require(not outputs["partial"].exists(), "predictor left a partial prediction file")
    sidecar = read_object(outputs["sidecar"])
    artifacts = contract["artifacts"]
    prediction = contract["prediction"]
    require(
        sidecar.get("run") == contract["fit_run"] and sidecar.get("split") == "valid",
        "sidecar run/split differs",
    )
    require(
        sidecar.get("epoch") == contract["fit_epoch"],
        "actual c337 sidecar uses `epoch`; it must identify fixed CP0",
    )
    require(
        sidecar.get("checkpoint_sha256") == artifacts["checkpoint_sha256"],
        "sidecar does not bind fixed CP0",
    )
    require(
        sidecar.get("split_manifest_sha256")
        == contract["recipe"]["membership"]["split_manifest_sha256"],
        "sidecar split fingerprint differs",
    )
    row_groups = np.asarray(sidecar.get("row_groups"), dtype="<i8")
    require(row_groups.shape == (prediction["row_group_count"],), "sidecar row-group shape differs")
    require(
        hashlib.sha256(row_groups.tobytes()).hexdigest() == prediction["row_group_ids_sha256"],
        "sidecar row order differs from canonical valid targets",
    )
    prediction_sha = sha256_file(outputs["prediction"])
    require(sidecar.get("prediction_sha256") == prediction_sha, "sidecar prediction hash differs")
    mapped = np.load(outputs["prediction"], mmap_mode="r", allow_pickle=False)
    require(
        mapped.shape == tuple(prediction["shape"])
        and mapped.dtype == np.dtype(prediction["dtype"]),
        "prediction NPY header differs",
    )
    require(
        outputs["prediction"].stat().st_size == prediction["npy_file_bytes"],
        "prediction NPY byte length differs",
    )
    for start in range(0, mapped.shape[0], 8):
        require(
            np.isfinite(mapped[start : start + 8]).all(),
            f"nonfinite prediction sequences at row {start}",
        )
    require(not outputs["sequence_sums"].exists(), "unexpected predictor sequence-sums file exists")
    return {
        "schema": "wnn.oldz4-hft-export-result.r3",
        "status": "PASS_ACTUAL_CP0_EXPORT",
        "export_attempt_id": contract["export_attempt_id"],
        "fit_attempt_id": contract["fit_attempt_id"],
        "fit_run": contract["fit_run"],
        "checkpoint_sha256": artifacts["checkpoint_sha256"],
        "prediction_sha256": prediction_sha,
        "prediction_bytes": outputs["prediction"].stat().st_size,
        "prediction_shape": list(mapped.shape),
        "prediction_dtype": str(mapped.dtype),
        "sidecar_sha256": sha256_file(outputs["sidecar"]),
        "sidecar_epoch_key": "epoch",
        "row_groups_sha256": prediction["row_group_ids_sha256"],
        "all_predictions_finite": True,
        "output_paths_absent_before_export": outputs_before,
        "root_gates": gates,
        "producer_preflight_status": audit["status"],
        "score_claim": None,
        "selection_scope": contract["recipe"]["provenance_scope"],
    }


def main() -> None:
    """Run a read-only preflight or one root-authorized prediction export."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("preflight", "export"), required=True)
    parser.add_argument("--terminal-gate-sha256")
    parser.add_argument("--export-gate-sha256")
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    if args.stage == "preflight":
        require(
            args.terminal_gate_sha256 is None and args.export_gate_sha256 is None,
            "preflight does not accept authorization pins",
        )
        result = audit_actual_producer(run_cli_probe=True)
        if args.receipt is not None:
            require(
                not args.receipt.exists(), f"refuse to overwrite preflight receipt: {args.receipt}"
            )
            args.receipt.parent.mkdir(parents=True, exist_ok=True)
            with args.receipt.open("x", encoding="utf-8") as stream:
                json.dump(result, stream, indent=2, sort_keys=True)
                stream.write("\n")
        print(json.dumps(result, sort_keys=True))
        return
    require(
        args.terminal_gate_sha256 is not None and args.export_gate_sha256 is not None,
        "export requires both root gate SHA pins",
    )
    contract = load_contract()
    expected_receipt = PROJECT_ROOT / contract["prediction"]["export_result_relative_path"]
    require(args.receipt is not None, "export requires the fixed result receipt path")
    require(
        args.receipt.resolve() == expected_receipt.resolve(),
        "export result receipt path differs from the frozen contract",
    )
    require(
        not expected_receipt.exists(), "refuse export because the result receipt already exists"
    )
    result = run_export(args.terminal_gate_sha256, args.export_gate_sha256)
    if args.receipt is not None:
        require(not args.receipt.exists(), f"refuse to overwrite export receipt: {args.receipt}")
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        with args.receipt.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2, sort_keys=True)
            stream.write("\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
