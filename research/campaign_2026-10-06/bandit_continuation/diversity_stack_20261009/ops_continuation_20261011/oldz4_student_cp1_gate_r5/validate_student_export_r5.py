from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

PACKET_ROOT = Path(__file__).resolve().parent
CONTRACT_PATH = (
    PACKET_ROOT.parent / "local_oldz4_f0_student_export_r4" / "STUDENT_EXPORT_CONTRACT_R4.json"
)
EXPECTED_CONTRACT_SHA256 = "665bdb087975ce5d74e485ea31aea88e7d07f8545af3dbee48e181d0bcba56fb"
LOGGER = logging.getLogger("oldz4_student_export_r5")


def sha256_file(path: Path) -> str:
    """Hash a file in bounded memory for durable artifact bindings."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object and reject other top-level values."""
    # JSON manifests have heterogeneous values and no single useful static value type.
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _check_update_diagnostics(diagnostics: object, history: object) -> dict[str, int]:
    """Require two completed fixed-schedule epochs and internally consistent counts."""
    _require(isinstance(diagnostics, list), "provenance lacks update diagnostics")
    _require(isinstance(history, list), "history is not an epoch list")
    _require(len(diagnostics) == 2 and len(history) == 2, "expected both fit epochs")
    totals = {
        "optimizer_attempts": 0,
        "optimizer_completed_updates": 0,
        "amp_skipped_updates": 0,
        "completed_batches": 0,
    }
    for epoch_index, (entry, history_entry) in enumerate(zip(diagnostics, history, strict=True)):
        _require(isinstance(entry, dict), f"epoch {epoch_index}: malformed diagnostics")
        _require(entry.get("epoch_index") == epoch_index, f"epoch {epoch_index}: wrong index")
        epoch = entry.get("epoch")
        _require(isinstance(epoch, dict), f"epoch {epoch_index}: missing counters")
        _require(
            isinstance(history_entry, dict) and history_entry.get("epoch") == epoch_index,
            f"epoch {epoch_index}: history row differs",
        )
        history_diagnostics = history_entry.get("update_diagnostics")
        expected_history_diagnostics = {
            key: value for key, value in entry.items() if key != "epoch_index"
        }
        _require(
            isinstance(history_diagnostics, dict)
            and history_diagnostics == expected_history_diagnostics,
            f"epoch {epoch_index}: history counters differ from provenance",
        )
        for key in totals:
            count = epoch.get(key)
            _require(type(count) is int and count >= 0, f"epoch {epoch_index}: invalid {key}")
            totals[key] += count
        _require(epoch["optimizer_attempts"] > 0, f"epoch {epoch_index}: no attempts")
        _require(
            epoch["optimizer_completed_updates"] > 0,
            f"epoch {epoch_index}: no completed optimizer update",
        )
        _require(epoch["completed_batches"] > 0, f"epoch {epoch_index}: no batches")
        _require(
            epoch["optimizer_attempts"]
            == epoch["optimizer_completed_updates"] + epoch["amp_skipped_updates"],
            f"epoch {epoch_index}: optimizer counters do not reconcile",
        )
        cumulative = entry.get("cumulative")
        _require(isinstance(cumulative, dict), f"epoch {epoch_index}: missing cumulative counts")
        for key, total in totals.items():
            _require(cumulative.get(key) == total, f"epoch {epoch_index}: cumulative {key} differs")
    return totals


def _assert_model_finite(checkpoint: dict[str, Any]) -> int:
    """Check every floating tensor in the inference model and saved readout."""
    # A trusted torch checkpoint is a heterogeneous nested structure by design.
    model_state = checkpoint.get("model")
    _require(isinstance(model_state, dict) and model_state, "checkpoint has no model state")
    tensors = [value for value in model_state.values() if isinstance(value, torch.Tensor)]
    _require(len(tensors) == len(model_state), "model state contains a non-tensor value")
    _require(bool(tensors), "checkpoint model state is empty")
    for name, tensor in model_state.items():
        if tensor.is_floating_point():
            _require(bool(torch.isfinite(tensor).all()), f"nonfinite model tensor: {name}")
    readout = checkpoint.get("readout")
    if readout is not None:
        _require(isinstance(readout, list), "checkpoint readout has an unexpected format")
        for index, tensor in enumerate(readout):
            _require(
                isinstance(tensor, torch.Tensor), "checkpoint readout contains non-tensor data"
            )
            _require(bool(torch.isfinite(tensor).all()), f"nonfinite readout tensor {index}")
    return len(tensors)


def _model_state_sha256(model_state: dict[str, Any]) -> str:
    """Hash all named model tensors in a stable order for exact checkpoint matching."""
    digest = hashlib.sha256()
    for name in sorted(model_state):
        tensor = model_state[name]
        _require(isinstance(tensor, torch.Tensor), f"model state contains non-tensor: {name}")
        contiguous = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(json.dumps(list(contiguous.shape), separators=(",", ":")).encode("ascii"))
        digest.update(b"\0")
        digest.update(contiguous.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _assert_tree_finite(value: Any, path: str = "state") -> tuple[int, int]:
    """Walk nested training state and count finite tensors and scalar floats."""
    tensor_count = 0
    scalar_count = 0
    if isinstance(value, torch.Tensor):
        tensor_count = 1
        if value.is_floating_point():
            _require(bool(torch.isfinite(value).all()), f"nonfinite training-state tensor: {path}")
    elif isinstance(value, np.ndarray):
        if np.issubdtype(value.dtype, np.floating):
            scalar_count = int(value.size)
            _require(bool(np.isfinite(value).all()), f"nonfinite training-state array: {path}")
    elif isinstance(value, np.generic):
        if np.issubdtype(value.dtype, np.floating):
            scalar_count = 1
            _require(bool(np.isfinite(value)), f"nonfinite training-state scalar: {path}")
    elif isinstance(value, dict):
        for key, child in value.items():
            child_tensors, child_scalars = _assert_tree_finite(child, f"{path}.{key}")
            tensor_count += child_tensors
            scalar_count += child_scalars
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            child_tensors, child_scalars = _assert_tree_finite(child, f"{path}[{index}]")
            tensor_count += child_tensors
            scalar_count += child_scalars
    elif isinstance(value, float):
        scalar_count = 1
        _require(math.isfinite(value), f"nonfinite training-state scalar: {path}")
    return tensor_count, scalar_count


def validate_training_state(
    state_path: Path,
    *,
    expected_epochs: int = 2,
    expected_next_epoch: int = 2,
    expected_totals: dict[str, int],
    expected_update_diagnostics: list[dict[str, Any]],
    expected_history: list[dict[str, Any]],
    expected_recipe: dict[str, Any],
) -> dict[str, Any]:
    """Validate full epoch-boundary continuation state, including optimizer state."""
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    _require(isinstance(state, dict), "training state is not a mapping")
    _require(state.get("format") == "wnn.epoch-training-state", "training-state format differs")
    _require(state.get("version") == 1, "training-state version differs")
    _require(state.get("boundary") == "epoch", "training-state boundary is not an epoch")
    _require(state.get("next_epoch") == expected_next_epoch, "training-state next epoch differs")
    contract = state.get("contract")
    _require(isinstance(contract, dict), "training state lacks its continuation contract")
    options = contract.get("options")
    _require(isinstance(options, dict), "training-state contract lacks fit options")
    _require(options.get("epochs") == expected_epochs, "training-state horizon differs")
    _require(
        options.get("checkpoint_policy") == "fixed", "training-state checkpoint policy differs"
    )
    _require(options.get("checkpoint_epoch") == 1, "training-state fixed checkpoint epoch differs")
    for option in (
        "seed",
        "arch",
        "cell",
        "hidden",
        "layers",
        "precision",
        "optim",
        "sched",
        "init",
    ):
        if option in options:
            _require(
                options[option] == expected_recipe[option],
                f"training-state recipe option differs: {option}",
            )
    required = {
        "model",
        "optimizer",
        "scheduler",
        "scaler",
        "loader",
        "progress",
        "rng",
        "auxiliary",
    }
    _require(not (required - state.keys()), "training state lacks required continuation fields")
    optimizer = state.get("optimizer")
    scheduler = state.get("scheduler")
    _require(isinstance(optimizer, dict) and optimizer, "training state lacks optimizer state")
    optimizer_slots = optimizer.get("state")
    parameter_groups = optimizer.get("param_groups")
    _require(isinstance(optimizer_slots, dict) and optimizer_slots, "optimizer slots are empty")
    _require(
        isinstance(parameter_groups, list) and parameter_groups,
        "optimizer parameter groups are missing",
    )
    _require(isinstance(scheduler, dict) and scheduler, "training state lacks scheduler state")
    loader = state.get("loader")
    _require(isinstance(loader, dict), "training state lacks loader boundary state")
    _require(loader.get("num_workers") == 0, "saved loader state has nonzero workers")
    progress = state.get("progress")
    _require(isinstance(progress, dict), "training state lacks progress counters")
    history = progress.get("history")
    _require(
        isinstance(history, list) and len(history) == expected_next_epoch,
        "training-state history boundary differs",
    )
    _require(history == expected_history, "training-state history differs from terminal history")
    cumulative = progress.get("cumulative_updates")
    _require(isinstance(cumulative, dict), "training state lacks cumulative optimizer counters")
    for key, expected in expected_totals.items():
        _require(cumulative.get(key) == expected, f"training-state counter differs: {key}")
    update_diagnostics = progress.get("update_diagnostics")
    _require(
        isinstance(update_diagnostics, list) and len(update_diagnostics) == 2,
        "training-state epoch diagnostics differ",
    )
    _require(
        update_diagnostics == expected_update_diagnostics,
        "training-state diagnostics differ from terminal provenance",
    )
    model_state = state.get("model")
    _require(isinstance(model_state, dict) and model_state, "training state lacks model tensors")
    training_model_tensor_count = len(model_state)
    training_model_sha256 = _model_state_sha256(model_state)
    optimizer_tensor_count, _ = _assert_tree_finite(optimizer, "optimizer")
    _require(optimizer_tensor_count > 0, "optimizer state has no tensors to validate")
    state_tensor_count, state_scalar_count = _assert_tree_finite(state)
    del state
    return {
        "format": "wnn.epoch-training-state",
        "version": 1,
        "boundary": "epoch",
        "next_epoch": expected_next_epoch,
        "epochs": expected_epochs,
        "optimizer_attempts": 14920,
        "optimizer_completed_updates": 14913,
        "amp_skipped_updates": 7,
        "training_state_tensor_count": state_tensor_count,
        "optimizer_tensor_count": optimizer_tensor_count,
        "model_tensor_count": training_model_tensor_count,
        "model_state_sha256": training_model_sha256,
        "finite_floating_scalar_count": state_scalar_count,
        "all_training_state_floats_finite": True,
    }


def validate_prediction_payload(
    prediction_path: Path,
    metadata: dict[str, Any],
    *,
    run: str,
    epoch: int,
    checkpoint_sha256: str,
    split_manifest_sha256: str,
    expected_row_groups: list[int],
    expected_shape: tuple[int, int, int],
    expected_bytes: int,
) -> str:
    """Validate a sidecar-bound, finite float32 prediction NPY in bounded chunks."""
    expected_sidecar = {
        "run": run,
        "split": "valid",
        "epoch": epoch,
        "checkpoint_sha256": checkpoint_sha256,
        "split_manifest_sha256": split_manifest_sha256,
        "row_groups": expected_row_groups,
        "inference_precision": "fp16",
        "quantization_preset": None,
        "recurrent_rank": None,
        "evaluation_membership_sha256": None,
        "fit_subset_manifest_sha256": None,
    }
    for key, expected in expected_sidecar.items():
        _require(metadata.get(key) == expected, f"prediction sidecar differs for {key}")
    _require(prediction_path.is_file(), "prediction NPY is missing")
    _require(prediction_path.stat().st_size == expected_bytes, "prediction NPY byte length differs")
    prediction_sha256 = sha256_file(prediction_path)
    _require(
        metadata.get("prediction_sha256") == prediction_sha256,
        "prediction hash differs from sidecar",
    )
    prediction = np.load(prediction_path, mmap_mode="r", allow_pickle=False)
    _require(prediction.shape == expected_shape, f"prediction shape differs: {prediction.shape}")
    _require(prediction.dtype == np.float32, f"prediction dtype differs: {prediction.dtype}")
    for start in range(0, len(prediction), 8):
        _require(
            bool(np.isfinite(prediction[start : start + 8]).all()),
            f"nonfinite prediction rows at {start}",
        )
    return prediction_sha256


def validate_checkpoint(project_root: Path) -> dict[str, Any]:
    """Validate the R4-admitted fit receipt, fixed CP1, finite state and update counts."""
    contract = read_json(CONTRACT_PATH)
    _require(
        sha256_file(CONTRACT_PATH) == EXPECTED_CONTRACT_SHA256,
        "preserved R4 export contract bytes changed",
    )
    root = project_root.resolve(strict=True)
    _require(str(root) == contract["project_root"], "project root differs from frozen contract")
    admission_path = root / contract["admitted_fit_contract_relative_path"]
    attempt_path = root / contract["fit_attempt_relative_path"]
    _require(admission_path.is_file(), "admitted fit contract is missing")
    _require(
        sha256_file(admission_path) == contract["admitted_fit_contract_sha256"],
        "fit admission contract changed",
    )
    admission = read_json(admission_path)
    _require(
        admission.get("attempt") == contract["fit_attempt_id"], "fit attempt differs from admission"
    )
    _require(admission.get("run") == contract["run"], "fit run differs from admission")
    _require(admission.get("launch_authorized") is True, "R4 fit was not root-admitted")
    _require(attempt_path.is_file(), "fit attempt receipt is missing")
    attempt = read_json(attempt_path)
    _require(attempt.get("id") == contract["fit_attempt_id"], "terminal receipt ID differs")
    _require(attempt.get("status") == "finished", "fit attempt is not terminal")
    _require(attempt.get("phase") == "terminal", "fit attempt phase is not terminal")
    _require(attempt.get("exit_status") == 0, "fit attempt did not exit successfully")
    _require(attempt.get("finished_utc"), "fit attempt has no terminal timestamp")
    _require(attempt.get("failure") is None, "fit attempt records a failure")
    _require(
        attempt.get("contract_sha256") == contract["admitted_fit_contract_sha256"],
        "fit attempt used a different admitted contract",
    )
    source_hashes = attempt.get("source_sha256")
    _require(isinstance(source_hashes, dict), "fit attempt lacks source hashes")
    _require(
        source_hashes.get(contract["fit_trainer_relative_path"]) == contract["fit_trainer_sha256"],
        "executed trainer source differs from the frozen fit source",
    )
    for event_field in ("memory_event_deltas", "final_memory_event_deltas"):
        events = attempt.get(event_field)
        _require(isinstance(events, dict), f"fit attempt lacks {event_field}")
        _require(events.get("oom", 0) == 0, f"fit attempt has {event_field} OOMs")
        _require(events.get("oom_group_kill", 0) == 0, f"fit attempt has {event_field} group OOMs")
        _require(events.get("oom_kill", 0) == 0, f"fit attempt has {event_field} killed processes")

    run = contract["run"]
    run_dir = root / "artifacts" / "runs" / run
    checkpoint_path = run_dir / "model_ep1.pt"
    provenance_path = run_dir / "provenance.json"
    history_path = run_dir / "history.json"
    state_path = run_dir / "training_state.pt"
    for path in (checkpoint_path, provenance_path, history_path, state_path):
        _require(path.is_file(), f"required terminal fit artifact is missing: {path.name}")
    checkpoint_sha256 = sha256_file(checkpoint_path)
    provenance = read_json(provenance_path)
    _require(provenance.get("selected_epoch") == 1, "provenance did not select fixed epoch 1")
    _require(
        provenance.get("checkpoint_sha256", {}).get("model_ep1.pt") == checkpoint_sha256,
        "provenance checkpoint hash differs from CP1 bytes",
    )
    _require(
        provenance.get("args", {}).get("split_manifest_sha256")
        == contract["fit_recipe_pins"]["split_manifest_sha256"],
        "provenance split differs from the fixed fold-0 split",
    )
    split_path = root / contract["fit_recipe_pins"]["split_manifest_relative_path"]
    split = read_json(split_path)
    _require(
        sha256_file(split_path) == contract["fit_recipe_pins"]["split_manifest_sha256"],
        "fold-0 split changed",
    )
    evaluation_groups = split.get("evaluation", {}).get("valid")
    fit_groups = provenance.get("fit_row_groups")
    _require(
        isinstance(evaluation_groups, list) and len(evaluation_groups) == 468,
        "fold-0 eval membership differs",
    )
    _require(isinstance(fit_groups, dict), "provenance lacks source-qualified fit membership")
    actual_train = set(map(int, fit_groups.get("train", [])))
    actual_valid = set(map(int, fit_groups.get("valid", [])))
    expected_train = set(map(int, split.get("fit", {}).get("train", [])))
    expected_valid = set(map(int, split.get("fit", {}).get("valid", [])))
    _require(
        actual_train == expected_train, "train.parquet fit membership differs from fold-0 split"
    )
    _require(
        actual_valid == expected_valid, "valid.parquet fit membership differs from fold-0 split"
    )
    _require(
        not actual_valid.intersection(map(int, evaluation_groups)),
        "student fit includes fold-0 evaluation groups",
    )
    _require(
        len(actual_train) == 10535 and len(actual_valid) == 1405, "fit membership counts differ"
    )
    history = json.loads(history_path.read_text(encoding="utf-8"))
    totals = _check_update_diagnostics(provenance.get("update_diagnostics"), history)
    training_state_validation = validate_training_state(
        state_path,
        expected_totals=totals,
        expected_update_diagnostics=provenance["update_diagnostics"],
        expected_history=history,
        expected_recipe=contract["checkpoint_recipe_assertions"],
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    _require(isinstance(checkpoint, dict), "CP1 is not a checkpoint mapping")
    _require(checkpoint.get("epoch") == 1, "checkpoint payload is not epoch 1")
    args = checkpoint.get("args")
    _require(isinstance(args, dict), "CP1 lacks recipe arguments")
    expected_args = contract["checkpoint_recipe_assertions"]
    for name, expected in expected_args.items():
        _require(args.get(name) == expected, f"CP1 recipe argument changed: {name}")
    embedded_provenance = checkpoint.get("provenance")
    _require(isinstance(embedded_provenance, dict), "CP1 lacks embedded provenance")
    _require(
        embedded_provenance.get("update_diagnostics") == provenance.get("update_diagnostics"),
        "CP1 update diagnostics differ from terminal provenance",
    )
    model_tensor_count = _assert_model_finite(checkpoint)
    model_sha256 = _model_state_sha256(checkpoint["model"])
    _require(
        training_state_validation["model_tensor_count"] == model_tensor_count,
        "training-state model tensor count differs from CP1",
    )
    _require(
        training_state_validation["model_state_sha256"] == model_sha256,
        "training-state model tensors differ bitwise from CP1",
    )
    contract_sha256 = sha256_file(CONTRACT_PATH)
    return {
        "schema": "wnn.oldz4.student-cp1-terminal-gate.r5",
        "status": "pass",
        "validator_source_path": str(Path(__file__).resolve()),
        "validator_source_sha256": sha256_file(Path(__file__).resolve()),
        "fit_attempt_id": contract["fit_attempt_id"],
        "fit_attempt_receipt_sha256": sha256_file(attempt_path),
        "admitted_fit_contract_sha256": contract["admitted_fit_contract_sha256"],
        "contract_sha256": contract_sha256,
        "run": run,
        "checkpoint_epoch": 1,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "provenance_sha256": sha256_file(provenance_path),
        "history_sha256": sha256_file(history_path),
        "training_state_sha256": sha256_file(state_path),
        "model_tensor_count": model_tensor_count,
        "model_state_sha256": model_sha256,
        "model_tensors_finite": True,
        "optimizer_update_totals": totals,
        "amp_skipped_updates": totals["amp_skipped_updates"],
        "gradient_finiteness_claim": "not independently measured; AMP skip counts are reported",
        "split_manifest_sha256": contract["fit_recipe_pins"]["split_manifest_sha256"],
        "exposure": contract["grade_semantics"]["exposure"],
        "training_state_validation": training_state_validation,
        "created_utc": datetime.now(UTC).isoformat(),
    }


def validate_export(project_root: Path, gate_path: Path) -> dict[str, Any]:
    """Validate full-valid NPY framing, finite values, row order and export sidecar."""
    contract = read_json(CONTRACT_PATH)
    _require(
        sha256_file(CONTRACT_PATH) == EXPECTED_CONTRACT_SHA256,
        "preserved R4 export contract bytes changed",
    )
    root = project_root.resolve(strict=True)
    gate = read_json(gate_path)
    _require(
        gate.get("schema") == "wnn.oldz4.student-cp1-terminal-gate.r5", "wrong CP1 gate schema"
    )
    _require(gate.get("status") == "pass", "CP1 terminal gate did not pass")
    _require(gate.get("run") == contract["run"], "CP1 gate is for another run")
    _require(gate.get("checkpoint_epoch") == 1, "CP1 gate is for another epoch")
    checkpoint_path = root / "artifacts" / "runs" / contract["run"] / "model_ep1.pt"
    checkpoint_sha256 = sha256_file(checkpoint_path)
    _require(checkpoint_sha256 == gate.get("checkpoint_sha256"), "CP1 changed after terminal gate")
    prediction_path = root / contract["export"]["prediction_relative_path"]
    sidecar_path = prediction_path.with_suffix(".json")
    _require(
        prediction_path.is_file() and sidecar_path.is_file(), "prediction or sidecar is missing"
    )
    metadata = read_json(sidecar_path)
    target_index_path = root / "cache" / "valid_targets.npz"
    with np.load(target_index_path, allow_pickle=False) as target_index:
        expected_groups = [int(group) for group in target_index["row_group"]]
    _require(len(expected_groups) == 1873, "valid target index no longer has 1,873 rows")
    _require(
        expected_groups == sorted(set(expected_groups)),
        "valid target row groups are not unique and ordered",
    )
    expected_shape = (1873, 20000, 2)
    expected_bytes = 299_680_128
    prediction_sha256 = validate_prediction_payload(
        prediction_path,
        metadata,
        run=contract["run"],
        epoch=1,
        checkpoint_sha256=checkpoint_sha256,
        split_manifest_sha256=contract["fit_recipe_pins"]["split_manifest_sha256"],
        expected_row_groups=expected_groups,
        expected_shape=expected_shape,
        expected_bytes=expected_bytes,
    )
    return {
        "schema": "wnn.oldz4.student-full-valid-export-validation.r5",
        "status": "pass",
        "run": contract["run"],
        "checkpoint_epoch": 1,
        "checkpoint_sha256": checkpoint_sha256,
        "prediction_path": str(prediction_path),
        "prediction_sha256": prediction_sha256,
        "prediction_bytes": expected_bytes,
        "prediction_shape": list(expected_shape),
        "prediction_dtype": "float32",
        "row_groups": len(expected_groups),
        "row_groups_sha256": hashlib.sha256(
            np.asarray(expected_groups, dtype="<i8").tobytes()
        ).hexdigest(),
        "sidecar_sha256": sha256_file(sidecar_path),
        "split_manifest_sha256": contract["fit_recipe_pins"]["split_manifest_sha256"],
        "finite_values": True,
        "exposure": contract["export"]["exposure"],
    }


def write_once(path: Path, value: dict[str, Any]) -> None:
    """Write a receipt without replacing evidence from an earlier attempt."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    with path.open("x", encoding="utf-8") as handle:
        handle.write(payload)


def main() -> None:
    """Create a CP1 terminal gate or validate an already-completed full-valid export."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["checkpoint-gate", "export-gate"])
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--receipt-out", type=Path, required=True)
    parser.add_argument("--checkpoint-gate", type=Path)
    args = parser.parse_args()
    if args.action == "checkpoint-gate":
        receipt = validate_checkpoint(args.project_root)
    else:
        if args.checkpoint_gate is None:
            parser.error("export-gate requires --checkpoint-gate")
        receipt = validate_export(args.project_root, args.checkpoint_gate)
    write_once(args.receipt_out, receipt)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    LOGGER.info(json.dumps({"status": receipt["status"], "receipt": str(args.receipt_out)}))


if __name__ == "__main__":
    main()
