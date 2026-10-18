"""Delivery-only validation and command construction for completed PR15 fits.

This adapter intentionally has no fit command. It validates existing checkpoints
and produces only the pinned export and CPU score commands for an external
root-owned runner. It writes no files and never edits the sealed PR15 packet.
"""

from __future__ import annotations

import importlib.util
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

from checkpoint_identity import (
    validate_prediction_pin,
    validate_provenance_file_hashes,
    verify_checkpoint_pair,
)

PARENT_PROFILE_PACKET_SHA256 = "c18f1877b8579c03d2a0905534f5ecbd13ed56a54306ba66d63b47d892b66510"
PARENT_FIT_CONTRACT_SHA256 = "05aa087af1fdd68ac97146e6f56750157d4a33ebd4fce2a5ec2f5eb30cc6bcb8"
PINNED_SOURCE_MANIFEST_SHA256 = "873662aded939afd64c17c80b684011de023c04a31be5b2c74e0b9d97a18efc0"
PINNED_SCORE_SUPPORT_SHA256 = "698ae26944b96ffb1b2b2ccb2de11cab427d3c822c2ca74cdacff1b1ef7d6368"
PINNED_SCORE_HELPER_SHA256 = "fb7a85a5837f61fa6c65cd10a799768006067e87d4c18e92d4b813ba83b9d82e"
NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")


def _load_module(path: Path, name: str) -> Any:
    """Load one unchanged host-packet module by its exact local path."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load sealed host module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def verify_frozen_packet(packet_dir: Path) -> dict[str, Any]:
    """Verify the v2 packet members and their original profile ancestry."""
    packet_dir = packet_dir.resolve(strict=True)
    frozen_path = packet_dir / "FROZEN.json"
    contract_path = packet_dir / "FIT_CONTRACT.json"
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    if (
        frozen.get("schema") != "tabdata-pr15-delivery-repair-freeze.v2"
        or frozen.get("parent_profile_packet_sha256") != PARENT_PROFILE_PACKET_SHA256
        or frozen.get("parent_fit_contract_sha256") != PARENT_FIT_CONTRACT_SHA256
        or not isinstance(frozen.get("parent_execution_manifest_sha256"), str)
        or not isinstance(frozen.get("files"), dict)
    ):
        raise ValueError("v2 freeze does not bind the original PR15 profile packet")
    expected_names = set(frozen["files"]) | {"FROZEN.json", "EXECUTION_MANIFEST.json"}
    actual_names = {path.name for path in packet_dir.iterdir() if path.is_file()}
    if actual_names != expected_names:
        raise ValueError("v2 packet contains missing or unmanifested files")
    for relative, expected in frozen["files"].items():
        if Path(relative).name != relative:
            raise ValueError(f"unsafe v2 freeze member path: {relative}")
        member = packet_dir / relative
        if _sha256(member) != expected:
            raise ValueError(f"v2 frozen PR15 packet member changed: {relative}")
    parent_path = packet_dir / "PARENT_PACKET.json"
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    if (
        parent.get("frozen_sha256") != PARENT_PROFILE_PACKET_SHA256
        or parent.get("fit_contract_sha256") != PARENT_FIT_CONTRACT_SHA256
        or parent.get("execution_manifest_sha256") != frozen.get("parent_execution_manifest_sha256")
        or not isinstance(parent.get("files"), dict)
    ):
        raise ValueError("v2 parent file map changed the original packet identities")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if (
        contract.get("schema") != "tabdata-pr15-fit-contract.v2"
        or contract.get("profile_packet_sha256") != PARENT_PROFILE_PACKET_SHA256
        or contract.get("parent_fit_contract_sha256") != PARENT_FIT_CONTRACT_SHA256
        or contract.get("files")
        != {
            "fit_validation.py": _sha256(packet_dir / "fit_validation.py"),
            "checkpoint_identity.py": _sha256(packet_dir / "checkpoint_identity.py"),
        }
    ):
        raise ValueError("v2 fit contract does not bind its original parent")
    manifest_path = packet_dir / "EXECUTION_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_hashes = {
        "FROZEN.json": _sha256(frozen_path),
        "FIT_CONTRACT.json": _sha256(contract_path),
        "fit_validation.py": _sha256(packet_dir / "fit_validation.py"),
        "checkpoint_identity.py": _sha256(packet_dir / "checkpoint_identity.py"),
        "PARENT_PACKET.json": _sha256(parent_path),
    }
    if (
        manifest.get("schema") != "tabdata-pr15-delivery-repair-manifest.v2"
        or manifest.get("parent_profile_packet_sha256") != PARENT_PROFILE_PACKET_SHA256
        or manifest.get("files") != expected_hashes
    ):
        raise ValueError("v2 execution manifest does not match frozen members")
    return {
        "frozen_manifest_sha256": _sha256(frozen_path),
        "fit_contract_sha256": _sha256(contract_path),
        "fit_validation_sha256": expected_hashes["fit_validation.py"],
        "frozen_member_count": len(frozen["files"]),
    }


def _verify_scorer_copies(packet_dir: Path, source_root: Path, work: Path) -> dict[str, str]:
    """Bind the workspace scorer copies to the pinned PR15 source and helper."""
    source_manifest = json.loads((packet_dir / "source_manifest.json").read_text(encoding="utf-8"))
    if _sha256(packet_dir / "source_manifest.json") != PINNED_SOURCE_MANIFEST_SHA256:
        raise ValueError("PR15 source manifest differs from the frozen source pin")
    source_hashes = {row["path"]: row["sha256"] for row in source_manifest["files"]}
    source_scorer_hash = source_hashes.get("scripts/score_run.py")
    if source_scorer_hash is None:
        raise ValueError("pinned source manifest omits the native scorer")
    support_path = packet_dir / "score_support_manifest.json"
    support = json.loads(support_path.read_text(encoding="utf-8"))
    helper_rows = support.get("files", [])
    helper_hash = helper_rows[0].get("sha256") if len(helper_rows) == 1 else None
    if (
        _sha256(support_path) != PINNED_SCORE_SUPPORT_SHA256
        or support.get("source_commit") != "e238fc5c82ac8c4591f3ca3ef87c37b2b73b929e"
        or helper_hash != PINNED_SCORE_HELPER_SHA256
    ):
        raise ValueError("source-native scorer support pin is invalid")
    source_scorer = source_root / "scripts" / "score_run.py"
    if _sha256(source_scorer) != source_scorer_hash:
        raise ValueError("deployed native scorer differs from the source manifest")
    copies = {
        "score_run.py": _sha256(work / "scripts" / "score_run.py"),
        "grade_prediction_bank.py": _sha256(work / "scripts" / "grade_prediction_bank.py"),
    }
    expected = {
        "score_run.py": source_scorer_hash,
        "grade_prediction_bank.py": PINNED_SCORE_HELPER_SHA256,
    }
    if copies != expected:
        raise ValueError("workspace source-native scoring files differ from pinned bytes")
    return expected


def _require_fit_record(fit_validation: dict[str, Any], work: Path, name: str) -> dict[str, Any]:
    """Bind delivery-only phases to a completed strict fit-validation result."""
    request_path = work / "worker_request.json"
    failed_path = work / "worker_receipt.json"
    envelope_path = work / "envelope" / "fit" / "train_cell_resource_receipt.json"
    log_path = work / "logs" / "fit.log"
    if (
        fit_validation.get("schema") != "pr15-delivery-repair-fit-validation.v1"
        or fit_validation.get("status") != "existing_fit_validated_for_delivery"
        or fit_validation.get("name") != name
        or fit_validation.get("old_failure_preserved") is not True
        or fit_validation.get("request_sha256") != _sha256(request_path)
        or fit_validation.get("old_failed_worker_receipt_sha256") != _sha256(failed_path)
    ):
        raise ValueError("delivery phase lacks the strict validation for this fit")
    evidence = fit_validation.get("fit_actor_evidence", {})
    if evidence.get("resource_receipt_sha256") != _sha256(envelope_path) or evidence.get(
        "log_sha256"
    ) != _sha256(log_path):
        raise ValueError("fit execution evidence changed after strict validation")
    identity = fit_validation.get("checkpoint_identity")
    if not isinstance(identity, dict):
        raise ValueError("fit validation has no checkpoint identity receipt")
    run_dir = work / "artifacts" / "runs" / name
    current = verify_checkpoint_pair(run_dir / "model_ep2.pt", run_dir / "model.pt")
    provenance = json.loads((run_dir / "provenance.json").read_text(encoding="utf-8"))
    validate_provenance_file_hashes(provenance, current)
    if current.receipt() != identity:
        raise ValueError("checkpoint or provenance changed after strict fit validation")
    return identity


def validate_completed_fit(
    *,
    packet_dir: Path,
    root: Path,
    work: Path,
    name: str,
    fold: int,
    seed: int,
    control: str,
    gpu_uuid: str,
    expected_cpus: list[int],
) -> dict[str, Any]:
    """Validate a finished fit while preserving the original failed receipt.

    The old worker did not persist its in-memory successful phase record because
    its later raw-archive comparison failed. This routine records the surviving
    actor envelope and log as evidence, without inventing a missing exit code or
    phase timestamps.
    """
    if (
        not NAME_PATTERN.fullmatch(name)
        or type(fold) is not int
        or not 0 <= fold < 4
        or type(seed) is not int
        or not gpu_uuid
        or not isinstance(expected_cpus, list)
        or not expected_cpus
        or any(type(cpu) is not int or cpu < 0 for cpu in expected_cpus)
        or len(expected_cpus) != len(set(expected_cpus))
    ):
        raise ValueError("invalid PR15 fit identity or resource receipt expectation")
    packet_dir = packet_dir.resolve(strict=True)
    root = root.resolve(strict=True)
    work = work.resolve(strict=True)
    packet_identity = verify_frozen_packet(packet_dir)
    source_root = (root / "runtime" / "pr15-source-e238-v1").resolve(strict=True)
    request_path = work / "worker_request.json"
    failed_path = work / "worker_receipt.json"
    split_path = (root / "splits" / f"fold{fold}.json").resolve(strict=True)
    request = json.loads(request_path.read_text(encoding="utf-8"))
    old_failure = json.loads(failed_path.read_text(encoding="utf-8"))
    split_hash = _sha256(split_path)

    if control not in {"all", "moments_only", "dictionary_only", "none"}:
        raise ValueError("invalid PR15 fold or control")
    maximum_seed = 1008 if fold < 2 else 1007
    if seed > maximum_seed or seed < 1001:
        raise ValueError("seed is outside the original PR15 fold panel")
    if (
        old_failure.get("schema") != "pr15-host-worker-receipt.v1"
        or old_failure.get("status") != "failed"
        or old_failure.get("name") != name
        or old_failure.get("fold") != fold
        or old_failure.get("seed") != seed
        or old_failure.get("control") != control
        or "selected model.pt" not in str(old_failure.get("error", ""))
    ):
        raise ValueError("original failed worker receipt is not this serialization failure")

    validator = _load_module(packet_dir / "validation.py", "_pr15_repair_validation")
    source_identity = validator.verify_source_tree(source_root, packet_dir / "source_manifest.json")
    scorer_identity = _verify_scorer_copies(packet_dir, source_root, work)
    split = validator.read_json(split_path)
    if split.get("fold") != fold or split.get("n_folds") != 4:
        raise ValueError("canonical fold manifest has unexpected membership")
    if (
        request.get("schema") != "pr15-host-worker-request.v1"
        or request.get("mode") != "fit"
        or request.get("name") != name
        or Path(request.get("work", "")).resolve(strict=True) != work
        or Path(request.get("source_root", "")).resolve(strict=True) != source_root
    ):
        raise ValueError("saved worker request does not bind the existing fit")

    fit_validator_path = packet_dir / "fit_validation.py"
    fit_validator = _load_module(fit_validator_path, "_pr15_repair_fit_validation")
    canonical_train = fit_validator._canonical_argv(
        source_root,
        name,
        fold,
        seed,
        control,
        split_path,
    )
    canonical_predict = fit_validator._canonical_argv(
        source_root,
        name,
        fold,
        seed,
        control,
        split_path,
        "predict",
    )
    if (
        request.get("train_argv") != canonical_train
        or request.get("predict_argv") != canonical_predict
    ):
        raise ValueError("saved request differs from the exact pinned PR15 commands")

    # Validate exact serialization-independent payload identity first. Then
    # validate each raw archive hash against the trainer's provenance separately.
    run_dir = work / "artifacts" / "runs" / name
    fixed_path = run_dir / "model_ep2.pt"
    selected_path = run_dir / "model.pt"
    identity = verify_checkpoint_pair(fixed_path, selected_path)
    provenance = json.loads((run_dir / "provenance.json").read_text(encoding="utf-8"))
    validate_provenance_file_hashes(provenance, identity)

    # Reuse the corrected v2 full recipe/source/update validator. Its checkpoint
    # gate compares semantic payload identity and always hashes actual file bytes.
    contract = fit_validator._check_contract()
    fit_record = fit_validator._check_checkpoint(
        work,
        name,
        fold,
        seed,
        control,
        split_hash,
        source_root,
        contract,
    )
    if fit_record.get("checkpoint_identity") != identity.receipt():
        raise ValueError("corrected fit validator changed checkpoint identity")

    envelope_path = work / "envelope" / "fit" / "train_cell_resource_receipt.json"
    fit_log_path = work / "logs" / "fit.log"
    envelope = json.loads(envelope_path.read_text(encoding="utf-8"))
    if (
        envelope.get("schema") != "tabdata-b300-process-envelope.v1"
        or type(envelope.get("pid")) is not int
        or envelope["pid"] <= 0
        or envelope.get("script") != "train_cell.py"
        or envelope.get("status") != "actor_returned"
        or envelope.get("visible_devices") != gpu_uuid
        or envelope.get("cpu_affinity") != expected_cpus
        or envelope.get("allocator_budget_bytes") != 6 * 1024**3
        or envelope.get("allocator_limit_is_not_a_device_memory_reservation") is not True
        or "B300" not in str(envelope.get("gpu", ""))
    ):
        raise ValueError("surviving fit actor envelope differs from assigned resources")
    elapsed = envelope.get("elapsed_seconds")
    if (
        isinstance(elapsed, bool)
        or not isinstance(elapsed, (int, float))
        or not math.isfinite(elapsed)
        or elapsed <= 0
    ):
        raise ValueError("fit actor envelope lacks valid elapsed-time evidence")
    for key in ("cuda_peak_allocated_bytes", "cuda_peak_reserved_bytes"):
        value = envelope.get(key)
        if type(value) is not int or not 0 <= value <= 6 * 1024**3:
            raise ValueError(f"fit actor exceeded or omitted allocator evidence: {key}")
    log_bytes = fit_log_path.read_bytes()
    if not log_bytes:
        raise ValueError("fit actor log is empty")
    envelope_lines = [
        line.removeprefix("PROCESS_ENVELOPE ")
        for line in log_bytes.decode("utf-8", errors="replace").splitlines()
        if line.startswith("PROCESS_ENVELOPE ")
    ]
    if len(envelope_lines) != 1:
        raise ValueError("fit log does not contain one process-envelope record")
    logged_envelope = json.loads(envelope_lines[0])
    for key in (
        "schema",
        "pid",
        "script",
        "torch",
        "cuda",
        "gpu",
        "visible_devices",
        "cpu_affinity",
        "torch_threads",
        "allocator_budget_bytes",
    ):
        if logged_envelope.get(key) != envelope.get(key):
            raise ValueError(f"fit log and resource envelope disagree on {key}")

    return {
        "schema": "pr15-delivery-repair-fit-validation.v1",
        "status": "existing_fit_validated_for_delivery",
        "name": name,
        "fold": fold,
        "seed": seed,
        "control": control,
        "source": source_identity,
        "packet": packet_identity,
        "scorer_files": scorer_identity,
        "split_manifest_sha256": split_hash,
        "request_sha256": _sha256(request_path),
        "old_failed_worker_receipt_sha256": _sha256(failed_path),
        "old_failure_preserved": True,
        "fit_actor_evidence": {
            "kind": "existing actor-returned resource envelope plus hashed fit log",
            "actor_status": "actor_returned",
            "outer_phase_exit_code": None,
            "outer_phase_timestamps": None,
            "log_sha256": _sha256(fit_log_path),
            "resource_receipt_sha256": _sha256(envelope_path),
        },
        "checkpoint_identity": identity.receipt(),
        "fit_contract": fit_record,
    }


def delivery_commands(
    *,
    packet_dir: Path,
    work: Path,
    name: str,
    split_path: Path,
    fit_validation: dict[str, Any],
) -> dict[str, Any]:
    """Return only the existing export actor and CPU score commands; never fit."""
    packet_dir = packet_dir.resolve(strict=True)
    work = work.resolve(strict=True)
    split_path = split_path.resolve(strict=True)
    verify_frozen_packet(packet_dir)
    _require_fit_record(fit_validation, work, name)
    request = work / "worker_request.json"
    if not request.is_file():
        raise FileNotFoundError(request)
    request_value = json.loads(request.read_text(encoding="utf-8"))
    if (
        request_value.get("schema") != "pr15-host-worker-request.v1"
        or request_value.get("mode") != "fit"
        or request_value.get("name") != name
        or not isinstance(request_value.get("predict_argv"), list)
        or not isinstance(request_value.get("train_argv"), list)
    ):
        raise ValueError("export request is not the saved PR15 fit request")
    source_root = Path(request_value["source_root"]).resolve(strict=True)
    source_manifest = packet_dir / "source_manifest.json"
    validation = _load_module(packet_dir / "validation.py", "_pr15_repair_command_validation")
    validation.verify_source_tree(source_root, source_manifest)
    _verify_scorer_copies(packet_dir, source_root, work)
    score_script = work / "scripts" / "score_run.py"
    if not score_script.is_file():
        raise FileNotFoundError("pinned workspace score_run.py is missing")
    export = [
        sys.executable,
        "-u",
        str(packet_dir / "gpu_entrypoint.py"),
        str(packet_dir / "export_cell.py"),
        "--request",
        str(request),
    ]
    score = [
        sys.executable,
        "-u",
        str(score_script),
        "--run",
        name,
        "--split",
        str(split_path),
        "--epoch",
        "2",
    ]
    if any("train_rnn.py" in token or "train_cell.py" in token for token in export + score):
        raise AssertionError("delivery repair must never construct a fit command")
    return {
        "export": export,
        "export_pythonpath_prefix": str(source_root / "scripts"),
        "score": score,
    }


def validate_exported_delivery(
    *,
    packet_dir: Path,
    work: Path,
    name: str,
    split_path: Path,
    fit_validation: dict[str, Any],
) -> dict[str, Any]:
    """Check exact prediction pin and run the unchanged official PR15 score gate."""
    packet_dir = packet_dir.resolve(strict=True)
    work = work.resolve(strict=True)
    split_path = split_path.resolve(strict=True)
    packet_identity = verify_frozen_packet(packet_dir)
    checkpoint_identity = _require_fit_record(fit_validation, work, name)
    validator = _load_module(packet_dir / "validation.py", "_pr15_repair_export_validation")
    request = json.loads((work / "worker_request.json").read_text(encoding="utf-8"))
    if (
        request.get("schema") != "pr15-host-worker-request.v1"
        or request.get("mode") != "fit"
        or request.get("name") != name
    ):
        raise ValueError("exported artifacts do not belong to this saved fit request")
    source_root = Path(request["source_root"]).resolve(strict=True)
    validator.verify_source_tree(source_root, packet_dir / "source_manifest.json")
    scorer_identity = _verify_scorer_copies(packet_dir, source_root, work)
    run_dir = work / "artifacts" / "runs" / name
    identity = verify_checkpoint_pair(run_dir / "model_ep2.pt", run_dir / "model.pt")
    provenance = json.loads((run_dir / "provenance.json").read_text(encoding="utf-8"))
    validate_provenance_file_hashes(provenance, identity)
    if identity.receipt() != checkpoint_identity:
        raise ValueError("checkpoint identity changed after fit validation")
    split = validator.read_json(split_path)
    split_hash = _sha256(split_path)
    prediction = work / "cache" / f"pred_{name}_valid.npy"
    sidecar_path = prediction.with_suffix(".json")
    sidecar = validator.read_json(sidecar_path)
    pin = validate_prediction_pin(
        sidecar,
        prediction,
        identity,
        name=name,
        split_manifest_sha256=split_hash,
        evaluation_groups=split["evaluation"]["valid"],
    )
    source_score = validator.validate_prediction_and_score(work, name, split, split_hash)
    fit_validator = _load_module(
        packet_dir / "fit_validation.py", "_pr15_repair_export_fit_validation"
    )
    fold = fit_validation.get("fold")
    if type(fold) is not int or fold not in fit_validator.EXPECTED_UPDATES:
        raise ValueError("strict fit validation record has an invalid fold")
    score = fit_validator._check_export_and_score(
        work,
        name,
        split_hash,
        identity.fixed_file_sha256,
        split["evaluation"]["valid"],
        fit_validator.EXPECTED_UPDATES[fold],
        {"score": source_score},
    )
    return {
        "schema": "pr15-delivery-repair-export-score.v1",
        "status": "evaluation_export_and_source_native_grade_validated",
        "name": name,
        "packet": packet_identity,
        "scorer_files": scorer_identity,
        "split_manifest_sha256": split_hash,
        "prediction_pin": pin,
        "prediction_sha256": _sha256(prediction),
        "sidecar_sha256": _sha256(sidecar_path),
        "source_native_validation": source_score,
        "score": score,
    }


def _sha256(path: Path) -> str:
    """Hash a file with bounded memory."""
    from checkpoint_identity import sha256_file

    return sha256_file(path)
