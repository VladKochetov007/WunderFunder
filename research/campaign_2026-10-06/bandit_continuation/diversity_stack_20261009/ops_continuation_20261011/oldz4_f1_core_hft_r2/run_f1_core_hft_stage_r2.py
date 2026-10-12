from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import Any

PACKET_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = Path("/home/vlad/development/wunder/wnn_connectome_starterpack")
OPS_ROOT = PROJECT_ROOT / (
    "research/campaign_2026-10-06/bandit_continuation/"
    "diversity_stack_20261009/ops_continuation_20261011"
)
R4_PACKET = OPS_ROOT / "old_z4_f1_pipeline_successor_20261012_r4"
R4_INPUTS = R4_PACKET / "INPUT_BINDINGS_R4.json"
R4_SOURCES = R4_PACKET / "SOURCE_BINDINGS_R4.json"
R4_STUDENT_FIT = R4_PACKET / "contracts/oldz4-r4-f1-s3-student-fit.json"
R5_CROP_PACKET = OPS_ROOT / "oldz4_student_core_hft_f0_r5"
R5_CROP_BINDINGS = R5_CROP_PACKET / "SOURCE_BINDINGS_R5.json"
R5_CROP_VERIFIER = R5_CROP_PACKET / "prepare_core_crop_r5.py"
R6_CROP_PACKET = OPS_ROOT / "oldz4_student_core_hft_f0_r6"
R6_CROP_RETRY = R6_CROP_PACKET / "CORE_CROP_RETRY_CONTRACT_R6.json"
R6_CROP_CONSUMER = R6_CROP_PACKET / "prepare_core_crop_r6.py"
INPUT_BINDINGS_PATH = PACKET_ROOT / "SOURCE_INPUT_BINDINGS_R2.json"
CORE_CONTRACT_PATH = PACKET_ROOT / "CORE_CROP_CONTRACT_R2.json"
CORE_RECEIPT_PATH = PACKET_ROOT / "CORE_CROP_PRODUCER_RECEIPT_R2.json"
PROFILE_CONTRACT_PATH = PACKET_ROOT / "HFT_PROFILE_CONTRACT_R2.json"
FIT_CONTRACT_PATH = PACKET_ROOT / "HFT_FIT_CONTRACT_R2.json"
EXPORT_CONTRACT_PATH = PACKET_ROOT / "HFT_EXPORT_CONTRACT_R2.json"


class _StopBeforeDevice(RuntimeError):
    """Stop actual trainer CLI validation before device or fitting work."""


def sha256_file(path: Path) -> str:
    """Hash a file incrementally to keep artifact checks bounded in memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_object(path: Path) -> dict[str, Any]:
    """Read one JSON object and reject malformed packet or producer receipts."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _import_from_path(path: Path, module_name: str) -> ModuleType:
    """Import one exact frozen producer module without writing bytecode."""
    if os.environ.get("PYTHONDONTWRITEBYTECODE") != "1":
        raise ValueError("set PYTHONDONTWRITEBYTECODE=1 before frozen-source imports")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import pinned module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _verify_file_manifest(
    root: Path,
    files: dict[str, str],
    label: str,
    supplemental_files: dict[str, str] | None = None,
) -> None:
    """Require the full manifest plus any separately pinned source extras."""
    supplemental_files = supplemental_files or {}
    if set(files).intersection(supplemental_files):
        raise ValueError(f"{label} supplemental paths overlap the frozen manifest")
    actual = sorted(path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file())
    expected_paths = sorted(set(files).union(supplemental_files))
    if actual != expected_paths:
        raise ValueError(f"{label} source file list differs from its frozen manifest")
    for relative, expected in {**files, **supplemental_files}.items():
        if sha256_file(root / relative) != expected:
            raise ValueError(f"{label} source bytes changed: {relative}")


def _project_path(path: str | Path) -> Path:
    """Resolve source-manifest project-relative paths independently of cwd."""
    candidate = Path(path)
    return candidate.resolve() if candidate.is_absolute() else (PROJECT_ROOT / candidate).resolve()


def verify_packet_bindings() -> dict[str, Any]:
    """Check the exact F1 inputs and unchanged F0 crop/HFT source primitives."""
    sealed = read_object(INPUT_BINDINGS_PATH)
    r4_inputs_sha = sha256_file(R4_INPUTS)
    r4_sources_sha = sha256_file(R4_SOURCES)
    r4_fit_sha = sha256_file(R4_STUDENT_FIT)
    r5_bindings_sha = sha256_file(R5_CROP_BINDINGS)
    r6_retry_sha = sha256_file(R6_CROP_RETRY)
    r6_consumer_sha = sha256_file(R6_CROP_CONSUMER)
    r5_verifier_sha = sha256_file(R5_CROP_VERIFIER)
    r4_pins = sealed["r4"]
    if (
        r4_pins.get("input_bindings_path") != str(R4_INPUTS.resolve())
        or r4_pins.get("input_bindings_sha256") != r4_inputs_sha
    ):
        raise ValueError("F1 R4 input bindings changed after this packet was prepared")
    if (
        r4_pins.get("source_bindings_path") != str(R4_SOURCES.resolve())
        or r4_pins.get("source_bindings_sha256") != r4_sources_sha
    ):
        raise ValueError("F1 R4 source bindings changed after this packet was prepared")
    if (
        r4_pins.get("student_fit_contract_path") != str(R4_STUDENT_FIT.resolve())
        or r4_pins.get("student_fit_contract_sha256") != r4_fit_sha
    ):
        raise ValueError("F1 student recipe changed after this packet was prepared")
    crop_pins = sealed["core_crop"]
    if crop_pins.get("r5_source_bindings_path") != str(R5_CROP_BINDINGS.resolve()):
        raise ValueError("F0 crop source-binding path differs from the frozen packet")
    if crop_pins.get("r5_source_bindings_sha256") != r5_bindings_sha:
        raise ValueError("validated F0 crop source bindings changed")
    if crop_pins.get("r6_retry_contract_sha256") != r6_retry_sha:
        raise ValueError("corrected F0 R6 crop retry contract changed")
    if crop_pins.get("r6_corrected_consumer_sha256") != r6_consumer_sha:
        raise ValueError("corrected F0 R6 crop consumer changed")
    if crop_pins.get("r5_actual_output_verifier_sha256") != r5_verifier_sha:
        raise ValueError("F0 crop actual-output verifier changed")

    r4_sources = read_object(R4_SOURCES)
    source_manifest_path = Path(r4_sources["r3_source_snapshot_bindings_path"])
    if sha256_file(source_manifest_path) != r4_sources["r3_source_snapshot_bindings_sha256"]:
        raise ValueError("F1 c337 source snapshot manifest hash mismatch")
    source_manifest = read_object(source_manifest_path)
    hft = source_manifest["hft"]
    hft_root = _project_path(hft["snapshot_root"])
    c337_pins = sealed["c337_source"]
    if c337_pins.get("files") != hft["files"]:
        raise ValueError("F1 c337 source file pins differ from the original snapshot manifest")
    supplemental_files = c337_pins.get("supplemental_unmanifested_files", {})
    if not isinstance(supplemental_files, dict):
        raise TypeError("F1 supplemental c337 source pins must be a path-to-SHA map")
    _verify_file_manifest(
        hft_root,
        hft["files"],
        "c337 HFT/predictor",
        supplemental_files,
    )
    student_pins = sealed.get("f1_student_cp1")
    if (
        not isinstance(student_pins, dict)
        or student_pins.get("run") != "oldz4_h230_student_f1_s3_r4"
    ):
        raise ValueError("F1 source bindings do not identify the reviewed student CP1")
    review_path = Path(student_pins.get("root_consumer_review_path", ""))
    if (
        not review_path.is_absolute()
        or not review_path.is_file()
        or sha256_file(review_path) != student_pins.get("root_consumer_review_sha256")
    ):
        raise ValueError("root CP1 consumer review differs from the sealed source binding")
    for artifact_name, relative_path in {
        "checkpoint": "artifacts/runs/oldz4_h230_student_f1_s3_r4/model_ep1.pt",
        "provenance": "artifacts/runs/oldz4_h230_student_f1_s3_r4/provenance.json",
        "training_state": "artifacts/runs/oldz4_h230_student_f1_s3_r4/training_state.pt",
    }.items():
        artifact_path = PROJECT_ROOT / relative_path
        expected_path = str(artifact_path.resolve())
        if student_pins.get("paths", {}).get(artifact_name) != expected_path:
            raise ValueError(f"F1 student CP1 path binding differs at {artifact_name}")
        if sha256_file(artifact_path) != student_pins.get("sha256", {}).get(artifact_name):
            raise ValueError(f"F1 student CP1 bytes differ at {artifact_name}")
    if hft["files"].get("scripts/train_rnn.py") != r4_sources["hft_snapshot"]["trainer_sha256"]:
        raise ValueError("F1 trainer digest disagrees with the reviewed R4 source binding")
    if hft["files"].get("scripts/predict_run.py") != r4_sources["hft_snapshot"]["predictor_sha256"]:
        raise ValueError("F1 predictor digest disagrees with the reviewed R4 source binding")

    crop_bindings = read_object(R5_CROP_BINDINGS)
    crop = crop_bindings["snapshots"]["core_crop"]
    crop_root = R5_CROP_PACKET / crop["root_relative_path"]
    _verify_file_manifest(crop_root, crop["files"], "F0 crop primitive")
    crop_source_sha = crop["files"]["scripts/make_core_checkpoint.py"]
    if crop_source_sha != crop_pins.get("r5_crop_source_sha256"):
        raise ValueError("F0 crop primitive differs from the prepared F1 contract")
    return {
        "r4_input_bindings_sha256": r4_inputs_sha,
        "r4_source_bindings_sha256": r4_sources_sha,
        "r4_student_fit_contract_sha256": r4_fit_sha,
        "r3_source_snapshot_bindings_sha256": r4_sources["r3_source_snapshot_bindings_sha256"],
        "trainer_sha256": hft["files"]["scripts/train_rnn.py"],
        "predictor_sha256": hft["files"]["scripts/predict_run.py"],
        "supplemental_c337_source_files": supplemental_files,
        "f0_crop_source_bindings_sha256": r5_bindings_sha,
        "f0_crop_source_sha256": crop_source_sha,
        "f0_crop_retry_contract_sha256": r6_retry_sha,
        "f0_crop_corrected_consumer_sha256": r6_consumer_sha,
        "f0_crop_actual_output_verifier_sha256": r5_verifier_sha,
    }


def _require_environment() -> None:
    """Reject accidental loader/thread changes from the frozen zero-worker profile."""
    expected = {
        "WNN_LOADER_WORKERS": "0",
        "WNN_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "PYTHONHASHSEED": "0",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    mismatches = {
        key: os.environ.get(key) for key, value in expected.items() if os.environ.get(key) != value
    }
    if mismatches:
        raise ValueError(f"bounded environment does not match the recipe: {mismatches}")


def import_trainer() -> tuple[ModuleType, Path]:
    """Import the sealed c337 trainer, then bind its real module ROOT to MAIN."""
    bindings = verify_packet_bindings()
    r4_sources = read_object(R4_SOURCES)
    manifest = read_object(Path(r4_sources["r3_source_snapshot_bindings_path"]))
    hft_root = _project_path(manifest["hft"]["snapshot_root"])
    trainer_path = hft_root / "scripts/train_rnn.py"
    if sha256_file(trainer_path) != bindings["trainer_sha256"]:
        raise ValueError("actual imported trainer differs from the sealed F1 source")
    for candidate in (hft_root / "src", hft_root / "scripts"):
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
    existing = sys.modules.get("train_rnn")
    if existing is not None:
        if Path(existing.__file__).resolve() != trainer_path.resolve():
            raise RuntimeError("a different train_rnn module was already imported")
        trainer = existing
    else:
        trainer = _import_from_path(trainer_path, "train_rnn")
    trainer.ROOT = PROJECT_ROOT.resolve()
    sequence_path = Path(trainer.Sequences("valid", []).path).resolve()
    expected_path = PROJECT_ROOT.resolve() / "datasets/valid.parquet"
    if sequence_path != expected_path:
        raise RuntimeError(
            f"actual c337 Sequences path is {sequence_path}, expected {expected_path}"
        )
    return trainer, trainer_path


def import_predictor(trainer: ModuleType) -> tuple[ModuleType, Path]:
    """Import frozen c337 prediction code and bind both actual module roots."""
    bindings = verify_packet_bindings()
    r4_sources = read_object(R4_SOURCES)
    manifest = read_object(Path(r4_sources["r3_source_snapshot_bindings_path"]))
    hft_root = _project_path(manifest["hft"]["snapshot_root"])
    predictor_path = hft_root / "scripts/predict_run.py"
    if sha256_file(predictor_path) != bindings["predictor_sha256"]:
        raise ValueError("actual imported predictor differs from the sealed F1 source")
    predictor = _import_from_path(predictor_path, "predict_run_f1_hft_r2")
    predictor.ROOT = PROJECT_ROOT.resolve()
    predictor.train_rnn.ROOT = PROJECT_ROOT.resolve()
    if predictor.train_rnn is not trainer:
        raise RuntimeError("predictor did not reuse the exact pinned c337 trainer module")
    sequence_path = Path(predictor.Sequences("valid", []).path).resolve()
    expected_path = PROJECT_ROOT.resolve() / "datasets/valid.parquet"
    if sequence_path != expected_path:
        raise RuntimeError(
            f"actual predictor Sequences path is {sequence_path}, expected {expected_path}"
        )
    return predictor, predictor_path


def validate_inputs(trainer: ModuleType) -> dict[str, Any]:
    """Validate F1 source-qualified membership through the actual c337 split reader."""
    inputs = read_object(R4_INPUTS)
    fold = inputs["fold"]
    split_info = fold["source_qualified_fit_membership"]
    split_path = PROJECT_ROOT / fold["split_manifest"]
    if sha256_file(split_path) != fold["split_manifest_sha256"]:
        raise ValueError("F1 split manifest differs from the reviewed R4 source binding")
    split = trainer.load_campaign_split(split_path, PROJECT_ROOT)
    if split.fold != 1 or split.n_folds != 4:
        raise ValueError("actual c337 split reader returned an unexpected F1 fold")
    observed = {
        "train": len(split.train_groups),
        "valid_fit": len(split.valid_groups),
        "valid_evaluation": len(split.evaluation_groups),
    }
    expected = {
        "train": split_info["train.parquet"]["count"],
        "valid_fit": split_info["valid.parquet"]["count"],
        "valid_evaluation": split_info["evaluation_valid_parquet"]["count"],
    }
    if observed != expected or observed != {
        "train": 10533,
        "valid_fit": 1416,
        "valid_evaluation": 457,
    }:
        raise ValueError(f"actual F1 membership counts differ: {observed}")
    if len(set(split.valid_groups) & set(split.evaluation_groups)) != 0:
        raise ValueError("F1 fitting and evaluation validation groups overlap")
    split.assert_fit_groups(split.train_groups, split.valid_groups)
    return {
        "split_manifest_sha256": split.sha256,
        "fold": split.fold,
        "counts": observed,
        "source_qualified": True,
        "component_disjoint": True,
        "outer_evaluation_fit_overlap": 0,
        "full_parquet_rehashed": False,
    }


def verify_student_cp1() -> tuple[Path, Path, str, str, dict[str, Any]]:
    """Require a root-validated F1 student CP1 and its exact fit provenance."""
    contract = read_object(R4_STUDENT_FIT)
    student_pins = read_object(INPUT_BINDINGS_PATH)["f1_student_cp1"]
    gate_path = PACKET_ROOT / "ROOT_F1_STUDENT_CP1_TERMINAL_GATE_R2.json"
    gate = read_object(gate_path)
    fit_sha = sha256_file(R4_STUDENT_FIT)
    if (
        gate.get("schema") != "wnn.root.f1-student-cp1-terminal-gate.r1"
        or gate.get("status") != "pass"
        or gate.get("run") != contract["run"]
        or gate.get("epoch") != 1
        or gate.get("fit_contract_sha256") != fit_sha
    ):
        raise ValueError("root terminal gate does not release this exact F1 student CP1")
    cp_path = Path(contract["expected_output"])
    if cp_path != PROJECT_ROOT / "artifacts/runs" / contract["run"] / "model_ep1.pt":
        raise ValueError("R4 student CP1 path differs from the frozen fit contract")
    provenance_path = cp_path.parent / "provenance.json"
    state_path = cp_path.parent / "training_state.pt"
    cp_sha, provenance_sha, state_sha = (
        sha256_file(cp_path),
        sha256_file(provenance_path),
        sha256_file(state_path),
    )
    if (
        student_pins.get("sha256", {}).get("checkpoint") != cp_sha
        or student_pins.get("sha256", {}).get("provenance") != provenance_sha
        or student_pins.get("sha256", {}).get("training_state") != state_sha
    ):
        raise ValueError("actual F1 student parent changed from the sealed packet binding")
    if (
        gate.get("checkpoint_sha256") != cp_sha
        or gate.get("provenance_sha256") != provenance_sha
        or gate.get("training_state_sha256") != state_sha
    ):
        raise ValueError("F1 student CP1 or provenance differs from the root terminal gate")
    provenance = read_object(provenance_path)
    args = provenance.get("args", {})
    expected_args = {
        "name": contract["run"],
        "seed": 3,
        "valid_fold": 1,
        "train_source": "train+valid",
        "checkpoint_policy": "fixed",
        "checkpoint_epoch": 1,
        "no_eval_during_fit": True,
        "aux_struct": 0.3,
        "aux_next": 0.3,
        "distill_alpha": 0.7,
        "distill_family_head": True,
        "distill_family_target_mode": "raw",
    }
    for key, value in expected_args.items():
        if args.get(key) != value:
            raise ValueError(f"actual student CP1 provenance differs at args.{key}")
    if provenance.get("selected_epoch") != 1:
        raise ValueError("actual F1 student checkpoint is not selected fixed epoch 1")
    return cp_path, provenance_path, cp_sha, provenance_sha, gate


def _require_root_gate(
    path: Path, schema: str, fields: dict[str, Any]
) -> tuple[dict[str, Any], str]:
    """Fail closed unless the root has released this exact stage and its parents."""
    gate = read_object(path)
    if gate.get("schema") != schema or gate.get("decision") != "PASS":
        raise ValueError(f"root gate does not release this stage: {path.name}")
    for key, value in fields.items():
        if gate.get(key) != value:
            raise ValueError(f"root gate binding mismatch at {key}: {path.name}")
    return gate, sha256_file(path)


def validate_core_receipt() -> tuple[dict[str, Any], str]:
    """Verify the write-once F1 31-to-23 crop before any HFT stage."""
    receipt = read_object(CORE_RECEIPT_PATH)
    contract = read_object(CORE_CONTRACT_PATH)
    if (
        receipt.get("schema") != "wnn.f1-core-crop-producer-receipt.r2"
        or receipt.get("status") != "PASS_ACTUAL_CORE_CROP"
        or receipt.get("contract_sha256") != sha256_file(CORE_CONTRACT_PATH)
        or receipt.get("destination_run") != contract["destination_run"]
    ):
        raise ValueError("F1 crop receipt does not bind the frozen crop contract")
    checkpoint = PROJECT_ROOT / receipt["output_checkpoint_relative_path"]
    provenance = PROJECT_ROOT / receipt["output_provenance_relative_path"]
    if sha256_file(checkpoint) != receipt["verification"]["output_checkpoint_sha256"]:
        raise ValueError("F1 cropped core checkpoint changed after its receipt")
    if sha256_file(provenance) != receipt["verification"]["output_provenance_sha256"]:
        raise ValueError("F1 cropped core provenance changed after its receipt")
    _, _, source_cp_sha, source_prov_sha, student_gate = verify_student_cp1()
    if (
        receipt.get("source_checkpoint_sha256") != source_cp_sha
        or receipt.get("source_provenance_sha256") != source_prov_sha
        or receipt.get("source_student_terminal_gate_sha256")
        != sha256_file(PACKET_ROOT / "ROOT_F1_STUDENT_CP1_TERMINAL_GATE_R2.json")
        or student_gate.get("status") != "pass"
    ):
        raise ValueError("F1 crop receipt no longer binds the actual student CP1 terminal gate")
    return receipt, sha256_file(CORE_RECEIPT_PATH)


def _trainer_argv(stage: str) -> list[str]:
    """Select the frozen profile-only or full one-epoch c337 command."""
    contract_path = PROFILE_CONTRACT_PATH if stage == "profile" else FIT_CONTRACT_PATH
    return read_object(contract_path)["trainer_argv"]


def _validate_actual_cli(
    trainer: ModuleType, trainer_path: Path, project_root: Path
) -> dict[str, Any]:
    """Parse both frozen c337 commands and stop before CUDA initialization."""
    reached_device = False

    def stop_before_device() -> None:
        nonlocal reached_device
        reached_device = True
        raise _StopBeforeDevice("actual c337 CLI reached configure_torch boundary")

    original_configure = trainer.configure_torch
    original_argv = sys.argv[:]
    checked: list[dict[str, Any]] = []
    try:
        trainer.configure_torch = stop_before_device
        for stage in ("profile", "fit"):
            contract = read_object(
                PROFILE_CONTRACT_PATH if stage == "profile" else FIT_CONTRACT_PATH
            )
            output = project_root / "artifacts/runs" / contract["run"]
            if output.exists():
                raise FileExistsError(f"preflight run output already exists: {output}")
            reached_device = False
            sys.argv = [str(trainer_path), *_trainer_argv(stage)]
            with contextlib.suppress(_StopBeforeDevice):
                trainer.main()
            if not reached_device:
                raise RuntimeError(f"actual c337 {stage} CLI failed before device boundary")
            if output.exists():
                raise RuntimeError(f"actual c337 CLI preflight created a run: {output}")
            checked.append({"stage": stage, "actual_c337_main_reached_device_boundary": True})
    finally:
        trainer.configure_torch = original_configure
        sys.argv = original_argv
    return {"status": "PASS_ACTUAL_C337_CLI_NO_DEVICE", "checks": checked, "device_started": False}


def _run_crop(project_root: Path) -> dict[str, Any]:
    """Derive the F1 core from actual student CP1 using the sealed crop primitive."""
    bindings = verify_packet_bindings()
    inputs = validate_inputs(import_trainer()[0])
    source_checkpoint, source_provenance, cp_sha, provenance_sha, student_gate = (
        verify_student_cp1()
    )
    contract = read_object(CORE_CONTRACT_PATH)
    contract_sha = sha256_file(CORE_CONTRACT_PATH)
    _require_root_gate(
        PACKET_ROOT / "ROOT_F1_CORE_CROP_ADMISSION_R2.json",
        "wnn.f1-core-crop-admission.r1",
        {
            "decision": "PASS",
            "crop_contract_sha256": contract_sha,
            "student_terminal_gate_sha256": sha256_file(
                PACKET_ROOT / "ROOT_F1_STUDENT_CP1_TERMINAL_GATE_R2.json"
            ),
            "source_checkpoint_sha256": cp_sha,
            "source_provenance_sha256": provenance_sha,
        },
    )
    destination_run = contract["destination_run"]
    destination = project_root / "artifacts/runs" / destination_run
    if destination.exists():
        raise FileExistsError(f"refuse to overwrite existing F1 crop output: {destination}")
    r5_bindings = read_object(R5_CROP_BINDINGS)
    crop_binding = r5_bindings["snapshots"]["core_crop"]
    crop_path = (
        R5_CROP_PACKET / crop_binding["root_relative_path"] / "scripts/make_core_checkpoint.py"
    )
    if sha256_file(crop_path) != bindings["f0_crop_source_sha256"]:
        raise ValueError("crop primitive changed immediately before use")
    crop_module = _import_from_path(crop_path, "oldz4_core_crop_f1_r2")
    crop_module.ROOT = project_root.resolve()
    derived = crop_module.derive_core(
        project_root.resolve(), contract["source_run"], contract["source_epoch"], destination_run
    )
    output_checkpoint = destination / f"model_ep{contract['source_epoch']}.pt"
    output_provenance = destination / "provenance.json"
    verifier = _import_from_path(R5_CROP_VERIFIER, "oldz4_core_crop_verifier_f1_r2")
    verification = verifier.verify_crop(
        source_checkpoint,
        source_provenance,
        output_checkpoint,
        output_provenance,
        source_run=contract["source_run"],
        source_epoch=contract["source_epoch"],
    )
    if derived.get("checkpoint_sha256") != verification["output_checkpoint_sha256"]:
        raise ValueError("F1 crop primitive and independent verifier disagree")
    if output_checkpoint.stat().st_size > contract["maximum_checkpoint_bytes"]:
        raise ValueError("F1 cropped checkpoint exceeds its bounded artifact size")
    receipt = {
        "schema": "wnn.f1-core-crop-producer-receipt.r2",
        "status": "PASS_ACTUAL_CORE_CROP",
        "created_utc": datetime.now(UTC).isoformat(),
        "contract_sha256": contract_sha,
        "r4_input_bindings_sha256": bindings["r4_input_bindings_sha256"],
        "r4_source_bindings_sha256": bindings["r4_source_bindings_sha256"],
        "source_student_terminal_gate_sha256": sha256_file(
            PACKET_ROOT / "ROOT_F1_STUDENT_CP1_TERMINAL_GATE_R2.json"
        ),
        "source_checkpoint_sha256": cp_sha,
        "source_provenance_sha256": provenance_sha,
        "source_run": contract["source_run"],
        "source_epoch": 1,
        "source_qualified_membership": inputs,
        "student_terminal_gate_status": student_gate["status"],
        "crop_source_sha256": bindings["f0_crop_source_sha256"],
        "destination_run": destination_run,
        "output_checkpoint_relative_path": str(output_checkpoint.relative_to(project_root)),
        "output_provenance_relative_path": str(output_provenance.relative_to(project_root)),
        "verification": verification,
    }
    with CORE_RECEIPT_PATH.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return receipt


def run_stage(stage: str, project_root: Path) -> dict[str, Any]:
    """Run one individually admitted F1 crop, HFT profile, fit, or export stage."""
    if project_root.resolve() != PROJECT_ROOT.resolve():
        raise ValueError(f"project root must be the MAIN project: {PROJECT_ROOT}")
    _require_environment()
    bindings = verify_packet_bindings()
    trainer, trainer_path = import_trainer()
    inputs = validate_inputs(trainer)
    if stage == "validate-import":
        predictor, predictor_path = import_predictor(trainer)
        return {
            "status": "PASS_IMPORTS_AND_F1_MEMBERSHIP",
            "bindings": bindings,
            "inputs": inputs,
            "trainer_path": str(trainer_path),
            "trainer_root": str(trainer.ROOT),
            "predictor_path": str(predictor_path),
            "predictor_root": str(predictor.ROOT),
            "dataset_path": str(Path(trainer.Sequences("valid", []).path).resolve()),
            "device_started": False,
            "full_parquet_rehashed": False,
        }
    if stage == "validate-cli":
        _, crop_sha = validate_core_receipt()
        cli = _validate_actual_cli(trainer, trainer_path, project_root)
        return {"status": cli["status"], "crop_receipt_sha256": crop_sha, **cli}
    if stage == "crop":
        return _run_crop(project_root)

    _core_receipt, core_receipt_sha = validate_core_receipt()
    core_contract = read_object(CORE_CONTRACT_PATH)
    stage_contract_path = (
        PROFILE_CONTRACT_PATH
        if stage == "profile"
        else FIT_CONTRACT_PATH
        if stage == "fit"
        else EXPORT_CONTRACT_PATH
    )
    contract = read_object(stage_contract_path)
    if (
        contract.get("source_snapshot_bindings_sha256")
        != bindings["r3_source_snapshot_bindings_sha256"]
    ):
        raise ValueError("HFT stage contract source bindings changed")
    if contract.get("initializer") != f"{core_contract['destination_run']}@ep1":
        raise ValueError("HFT stage initializer differs from the actual cropped student CP1")
    run_dir = project_root / "artifacts/runs" / contract["run"]

    if stage == "profile":
        if run_dir.exists():
            raise FileExistsError(f"profile output exists; require a new attempt: {run_dir}")
        _require_root_gate(
            PACKET_ROOT / "ROOT_F1_HFT_PROFILE_LAUNCH_R2.json",
            "wnn.oldz4-hft-profile-launch.v1",
            {
                "decision": "PASS",
                "profile_run": contract["run"],
                "profile_contract_sha256": sha256_file(PROFILE_CONTRACT_PATH),
                "core_crop_receipt_sha256": core_receipt_sha,
            },
        )
        argv = contract["trainer_argv"]
        previous_argv = sys.argv[:]
        try:
            sys.argv = [str(trainer_path), *argv]
            trainer.main()
        finally:
            sys.argv = previous_argv
        return {"stage": stage, "run": contract["run"], "core_receipt_sha256": core_receipt_sha}

    if stage == "fit":
        if run_dir.exists():
            raise FileExistsError(f"fit output exists; require a new attempt: {run_dir}")
        profile_path = PACKET_ROOT / "ROOT_F1_HFT_PROFILE_ADMISSION_R2.json"
        profile_gate, profile_sha = _require_root_gate(
            profile_path,
            "wnn.oldz4-hft-profile-admission.v1",
            {
                "decision": "PASS",
                "profile_run": read_object(PROFILE_CONTRACT_PATH)["run"],
                "profile_contract_sha256": sha256_file(PROFILE_CONTRACT_PATH),
                "core_crop_receipt_sha256": core_receipt_sha,
            },
        )
        evidence = profile_gate.get("profile_update_evidence")
        if (
            not isinstance(evidence, dict)
            or evidence.get("valid_for_throughput") is not True
            or evidence.get("cuda_device_confirmed") is not True
            or evidence.get("finite_loss") is not True
            or evidence.get("nonfinite_gradient_updates") != 0
            or not isinstance(evidence.get("optimizer_completed_updates"), int)
            or evidence["optimizer_completed_updates"] < 1
            or evidence.get("oom_delta") != 0
        ):
            raise ValueError(
                "root HFT profile gate lacks finite actual-CUDA update/resource evidence"
            )
        _require_root_gate(
            PACKET_ROOT / "ROOT_F1_HFT_FIT_ADMISSION_R2.json",
            "wnn.oldz4-hft-fit-admission.v1",
            {
                "decision": "PASS",
                "fit_run": contract["run"],
                "fit_contract_sha256": sha256_file(FIT_CONTRACT_PATH),
                "profile_admission_sha256": profile_sha,
                "core_crop_receipt_sha256": core_receipt_sha,
            },
        )
        previous_argv = sys.argv[:]
        try:
            sys.argv = [str(trainer_path), *contract["trainer_argv"]]
            trainer.main()
        finally:
            sys.argv = previous_argv
        return {"stage": stage, "run": contract["run"], "core_receipt_sha256": core_receipt_sha}

    if stage == "export":
        fit_contract = read_object(FIT_CONTRACT_PATH)
        terminal_path = PACKET_ROOT / "ROOT_F1_HFT_TERMINAL_ADMISSION_R2.json"
        terminal, terminal_sha = _require_root_gate(
            terminal_path,
            "wnn.oldz4-hft-terminal-admission.v1",
            {
                "decision": "PASS",
                "fit_run": fit_contract["run"],
                "fit_contract_sha256": sha256_file(FIT_CONTRACT_PATH),
                "core_crop_receipt_sha256": core_receipt_sha,
            },
        )
        checkpoint = project_root / fit_contract["output_relative_paths"]["checkpoint"]
        provenance = project_root / fit_contract["output_relative_paths"]["provenance"]
        training_state = project_root / fit_contract["output_relative_paths"]["training_state"]
        if terminal.get("checkpoint_epoch") != 0 or not training_state.is_file():
            raise ValueError("root terminal gate or HFT training state is missing")
        checkpoint_sha, provenance_sha = sha256_file(checkpoint), sha256_file(provenance)
        if (
            terminal.get("checkpoint_sha256") != checkpoint_sha
            or terminal.get("provenance_sha256") != provenance_sha
        ):
            raise ValueError("HFT checkpoint/provenance differs from its root terminal gate")
        fit_provenance = read_object(provenance)
        fit_args = fit_provenance.get("args", {})
        expected_fit_args = {
            "name": fit_contract["run"],
            "seed": fit_contract["seed"],
            "valid_fold": fit_contract["fold"],
            "train_source": "valid",
            "mask_weight": True,
            "checkpoint_policy": "fixed",
            "checkpoint_epoch": 0,
            "no_eval_during_fit": True,
            "freeze": "inp,gru",
            "inherit_init_preprocessing": True,
            "aux_struct": 0.3,
            "aux_next": 0.3,
        }
        for key, value in expected_fit_args.items():
            if fit_args.get(key) != value:
                raise ValueError(f"F1 HFT provenance differs at args.{key}")
        if "--aux-struct" in contract["trainer_argv"] or "--aux-next" in contract["trainer_argv"]:
            raise ValueError(
                "HFT auxiliary coefficients must be inherited from the exact core initializer"
            )
        split = trainer.load_campaign_split(
            PROJECT_ROOT / read_object(R4_INPUTS)["fold"]["split_manifest"], PROJECT_ROOT
        )
        fit_row_groups = fit_provenance.get("fit_row_groups", {})
        if set(fit_row_groups.get("train", [])) or set(fit_row_groups.get("valid", [])) != set(
            split.valid_groups
        ):
            raise ValueError(
                "F1 HFT direct fit row groups do not equal the 1,416 source-qualified valid fit groups"
            )
        fit_identity = fit_provenance.get("fit_membership_identity", {})
        if (
            fit_identity.get("train", {}).get("count") != 0
            or fit_identity.get("valid", {}).get("count") != 1416
        ):
            raise ValueError("F1 HFT provenance fit membership counts differ")
        if set(fit_provenance.get("evaluation_row_groups", [])) != set(split.evaluation_groups):
            raise ValueError(
                "F1 HFT evaluation row groups differ from the 457 outer F1 evaluation groups"
            )
        core_provenance_path = PROJECT_ROOT / core_contract["output_relative_paths"]["provenance"]
        core_provenance = read_object(core_provenance_path)
        if fit_provenance.get("initial_checkpoint_sha256") != sha256_file(
            PROJECT_ROOT / core_contract["output_relative_paths"]["checkpoint"]
        ):
            raise ValueError("F1 HFT initial checkpoint hash differs from its actual cropped core")
        if fit_provenance.get("normalization_identity") != core_provenance.get(
            "normalization_identity"
        ):
            raise ValueError("F1 HFT normalizer differs from its exact core initializer")
        output = project_root / contract["prediction_relative_path"]
        sidecar = project_root / contract["sidecar_relative_path"]
        if output.exists() or sidecar.exists():
            raise FileExistsError("refuse to overwrite an existing F1 HFT prediction")
        predictor, predictor_path = import_predictor(trainer)
        previous_argv = sys.argv[:]
        try:
            sys.argv = [str(predictor_path), *contract["predictor_argv"]]
            predictor.main()
        finally:
            sys.argv = previous_argv
        if not output.is_file() or not sidecar.is_file():
            raise FileNotFoundError("frozen c337 predictor did not create both F1 outputs")
        sidecar_obj = read_object(sidecar)
        if (
            sidecar_obj.get("run") != fit_contract["run"]
            or sidecar_obj.get("split") != "valid"
            or sidecar_obj.get("checkpoint_epoch") != 0
            or sidecar_obj.get("checkpoint_sha256") != checkpoint_sha
            or sidecar_obj.get("split_manifest_sha256")
            != sha256_file(PROJECT_ROOT / read_object(R4_INPUTS)["fold"]["split_manifest"])
            or sidecar_obj.get("prediction_sha256") != sha256_file(output)
        ):
            raise ValueError("F1 HFT prediction sidecar is not bound to the terminal CP0")
        import numpy as np

        predictions = np.load(output, mmap_mode="r", allow_pickle=False)
        if predictions.shape != (1873, 20000, 2) or predictions.dtype != np.dtype("float32"):
            raise ValueError("F1 HFT prediction header differs from the full-valid contract")
        if output.stat().st_size != contract["expected_npy_file_bytes"]:
            raise ValueError("F1 HFT prediction byte size differs from the expected NPY envelope")
        for start in range(0, predictions.shape[0], 8):
            if not np.isfinite(predictions[start : start + 8]).all():
                raise ValueError(
                    f"F1 HFT prediction has non-finite values in sequence rows {start}.."
                )
        return {
            "stage": stage,
            "run": fit_contract["run"],
            "terminal_admission_sha256": terminal_sha,
            "checkpoint_sha256": checkpoint_sha,
            "prediction_relative_path": str(output.relative_to(project_root)),
            "prediction_bytes": output.stat().st_size,
            "sidecar_sha256": sha256_file(sidecar),
        }
    raise ValueError(f"unsupported stage: {stage}")


def main() -> None:
    """Run one prepared stage; every mutating stage requires a root-owned gate."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("validate-import", "validate-cli", "crop", "profile", "fit", "export"),
        required=True,
    )
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    args = parser.parse_args()
    result = run_stage(args.stage, args.project_root)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
