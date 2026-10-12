from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import importlib.util
import io
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

ROOT = Path("/home/vlad/development/wunder/wnn_connectome_starterpack")
PACKET = Path(__file__).resolve().parent
CONTRACT_PATH = PACKET / "EXPORT_CONTRACT_R1.json"
RELEASE_PATH = (
    ROOT
    / "research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/ops_continuation_20261011/e529_muon512_f0_cp2_export_r1/ROOT_EXPORT_RELEASE_R1.json"
)
RUN_ID = "bd26_gru512_full_s9_f0_muon_seeded_b8_amp4096_capacityv5r2_fit_20261009"
OUTPUT_NAME = "e529_muon512_f0_cp2_prefix4_retained_r1"
SNAPSHOT = Path(
    "/home/vlad/development/wunder/wnn_old_pipeline_reconstruct_20261011_r1/research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/ops_continuation_20261011/old_z4_cleanfold_successor_20261011_r3/source_snapshots/hft_c337"
)
SOURCE_BINDINGS_PATH = (
    ROOT
    / "research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/ops_continuation_20261011/oldz4_f1_core_hft_r2/SOURCE_INPUT_BINDINGS_R2.json"
)
SOURCE_SNAPSHOT_BINDINGS_PATH = Path(
    "/home/vlad/development/wunder/wnn_old_pipeline_reconstruct_20261011_r1/research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/ops_continuation_20261011/old_z4_cleanfold_successor_20261011_r3/SOURCE_SNAPSHOT_BINDINGS_R3.json"
)
E529_BASE = (
    ROOT
    / "research/campaign_2026-10-06/bandit_continuation/continuous_execution_E529/gru512_768_muon_fit_v5r2_20261009"
)
SPLIT_PATH = E529_BASE / "packet/fold0.json"
FIT_INPUT_PATH = (
    E529_BASE / "terminal_collect_r1/outputs/gru_capacity_v5r2_project/fit_input_binding.json"
)
FIT_TERMINAL_PATH = (
    E529_BASE
    / "terminal_collect_r1/outputs/gru_capacity_v5r2_project/capacity_receipts/muon512/fit_terminal.json"
)
RUNNER_RESULT_PATH = (
    E529_BASE / "terminal_collect_r1/outputs/gru_capacity_v5r2_runner_results/muon512/result.json"
)
AUDIT_PATH = (
    PACKET.parent
    / "e529_gru512_cp2_successor_audit_20261012_r1/E529_GRU512_CP2_SUCCESSOR_AUDIT_R1.json"
)
CHECKPOINT_PATH = ROOT / "artifacts/runs" / RUN_ID / "model_ep2.pt"
PROVENANCE_PATH = ROOT / "artifacts/runs" / RUN_ID / "provenance.json"
OLD_STUDENT_RUN = "oldz4_h230_student_f0_s3_r3"
OLD_STUDENT_CHECKPOINT = ROOT / "artifacts/runs" / OLD_STUDENT_RUN / "model_ep1.pt"
OLD_STUDENT_PROVENANCE = ROOT / "artifacts/runs" / OLD_STUDENT_RUN / "provenance.json"
OLD_STUDENT_PREDICTION = ROOT / "cache" / f"pred_{OLD_STUDENT_RUN}_valid.npy"
OLD_STUDENT_SIDECAR = ROOT / "cache" / f"pred_{OLD_STUDENT_RUN}_valid.json"
FOLD0_EVAL_ROW_GROUP_SHA256 = "2aac564e257cf5857d20f3a54a44de52e23ba5c9fa79c01df4f59e2d5cdd1ec1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _verify_hash(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} missing: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"{label} SHA-256 mismatch: {actual}")


def _verify_source_snapshot(contract: dict[str, Any]) -> dict[str, Any]:
    source = contract["inference_source"]
    _verify_hash(
        SOURCE_BINDINGS_PATH,
        source["source_input_bindings_sha256"],
        "source input bindings",
    )
    source_input = _json(SOURCE_BINDINGS_PATH)["c337_source"]
    if source_input["trainer_sha256"] != source["trainer_sha256"]:
        raise ValueError("source input bindings disagree on the trainer hash")
    if source_input["predictor_sha256"] != source["predictor_sha256"]:
        raise ValueError("source input bindings disagree on the predictor hash")
    if source_input["snapshot_bindings_sha256"] != source["snapshot_bindings_sha256"]:
        raise ValueError("source input bindings disagree on the snapshot manifest")
    _verify_hash(
        SOURCE_SNAPSHOT_BINDINGS_PATH,
        source["snapshot_bindings_sha256"],
        "C337 source snapshot bindings",
    )
    manifest = _json(SOURCE_SNAPSHOT_BINDINGS_PATH)
    source_files = manifest["hft"]["files"]
    if source_input.get("files") != source_files:
        raise ValueError("source input map differs from the frozen C337 snapshot manifest")
    if Path(source_input["snapshot_bindings_path"]).resolve(strict=True) != (
        SOURCE_SNAPSHOT_BINDINGS_PATH.resolve(strict=True)
    ):
        raise ValueError("source input binding points to a different snapshot manifest")
    expected_sources = dict(source_files)
    expected_sources.update(source_input.get("supplemental_unmanifested_files", {}))
    if not expected_sources:
        raise ValueError("C337 source snapshot manifest has no files")
    checked = 0
    for relative, expected_hash in sorted(expected_sources.items()):
        path = (SNAPSHOT / relative).resolve(strict=True)
        if not path.is_relative_to(SNAPSHOT.resolve(strict=True)):
            raise ValueError(f"C337 source path escapes snapshot: {relative}")
        _verify_hash(path, expected_hash, f"C337 source {relative}")
        checked += 1
    if expected_sources.get("scripts/train_rnn.py") != source["trainer_sha256"]:
        raise ValueError("manifest trainer hash mismatch")
    if expected_sources.get("scripts/predict_run.py") != source["predictor_sha256"]:
        raise ValueError("manifest predictor hash mismatch")
    return {
        "source_files_checked": checked,
        "source_snapshot_bindings_sha256": source["snapshot_bindings_sha256"],
        "trainer_sha256": source["trainer_sha256"],
        "predictor_sha256": source["predictor_sha256"],
    }


def _output_paths() -> dict[str, Path]:
    prediction = ROOT / "cache" / f"pred_{OUTPUT_NAME}_valid.npy"
    return {
        "prediction": prediction,
        "partial": prediction.with_name(prediction.name + ".partial"),
        "sidecar": prediction.with_suffix(".json"),
        "seqsums": ROOT / "cache" / f"seqsums_{OUTPUT_NAME}.npz",
        "seqsums_partial": ROOT / "cache" / f"seqsums_{OUTPUT_NAME}.npz.partial",
    }


def _assert_outputs_absent() -> None:
    collisions = [
        str(path) for path in _output_paths().values() if path.exists() or path.is_symlink()
    ]
    if collisions:
        raise FileExistsError(f"refusing to overwrite output paths: {collisions}")


def _verify_fold_and_inputs(contract: dict[str, Any]) -> dict[str, Any]:
    input_contract = contract["inputs"]
    _verify_hash(SPLIT_PATH, input_contract["fold0_manifest_sha256"], "fold-0 manifest")
    _verify_hash(FIT_INPUT_PATH, input_contract["fit_input_binding_sha256"], "fit input binding")
    split = _json(SPLIT_PATH)
    if split.get("fold") != 0:
        raise ValueError("split manifest is not fold 0")
    if len(split["fit"]["train"]) != 10535 or len(split["fit"]["valid"]) != 1405:
        raise ValueError("fold-0 fit membership counts changed")
    if len(split["evaluation"]["valid"]) != 468:
        raise ValueError("fold-0 evaluation membership count changed")
    fit_input = _json(FIT_INPUT_PATH)
    valid_binding = fit_input["sources"]["datasets/valid.parquet"]
    targets_binding = fit_input["sources"]["cache/valid_targets.npz"]
    if valid_binding.get("sha256") != input_contract["valid_parquet_sha256_from_fit_binding"]:
        raise ValueError("fit input binding has a different valid.parquet identity")
    if targets_binding.get("sha256") != input_contract["valid_targets_sha256_from_fit_binding"]:
        raise ValueError("fit input binding has a different valid_targets identity")

    valid_path = ROOT / "datasets/valid.parquet"
    targets_path = ROOT / "cache/valid_targets.npz"
    if (
        not valid_path.is_file()
        or valid_path.stat().st_size != input_contract["valid_parquet_bytes"]
    ):
        raise ValueError("MAIN valid.parquet is missing or has a different size")
    if (
        not targets_path.is_file()
        or targets_path.stat().st_size != input_contract["valid_targets_bytes"]
    ):
        raise ValueError("MAIN valid_targets.npz is missing or has a different size")
    import numpy as np
    import pyarrow.parquet as parquet

    metadata = parquet.ParquetFile(valid_path).metadata
    if metadata.num_rows != 37_460_000 or metadata.num_row_groups != 1873:
        raise ValueError("MAIN valid.parquet footer differs from the bound producer")
    valid_rows = split["sequences"]["valid"]
    canonical_groups = [row["row_group"] for row in valid_rows]
    if canonical_groups != list(range(1873)):
        raise ValueError("fold-0 manifest no longer defines canonical valid row order")
    with np.load(targets_path, allow_pickle=False) as target_archive:
        actual_groups = np.asarray(target_archive["row_group"], dtype=np.int64).tolist()
    if actual_groups != canonical_groups:
        raise ValueError("actual valid-target row order differs from the fold manifest")

    eval_groups = split["evaluation"]["valid"]
    if eval_groups != sorted(set(eval_groups)):
        raise ValueError("fold-0 evaluation row_group IDs are not sorted and unique")
    if not set(eval_groups).issubset(set(canonical_groups)):
        raise ValueError("fold-0 evaluation IDs are outside valid prediction row order")
    fit_valid_groups = set(split["fit"]["valid"])
    if fit_valid_groups.intersection(eval_groups):
        raise ValueError("fold-0 evaluation groups overlap model fitting membership")
    if fit_valid_groups.union(eval_groups) != set(canonical_groups):
        raise ValueError("fold-0 fit-valid and evaluation groups do not partition valid rows")
    by_group = {row["row_group"]: row for row in valid_rows}
    evaluation_components = {by_group[group]["component"] for group in eval_groups}
    train_groups = set(split["fit"]["train"])
    fit_components = {
        row["component"] for row in split["sequences"]["train"] if row["row_group"] in train_groups
    }
    fit_components.update(by_group[group]["component"] for group in split["fit"]["valid"])
    if fit_components.intersection(evaluation_components):
        raise ValueError("fold-0 evaluation components overlap fitting components")
    membership_digest = hashlib.sha256(np.asarray(eval_groups, dtype="<i8").tobytes()).hexdigest()
    if membership_digest != FOLD0_EVAL_ROW_GROUP_SHA256:
        raise ValueError("fold-0 evaluation row_group digest differs from the frozen contract")
    if membership_digest != input_contract["f0_evaluation_row_group_sha256"]:
        raise ValueError("fold-0 evaluation row_group digest differs from the contract")
    if len(evaluation_components) != 141:
        raise ValueError("fold-0 evaluation component count changed")
    return {
        "valid_parquet_bytes": valid_path.stat().st_size,
        "valid_parquet_rows": metadata.num_rows,
        "valid_row_groups": metadata.num_row_groups,
        "valid_targets_bytes": targets_path.stat().st_size,
        "canonical_row_group_order_sha256": hashlib.sha256(
            np.asarray(canonical_groups, dtype="<i8").tobytes()
        ).hexdigest(),
        "evaluation_row_groups": eval_groups,
        "evaluation_row_group_sha256": membership_digest,
        "evaluation_components": len(evaluation_components),
        "fit_evaluation_component_overlap": 0,
    }


def _verify_producer(contract: dict[str, Any]) -> dict[str, Any]:
    producer = contract["producer"]
    _verify_hash(CHECKPOINT_PATH, producer["checkpoint_sha256"], "H512 checkpoint")
    _verify_hash(PROVENANCE_PATH, producer["provenance_sha256"], "H512 provenance")
    _verify_hash(FIT_TERMINAL_PATH, producer["fit_terminal_sha256"], "fit terminal receipt")
    _verify_hash(RUNNER_RESULT_PATH, producer["runner_result_sha256"], "runner result")
    provenance = _json(PROVENANCE_PATH)
    if (
        provenance.get("seed") != 9
        or provenance.get("split_manifest_sha256") != contract["inputs"]["fold0_manifest_sha256"]
    ):
        raise ValueError("H512 provenance seed or split identity changed")
    args = provenance.get("args", {})
    if args.get("name") != RUN_ID or args.get("checkpoint_epoch") != 3:
        raise ValueError("H512 provenance does not identify the fixed four-epoch recipe")
    if args.get("init") or args.get("distill") or args.get("propensity") or args.get("prop_input"):
        raise ValueError("H512 checkpoint ancestry is no longer the audited fresh model")
    if provenance.get("checkpoint_sha256", {}).get("model_ep2.pt") != producer["checkpoint_sha256"]:
        raise ValueError("H512 provenance does not bind the selected model_ep2 checkpoint")
    terminal = _json(FIT_TERMINAL_PATH)
    runner = _json(RUNNER_RESULT_PATH)
    if (
        terminal.get("run_name") != RUN_ID
        or terminal.get("status") != "fit_failed"
        or terminal.get("fit_subprocess", {}).get("exit_code") != -15
    ):
        raise ValueError("H512 terminal receipt is not the recorded interrupted producer")
    markers = terminal.get("fit_subprocess", {}).get("markers", {})
    if markers.get("fit_epoch", {}).get("epoch") != 2:
        raise ValueError("terminal receipt no longer identifies the completed epoch-2 prefix")
    updates = markers.get("fit_epoch", {}).get("update_diagnostics", {}).get("cumulative", {})
    if (
        updates.get("completed_batches") != 4476
        or updates.get("optimizer_attempts") != 44760
        or updates.get("optimizer_completed_updates") != 44740
        or updates.get("amp_skipped_updates") != 20
    ):
        raise ValueError("terminal fit-prefix counters differ from the preserved audit")
    if runner.get("status") != "failed" or runner.get("returncode") != 1:
        raise ValueError("H512 runner status differs from the preserved timeout")
    files = {item["path"]: item for item in runner.get("artifacts", {}).get("files", [])}
    for path, expected_sha in (
        (str(CHECKPOINT_PATH.relative_to(ROOT)), producer["checkpoint_sha256"]),
        (str(PROVENANCE_PATH.relative_to(ROOT)), producer["provenance_sha256"]),
    ):
        if files.get(path, {}).get("sha256") != expected_sha:
            raise ValueError(f"terminal runner artifact list does not bind {path}")
    if runner.get("artifacts", {}).get("missing"):
        raise ValueError("terminal runner receipt lists missing artifacts")
    if (ROOT / "artifacts/runs" / RUN_ID / "training_state.pt").exists():
        raise ValueError("unexpected optimizer training state exists for this interrupted fit")
    if (ROOT / "artifacts/runs" / RUN_ID / "model_ep3.pt").exists():
        raise ValueError("unexpected epoch-3 checkpoint exists for this interrupted fit")
    return {
        "run_id": RUN_ID,
        "checkpoint_path": str(CHECKPOINT_PATH),
        "checkpoint_sha256": producer["checkpoint_sha256"],
        "checkpoint_bytes": CHECKPOINT_PATH.stat().st_size,
        "provenance_sha256": producer["provenance_sha256"],
        "fit_terminal_sha256": producer["fit_terminal_sha256"],
        "runner_result_sha256": producer["runner_result_sha256"],
        "completed_epoch_index": 2,
        "requested_schedule_epochs": 4,
        "fit_status": "interrupted_after_epoch2_during_epoch_index3",
        "optimizer_state_present": False,
    }


def _verify_base_student(contract: dict[str, Any]) -> dict[str, Any]:
    base = contract["fixed_blend"]["old_student"]
    _verify_hash(OLD_STUDENT_CHECKPOINT, base["checkpoint_sha256"], "F0 student checkpoint")
    _verify_hash(OLD_STUDENT_PROVENANCE, base["provenance_sha256"], "F0 student provenance")
    _verify_hash(OLD_STUDENT_SIDECAR, base["prediction_sidecar_sha256"], "F0 student sidecar")
    if not OLD_STUDENT_PREDICTION.is_file() or OLD_STUDENT_PREDICTION.stat().st_size != 299_680_128:
        raise ValueError("fixed F0 student full-valid prediction is missing or has changed size")
    sidecar = _json(OLD_STUDENT_SIDECAR)
    if (
        sidecar.get("run") != OLD_STUDENT_RUN
        or sidecar.get("split") != "valid"
        or sidecar.get("epoch") != 1
        or sidecar.get("checkpoint_sha256") != base["checkpoint_sha256"]
        or sidecar.get("split_manifest_sha256") != contract["inputs"]["fold0_manifest_sha256"]
        or sidecar.get("row_groups") != list(range(1873))
    ):
        raise ValueError("fixed F0 student prediction sidecar does not match its endpoint")
    return {
        "run_id": OLD_STUDENT_RUN,
        "checkpoint_sha256": base["checkpoint_sha256"],
        "provenance_sha256": base["provenance_sha256"],
        "prediction_path": str(OLD_STUDENT_PREDICTION),
        "prediction_bytes": OLD_STUDENT_PREDICTION.stat().st_size,
        "prediction_sidecar_sha256": base["prediction_sidecar_sha256"],
        "sidecar_prediction_sha256": sidecar.get("prediction_sha256"),
        "weight_in_fixed_blend": 0.75,
    }


def _import_actual_predictor(contract: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    script_dir = SNAPSHOT / "scripts"
    src_dir = SNAPSHOT / "src"
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    if "train_rnn" in sys.modules or "wnn" in sys.modules:
        raise RuntimeError("pinned C337 modules were imported before source validation")
    train_rnn = importlib.import_module("train_rnn")
    trainer_path = Path(train_rnn.__file__).resolve(strict=True)
    expected_trainer = SNAPSHOT / "scripts/train_rnn.py"
    if trainer_path != expected_trainer.resolve(strict=True):
        raise ValueError(f"actual trainer import came from {trainer_path}")
    predictor_path = SNAPSHOT / "scripts/predict_run.py"
    spec = importlib.util.spec_from_file_location("e529_c337_predict_run", predictor_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not construct import spec for exact C337 predictor")
    predictor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(predictor)
    if Path(predictor.__file__).resolve(strict=True) != predictor_path.resolve(strict=True):
        raise ValueError("actual predictor import path differs from pinned C337 source")
    if predictor.train_rnn is not train_rnn:
        raise ValueError("predictor does not use the pinned imported trainer module")
    module_paths = {
        name: str(Path(sys.modules[name].__file__).resolve())
        for name in ("wnn", "wnn.data", "wnn.resources", "wnn.splits")
        if name in sys.modules and getattr(sys.modules[name], "__file__", None)
    }
    for name, module_path in module_paths.items():
        if not Path(module_path).is_relative_to(src_dir.resolve(strict=True)):
            raise ValueError(f"{name} imported outside the pinned C337 source: {module_path}")
    c337_root = SNAPSHOT.resolve(strict=True)
    if Path(train_rnn.ROOT).resolve() != c337_root or Path(predictor.ROOT).resolve() != c337_root:
        raise ValueError("C337 module roots do not match the frozen snapshot before rebinding")
    train_rnn.ROOT = ROOT
    predictor.ROOT = ROOT
    if Path(train_rnn.ROOT).resolve(strict=True) != ROOT.resolve(strict=True):
        raise ValueError("trainer ROOT did not bind to MAIN")
    if Path(predictor.ROOT).resolve(strict=True) != ROOT.resolve(strict=True):
        raise ValueError("predictor ROOT did not bind to MAIN")
    if predictor.train_rnn.ROOT != ROOT:
        raise ValueError("predictor trainer ROOT did not bind to MAIN")
    sequences = train_rnn.Sequences("valid", [])
    if sequences.path.resolve(strict=True) != (ROOT / "datasets/valid.parquet").resolve(
        strict=True
    ):
        raise ValueError("actual C337 Sequences(valid, []) does not resolve to MAIN data")
    if getattr(sequences, "_file", None) is not None:
        raise RuntimeError("empty-membership producer check unexpectedly opened valid.parquet")

    help_output = io.StringIO()
    old_argv = sys.argv
    try:
        sys.argv = [str(predictor_path), "--help"]
        with contextlib.redirect_stdout(help_output), contextlib.redirect_stderr(help_output):
            try:
                predictor.main()
            except SystemExit as exc:
                if exc.code != 0:
                    raise
            else:
                raise RuntimeError("actual predictor --help did not terminate at argparse")
    finally:
        sys.argv = old_argv
    help_text = help_output.getvalue()
    if "--epoch" not in help_text or "--output-name" not in help_text:
        raise ValueError("actual predictor CLI lacks the fixed epoch/output-name options")
    import torch

    if torch.cuda.is_initialized():
        raise RuntimeError("CPU-only source/CLI preflight unexpectedly initialized CUDA")
    return predictor, {
        "trainer_import_path": str(trainer_path),
        "predictor_import_path": str(predictor_path.resolve(strict=True)),
        "module_paths": module_paths,
        "trainer_root_after_binding": str(Path(train_rnn.ROOT).resolve()),
        "predictor_root_after_binding": str(Path(predictor.ROOT).resolve()),
        "sequences_valid_path": str(sequences.path.resolve()),
        "sequences_file_opened": False,
        "cli_help_confirmed": True,
        "cli_help_sha256": hashlib.sha256(help_text.encode()).hexdigest(),
        "cuda_initialized": False,
    }


def _preflight(contract: dict[str, Any]) -> tuple[dict[str, Any], Any]:
    if contract.get("schema") != "wnn.e529-h512-cp2-retained-prediction-export-contract.r1":
        raise ValueError("unexpected H512 export contract schema")
    if (
        contract.get("status") != "prepared_no_launch"
        or contract.get("launch_authorized") is not False
    ):
        raise PermissionError("preparation contract itself must never be launch authority")
    if ROOT.resolve(strict=True) != Path(contract["project_root"]).resolve(strict=True):
        raise ValueError("project root differs from frozen contract")
    declared_release = (ROOT / contract["root_release"]["path"]).resolve(strict=False)
    if declared_release != RELEASE_PATH.resolve(strict=False):
        raise ValueError("root release path differs from the contract")
    if os.environ.get("WNN_PROJECT_ROOT") != str(ROOT):
        raise ValueError("WNN_PROJECT_ROOT must point to MAIN")
    if os.environ.get("WNN_LOADER_WORKERS") != "0" or os.environ.get("WNN_THREADS") != "1":
        raise ValueError("export preflight requires zero workers and one thread")
    contract_sha = sha256_file(CONTRACT_PATH)
    helper_sha = sha256_file(Path(__file__).resolve(strict=True))
    if helper_sha != contract["helper_sha256"]:
        raise ValueError("helper SHA-256 differs from frozen contract")
    producer = _verify_producer(contract)
    inputs = _verify_fold_and_inputs(contract)
    base_student = _verify_base_student(contract)
    source = _verify_source_snapshot(contract)
    _verify_hash(AUDIT_PATH, contract["producer"]["audit_sha256"], "CP2 successor audit")
    _assert_outputs_absent()
    predictor, cli = _import_actual_predictor(contract)
    return (
        {
            "status": "preflight_pass_no_export",
            "contract_sha256": contract_sha,
            "helper_sha256": helper_sha,
            "producer": producer,
            "inputs": inputs,
            "fixed_blend_base": base_student,
            "source": source,
            "actual_predictor_preflight": cli,
            "output_paths_absent": {
                key: not (path.exists() or path.is_symlink())
                for key, path in _output_paths().items()
            },
            "launch_authorized_by_contract": False,
            "standalone_grade_performed": False,
        },
        predictor,
    )


def _verify_root_release(contract: dict[str, Any]) -> dict[str, Any]:
    release = _json(RELEASE_PATH)
    if release.get("schema") != "wnn.root.e529-cp2-retained-prediction-export-release.r1":
        raise ValueError("root release schema is not recognized")
    if (
        release.get("status") != "authorized_one_export"
        or release.get("launch_authorized") is not True
    ):
        raise PermissionError("root release does not authorize this one export")
    expected = {
        "attempt": contract["attempt"],
        "contract_sha256": sha256_file(CONTRACT_PATH),
        "helper_sha256": contract["helper_sha256"],
        "run_id": RUN_ID,
        "checkpoint_epoch_arg": 2,
        "checkpoint_sha256": contract["producer"]["checkpoint_sha256"],
        "provenance_sha256": contract["producer"]["provenance_sha256"],
        "fit_terminal_sha256": contract["producer"]["fit_terminal_sha256"],
        "runner_result_sha256": contract["producer"]["runner_result_sha256"],
        "fold0_manifest_sha256": contract["inputs"]["fold0_manifest_sha256"],
        "source_snapshot_bindings_sha256": contract["inference_source"]["snapshot_bindings_sha256"],
        "predictor_sha256": contract["inference_source"]["predictor_sha256"],
        "trainer_sha256": contract["inference_source"]["trainer_sha256"],
        "output_name": OUTPUT_NAME,
        "prediction_path": str(_output_paths()["prediction"]),
        "prediction_bytes": contract["export"]["expected_npy_bytes"],
    }
    for key, value in expected.items():
        if release.get(key) != value:
            raise PermissionError(f"root release does not match {key}")
    blend = release.get("fixed_blend")
    expected_blend = contract["fixed_blend"]
    if blend != expected_blend:
        raise PermissionError("root release does not bind the predeclared 0.75/0.25 blend")
    resource = release.get("resource_admission", {})
    expected_resource = contract["resource_profile"]
    for key in (
        "cpu_cores",
        "gpu_count",
        "memory_reserve_bytes",
        "allocator_budget_bytes",
        "observed_vram_limit_bytes",
        "loader_workers",
        "threads",
        "swap_bytes",
        "worker_timeout_seconds",
    ):
        value = resource.get(key)
        expected_value = expected_resource[key]
        if type(value) is not type(expected_value) or value != expected_value:
            raise PermissionError(f"root release resource admission differs at {key}")
    if resource.get("fresh_snapshot_passed") is not True:
        raise PermissionError("root release lacks a fresh resource snapshot pass")
    return {
        "release_path": str(RELEASE_PATH),
        "release_sha256": sha256_file(RELEASE_PATH),
        "release_schema": release["schema"],
        "release_status": release["status"],
    }


def _verify_prediction_output(contract: dict[str, Any]) -> dict[str, Any]:
    import numpy as np

    paths = _output_paths()
    final = paths["prediction"]
    if not final.is_file() or paths["partial"].exists():
        raise RuntimeError("predictor did not atomically complete the expected NPY")
    if final.stat().st_size != contract["export"]["expected_npy_bytes"]:
        raise ValueError("retained NPY byte length differs from the fixed output contract")
    if not paths["sidecar"].is_file():
        raise RuntimeError("predictor did not write the expected sidecar")
    if paths["seqsums"].exists() or paths["seqsums_partial"].exists():
        raise RuntimeError("unexpected seqsums artifact created by this predictor")
    sidecar = _json(paths["sidecar"])
    checkpoint = contract["producer"]["checkpoint_sha256"]
    if (
        sidecar.get("run") != RUN_ID
        or sidecar.get("split") != "valid"
        or sidecar.get("checkpoint_sha256") != checkpoint
        or sidecar.get("split_manifest_sha256") != contract["inputs"]["fold0_manifest_sha256"]
    ):
        raise ValueError(
            "prediction sidecar is not bound to the expected producer/checkpoint/split"
        )
    prediction = np.load(final, mmap_mode="r", allow_pickle=False)
    if prediction.shape != (1873, 20000, 2) or prediction.dtype != np.float32:
        raise ValueError("retained H512 prediction has an unexpected shape or dtype")
    if sidecar.get("row_groups") != list(range(1873)):
        raise ValueError("retained H512 prediction sidecar is not in canonical row order")
    for start in range(0, prediction.shape[0], 16):
        if not np.isfinite(prediction[start : start + 16]).all():
            raise ValueError(f"non-finite H512 output at rows starting {start}")
    del prediction
    return {
        "prediction_path": str(final),
        "prediction_bytes": final.stat().st_size,
        "prediction_sha256": sha256_file(final),
        "sidecar_path": str(paths["sidecar"]),
        "sidecar_sha256": sha256_file(paths["sidecar"]),
        "shape": [1873, 20000, 2],
        "dtype": "float32",
        "canonical_row_order": True,
        "finite": True,
        "fixed_fold0_evaluation_row_groups": 468,
        "fold0_evaluation_row_group_sha256": FOLD0_EVAL_ROW_GROUP_SHA256,
        "standalone_grade_performed": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="validate actual producer, source and data bindings without launch authority",
    )
    args = parser.parse_args(argv)
    contract = _json(CONTRACT_PATH)
    preflight, predictor = _preflight(contract)
    if args.preflight_only:
        sys.stdout.write(json.dumps(preflight, sort_keys=True) + "\n")
        return 0
    release = _verify_root_release(contract)
    _assert_outputs_absent()
    cli = preflight["actual_predictor_preflight"]
    sys.argv = [
        str(SNAPSHOT / "scripts/predict_run.py"),
        RUN_ID,
        "valid",
        "--epoch",
        "2",
        "--output-name",
        OUTPUT_NAME,
    ]
    predictor.main()
    verified = _verify_prediction_output(contract)
    sys.stdout.write(
        json.dumps(
            {
                "status": "export_complete_ungraded",
                "root_release": release,
                "actual_predictor_preflight": cli,
                "output": verified,
            },
            sort_keys=True,
        )
        + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
