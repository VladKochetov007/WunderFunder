from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import json
import logging
import os
import runpy
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

from validate_student_export_r4 import (
    CONTRACT_PATH,
    PACKET_ROOT,
    read_json,
    sha256_file,
    validate_checkpoint,
    validate_export,
    write_once,
)

LOGGER = logging.getLogger("oldz4_student_export_r4")
SOURCE_BINDINGS_PATH = PACKET_ROOT / "SOURCE_BINDINGS_R4.json"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _verify_bound_sources(project_root: Path) -> tuple[Path, Path]:
    """Verify and import only the frozen H230 source tree that produced the fit."""
    contract = read_json(CONTRACT_PATH)
    source_bindings_path = PACKET_ROOT / "SOURCE_BINDINGS_R4.json"
    _require(source_bindings_path.is_file(), "source bindings file is missing")
    _require(
        sha256_file(source_bindings_path) == contract["source_bindings_sha256"],
        "source bindings file changed",
    )
    bindings = read_json(source_bindings_path)
    source_root = Path(bindings["fit_source_root"]).resolve(strict=True)
    trainer_path = Path(bindings["fit_trainer_path"]).resolve(strict=True)
    predictor_path = project_root / contract["predictor_relative_path"]
    _require(
        sha256_file(trainer_path) == contract["fit_trainer_sha256"],
        "frozen fit trainer bytes changed",
    )
    _require(
        trainer_path == source_root / "scripts" / "train_rnn.py",
        "trainer path is not the bound entrypoint",
    )
    _require(predictor_path.is_file(), "MAIN predictor entrypoint is missing")
    _require(sha256_file(predictor_path) == contract["predictor_sha256"], "predictor bytes changed")
    for item in bindings["fit_python_modules"]:
        path = source_root / item["path"]
        _require(path.is_file(), f"bound source module is missing: {item['path']}")
        _require(
            sha256_file(path) == item["sha256"], f"bound source module changed: {item['path']}"
        )
    _require(
        hashlib.sha256(Path(bindings["fit_trainer_path"]).read_bytes()).hexdigest()
        == bindings["fit_trainer_sha256"],
        "trainer source binding does not close",
    )
    return source_root, predictor_path


def _verify_root_export_admission(
    project_root: Path, gate_path: Path, admission_path: Path
) -> dict[str, Any]:
    """Require root's exact export-only admission before starting the predictor."""
    # Root admission receipts are heterogeneous external JSON objects.
    contract = read_json(CONTRACT_PATH)
    gate = read_json(gate_path)
    _require(
        gate.get("schema") == "wnn.oldz4.student-cp1-terminal-gate.r4", "wrong CP1 gate schema"
    )
    _require(gate.get("status") == "pass", "CP1 terminal gate did not pass")
    admission = read_json(admission_path)
    _require(
        admission.get("schema") == "wnn.oldz4.student-export-root-admission.r4",
        "root export admission schema differs",
    )
    _require(admission.get("status") == "admitted", "root has not admitted this export")
    _require(admission.get("attempt_id") == contract["attempt_id"], "admission attempt ID differs")
    _require(admission.get("run") == contract["run"], "admission run differs")
    _require(
        admission.get("contract_sha256") == sha256_file(CONTRACT_PATH),
        "admission does not bind this frozen packet",
    )
    _require(
        admission.get("checkpoint_gate_sha256") == sha256_file(gate_path),
        "admission does not bind the terminal CP1 gate",
    )
    _require(
        admission.get("checkpoint_sha256") == gate.get("checkpoint_sha256"),
        "admission CP1 hash differs from the terminal gate",
    )
    _require(
        admission.get("authorized_action") == "fixed-checkpoint-export-only",
        "admission does not authorize only the fixed-checkpoint export",
    )
    _require(admission.get("admitted_utc"), "admission lacks a timestamp")
    _require(admission.get("resource_profile_receipt"), "admission lacks a resource profile")
    return gate


def _import_exact_trainer(source_root: Path, project_root: Path) -> ModuleType:
    """Import the training module, bind its real global ROOT, then verify valid data path."""
    source_scripts = source_root / "scripts"
    source_python = source_root / "src"
    sys.path.insert(0, str(source_scripts))
    sys.path.insert(0, str(source_python))
    if "train_rnn" in sys.modules:
        del sys.modules["train_rnn"]
    train_rnn = importlib.import_module("train_rnn")
    trainer_path = source_scripts / "train_rnn.py"
    _require(
        Path(train_rnn.__file__).resolve() == trainer_path.resolve(),
        "imported training module is not the bound trainer",
    )
    _require(
        sha256_file(Path(train_rnn.__file__).resolve())
        == "09902e4b4d886f863d5090ff58ba9ec4b7c677f995bb0192f0538c819a693bbb",
        "imported training module bytes differ from the fitted trainer",
    )
    train_rnn.ROOT = project_root
    valid_path = project_root / "datasets" / "valid.parquet"
    _require(valid_path.is_file(), "project valid.parquet is missing")
    _require(valid_path.stat().st_size == 5_615_509_128, "project valid.parquet size differs")
    resolved = Path(train_rnn.Sequences("valid", []).path).resolve(strict=True)
    _require(
        resolved == valid_path.resolve(strict=True),
        "trainer still resolves valid data under its snapshot",
    )
    import wnn
    from wnn.resources import dataloader_budget

    _require(
        Path(wnn.__file__).resolve().is_relative_to(source_python / "wnn"),
        "wnn package resolved outside the checkpoint's frozen source tree",
    )
    _require(
        os.environ.get("WNN_LOADER_WORKERS") == "0",
        "export requires the explicit zero-worker loader environment",
    )
    loader_probe = train_rnn.Sequences("valid", [])
    loader_workers, prefetch = dataloader_budget(loader_probe.batch_tensor_bytes(16), max_workers=2)
    _require(
        loader_workers == 0 and prefetch == 0, "imported loader helper did not enforce zero workers"
    )
    return train_rnn


def main() -> None:
    """Run one root-admitted fixed-CP1 full-valid prediction and validate its bytes."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--checkpoint-gate", type=Path, required=True)
    parser.add_argument("--root-admission", type=Path, required=True)
    parser.add_argument("--export-receipt", type=Path, required=True)
    args = parser.parse_args()
    project_root = args.project_root.resolve(strict=True)
    contract = read_json(CONTRACT_PATH)
    _require(str(project_root) == contract["project_root"], "project root differs from contract")
    _verify_root_export_admission(project_root, args.checkpoint_gate, args.root_admission)
    gate = validate_checkpoint(project_root)
    _require(
        gate["checkpoint_sha256"] == read_json(args.checkpoint_gate).get("checkpoint_sha256"),
        "CP1 changed after gate",
    )
    source_root, predictor_path = _verify_bound_sources(project_root)
    train_rnn = _import_exact_trainer(source_root, project_root)
    prediction_path = project_root / contract["export"]["prediction_relative_path"]
    collision_paths = (
        prediction_path,
        prediction_path.with_suffix(".json"),
        prediction_path.with_name(prediction_path.name + ".partial"),
        project_root / "cache" / f"seqsums_{contract['run']}.npz",
    )
    existing = [str(path) for path in collision_paths if path.exists()]
    _require(not existing, f"export would overwrite existing artifacts: {existing}")
    del gate
    gc.collect()
    sys.argv = [str(predictor_path), *contract["export"]["predictor_arguments"]]
    runpy.run_path(str(predictor_path), run_name="__main__")
    _require(
        sys.modules.get("train_rnn") is train_rnn
        and Path(train_rnn.ROOT).resolve() == project_root,
        "predictor replaced the bound trainer or ROOT",
    )
    receipt = validate_export(project_root, args.checkpoint_gate)
    receipt["export_attempt_id"] = contract["attempt_id"]
    receipt["root_admission_sha256"] = sha256_file(args.root_admission)
    write_once(args.export_receipt, receipt)
    LOGGER.info(json.dumps({"status": "pass", "prediction_sha256": receipt["prediction_sha256"]}))


if __name__ == "__main__":
    main()
