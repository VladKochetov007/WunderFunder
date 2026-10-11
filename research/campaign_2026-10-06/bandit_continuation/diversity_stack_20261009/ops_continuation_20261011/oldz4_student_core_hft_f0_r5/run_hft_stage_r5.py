from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

PACKET_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = Path("/home/vlad/development/wunder/wnn_connectome_starterpack")
SOURCE_BINDINGS_PATH = PACKET_ROOT / "SOURCE_BINDINGS_R5.json"
CORE_CONTRACT_PATH = PACKET_ROOT / "CORE_CROP_CONTRACT_R5.json"
CORE_RECEIPT_PATH = PACKET_ROOT / "CORE_CROP_PRODUCER_RECEIPT_R5.json"
PROFILE_CONTRACT_PATH = PACKET_ROOT / "HFT_PROFILE_CONTRACT_R5.json"
FIT_CONTRACT_PATH = PACKET_ROOT / "HFT_FIT_CONTRACT_R5.json"
EXPORT_CONTRACT_PATH = PACKET_ROOT / "HFT_EXPORT_CONTRACT_R5.json"


def sha256_file(path: Path) -> str:
    """Hash a file without loading its full contents into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_object(path: Path) -> dict[str, Any]:
    """Read one JSON object from a frozen contract or root receipt."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return value


def _load_bindings() -> dict[str, Any]:
    """Verify every file in the committed frozen HFT and crop snapshots."""
    bindings = read_object(SOURCE_BINDINGS_PATH)
    for name, snapshot in bindings["snapshots"].items():
        snapshot_root = PACKET_ROOT / snapshot["root_relative_path"]
        observed_files = sorted(
            path.relative_to(snapshot_root).as_posix()
            for path in snapshot_root.rglob("*")
            if path.is_file()
        )
        expected_files = sorted(snapshot["files"])
        if observed_files != expected_files:
            raise ValueError(f"{name} snapshot file list differs from its seal")
        for relative_path, expected_hash in snapshot["files"].items():
            path = snapshot_root / relative_path
            if sha256_file(path) != expected_hash:
                raise ValueError(f"{name} snapshot source changed: {relative_path}")
    return bindings


def verify_project_inputs(project_root: Path) -> dict[str, Any]:
    """Bind the source CP1 and F0 split to the exact input contract."""
    project_root = project_root.resolve()
    if project_root != PROJECT_ROOT:
        raise ValueError(f"project root must be {PROJECT_ROOT}")
    contract = read_object(CORE_CONTRACT_PATH)
    if sha256_file(SOURCE_BINDINGS_PATH) != contract["source_snapshot_bindings_sha256"]:
        raise ValueError("source snapshot bindings differ from the core crop contract")
    checkpoint = project_root / contract["source_checkpoint_relative_path"]
    provenance = project_root / contract["source_provenance_relative_path"]
    split = project_root / contract["split_manifest_relative_path"]
    parent_gate = project_root / contract["parent_gate_relative_path"]
    checks = (
        (checkpoint, contract["source_checkpoint_sha256"], "student CP1"),
        (provenance, contract["source_provenance_sha256"], "student provenance"),
        (split, contract["split_manifest_sha256"], "F0 split manifest"),
        (parent_gate, contract["parent_gate_sha256"], "student CP1 terminal gate"),
    )
    observed: dict[str, str] = {}
    for path, expected, label in checks:
        digest = sha256_file(path)
        if digest != expected:
            raise ValueError(f"{label} hash differs from the frozen R5 binding")
        observed[label] = digest

    provenance_data = read_object(provenance)
    split_data = read_object(split)
    parent_gate_data = read_object(parent_gate)
    args = provenance_data["args"]
    if (
        args.get("split_manifest_sha256") != contract["split_manifest_sha256"]
        or args.get("valid_fold") != 0
        or args.get("checkpoint_epoch") != 1
        or args.get("checkpoint_policy") != "fixed"
        or not args.get("no_eval_during_fit")
        or provenance_data.get("selected_epoch") != 1
    ):
        raise ValueError("parent CP1 provenance is not the expected clean F0 selected epoch")
    parent_membership = provenance_data["fit_row_groups"]
    if set(map(int, parent_membership["train"])) != set(map(int, split_data["fit"]["train"])):
        raise ValueError("parent train fit membership differs from the frozen F0 split")
    if set(map(int, parent_membership["valid"])) != set(map(int, split_data["fit"]["valid"])):
        raise ValueError("parent valid fit membership differs from the frozen F0 split")
    if set(map(int, parent_membership["valid"])) & set(map(int, split_data["evaluation"]["valid"])):
        raise ValueError("parent valid fit membership overlaps the held F0 evaluation groups")
    if (
        parent_gate_data.get("status") != "pass"
        or parent_gate_data.get("checkpoint_sha256") != contract["source_checkpoint_sha256"]
        or parent_gate_data.get("provenance_sha256") != contract["source_provenance_sha256"]
        or parent_gate_data.get("checkpoint_epoch") != 1
        or parent_gate_data.get("split_manifest_sha256") != contract["split_manifest_sha256"]
        or parent_gate_data.get("model_tensors_finite") is not True
    ):
        raise ValueError("actual CP1 terminal gate does not validate this source parent")
    return observed


def verify_resource_environment() -> dict[str, Any]:
    """Verify the exact imported zero-worker helper before any HFT stage."""
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
        key: {"expected": value, "actual": os.environ.get(key)}
        for key, value in expected.items()
        if os.environ.get(key) != value
    }
    if mismatches:
        raise ValueError(f"bounded environment is incomplete: {mismatches}")
    bindings = _load_bindings()
    hft_root = PACKET_ROOT / bindings["snapshots"]["hft"]["root_relative_path"]
    source_paths = [hft_root / "src", hft_root / "scripts"]
    for path in reversed(source_paths):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    import wnn.resources as resources

    if not Path(resources.__file__).resolve().is_relative_to(hft_root / "src"):
        raise ValueError("zero-worker helper did not import from the frozen HFT snapshot")
    workers, prefetch = resources.dataloader_budget(
        batch_bytes=1024,
        ram_budget_bytes=1024 * 1024,
        max_workers=4,
    )
    if (workers, prefetch) != (0, 0):
        raise ValueError(f"actual imported loader helper returned {(workers, prefetch)}")
    return {
        "environment": expected,
        "resources_module": str(Path(resources.__file__).resolve()),
        "workers": workers,
        "prefetch_factor": prefetch,
        "source_sha256": bindings["snapshots"]["hft"]["files"]["src/wnn/resources.py"],
    }


def import_trainer(project_root: Path) -> tuple[ModuleType, Path]:
    """Import the exact c337 trainer and set its actual module-global ROOT."""
    bindings = _load_bindings()
    hft = bindings["snapshots"]["hft"]
    hft_root = PACKET_ROOT / hft["root_relative_path"]
    trainer_path = hft_root / "scripts/train_rnn.py"
    if sha256_file(trainer_path) != hft["files"]["scripts/train_rnn.py"]:
        raise ValueError("trainer source hash does not match c337")
    for path in (hft_root / "src", hft_root / "scripts"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    existing = sys.modules.get("train_rnn")
    if existing is not None:
        existing_path = Path(existing.__file__).resolve()
        if existing_path != trainer_path.resolve():
            raise RuntimeError(f"conflicting train_rnn module already imported: {existing_path}")
        module = existing
    else:
        spec = importlib.util.spec_from_file_location("train_rnn", trainer_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot import frozen trainer: {trainer_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules["train_rnn"] = module
        spec.loader.exec_module(module)

    module.ROOT = project_root.resolve()
    if Path(module.ROOT).resolve() != project_root.resolve():
        raise RuntimeError("trainer module-global ROOT was not rebound")
    if Path(module.Sequences("valid", [0]).path).resolve() != (
        project_root.resolve() / "datasets/valid.parquet"
    ):
        raise RuntimeError("actual trainer Sequences path does not resolve to project data")
    return module, trainer_path


def _load_crop_receipt(project_root: Path) -> tuple[dict[str, Any], str]:
    """Require the verified crop receipt before profile, fit or export."""
    if not CORE_RECEIPT_PATH.is_file():
        raise FileNotFoundError("verified actual CP1 core crop receipt is missing")
    receipt = read_object(CORE_RECEIPT_PATH)
    contract = read_object(CORE_CONTRACT_PATH)
    if receipt.get("schema") != "wnn.oldz4-core-crop-producer-receipt.v1":
        raise ValueError("core crop receipt schema mismatch")
    if receipt.get("status") != "PASS_ACTUAL_CORE_CROP":
        raise ValueError("core crop receipt does not report a successful actual crop")
    if receipt.get("contract_sha256") != sha256_file(CORE_CONTRACT_PATH):
        raise ValueError("core crop receipt is bound to a different contract")
    if receipt.get("destination_run") != contract["destination_run"]:
        raise ValueError("core crop receipt names a different output run")
    checkpoint_path = project_root / receipt["output_checkpoint_relative_path"]
    provenance_path = project_root / receipt["output_provenance_relative_path"]
    if sha256_file(checkpoint_path) != receipt["verification"]["output_checkpoint_sha256"]:
        raise ValueError("derived core checkpoint changed after its producer receipt")
    if sha256_file(provenance_path) != receipt["verification"]["output_provenance_sha256"]:
        raise ValueError("derived core provenance changed after its producer receipt")
    return receipt, sha256_file(CORE_RECEIPT_PATH)


def _require_root_receipt(
    path: Path,
    *,
    schema: str,
    decision: str,
    bindings: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    """Fail closed unless a write-once root receipt releases this exact stage."""
    if not path.is_file():
        raise FileNotFoundError(f"root-owned stage release is missing: {path.name}")
    receipt = read_object(path)
    if receipt.get("schema") != schema or receipt.get("decision") != decision:
        raise ValueError(f"root receipt does not release this stage: {path.name}")
    for key, value in bindings.items():
        if receipt.get(key) != value:
            raise ValueError(f"root receipt has wrong {key}: {path.name}")
    return receipt, sha256_file(path)


def _contract_for(stage: str) -> tuple[Path, dict[str, Any]]:
    """Return the frozen argv contract for one stage."""
    path = {
        "profile": PROFILE_CONTRACT_PATH,
        "fit": FIT_CONTRACT_PATH,
        "export": EXPORT_CONTRACT_PATH,
    }[stage]
    return path, read_object(path)


def _load_predictor(project_root: Path, trainer: ModuleType) -> tuple[ModuleType, Path]:
    """Import frozen c337 predictor and bind its actual module roots."""
    bindings = _load_bindings()
    hft_root = PACKET_ROOT / bindings["snapshots"]["hft"]["root_relative_path"]
    predictor_path = hft_root / "scripts/predict_run.py"
    if (
        sha256_file(predictor_path)
        != bindings["snapshots"]["hft"]["files"]["scripts/predict_run.py"]
    ):
        raise ValueError("frozen c337 predictor source hash mismatch")
    sys.modules.pop("predict_run", None)
    spec = importlib.util.spec_from_file_location("predict_run", predictor_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import frozen predictor: {predictor_path}")
    predictor = importlib.util.module_from_spec(spec)
    sys.modules["predict_run"] = predictor
    spec.loader.exec_module(predictor)
    predictor.ROOT = project_root
    predictor.train_rnn.ROOT = project_root
    if predictor.train_rnn is not trainer:
        raise RuntimeError("predictor did not bind to the exact pinned c337 trainer")
    if Path(predictor.ROOT).resolve() != project_root:
        raise RuntimeError("predictor module-global ROOT was not rebound")
    if Path(predictor.Sequences("valid", [0]).path).resolve() != (
        project_root / "datasets/valid.parquet"
    ):
        raise RuntimeError("actual predictor sequence path is not the project data")
    return predictor, predictor_path


def validate_import(project_root: Path) -> dict[str, Any]:
    """Check frozen imports and module-global data root without reading data."""
    project_root = project_root.resolve()
    project_inputs = verify_project_inputs(project_root)
    loader = verify_resource_environment()
    trainer, trainer_path = import_trainer(project_root)
    predictor, predictor_path = _load_predictor(project_root, trainer)
    return {
        "status": "PASS_IMPORT_AND_ROOT_BINDING",
        "project_inputs": project_inputs,
        "trainer_path": str(trainer_path.resolve()),
        "trainer_root": str(Path(trainer.ROOT).resolve()),
        "predictor_path": str(predictor_path.resolve()),
        "predictor_root": str(Path(predictor.ROOT).resolve()),
        "loader": loader,
    }


def run_stage(stage: str, project_root: Path) -> None:
    """Run a frozen stage only after its parent and root-owned gates validate."""
    project_root = project_root.resolve()
    verify_project_inputs(project_root)
    loader_receipt = verify_resource_environment()
    core_receipt, core_receipt_sha = _load_crop_receipt(project_root)
    _, contract = _contract_for(stage)
    fit = read_object(FIT_CONTRACT_PATH)
    profile = read_object(PROFILE_CONTRACT_PATH)
    export = read_object(EXPORT_CONTRACT_PATH)
    core_run = read_object(CORE_CONTRACT_PATH)["destination_run"]
    if contract.get("source_snapshot_bindings_sha256") != sha256_file(SOURCE_BINDINGS_PATH):
        raise ValueError("stage contract source bindings differ from the frozen snapshot manifest")
    if contract.get("initializer") != f"{core_run}@ep1":
        raise ValueError("stage initializer differs from the actual cropped core checkpoint")

    hft_run = contract["run"]
    run_dir = project_root / "artifacts/runs" / hft_run
    if stage in ("profile", "fit") and run_dir.exists():
        raise FileExistsError(f"stage output already exists; use a new attempt ID: {run_dir}")

    if stage == "profile":
        _require_root_receipt(
            PACKET_ROOT / "ROOT_HFT_PROFILE_LAUNCH_R5.json",
            schema="wnn.oldz4-hft-profile-launch.v1",
            decision="PASS",
            bindings={
                "profile_run": profile["run"],
                "profile_contract_sha256": sha256_file(PROFILE_CONTRACT_PATH),
                "core_crop_receipt_sha256": core_receipt_sha,
            },
        )
        if profile.get("fit_contract_sha256") != sha256_file(FIT_CONTRACT_PATH):
            raise ValueError("profile contract is bound to a different HFT fit contract")
    elif stage == "fit":
        if fit.get("source_snapshot_bindings_sha256") != sha256_file(SOURCE_BINDINGS_PATH):
            raise ValueError(
                "fit contract source bindings differ from the sealed snapshot manifest"
            )
        if fit.get("core_contract_sha256") != sha256_file(CORE_CONTRACT_PATH):
            raise ValueError("fit contract is bound to a different core crop contract")
        if profile.get("fit_contract_sha256") != sha256_file(FIT_CONTRACT_PATH):
            raise ValueError("profile contract is bound to a different HFT fit contract")
        profile_receipt, profile_sha = _require_root_receipt(
            PACKET_ROOT / "ROOT_HFT_PROFILE_ADMISSION_R5.json",
            schema="wnn.oldz4-hft-profile-admission.v1",
            decision="PASS",
            bindings={
                "profile_run": profile["run"],
                "profile_contract_sha256": sha256_file(PROFILE_CONTRACT_PATH),
                "core_crop_receipt_sha256": core_receipt_sha,
                "initializer_checkpoint_sha256": core_receipt["verification"][
                    "output_checkpoint_sha256"
                ],
            },
        )
        evidence = profile_receipt.get("profile_update_evidence")
        if not isinstance(evidence, dict) or evidence.get("valid_for_throughput") is not True:
            raise ValueError("root profile receipt lacks finite update evidence")
        _require_root_receipt(
            PACKET_ROOT / "ROOT_HFT_FIT_ADMISSION_R5.json",
            schema="wnn.oldz4-hft-fit-admission.v1",
            decision="PASS",
            bindings={
                "fit_run": fit["run"],
                "fit_contract_sha256": sha256_file(FIT_CONTRACT_PATH),
                "profile_admission_sha256": profile_sha,
                "core_crop_receipt_sha256": core_receipt_sha,
            },
        )
    elif stage == "export":
        if export.get("source_snapshot_bindings_sha256") != sha256_file(SOURCE_BINDINGS_PATH):
            raise ValueError(
                "export contract source bindings differ from the sealed snapshot manifest"
            )
        if export.get("fit_contract_sha256") != sha256_file(FIT_CONTRACT_PATH):
            raise ValueError("export contract is bound to a different HFT fit contract")
        checkpoint = run_dir / "model_ep0.pt"
        terminal_receipt, _ = _require_root_receipt(
            PACKET_ROOT / "ROOT_HFT_TERMINAL_ADMISSION_R5.json",
            schema="wnn.oldz4-hft-terminal-admission.v1",
            decision="PASS",
            bindings={
                "fit_run": fit["run"],
                "fit_contract_sha256": sha256_file(FIT_CONTRACT_PATH),
                "core_crop_receipt_sha256": core_receipt_sha,
            },
        )
        if terminal_receipt.get("checkpoint_epoch") != 0 or not checkpoint.is_file():
            raise ValueError("terminal receipt or fixed HFT CP0 is missing")
        if terminal_receipt.get("checkpoint_sha256") != sha256_file(checkpoint):
            raise ValueError("terminal receipt is bound to different HFT checkpoint bytes")
        output = project_root / export["prediction_relative_path"]
        sidecar = project_root / export["sidecar_relative_path"]
        if output.exists() or sidecar.exists():
            raise FileExistsError("HFT export path exists; refuse to overwrite prior evidence")

    if stage == "export":
        trainer, _ = import_trainer(project_root)
        predictor, predictor_path = _load_predictor(project_root, trainer)
        sys.argv = [str(predictor_path), *export["predictor_argv"]]
        predictor.main()
        return

    trainer, trainer_path = import_trainer(project_root)
    argv = profile["trainer_argv"] if stage == "profile" else fit["trainer_argv"]
    sys.argv = [str(trainer_path), *argv]
    trainer.main()
    print(json.dumps({"stage": stage, "loader": loader_receipt}, sort_keys=True))


def main() -> None:
    """Run an import check, profile, admitted fit or admitted export."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=("validate-import", "profile", "fit", "export"), required=True
    )
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    if args.stage == "validate-import":
        result = validate_import(args.project_root)
        print(json.dumps(result, sort_keys=True))
    else:
        run_stage(args.stage, args.project_root)


if __name__ == "__main__":
    main()
