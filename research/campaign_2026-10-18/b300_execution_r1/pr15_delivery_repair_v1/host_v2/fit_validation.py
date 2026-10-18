"""Independent post-fit contract for one frozen PR15 host cell."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

from checkpoint_identity import (
    validate_provenance_file_hashes,
    verify_checkpoint_pair,
)

SOURCE_COMMIT = "e238fc5c82ac8c4591f3ca3ef87c37b2b73b929e"
SOURCE_MANIFEST_SHA256 = "873662aded939afd64c17c80b684011de023c04a31be5b2c74e0b9d97a18efc0"
SCORE_SUPPORT_SHA256 = "fb7a85a5837f61fa6c65cd10a799768006067e87d4c18e92d4b813ba83b9d82e"
SCORE_SUPPORT_MANIFEST_SHA256 = "698ae26944b96ffb1b2b2ccb2de11cab427d3c822c2ca74cdacff1b1ef7d6368"
EXPECTED_UPDATES = {0: 44760, 1: 44790, 2: 44850, 3: 44640}


def _sha256(path: Path) -> str:
    """Hash one artifact with bounded memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    """Read a JSON object from one immutable receipt."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object at {path}")
    return value


def _bound_path(work: Path, text: str) -> Path:
    """Resolve one receipt path and require it to remain inside the worker root."""
    path = Path(text).resolve(strict=True)
    if not path.is_relative_to(work.resolve(strict=True)):
        raise ValueError(f"receipt path escaped the worker workspace: {path}")
    return path


def _check_contract() -> dict[str, Any]:
    """Require the externally sealed fit-only contract and this exact validator."""
    package = Path(__file__).resolve().parent
    contract_path = package / "FIT_CONTRACT.json"
    contract = _json(contract_path)
    expected_files = {
        "fit_validation.py": _sha256(Path(__file__).resolve()),
        "checkpoint_identity.py": _sha256(package / "checkpoint_identity.py"),
    }
    if (
        contract.get("schema") != "tabdata-pr15-fit-contract.v2"
        or contract.get("files") != expected_files
        or contract.get("profile_packet_sha256")
        != "c18f1877b8579c03d2a0905534f5ecbd13ed56a54306ba66d63b47d892b66510"
        or contract.get("parent_fit_contract_sha256")
        != "05aa087af1fdd68ac97146e6f56750157d4a33ebd4fce2a5ec2f5eb30cc6bcb8"
    ):
        raise ValueError("fit-only validator differs from externally sealed FIT_CONTRACT.json")
    return contract


def _canonical_argv(
    source_root: Path,
    name: str,
    fold: int,
    seed: int,
    control: str,
    split_path: Path,
    phase: str = "train",
) -> list[str]:
    """Rebuild the exact frozen PR15 command, replacing only its split-file path."""
    builder_path = source_root / "scripts" / "geometry_b300" / "common.py"
    manifest = _json(Path(__file__).with_name("source_manifest.json"))
    records = {row["path"]: row["sha256"] for row in manifest["files"]}
    relative = "scripts/geometry_b300/common.py"
    if relative not in records or _sha256(builder_path) != records[relative]:
        raise ValueError(
            "PR15 canonical command builder is absent or differs from its source manifest"
        )
    builder_dir = str(builder_path.parent)
    module_name = "_pr15_fit_validation_common"
    spec = importlib.util.spec_from_file_location(module_name, builder_path)
    if spec is None or spec.loader is None:
        raise ImportError("cannot load the hash-pinned PR15 command builder")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, builder_dir)
    try:
        spec.loader.exec_module(module)
        command = (
            module.prediction_command(name, fold)
            if phase == "predict"
            else module.train_command(name, fold, seed, control)
        )
    finally:
        sys.path.remove(builder_dir)
        sys.modules.pop("mounts", None)
    if (
        not isinstance(command, list)
        or len(command) < 4
        or command[2]
        != ("scripts/predict_run.py" if phase == "predict" else "scripts/train_rnn.py")
    ):
        raise ValueError("PR15 command builder returned an unsupported trainer command")
    argv = [str(value) for value in command[3:]]
    try:
        split_index = argv.index(
            "--campaign-split-manifest" if phase == "predict" else "--split-manifest"
        )
    except ValueError as error:
        raise ValueError("PR15 canonical command omitted its campaign split manifest") from error
    if split_index + 1 >= len(argv):
        raise ValueError("PR15 canonical command has no split manifest value")
    argv[split_index + 1] = str(split_path)
    return argv


def _check_request_and_split(
    work: Path,
    request: dict[str, Any],
    worker: dict[str, Any],
    *,
    name: str,
    fold: int,
    seed: int,
    control: str,
    split_sha256: str,
) -> tuple[Path, dict[str, Any]]:
    """Bind worker request and canonical fold file to the claimed cell identity."""
    request_path = work / "worker_request.json"
    workspace = worker.get("workspace", {})
    source_root = Path(request.get("source_root", "")).resolve(strict=True)
    project_root = source_root.parent.parent
    split_path = (project_root / "splits" / f"fold{fold}.json").resolve(strict=True)
    if (
        request.get("schema") != "pr15-host-worker-request.v1"
        or request.get("mode") != "fit"
        or request.get("name") != name
        or Path(request.get("work", "")).resolve(strict=True) != work
        or source_root != (project_root / "runtime" / "pr15-source-e238-v1").resolve(strict=True)
        or worker.get("request_sha256") != _sha256(request_path)
        or worker.get("split_hash") != split_sha256
        or workspace.get("fold") != fold
        or workspace.get("split_manifest") != str(split_path)
        or workspace.get("split_manifest_sha256") != split_sha256
        or _sha256(split_path) != split_sha256
    ):
        raise ValueError("worker request or fold manifest differs from the requested cell")
    expected_predict = _canonical_argv(
        source_root, name, fold, seed, control, split_path, "predict"
    )
    if request.get("predict_argv") != expected_predict:
        raise ValueError("Prediction request differs from the exact canonical epoch-2 command")
    train_argv = request.get("train_argv")
    expected_argv = _canonical_argv(source_root, name, fold, seed, control, split_path)
    if train_argv != expected_argv:
        raise ValueError("worker trainer argv differs from the frozen PR15 builder")
    split = _json(split_path)
    if (
        split.get("fold") != fold
        or split.get("n_folds") != 4
        or not isinstance(split.get("evaluation", {}).get("valid"), list)
    ):
        raise ValueError("canonical split manifest has the wrong fold or evaluation membership")
    return source_root, split


def _phase_receipt(
    work: Path, receipt: dict[str, Any], *, actor: str | None, gpu_uuid: str
) -> dict[str, Any]:
    """Verify successful phase log and optional GPU resource-envelope receipt hashes."""
    if receipt.get("exit_code") != 0:
        raise ValueError("phase did not exit successfully")
    log_path = _bound_path(work, str(receipt.get("log", "")))
    log_hash = _sha256(log_path)
    if log_hash != receipt.get("log_sha256"):
        raise ValueError(f"phase log hash mismatch: {log_path}")
    result: dict[str, Any] = {"log": str(log_path), "log_sha256": log_hash}
    if actor is not None:
        resource_path = _bound_path(work, str(receipt.get("resource_receipt", "")))
        resource_hash = _sha256(resource_path)
        if resource_hash != receipt.get("resource_receipt_sha256"):
            raise ValueError(f"phase resource receipt hash mismatch: {resource_path}")
        resource = _json(resource_path)
        if (
            resource.get("script") != actor
            or resource.get("status") != "actor_returned"
            or resource.get("visible_devices") != gpu_uuid
            or resource.get("allocator_budget_bytes") != 6 * 1024**3
            or "B300" not in str(resource.get("gpu", ""))
            or resource.get("allocator_limit_is_not_a_device_memory_reservation") is not True
        ):
            raise ValueError(f"phase process envelope does not match the assigned B300: {actor}")
        for key in ("cuda_peak_allocated_bytes", "cuda_peak_reserved_bytes"):
            value = resource.get(key)
            if type(value) is not int or not 0 <= value <= 6 * 1024**3:
                raise ValueError(f"GPU peak envelope evidence exceeds its budget: {key}={value}")
        result.update(resource_receipt=str(resource_path), resource_receipt_sha256=resource_hash)
    return result


def _check_recipe(
    args: dict[str, Any],
    *,
    name: str,
    fold: int,
    seed: int,
    control: str,
    split_sha256: str,
) -> None:
    """Require the PR15 command's unchanged model, optimizer, control and endpoint."""
    expected: dict[str, Any] = {
        "name": name,
        "seed": seed,
        "valid_fold": fold,
        "n_valid_folds": 4,
        "eval_source": "vfold",
        "arch": "gru",
        "cell": "gru",
        "enc": "linear",
        "hidden": 512,
        "layers": 1,
        "dropout": 0.0,
        "head_layers": 1,
        "batch": 8,
        "chunk": 2000,
        "epochs": 3,
        "checkpoint_policy": "fixed",
        "checkpoint_epoch": 2,
        "no_eval_during_fit": True,
        "save_training_state": True,
        "train_source": "train+valid",
        "train_seqs": 10607,
        "loss": "wmse",
        "optim": "adamw",
        "sched": "onecycle",
        "lr": 0.001,
        "wd": 0.01,
        "clip": 1.0,
        "precision": "fp32",
        "deterministic": True,
        "grad_stats": True,
        "rnn_init": "default",
        "eval_ema": False,
        "numeric_map": "none",
        "research_features": "causal_geometry_discrete",
        "research_feature_control": control,
        "split_manifest_sha256": split_sha256,
    }
    for key, expected_value in expected.items():
        observed = args.get(key)
        if observed != expected_value:
            raise ValueError(
                f"checkpoint recipe field {key} expected {expected_value!r}, observed {observed!r}"
            )
    if args.get("device") != "cuda":
        raise ValueError("PR15 checkpoint was not trained on the assigned CUDA device")


def _check_update_totals(provenance: dict[str, Any], history: list[Any], fold: int) -> int:
    """Require full three-epoch attempted/completed counts and zero skipped updates."""
    expected = EXPECTED_UPDATES[fold]
    sources = [
        provenance.get("cumulative_updates"),
        history[-1].get("update_diagnostics", {}).get("cumulative"),
    ]
    for counts in sources:
        if not isinstance(counts, dict):
            raise TypeError("checkpoint/history lacks cumulative optimizer counters")
        if (
            type(counts.get("optimizer_attempts")) is not int
            or counts["optimizer_attempts"] != expected
            or type(counts.get("optimizer_completed_updates")) is not int
            or counts["optimizer_completed_updates"] != expected
            or type(counts.get("amp_skipped_updates")) is not int
            or counts["amp_skipped_updates"] != 0
        ):
            raise ValueError(
                f"fold {fold} optimizer totals must be {expected} completed attempts, zero skipped"
            )
    epoch_totals = {
        "optimizer_attempts": 0,
        "optimizer_completed_updates": 0,
        "amp_skipped_updates": 0,
    }
    for epoch, row in enumerate(history):
        diagnostics = row.get("update_diagnostics", {})
        current = diagnostics.get("epoch")
        if not isinstance(current, dict):
            raise TypeError("fit history lacks per-epoch optimizer counters")
        for key in epoch_totals:
            value = current.get(key)
            if type(value) is not int:
                raise ValueError(f"fit history has invalid per-epoch counter {key}")
            epoch_totals[key] += value
        if current["optimizer_attempts"] != current["optimizer_completed_updates"]:
            raise ValueError(f"epoch {epoch} contains skipped optimizer updates")
        cumulative = diagnostics.get("cumulative", {})
        if not isinstance(cumulative, dict) or any(
            cumulative.get(key) != epoch_totals[key] for key in epoch_totals
        ):
            raise ValueError(f"epoch {epoch} cumulative counters do not match its history")
    if (
        epoch_totals["optimizer_attempts"] != expected
        or epoch_totals["optimizer_completed_updates"] != expected
        or epoch_totals["amp_skipped_updates"] != 0
    ):
        raise ValueError("per-epoch optimizer totals do not match the full PR15 fold recipe")
    provenance_diagnostics = provenance.get("update_diagnostics")
    if not isinstance(provenance_diagnostics, list) or len(provenance_diagnostics) != 3:
        raise ValueError("checkpoint provenance lacks all three epoch update records")
    for epoch, (history_row, provenance_row) in enumerate(
        zip(history, provenance_diagnostics, strict=True)
    ):
        if (
            provenance_row.get("epoch_index") != epoch
            or provenance_row.get("epoch") != history_row["update_diagnostics"].get("epoch")
            or provenance_row.get("cumulative")
            != history_row["update_diagnostics"].get("cumulative")
        ):
            raise ValueError("checkpoint and history per-epoch update records differ")
    return expected


def _check_trainer_source(
    provenance: dict[str, Any], source_root: Path, contract: dict[str, Any]
) -> None:
    """Bind checkpoint provenance to the PR15 manifest's public trainer files."""
    source_manifest_path = Path(__file__).with_name("source_manifest.json")
    if _sha256(source_manifest_path) != SOURCE_MANIFEST_SHA256:
        raise ValueError("PR15 source manifest does not match the fit contract")
    manifest = _json(source_manifest_path)
    all_files = {row["path"]: row["sha256"] for row in manifest["files"]}
    expected_paths = {"scripts/train_rnn.py", "pyproject.toml", "uv.lock"}
    expected_paths.update(
        path.relative_to(source_root).as_posix()
        for path in (source_root / "src" / "wnn" / "training").glob("*.py")
    )
    if not expected_paths.issubset(all_files):
        raise ValueError("PR15 manifest omits one or more trainer implementation files")
    expected = {path: all_files[path] for path in expected_paths}
    implementation = provenance.get("trainer_implementation", {})
    if implementation.get("files") != expected:
        raise ValueError("checkpoint trainer implementation hashes differ from frozen PR15 source")
    aggregate = hashlib.sha256(
        json.dumps(expected, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if implementation.get("sha256") != aggregate:
        raise ValueError("checkpoint aggregate trainer source identity is invalid")


def _check_model_state(checkpoint: dict[str, Any], torch: Any) -> None:
    """Require the actual checkpoint tensors to be the 271x512 one-layer FP32 GRU."""
    state = checkpoint.get("model")
    expected_shapes = {
        "mean": (271,),
        "std": (271,),
        "inp.weight": (512, 271),
        "inp.bias": (512,),
        "gru.weight_ih_l0": (1536, 512),
        "gru.weight_hh_l0": (1536, 512),
        "gru.bias_ih_l0": (1536,),
        "gru.bias_hh_l0": (1536,),
        "head.0.weight": (512, 1024),
        "head.0.bias": (512,),
        "head.2.weight": (2, 512),
        "head.2.bias": (2,),
    }
    if not isinstance(state, dict) or set(state) != set(expected_shapes):
        raise ValueError("checkpoint tensor keys do not match the unmodified H512x1 GRU")
    for name, shape in expected_shapes.items():
        tensor = state[name]
        if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != shape:
            raise ValueError(f"checkpoint model tensor {name} has an unexpected shape")
        if tensor.dtype != torch.float32 or not torch.isfinite(tensor).all().item():
            raise ValueError(f"checkpoint model tensor {name} is not finite strict FP32")
    if not torch.all(state["std"] > 0).item():
        raise ValueError("checkpoint normalization scale contains nonpositive values")


def _check_source_and_scorer(package: Path, work: Path, worker: dict[str, Any]) -> dict[str, str]:
    """Verify the worker's frozen source and byte-identical native scorer copies."""
    manifest_path = package / "source_manifest.json"
    manifest = _json(manifest_path)
    if (
        _sha256(manifest_path) != SOURCE_MANIFEST_SHA256
        or manifest.get("source_commit") != SOURCE_COMMIT
    ):
        raise ValueError("PR15 source manifest is not the pinned e238 snapshot")
    source_files = {row["path"]: row["sha256"] for row in manifest["files"]}
    source_identity = worker.get("source", {})
    if (
        source_identity.get("source_tree_sha256") != manifest.get("source_tree_sha256")
        or source_identity.get("archive_sha256") != manifest.get("archive_sha256")
        or source_identity.get("file_count") != manifest.get("file_count")
    ):
        raise ValueError("worker source receipt differs from the frozen PR15 source manifest")
    scorer_manifest_path = package / "score_support_manifest.json"
    scorer_manifest = _json(scorer_manifest_path)
    helper_rows = scorer_manifest.get("files", [])
    helper_record = helper_rows[0] if len(helper_rows) == 1 else {}
    scorer_support = worker.get("scorer_support", {})
    scorer_hashes = {
        "source_scorer_sha256": source_files.get("scripts/score_run.py", ""),
        "support_helper_sha256": SCORE_SUPPORT_SHA256,
    }
    if (
        _sha256(scorer_manifest_path) != SCORE_SUPPORT_MANIFEST_SHA256
        or scorer_manifest.get("source_commit") != SOURCE_COMMIT
        or helper_record.get("sha256") != SCORE_SUPPORT_SHA256
        or scorer_support.get("manifest_sha256") != SCORE_SUPPORT_MANIFEST_SHA256
        or scorer_support.get("helper_sha256") != SCORE_SUPPORT_SHA256
        or scorer_support.get("git_blob_oid") != "7c8450d40bdf3e355ced2cddeb2b846598ce1554"
        or scorer_hashes["source_scorer_sha256"] == ""
    ):
        raise ValueError("source-native PR15 scorer identity differs from its pinned manifests")
    copies = worker.get("workspace", {}).get("scorer_copy_sha256", {})
    actual_copies = {
        "score_run.py": _sha256(work / "scripts" / "score_run.py"),
        "grade_prediction_bank.py": _sha256(work / "scripts" / "grade_prediction_bank.py"),
    }
    expected_copies = {
        "score_run.py": scorer_hashes["source_scorer_sha256"],
        "grade_prediction_bank.py": SCORE_SUPPORT_SHA256,
    }
    if copies != actual_copies or actual_copies != expected_copies:
        raise ValueError("worker scorer files differ from the exact PR15 scorer source")
    return scorer_hashes


def _check_checkpoint(
    work: Path,
    name: str,
    fold: int,
    seed: int,
    control: str,
    split_sha256: str,
    source_root: Path,
    contract: dict[str, Any],
) -> dict[str, Any]:
    """Load the trusted checkpoint on CPU and verify architecture, recipe and full fit counts."""
    run_dir = work / "artifacts" / "runs" / name
    checkpoint_path = run_dir / "model_ep2.pt"
    selected_path = run_dir / "model.pt"
    provenance_path = run_dir / "provenance.json"
    history_path = run_dir / "history.json"
    state_path = run_dir / "training_state.pt"
    for path in (
        checkpoint_path,
        selected_path,
        provenance_path,
        history_path,
        state_path,
    ):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"fit artifact is missing or empty: {path}")
    checkpoint_identity = verify_checkpoint_pair(checkpoint_path, selected_path)
    checkpoint_hash = checkpoint_identity.fixed_file_sha256
    import torch

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if (
        not isinstance(checkpoint, dict)
        or checkpoint.get("epoch") != 2
        or checkpoint.get("n_in") != 271
    ):
        raise ValueError(
            "checkpoint is not the fixed epoch-2 H512 geometry model with input width 271"
        )
    _check_model_state(checkpoint, torch)
    args = checkpoint.get("args", {})
    _check_recipe(
        args,
        name=name,
        fold=fold,
        seed=seed,
        control=control,
        split_sha256=split_sha256,
    )
    provenance = _json(provenance_path)
    validate_provenance_file_hashes(provenance, checkpoint_identity)
    if (
        provenance.get("split_manifest_sha256") != split_sha256
        or provenance.get("args") != args
        or provenance.get("seed") != seed
        or checkpoint.get("provenance", {}).get("split_manifest_sha256") != split_sha256
    ):
        raise ValueError("checkpoint/provenance split identity differs from requested fold")
    _check_trainer_source(provenance, source_root, contract)
    history = json.loads(history_path.read_text(encoding="utf-8"))
    if (
        not isinstance(history, list)
        or len(history) != 3
        or [row.get("epoch") for row in history] != [0, 1, 2]
        or any(row.get("evaluation") != "withheld" for row in history)
    ):
        raise ValueError("fit history does not contain three evaluation-withheld epochs")
    update_total = _check_update_totals(provenance, history, fold)
    if (
        checkpoint.get("provenance", {}).get("cumulative_updates", {}).get("optimizer_attempts")
        != update_total
    ):
        raise ValueError("epoch-2 checkpoint provenance does not bind the full optimizer count")
    del checkpoint
    return {
        "run_dir": str(run_dir),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_identity": checkpoint_identity.receipt(),
        "provenance_sha256": _sha256(provenance_path),
        "history_sha256": _sha256(history_path),
        "training_state_sha256": _sha256(state_path),
        "epochs": 3,
        "epoch": 2,
        "input_width": 271,
        "hidden": 512,
        "layers": 1,
        "optimizer_attempts": update_total,
        "optimizer_completed_updates": update_total,
        "amp_skipped_updates": 0,
    }


def _check_export_and_score(
    work: Path,
    name: str,
    split_sha256: str,
    checkpoint_sha256: str,
    expected_groups: list[int],
    expected_updates: int,
    score_receipt: dict[str, Any],
) -> dict[str, Any]:
    """Bind evaluation-only predictions to the fitted checkpoint and finite official score."""
    import numpy as np

    prediction = work / "cache" / f"pred_{name}_valid.npy"
    sidecar_path = prediction.with_suffix(".json")
    score_path = work / "artifacts" / "runs" / name / "score.json"
    for path in (prediction, sidecar_path, score_path):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"prediction/score artifact is missing: {path}")
    sidecar, score = _json(sidecar_path), _json(score_path)
    expected_score = score_receipt.get("score", {}).get("score")
    if score != expected_score:
        raise ValueError("score receipt differs from the source-native score.json bytes")
    groups = sidecar.get("row_groups")
    if (
        sidecar.get("run") != name
        or sidecar.get("epoch") != 2
        or sidecar.get("campaign_export_scope") != "evaluation"
        or sidecar.get("campaign_split_manifest_sha256") != split_sha256
        or sidecar.get("split_manifest_sha256") != split_sha256
        or sidecar.get("checkpoint_sha256") != checkpoint_sha256
        or not isinstance(groups, list)
        or groups != expected_groups
        or sidecar.get("prediction_sha256") != _sha256(prediction)
    ):
        raise ValueError("prediction sidecar is not hash-bound to this fold/checkpoint")
    array = np.load(prediction, mmap_mode="r", allow_pickle=False)
    if array.shape != (len(groups), 20_000, 2) or array.dtype != np.float32:
        raise ValueError(f"unexpected PR15 prediction shape/dtype: {array.shape}/{array.dtype}")
    for key in ("wp", "t0", "t1", "finite_fraction"):
        value = score.get(key)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError(f"official score field {key} is missing or nonfinite")
    if (
        score.get("schema") != "wnn.hpo-score.v1"
        or score.get("run") != name
        or score.get("epoch") != 2
        or score.get("finite_fraction") != 1.0
        or score.get("amp_skipped") != 0
        or score.get("optimizer_attempts") != expected_updates
    ):
        raise ValueError("source-native PR15 score does not prove a finite fixed endpoint")
    return {
        "prediction_path": str(prediction),
        "prediction_sha256": _sha256(prediction),
        "sidecar_path": str(sidecar_path),
        "sidecar_sha256": _sha256(sidecar_path),
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "score_path": str(score_path),
        "score_sha256": _sha256(score_path),
        "metric_id": "global_wp_clip2_v1",
        "score": score,
    }


def validate_full_fit(
    work: Path, name: str, fold: int, seed: int, control: str, split_sha256: str
) -> dict[str, Any]:
    """Independently verify one PR15 checkpoint, evaluation export, score and receipts."""
    work = work.resolve(strict=True)
    contract = _check_contract()
    if not 0 <= fold < 4 or control not in {
        "all",
        "moments_only",
        "dictionary_only",
        "none",
    }:
        raise ValueError("invalid PR15 fold or feature control")
    maximum_seed = 1008 if fold < 2 else 1007
    if not 1001 <= seed <= maximum_seed or len(split_sha256) != 64:
        raise ValueError("seed or split hash is outside the frozen PR15 matrix")
    worker_path = work / "worker_receipt.json"
    fit_path = work / "fit_receipt.json"
    export_path = work / "export_receipt.json"
    score_receipt_path = work / "score_receipt.json"
    request_path = work / "worker_request.json"
    for path in (worker_path, fit_path, export_path, score_receipt_path, request_path):
        if not path.is_file():
            raise FileNotFoundError(f"terminal phase receipt is missing: {path}")
    worker = _json(worker_path)
    fit = _json(fit_path)
    export = _json(export_path)
    score_receipt = _json(score_receipt_path)
    request = _json(request_path)
    identity = (
        worker.get("name"),
        worker.get("fold"),
        worker.get("seed"),
        worker.get("control"),
    )
    if (
        worker.get("status") != "fit_export_grade_completed"
        or identity != (name, fold, seed, control)
        or worker.get("source", {}).get("source_commit") != SOURCE_COMMIT
        or worker.get("source", {}).get("manifest_sha256") != SOURCE_MANIFEST_SHA256
        or worker.get("source", {}).get("file_count") != 172
        or request.get("mode") != "fit"
        or request.get("name") != name
    ):
        raise ValueError("worker terminal receipt does not bind the requested PR15 cell")
    source_root, split = _check_request_and_split(
        work,
        request,
        worker,
        name=name,
        fold=fold,
        seed=seed,
        control=control,
        split_sha256=split_sha256,
    )
    scorer_hashes = _check_source_and_scorer(Path(__file__).resolve().parent, work, worker)
    for label, receipt, expected_status in (
        ("fit", fit, "fit_completed"),
        ("export", export, "export_completed"),
        ("score", score_receipt, "graded_source_native"),
    ):
        if receipt.get("status") != expected_status:
            raise ValueError(f"{label} phase receipt does not report {expected_status}")
    if (
        fit.get("name") != name
        or fit.get("fold") != fold
        or fit.get("seed") != seed
        or fit.get("control") != control
        or fit.get("request_sha256") != worker.get("request_sha256")
        or fit.get("source") != worker.get("source")
        or score_receipt.get("metric_id") != "global_wp_clip2_v1"
    ):
        raise ValueError("fit or score receipt does not bind the worker cell identity")
    checkpoint = _check_checkpoint(
        work, name, fold, seed, control, split_sha256, source_root, contract
    )
    if fit.get("fit_artifacts", {}).get("checkpoint_identity") != checkpoint.get(
        "checkpoint_identity"
    ):
        raise ValueError("fit receipt checkpoint identity differs from the fixed artifact")
    gpu_uuid = str(worker.get("gpu", {}).get("uuid", ""))
    if not gpu_uuid:
        raise ValueError("worker receipt does not record the assigned GPU UUID")
    phase_receipts = {
        "fit": _phase_receipt(work, fit.get("phase", {}), actor="train_cell.py", gpu_uuid=gpu_uuid),
        "export": _phase_receipt(
            work, export.get("phase", {}), actor="export_cell.py", gpu_uuid=gpu_uuid
        ),
        "score": _phase_receipt(
            work, score_receipt.get("phase", {}), actor=None, gpu_uuid=gpu_uuid
        ),
    }
    if (
        export.get("prediction_sha256") != _sha256(work / "cache" / f"pred_{name}_valid.npy")
        or export.get("sidecar_sha256")
        != _sha256((work / "cache" / f"pred_{name}_valid.npy").with_suffix(".json"))
        or export.get("evaluation_groups") != split["evaluation"]["valid"]
        or export.get("epoch") != 2
    ):
        raise ValueError(
            "export receipt does not bind the expected prediction bytes and fold groups"
        )
    score = _check_export_and_score(
        work,
        name,
        split_sha256,
        checkpoint["checkpoint_sha256"],
        split["evaluation"]["valid"],
        EXPECTED_UPDATES[fold],
        score_receipt,
    )
    receipt_hashes = {
        "worker_receipt_sha256": _sha256(worker_path),
        "fit_receipt_sha256": _sha256(fit_path),
        "export_receipt_sha256": _sha256(export_path),
        "score_receipt_sha256": _sha256(score_receipt_path),
        "request_sha256": _sha256(request_path),
    }
    return {
        "schema": "pr15-fit-validation-result.v1",
        "status": "fit_export_grade_completed",
        "source_commit": SOURCE_COMMIT,
        "name": name,
        "fold": fold,
        "seed": seed,
        "control": control,
        "checkpoint": checkpoint,
        "score": score,
        "phase_receipts": phase_receipts,
        "receipt_hashes": receipt_hashes,
        "scorer_hashes": scorer_hashes,
        "fit_contract_sha256": _sha256(Path(__file__).with_name("FIT_CONTRACT.json")),
        "validator_sha256": _sha256(Path(__file__).resolve()),
    }
