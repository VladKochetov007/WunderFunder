"""Score every retained R5 Transformer epoch on its bound inner split."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

from epoch_contract import (
    EXPECTED_EPOCHS,
    ROOT,
    canonical_sha256,
    file_sha256,
    read_object,
    safe_file,
    validate_epoch_index,
    validate_split_pair,
    write_json_new,
)
from epoch_sweep_receipts import validate_prediction, validate_score
from epoch_sweep_resume import prepare_sweep_output

SWEEP_SCHEMA = "wnn.recovery-r5-epoch-sweep.v1"


def _implementation_hashes() -> dict[str, str]:
    """Bind every local scoring component that can change the retained-epoch result."""
    return {
        "sweep": file_sha256(Path(__file__).resolve()),
        "receipts": file_sha256(Path(__file__).with_name("epoch_sweep_receipts.py")),
        "resume": file_sha256(Path(__file__).with_name("epoch_sweep_resume.py")),
        "predictor": file_sha256(ROOT / "src/wnn/training/sequence_predict.py"),
        "scorer": file_sha256(ROOT / "src/wnn/training/sequence_score.py"),
    }


def _evaluation_environment(nproc_per_node: int) -> dict[str, str]:
    """Require the admitted Linux B300 precision environment for GPU prediction."""
    if platform.system() != "Linux":
        raise RuntimeError("epoch evaluation is a receiving-host Linux GPU action")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible or len([item for item in visible.split(",") if item.strip()]) != nproc_per_node:
        raise RuntimeError("CUDA_VISIBLE_DEVICES must list exactly the requested GPU ranks")
    if os.environ.get("CUBLAS_EMULATE_SINGLE_PRECISION") != "1":
        raise RuntimeError("BF16/BF16x9 epoch evaluation requires cuBLAS emulation")
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src") + (
        os.pathsep + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else ""
    )
    environment.setdefault("OMP_NUM_THREADS", "1")
    environment.setdefault("MKL_NUM_THREADS", "1")
    environment.setdefault("OPENBLAS_NUM_THREADS", "1")
    return environment


def _execute(command: list[str], log_path: Path, environment: dict[str, str]) -> None:
    """Run one receiver command and preserve its complete log on success or failure."""
    with log_path.open("x", encoding="utf-8") as log:
        subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            check=True,
            stdout=log,
            stderr=subprocess.STDOUT,
        )


def evaluate_fold(
    fit_dir: Path,
    config_path: Path,
    data_root: Path,
    outer_split: Path,
    inner_split: Path,
    epoch_index: Path,
    epoch_index_sha256: str,
    output: Path,
    nproc_per_node: int = 8,
    resume: bool = False,
) -> dict[str, Any]:
    """Run indexed inference and canonical scoring for all 30 retained epochs."""
    config_path = config_path.expanduser().resolve(strict=True)
    data_root = data_root.expanduser().resolve(strict=True)
    config, _ = read_object(config_path)
    fold = config.get("expected_fold")
    if type(fold) is not int or nproc_per_node != config.get("required_world_size"):
        raise ValueError("R5 cell fold or receiving GPU count differs from its frozen config")
    split_binding = validate_split_pair(outer_split, inner_split, fold)
    fit = validate_epoch_index(fit_dir, epoch_index, epoch_index_sha256, config_path, split_binding)
    output = output.expanduser()
    if output.is_symlink():
        raise ValueError("epoch sweep output cannot be a symlink")
    output = output.resolve()
    implementation = _implementation_hashes()
    source_hash = implementation["sweep"]
    plan = {
        "schema": SWEEP_SCHEMA,
        "status": "planned",
        "fold": fold,
        "expected_epochs": EXPECTED_EPOCHS,
        "fit_dir": fit["fit_dir"],
        "config_path": str(config_path),
        "config_file_sha256": fit["config_file_sha256"],
        "config_recipe_sha256": fit["config_recipe_sha256"],
        "config_identity_sha256": fit["identity"]["config_sha256"],
        "data_root": str(data_root),
        "outer_split": split_binding["outer_path"],
        "outer_split_sha256": split_binding["outer_sha256"],
        "inner_split": split_binding["inner_path"],
        "inner_split_sha256": split_binding["inner_sha256"],
        "epoch_index": str(Path(epoch_index).resolve()),
        "epoch_index_sha256": fit["index_sha256"],
        "epoch_index_file_sha256": fit["index_file_sha256"],
        "final_model_sha256": fit["final_model_sha256"],
        "identity": fit["identity"],
        "runner_sha256": source_hash,
        "implementation_sha256": implementation,
        "nproc_per_node": nproc_per_node,
    }
    state = prepare_sweep_output(output, plan, fit, split_binding, resume)
    if state["sweep"] is not None:
        return state["sweep"]
    epochs_root = state["epochs_root"]
    records = list(state["records"])
    completed_records = {record["completed_epochs"]: record for record in records}
    pending = [
        entry for entry in fit["entries"] if entry["completed_epochs"] not in completed_records
    ]
    environment = _evaluation_environment(nproc_per_node) if pending else {}
    for entry in pending:
        completed = entry["completed_epochs"]
        epoch_dir = epochs_root / f"epoch-{completed:06d}"
        epoch_dir.mkdir()
        predictions = epoch_dir / "predictions"
        checkpoint = safe_file(Path(fit["fit_dir"]), entry["checkpoint_path"])
        predict_command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nnodes=1",
            f"--nproc-per-node={nproc_per_node}",
            str(ROOT / "scripts/predict_sequence.py"),
            "--checkpoint",
            str(checkpoint),
            "--epoch-index",
            str(Path(epoch_index).resolve()),
            "--epoch-index-sha256",
            fit["index_sha256"],
            "--data-root",
            str(data_root),
            "--split-manifest",
            split_binding["inner_path"],
            "--output",
            str(predictions),
            "--membership",
            "evaluation",
        ]
        _execute(predict_command, epoch_dir / "predict.log", environment)
        prediction_manifest = predictions / "manifest.json"
        manifest, _ = read_object(prediction_manifest)
        producer = validate_prediction(
            manifest, entry, fit["index"], fit["index_sha256"], split_binding["inner_sha256"]
        )
        score_path = epoch_dir / "score.json"
        score_command = [
            sys.executable,
            str(ROOT / "scripts/score_sequence.py"),
            "--data-root",
            str(data_root),
            "--split-manifest",
            split_binding["inner_path"],
            "--predictions",
            str(predictions),
            "--output",
            str(score_path),
        ]
        _execute(score_command, epoch_dir / "score.log", environment)
        score, _ = read_object(score_path)
        metric = validate_score(score, producer, prediction_manifest, split_binding["inner_sha256"])
        record = {
            "epoch": entry["epoch"],
            "completed_epochs": completed,
            "checkpoint_path": entry["checkpoint_path"],
            "checkpoint_sha256": entry["checkpoint_sha256"],
            "prediction_manifest_sha256": file_sha256(prediction_manifest),
            "score_sha256": file_sha256(score_path),
            "score": metric,
        }
        receipt = {
            "schema": "wnn.recovery-r5-epoch-score.v1",
            "fold": fold,
            "completed_epochs": completed,
            "checkpoint_sha256": entry["checkpoint_sha256"],
            "config_identity_sha256": fit["identity"]["config_sha256"],
            "outer_split_sha256": split_binding["outer_sha256"],
            "inner_split_sha256": split_binding["inner_sha256"],
            "epoch_index_sha256": fit["index_sha256"],
            "prediction_manifest_sha256": record["prediction_manifest_sha256"],
            "score_sha256": record["score_sha256"],
            "score": metric,
        }
        receipt_path = epoch_dir / "epoch.json"
        record["receipt_sha256"] = write_json_new(receipt_path, receipt)
        records.append(record)

    if _implementation_hashes() != implementation:
        raise ValueError("sweep/scoring source changed while the epoch evaluation was running")
    final_split = validate_split_pair(outer_split, inner_split, fold)
    final_fit = validate_epoch_index(
        fit_dir, epoch_index, epoch_index_sha256, config_path, final_split
    )
    if (
        final_split["outer_sha256"] != split_binding["outer_sha256"]
        or final_split["inner_sha256"] != split_binding["inner_sha256"]
        or final_fit["index_sha256"] != fit["index_sha256"]
        or final_fit["index_file_sha256"] != fit["index_file_sha256"]
    ):
        raise ValueError("split or retained-epoch index changed during the sweep")

    unsigned_sweep = {
        "schema": SWEEP_SCHEMA,
        "status": "complete",
        "fold": fold,
        "expected_epochs": EXPECTED_EPOCHS,
        "plan_sha256": file_sha256(output / "plan.json"),
        "runner_sha256": source_hash,
        "epoch_index_sha256": fit["index_sha256"],
        "config_identity_sha256": fit["identity"]["config_sha256"],
        "outer_split_sha256": split_binding["outer_sha256"],
        "inner_split_sha256": split_binding["inner_sha256"],
        "epochs": records,
    }
    sweep = unsigned_sweep | {"sweep_sha256": canonical_sha256(unsigned_sweep)}
    write_json_new(output / "sweep.json", sweep)
    return sweep


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--outer-split", type=Path, required=True)
    parser.add_argument("--inner-split", type=Path, required=True)
    parser.add_argument("--epoch-index", type=Path, required=True)
    parser.add_argument("--epoch-index-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--nproc-per-node", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    result = evaluate_fold(
        args.fit_dir,
        args.config,
        args.data_root,
        args.outer_split,
        args.inner_split,
        args.epoch_index,
        args.epoch_index_sha256,
        args.output,
        args.nproc_per_node,
        args.resume,
    )
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
