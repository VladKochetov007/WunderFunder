"""Run one isolated, non-CP native-feature engineering smoke through B300 actors."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
CODE_DIR = SCRIPT_DIR / "code"
sys.path[:0] = [str(SCRIPT_DIR), str(CODE_DIR)]

# Any is reserved for JSON and runtime-module payloads validated at their boundaries.
from feature_common import (  # noqa: E402
    ARMS,
    GRADER_SHA256,
    ROOT,
    sha,
    training_arguments,
    verify_source,
    write,
)
from smoke_native_subset import make_subset  # noqa: E402
from smoke_native_validation import (  # noqa: E402
    grader_call,
    load_grader,
    options,
    set_option,
    validate_fit,
)

FIT_SEED = 990019
SMOKE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,54}\Z")
GPU_UUID = re.compile(r"GPU-[0-9A-Fa-f-]{8,}\Z")


def execute(root: Path, arm: str, gpu_uuid: str, smoke_id: str) -> dict[str, Any]:
    """Run one explicit feature probe, preserving all original evaluation groups."""
    if arm not in {"field", "innovations"} or not GPU_UUID.fullmatch(gpu_uuid):
        raise ValueError("smoke arm or B300 UUID is invalid")
    if not SMOKE_NAME.fullmatch(smoke_id):
        raise ValueError("smoke id is not a safe bounded identifier")
    root = root.resolve(strict=True)
    state = root / "state/native-feature-smoke-v1" / smoke_id
    state.mkdir(parents=True, exist_ok=False)
    work = state / "work"
    work.mkdir()
    source = root / "runtime/source-a266-lf-subset-v1"
    identity = verify_source(root)
    legacy = importlib.import_module("feature_common").legacy(root)
    split_path = root / "splits/fold0.json"
    job_id = f"native_smoke_{arm}_f0_s{FIT_SEED}_{smoke_id}"
    job = {
        "id": job_id,
        "arm": arm,
        "family": ARMS[arm][0],
        "fold": 0,
        "seed": FIT_SEED,
        "gpu_uuid": gpu_uuid,
    }

    inputs = legacy["fit_workspace"].prepare_workspace(work, root / "data", split_path, 0)
    inputs.update(
        trainer_source_identity=identity["trainer"], source_tree_identity=identity["tree"]
    )
    write(work / "INPUTS.json", inputs, fresh=True)
    subset = make_subset(work, source, split_path, job_id)
    write(work / "SUBSET.json", subset, fresh=True)

    argv = training_arguments(root, job)
    for flag, value in (
        ("--hidden", "16"),
        ("--layers", "1"),
        ("--epochs", "1"),
        ("--checkpoint-epoch", "0"),
        ("--batch", "2"),
        ("--chunk", "2000"),
    ):
        set_option(argv, flag, value)
    argv.extend(
        (
            "--fit-subset-manifest",
            subset["path"],
            "--fit-subset-experiment-id",
            subset["experiment_id"],
        )
    )
    parsed = options(argv)
    expected = {
        "hidden": "16",
        "layers": "1",
        "epochs": "1",
        "checkpoint-policy": "fixed",
        "checkpoint-epoch": "0",
        "batch": "2",
        "chunk": "2000",
        "precision": "fp32",
        "train-source": "train+valid",
        "train-seqs": "10607",
        "valid-train-fraction": "1",
        "valid-repeat": "1",
        "research-features": ARMS[arm][1],
        "research-feature-control": ARMS[arm][2],
        "fit-subset-manifest": subset["path"],
        "fit-subset-experiment-id": subset["experiment_id"],
        "no-eval-during-fit": True,
        "save-training-state": True,
        "grad-stats": True,
    }
    if any(parsed.get(key) != value for key, value in expected.items()):
        raise ValueError("canonical training arguments differ from the engineering smoke recipe")
    predict = [
        job_id,
        "valid",
        "--epoch",
        "0",
        "--campaign-split-manifest",
        str(split_path),
        "--campaign-evaluation-only",
        "--device",
        "cuda",
        "--inference-precision",
        "fp32",
    ]
    request = work / "worker_request.json"
    legacy["worker"]._make_request(
        request,
        mode="fit",
        name=job_id,
        source_root=source,
        work=work,
        train_argv=argv,
        predict_argv=predict,
    )

    started = time.time()
    package = root / "runtime/pr15-host-v2"
    base_env = dict(os.environ, PYTHONPATH=str(source / "scripts"))
    fit = legacy["worker"]._run_gpu_actor(
        package=package,
        work=work,
        request=request,
        actor="train_cell.py",
        phase="fit",
        base_env=base_env,
        gpu_uuid=gpu_uuid,
        source_root=source,
    )
    write(work / "fit_phase.json", fit, fresh=True)
    fit_evidence = validate_fit(work, job, subset, sha(split_path), legacy)
    export = legacy["worker"]._run_gpu_actor(
        package=package,
        work=work,
        request=request,
        actor="export_cell.py",
        phase="export",
        base_env=base_env,
        gpu_uuid=gpu_uuid,
        source_root=source,
    )
    write(work / "export_phase.json", export, fresh=True)

    grader = load_grader(root, source)
    split = grader.load_campaign_split(split_path, work)
    checkpoint_binding = grader.validate_checkpoint(work, job_id, 0, split)
    prediction, prediction_binding = grader.validate_prediction(
        work, job_id, 0, list(split.evaluation_groups), split
    )
    prediction_layout = {
        "shape": list(prediction.shape),
        "dtype": str(prediction.dtype),
        "row_groups": len(split.evaluation_groups),
    }
    del prediction
    grade_path = work / "SMOKE_GRADE.json"
    grade = grader_call(grader, work, job_id, split_path, grade_path)
    if verify_source(root) != identity:
        raise ValueError("pinned source changed during the engineering smoke")
    result = {
        "schema": "tabdata-native-feature-engineering-smoke.v1",
        "status": "completed",
        "engineering_smoke": True,
        "scientific_cells": 0,
        "root": str(root),
        "smoke_id": smoke_id,
        "job": job,
        "source_identity": identity,
        "official_grader_sha256": GRADER_SHA256,
        "subset": subset,
        "training_argv": argv,
        "prediction_argv": predict,
        "fit_phase": fit,
        "fit_evidence": fit_evidence,
        "export_phase": export,
        "checkpoint_binding": checkpoint_binding,
        "prediction_binding": prediction_binding,
        "prediction_layout": prediction_layout,
        "grade_path": str(grade_path),
        "grade_sha256": sha(grade_path),
        "grader_api": {
            "name": "grade_endpoint",
            "signature": grade["signature"],
            "arguments": grade["arguments"],
        },
        "score": grade["score"],
        "evaluation_group_count": len(split.evaluation_groups),
        "elapsed_seconds": time.time() - started,
    }
    write(state / "SMOKE.json", result, fresh=True)
    return result


def main() -> None:
    """Run one explicitly named engineering probe and preserve a failure receipt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--arm", choices=("field", "innovations"), required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--smoke-id", required=True)
    args = parser.parse_args()
    failure_path = args.root / "state/native-feature-smoke-v1" / args.smoke_id / "FAILED.json"
    try:
        result = execute(args.root, args.arm, args.gpu_uuid, args.smoke_id)
    except BaseException as error:
        if failure_path.parent.is_dir() and not failure_path.exists():
            write(
                failure_path,
                {
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "time_unix": time.time(),
                },
                fresh=True,
            )
        raise
    print(
        json.dumps(
            {key: result[key] for key in ("status", "smoke_id", "job", "score", "elapsed_seconds")},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
