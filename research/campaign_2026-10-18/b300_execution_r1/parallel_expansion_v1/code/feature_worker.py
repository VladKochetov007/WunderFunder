"""Deliver one fresh native-feature fit and its source-bound evaluation predictions."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from feature_common import (
    legacy,
    read,
    sha,
    training_arguments,
    verify_config,
    verify_source,
    write,
)


def execute(config: dict, job: dict) -> dict:
    """Run the unchanged canonical trainer and predictor through the bounded GPU envelope."""
    root = Path(config["root"])
    state = root / "state/parallel-feature-execution-v1"
    work = state / "work" / job["id"]
    work.mkdir(parents=True, exist_ok=False)
    identity = verify_source(root)
    source = root / "runtime/source-a266-lf-subset-v1"
    modules = legacy(root)
    common = modules["worker"]
    split = root / "splits" / f"fold{job['fold']}.json"
    inputs = modules["fit_workspace"].prepare_workspace(work, root / "data", split, job["fold"])
    inputs["trainer_source_identity"] = identity["trainer"]
    inputs["source_tree_identity"] = identity["tree"]
    inputs["deployed_source_tree"] = identity["tree"]
    write(work / "INPUTS.json", inputs, fresh=True)
    argv = training_arguments(root, job)
    predict = [
        job["id"],
        "valid",
        "--epoch",
        "5",
        "--campaign-split-manifest",
        str(split),
        "--campaign-evaluation-only",
        "--device",
        "cuda",
        "--inference-precision",
        "fp32",
    ]
    request = work / "worker_request.json"
    common._make_request(
        request,
        mode="fit",
        name=job["id"],
        source_root=source,
        work=work,
        train_argv=argv,
        predict_argv=predict,
    )
    env = dict(os.environ, PYTHONPATH=str(source / "scripts"))
    package = root / "runtime/pr15-host-v2"
    started = time.time()
    fit = common._run_gpu_actor(
        package=package,
        work=work,
        request=request,
        actor="train_cell.py",
        phase="fit",
        base_env=env,
        gpu_uuid=job["gpu_uuid"],
        source_root=source,
    )
    write(work / "fit_phase.json", fit, fresh=True)
    from feature_validation import validate_fit

    evidence = validate_fit(root, work, job)
    write(
        work / "FIT.json",
        dict(job=job, validation=evidence, phase=fit, source=identity),
        fresh=True,
    )
    export = common._run_gpu_actor(
        package=package,
        work=work,
        request=request,
        actor="export_cell.py",
        phase="export",
        base_env=env,
        gpu_uuid=job["gpu_uuid"],
        source_root=source,
    )
    from feature_outputs import validate_export

    prediction = validate_export(root, work, job, evidence["checkpoint_sha256"])
    result = dict(
        schema="tabdata-native-feature-endpoint.v1",
        status="fit_export_completed",
        job=job,
        source=identity,
        fit=evidence,
        export=export,
        prediction=prediction,
        fit_receipt_sha256=sha(work / "FIT.json"),
        elapsed_seconds=time.time() - started,
    )
    write(work / "EXPORTED.json", result, fresh=True)
    return result


def main() -> None:
    """Accept only a frozen controller-dispatched scientific cell."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--job", type=Path, required=True)
    args = parser.parse_args()
    config = verify_config(args.config, args.config_sha256)
    job = read(args.job)
    if sorted(os.sched_getaffinity(0)) != job["cpus"]:
        raise ValueError("Worker CPU affinity differs from its claimed slot")
    try:
        execute(config, job)
    except BaseException as error:
        state = Path(config["root"]) / "state/parallel-feature-execution-v1"
        write(
            state / "failures" / (job["id"] + ".json"),
            dict(job=job, error_type=type(error).__name__, error=str(error), time_unix=time.time()),
            fresh=True,
        )
        raise


if __name__ == "__main__":
    main()
