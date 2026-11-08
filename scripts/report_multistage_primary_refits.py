"""Collect four terminal full-primary refits and score configured-final endpoints only."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import sys
import time
from pathlib import Path

from multistage_primary_refits_inputs import HARD_DEADLINE, ROOT, artifact, collect_inputs
from report_multistage_primary_ensemble_worker import (
    SCORE_SCRIPT_SHA256,
    SOURCE6_SHA256,
    _sha256,
    _verify_source6,
    _write_new,
)


def watchdog_command(
    worker_command: list[str], deadline: float, now: float
) -> tuple[list[str], int]:
    """An independent process-group watchdog survives loss of this collector."""
    wall = min(900, int(min(deadline, HARD_DEADLINE) - now))
    duration = wall - 10
    if duration <= 0:
        raise RuntimeError("insufficient deadline headroom for watchdog cleanup")
    return [
        "/usr/bin/timeout",
        "--signal=TERM",
        "--kill-after=5s",
        f"{duration}s",
        *worker_command,
    ], wall


def _worker(
    args: argparse.Namespace,
    request: dict[str, object],
    request_path: Path,
    digest: str,
    mode: str,
    index: int | None = None,
) -> dict[str, object]:
    from wnn.multistage_search_owned_command import run_owned_command
    from wnn.multistage_search_runtime import current_process_owner

    output = Path(request["output_dir"])
    name = "fixed-blends" if index is None else f"member-{index:02d}"
    command = [
        str(args.python),
        str(ROOT / "scripts/report_multistage_primary_refits_worker.py"),
        "--request",
        str(request_path),
        "--request-sha256",
        digest,
        "--mode",
        mode,
    ]
    if index is not None:
        command += ["--member-index", str(index)]
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": "" if index is None else request["members"][index]["gpu_uuid"],
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "CUBLAS_EMULATE_SINGLE_PRECISION": "0",
            "WNN_PROJECT_ROOT": request["data_root"],
            "WNN_PORTABLE_DATA_MANIFEST": request["portable_data_registry"]["path"],
            "PYTHONPATH": os.pathsep.join(
                (
                    str(Path(request["source6"]["root"]) / "src"),
                    str(args.dependency_root.resolve(strict=True)),
                )
            ),
        }
    )
    for key in (
        "RANK",
        "LOCAL_RANK",
        "WORLD_SIZE",
        "LOCAL_WORLD_SIZE",
        "MASTER_ADDR",
        "MASTER_PORT",
    ):
        env.pop(key, None)
    worker_command = list(command)
    try:
        command, remaining = watchdog_command(
            worker_command, request["score_deadline_epoch"], time.time()
        )
    except RuntimeError:
        return {"name": name, "status": "score_deadline_reached", "exit_code": None}
    if not Path(command[0]).is_file():
        raise FileNotFoundError("independent GNU timeout watchdog is missing at /usr/bin/timeout")
    process_path = output / "process_receipts" / f"{name}.json"
    watchdog_path = output / "process_receipts" / f"{name}.watchdog.json"
    _write_new(
        watchdog_path,
        {
            "schema": "wnn.primary-refit-independent-watchdog.v1",
            "wrapper_command": command,
            "worker_command": worker_command,
            "max_wall_seconds": remaining,
            "kill_after_seconds": 5,
            "hard_deadline_epoch": HARD_DEADLINE,
            "independent_of_collector_lifetime": True,
        },
    )
    try:
        code, timed_out, process = run_owned_command(
            command,
            cwd=Path(request["source6"]["root"]),
            env=env,
            log_path=output / "logs" / f"{name}.log",
            receipt_path=process_path,
            owner=current_process_owner(),
            stage=4,
            trial_number=index if index is not None else -1,
            mode="report_only_primary_refit_score",
            gpu_key=env["CUDA_VISIBLE_DEVICES"],
            output_dir=output,
            timeout_seconds=remaining,
        )
        return {
            "name": name,
            "status": "censored_timeout"
            if timed_out or code in {124, 137}
            else "completed"
            if code == 0
            else "failed",
            "exit_code": code,
            "process_receipt": artifact(process_path),
            "watchdog_receipt": artifact(watchdog_path),
            "log_path": str(output / "logs" / f"{name}.log"),
            "started_at_epoch": process.get("started_at_epoch"),
            "finished_at_epoch": process.get("finished_at_epoch"),
        }
    except (OSError, ValueError, RuntimeError) as error:
        return {
            "name": name,
            "status": "launch_failed",
            "error": f"{type(error).__name__}: {error}",
        }


def _preserve_inputs(
    output: Path, members: list[dict[str, object]], context: dict[str, object]
) -> None:
    for member in members:
        destination = output / "input_receipts" / member["fit_id"]
        destination.mkdir(parents=True)
        for name, reference in member["artifacts"].items():
            source = Path(reference["path"])
            if _sha256(source) != reference["sha256"]:
                raise ValueError("terminal fit evidence changed during collection")
            with (destination / f"{name}.json").open("xb") as stream:
                stream.write(source.read_bytes())
        config_source = Path(member["config_path"])
        with (destination / "config.json").open("xb") as stream:
            stream.write(config_source.read_bytes())
        _write_new(
            destination / "model_config_source_identity.json",
            {
                "fit_id": member["fit_id"],
                "configured_final_complete": member["configured_final_complete"],
                "model": member["checkpoint"],
                "config_sha256": member["config_sha256"],
                "identity": member["identity"],
            },
        )
        if member["configured_final_complete"]:
            model_source = Path(member["checkpoint"]["path"])
            model_directory = output / "models" / member["fit_id"]
            model_directory.mkdir(parents=True)
            collected_model = model_directory / "model.pt"
            os.link(model_source, collected_model)
            if _sha256(collected_model) != member["checkpoint"]["sha256"]:
                raise ValueError("collected configured-final checkpoint bytes changed")
            member["collected_checkpoint"] = artifact(collected_model)
    for name in ("selection", "frozen_weights", "stage4", "fit_budget"):
        reference = context[name]
        source = Path(reference["path"])
        if _sha256(source) != reference["sha256"]:
            raise ValueError("frozen selection or weights changed during collection")
        with (output / f"frozen_{name}.json").open("xb") as stream:
            stream.write(source.read_bytes())


def run(args: argparse.Namespace) -> dict[str, object]:
    started_at = time.time()
    os.environ.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
        }
    )
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(args.dependency_root.resolve(strict=True)))
    if not args.python.is_file():
        raise FileNotFoundError("the supplied receiving Python interpreter is missing")
    if time.time() >= HARD_DEADLINE:
        raise RuntimeError("campaign hard deadline reached; report scoring admission closed")
    runtime, prepared = (
        args.runtime_root.resolve(strict=True),
        args.prepared_root.resolve(strict=True),
    )
    output = args.output.resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError("full-primary refit report output must be fresh")
    members, weights, context = collect_inputs(runtime, prepared, args.gpu_uuid)
    source6 = {
        "root": str(args.source6_root.resolve(strict=True)),
        "source_sha256": SOURCE6_SHA256,
        "score_script_sha256": SCORE_SCRIPT_SHA256,
    }
    _verify_source6(Path(source6["root"]), source6)
    pipeline_deadline = min(started_at + 1100, HARD_DEADLINE)
    request = {
        "schema": "wnn.full-primary-refit-score-request.v1",
        "status": "all_four_owned_fits_terminal",
        "members": members,
        "weights": weights,
        "source6": source6,
        "output_dir": str(output),
        **context,
        "hard_deadline_epoch": HARD_DEADLINE,
        "pipeline_deadline_epoch": pipeline_deadline,
        "score_deadline_epoch": pipeline_deadline - 30,
        "output_collection_reserve_seconds": 30,
        "max_member_seconds": 900,
        "max_outer_seconds": 1100,
        "batch_size": 64,
        "cpu_threads_per_worker": 1,
        "objective_eligible": False,
        "primary_valid_used_for_weights": False,
        "model_endpoint_policy": "configured-final complete only; no retained/censored endpoint scoring or resume",
        "ensemble_policy": "all four configured-final models and all four scores required; no post-hoc renormalization",
        "reporter_sources": {
            name: _sha256(ROOT / "scripts" / name)
            for name in (
                "report_multistage_primary_refits.py",
                "multistage_primary_refits_inputs.py",
                "report_multistage_primary_refits_worker.py",
                "multistage_primary_refits_outputs.py",
                "report_multistage_primary_ensemble_worker.py",
                "report_multistage_primary_ensemble_blend.py",
            )
        },
    }
    output.mkdir(parents=True, exist_ok=False)
    for name in ("member_predictions", "member_receipts", "process_receipts", "logs"):
        (output / name).mkdir()
    for member in members:
        index = member["index"]
        member["predictions_path"] = str(output / "member_predictions" / f"member-{index:02d}.npy")
        member["receipt_path"] = str(output / "member_receipts" / f"member-{index:02d}.json")
    _preserve_inputs(output, members, context)
    request_path = output / "score_request.json"
    _write_new(request_path, request)
    digest = _sha256(request_path)
    outcomes = {}
    eligible = [member["index"] for member in members if member["configured_final_complete"]]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = {
            pool.submit(_worker, args, request, request_path, digest, "member", index): index
            for index in eligible
        }
        for future in concurrent.futures.as_completed(futures):
            index = futures[future]
            outcomes[str(index)] = future.result()
    _write_new(output / "scoring_execution_receipt.json", outcomes)
    can_blend = len(eligible) == 4 and all(
        outcomes[str(index)].get("status") == "completed" for index in range(4)
    )
    blend = (
        _worker(args, request, request_path, digest, "blend")
        if can_blend
        else {
            "status": "skipped",
            "reason": "all four configured-final complete models and four successful singleton scores are required",
        }
    )
    _write_new(output / "fixed_blend_execution_receipt.json", blend)
    from multistage_primary_refits_outputs import export_report

    return export_report(request, request_path, digest, outcomes, blend)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("runtime-root", "prepared-root", "source6-root", "dependency-root", "output"):
        parser.add_argument(f"--{flag}", required=True, type=Path)
    parser.add_argument("--python", required=True, type=Path)
    parser.add_argument("--gpu-uuid", required=True, action="append")
    args = parser.parse_args()
    try:
        result = run(args)
    except (OSError, RuntimeError, ValueError, KeyError, TypeError) as error:
        raise SystemExit(
            f"Full-primary refit report refused: {type(error).__name__}: {error}"
        ) from error
    sys.stdout.write(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
