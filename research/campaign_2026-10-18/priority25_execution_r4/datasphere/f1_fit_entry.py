#!/usr/bin/env python3
"""Stage one sealed F1 cell, run its matched pair, export, and bundle artifacts."""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
import traceback
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any, TextIO

from f1_fit_bundle import write_bundle
from f1_fit_outputs import collect_artifacts, run_prediction, validate_completed_fit
from f1_fit_recipe import ARMS, FOLD1_SHA256, SOURCE_COMMIT, training_args, validate_fit_spec
from f1_fit_staging import check_packet_binding, prepare_fit, read_object
from profile_staging import sealed_hash, sha256_file


class _Tee:
    """Write canonical trainer output to both provider logs and a durable arm log."""

    def __init__(self, stream: TextIO, log: TextIO) -> None:
        self.stream = stream
        self.log = log

    def write(self, value: str) -> int:
        self.stream.write(value)
        self.log.write(value)
        return len(value)

    def flush(self) -> None:
        self.stream.flush()
        self.log.flush()


def _run_train_arm(
    arm: dict[str, object], split_path: Path, source_root: Path, data_root: Path, output: Path
) -> Path:
    """Call the canonical trainer once, preserving fit-only telemetry in its run folder/log."""
    sys.path.insert(0, str(source_root / "src"))
    from wnn.training.environment import TrainingEnvironment
    from wnn.training.runner import run_training

    log_path = output / f"{arm['name']}.train.log"
    with log_path.open("w", encoding="utf-8") as log:
        tee = _Tee(sys.stdout, log)
        with redirect_stdout(tee), redirect_stderr(tee):
            run_training(
                training_args(arm, split_path),
                environment=TrainingEnvironment(source_root=source_root, root=data_root, argv=[]),
            )
    run_dir = data_root / "artifacts" / "runs" / str(arm["name"])
    validate_completed_fit(run_dir, arm)
    return run_dir


def _request_stop(signum: int, _frame: object) -> None:
    raise InterruptedError(f"received signal {signum}; archiving available fit artifacts")


def main() -> int:
    """Run one bounded pair and package all artifacts even when the pair is interrupted."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-spec", type=Path, required=True)
    parser.add_argument("--fit-spec", type=Path, required=True)
    parser.add_argument("--code", type=Path, required=True)
    parser.add_argument("--supplemental", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", required=True)
    args = parser.parse_args()
    started = time.monotonic()
    output = Path("fit-results").resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("fit-results already contains artifacts; refusing stale-state reuse")
    output.mkdir(exist_ok=True)
    # Provider receipts contain recipe-specific nested JSON values.
    receipt: dict[str, Any] = {"status": "STARTING", "quality_result": None, "arms": {}}
    signal.signal(signal.SIGTERM, _request_stop)
    signal.signal(signal.SIGALRM, _request_stop)
    data_root: Path | None = None
    try:
        profile = read_object(args.profile_spec, "profile specification")
        fit = read_object(args.fit_spec, "fit specification")
        internal, external = validate_fit_spec(fit)
        check_packet_binding(profile, fit, args.profile_spec)
        data_root = Path.cwd() / "rank01-f1-fit-work" / str(profile["data_root"])
        if sha256_file(args.code) != sealed_hash(
            profile.get("code_archive_sha256"), "code archive"
        ) or sha256_file(args.supplemental) != sealed_hash(
            profile.get("supplemental_archive_sha256"), "supplemental archive"
        ):
            raise ValueError("source or supplemental archive differs from the sealed profile")
        deadline = started + internal
        remaining = max(0.001, deadline - time.monotonic())
        signal.setitimer(signal.ITIMER_REAL, remaining)
        for name in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "WNN_THREADS",
            "WNN_LOADER_WORKERS",
        ):
            os.environ[name] = "0" if name == "WNN_LOADER_WORKERS" else "1"
        os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
        os.environ["WNN_GPU_FRACTION"] = "0.90"
        inputs = {
            "profile_spec_sha256": sha256_file(args.profile_spec),
            "fit_spec_sha256": sha256_file(args.fit_spec),
            "code_archive_sha256": sha256_file(args.code),
            "supplemental_archive_sha256": sha256_file(args.supplemental),
            "source_commit": SOURCE_COMMIT,
            "fold_manifest_sha256": FOLD1_SHA256,
            "external_cancel_deadline_seconds_from_submit": external,
        }
        receipt["inputs"] = inputs
        source_root, data_root, split_path, runtime, staging = prepare_fit(profile, fit, args)
        os.environ["WNN_PROJECT_ROOT"] = str(data_root)
        os.environ["WNN_PORTABLE_DATA_MANIFEST"] = str(data_root / "portable-sources.json")
        sys.path.insert(0, str(source_root / "src"))
        from wnn.splits import load_campaign_split

        split = load_campaign_split(split_path, data_root)
        if split.fold != 1 or split.sha256 != FOLD1_SHA256:
            raise ValueError("canonical split loader did not return the sealed fold-1 identity")
        receipt.update(status="RUNNING", runtime=runtime, split_sha256=split.sha256, **staging)
        for arm in ARMS:
            arm_receipt: dict[str, Any] = {"status": "TRAINING"}
            receipt["arms"][arm["name"]] = arm_receipt
            run_dir = _run_train_arm(arm, split_path, source_root, data_root, output)
            fit_artifacts = validate_completed_fit(run_dir, arm)
            arm_receipt.update(
                status="FIT_COMPLETE",
                run_dir=str(run_dir.relative_to(data_root)),
                fit_artifacts=fit_artifacts,
                training_log=f"{arm['name']}.train.log",
            )
            if time.monotonic() >= deadline:
                raise TimeoutError("fit pair reached its internal deadline before export")
            arm_receipt["status"] = "EXPORTING"
            arm_receipt["prediction_export"] = run_prediction(
                arm,
                split_path,
                source_root,
                data_root,
                output,
                deadline,
                expected_evaluation_groups=split.evaluation_groups,
                expected_checkpoint_sha256=fit_artifacts["model_ep5.pt"]["sha256"],
            )
            arm_receipt["status"] = "COMPLETE"
        receipt["status"] = "PAIR_FIT_AND_EXPORT_COMPLETE"
    except BaseException as error:
        receipt.update(
            status="INTERRUPTED"
            if isinstance(error, (InterruptedError, TimeoutError))
            else "FAILED",
            error=f"{type(error).__name__}: {error}",
            traceback=traceback.format_exc(),
        )
        print(receipt["traceback"], flush=True)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        receipt["allocated_seconds"] = round(time.monotonic() - started, 3)
        receipt["quality_result"] = None
        collection_errors = collect_artifacts(data_root, output, args.profile_spec, args.fit_spec)
        if collection_errors:
            receipt["collection_errors"] = collection_errors
        (output / "fit-pair-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
        write_bundle(args.out, output)
        print(f"Fit pair bundle saved: {args.out} status={receipt['status']}", flush=True)
    return 0 if receipt["status"] == "PAIR_FIT_AND_EXPORT_COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
