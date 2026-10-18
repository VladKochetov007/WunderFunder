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

from f1_fit_bundle import MANIFEST_NAME, write_bundle
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


def _request_stop(signum: int, _frame: object, state: dict[str, object]) -> None:
    """Stop fitting on signals; during collection, record SIGTERM and keep bounded work moving."""
    phase = state.get("phase")
    if signum == signal.SIGALRM:
        if phase == "collection":
            raise TimeoutError("fit artifact collection/finalization deadline expired")
        raise TimeoutError("fit deadline expired")
    if signum == signal.SIGTERM and phase == "collection":
        state["sigterm_requested"] = True
        return
    raise InterruptedError(f"received signal {signum}; archiving available fit artifacts")


def _write_receipt(output: Path, receipt: dict[str, Any]) -> None:
    (output / "fit-pair-receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )


def _collect_and_bundle(
    data_root: Path | None,
    output: Path,
    profile_spec: Path,
    fit_spec: Path,
    bundle_path: Path,
    receipt: dict[str, Any],
    signal_state: dict[str, object],
) -> bool:
    """Collect partial or complete state under the caller's still-armed deadline."""
    try:
        collection_errors = collect_artifacts(data_root, output, profile_spec, fit_spec)
        receipt["collection_status"] = "COMPLETE_WITH_ERRORS" if collection_errors else "COMPLETE"
    except (InterruptedError, TimeoutError) as error:
        collection_errors = [f"{type(error).__name__}: {error}"]
        receipt["collection_status"] = "DEADLINE_EXPIRED"
        receipt["bundle_status"] = "NOT_ATTEMPTED_COLLECTION_DEADLINE"
        receipt["collection_errors"] = collection_errors
        if signal_state.get("sigterm_requested"):
            receipt["sigterm_requested_during_collection"] = True
        _write_receipt(output, receipt)
        return False

    receipt["collection_errors"] = collection_errors
    receipt["bundle_status"] = "PACKAGING_REQUESTED"
    if signal_state.get("sigterm_requested"):
        receipt["sigterm_requested_during_collection"] = True
    _write_receipt(output, receipt)
    try:
        write_bundle(bundle_path, output)
    except (InterruptedError, TimeoutError) as error:
        receipt["bundle_status"] = "DEADLINE_EXPIRED"
        receipt["bundle_error"] = f"{type(error).__name__}: {error}"
        (output / MANIFEST_NAME).unlink(missing_ok=True)
        _write_receipt(output, receipt)
        print(f"Fit artifact finalization deadline expired: {error}", flush=True)
        return False
    except Exception as error:
        receipt["bundle_status"] = "FAILED"
        receipt["bundle_error"] = f"{type(error).__name__}: {error}"
        (output / MANIFEST_NAME).unlink(missing_ok=True)
        _write_receipt(output, receipt)
        print(f"Fit artifact bundle failed: {error}", flush=True)
        return False
    print(f"Fit pair bundle saved: {bundle_path} status={receipt['status']}", flush=True)
    return True


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
    signal_state: dict[str, object] = {"phase": "fit", "sigterm_requested": False}
    previous_sigterm = signal.getsignal(signal.SIGTERM)
    previous_sigalrm = signal.getsignal(signal.SIGALRM)

    def stop_handler(signum: int, frame: object) -> None:
        _request_stop(signum, frame, signal_state)

    signal.signal(signal.SIGTERM, stop_handler)
    signal.signal(signal.SIGALRM, stop_handler)
    data_root: Path | None = None
    internal_deadline: int | None = None
    collection_reserve: int | None = None
    external_deadline: int | None = None
    collection_deadline: float | None = None
    bundle_written = False
    try:
        profile = read_object(args.profile_spec, "profile specification")
        fit = read_object(args.fit_spec, "fit specification")
        internal, collection_reserve, external_deadline = validate_fit_spec(fit)
        internal_deadline = internal
        check_packet_binding(profile, fit, args.profile_spec)
        data_root = Path.cwd() / "rank01-f1-fit-work" / str(profile["data_root"])
        if sha256_file(args.code) != sealed_hash(
            profile.get("code_archive_sha256"), "code archive"
        ) or sha256_file(args.supplemental) != sealed_hash(
            profile.get("supplemental_archive_sha256"), "supplemental archive"
        ):
            raise ValueError("source or supplemental archive differs from the sealed profile")
        fit_deadline = started + internal_deadline
        collection_deadline = fit_deadline + collection_reserve
        remaining = max(0.001, fit_deadline - time.monotonic())
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
            "internal_deadline_seconds": internal_deadline,
            "collection_reserve_seconds": collection_reserve,
            "external_cancel_deadline_seconds_from_submit": external_deadline,
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
            if time.monotonic() >= fit_deadline:
                raise TimeoutError("fit pair reached its internal deadline before export")
            arm_receipt["status"] = "EXPORTING"
            arm_receipt["prediction_export"] = run_prediction(
                arm,
                split_path,
                source_root,
                data_root,
                output,
                fit_deadline,
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
        receipt["allocated_seconds"] = round(time.monotonic() - started, 3)
        receipt["quality_result"] = None
        try:
            if collection_deadline is None:
                receipt["collection_status"] = "SKIPPED_NO_SEALED_COLLECTION_DEADLINE"
                receipt["bundle_status"] = "NOT_ATTEMPTED_NO_SEALED_COLLECTION_DEADLINE"
                _write_receipt(output, receipt)
            else:
                receipt["collection_deadline_seconds_from_entry"] = (
                    internal_deadline + collection_reserve
                )
                signal_state["phase"] = "collection"
                signal.setitimer(
                    signal.ITIMER_REAL,
                    max(0.001, collection_deadline - time.monotonic()),
                )
                try:
                    bundle_written = _collect_and_bundle(
                        data_root,
                        output,
                        args.profile_spec,
                        args.fit_spec,
                        args.out,
                        receipt,
                        signal_state,
                    )
                except (InterruptedError, TimeoutError) as error:
                    receipt["collection_status"] = "DEADLINE_EXPIRED"
                    receipt["bundle_status"] = "NOT_ATTEMPTED_OR_INCOMPLETE"
                    receipt["finalization_error"] = f"{type(error).__name__}: {error}"
                    (output / MANIFEST_NAME).unlink(missing_ok=True)
                    signal.setitimer(signal.ITIMER_REAL, 0)
                    _write_receipt(output, receipt)
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGTERM, previous_sigterm)
            signal.signal(signal.SIGALRM, previous_sigalrm)
    return 0 if receipt["status"] == "PAIR_FIT_AND_EXPORT_COMPLETE" and bundle_written else 1


if __name__ == "__main__":
    raise SystemExit(main())
