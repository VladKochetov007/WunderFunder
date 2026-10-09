"""Collect a short CPU/CUDA trace through the frozen H230 trainer."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import logging
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType

LOGGER = logging.getLogger(__name__)
TRACE_WAIT_STEPS = 0
TRACE_WARMUP_STEPS = 3
TRACE_ACTIVE_STEPS = 3
EXPECTED_PROFILE_STEPS = TRACE_ACTIVE_STEPS
EXPECTED_PROFILE_WARMUP = TRACE_WARMUP_STEPS
EXPECTED_RUN_NAME = "bd26_h230_gpu_trace_s3_f0"
EXPECTED_INIT = "bd26_h230_hetero_s3_f0@ep1"


def _sha256(path: Path) -> str:
    """Return a file's SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _value_after(arguments: Sequence[str], option: str) -> str | None:
    """Return the value after a trainer option, supporting ``--key=value``."""
    for index, argument in enumerate(arguments):
        if argument.startswith(f"{option}="):
            return argument.split("=", maxsplit=1)[1]
        if argument == option and index + 1 < len(arguments):
            return arguments[index + 1]
    return None


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse trace options and retain the frozen trainer arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trainer-root", type=Path, required=True)
    parser.add_argument("--trainer-sha256", required=True)
    parser.add_argument("--helper-sha256", required=True)
    parser.add_argument("--trace-output", type=Path, required=True)
    parser.add_argument("--receipt-output", type=Path, required=True)
    parser.add_argument("trainer_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    args.trainer_args = list(args.trainer_args)
    if args.trainer_args[:1] == ["--"]:
        args.trainer_args = args.trainer_args[1:]
    return args


def _load_trainer(trainer_path: Path, expected_sha256: str) -> ModuleType:
    """Load the pinned trainer only after validating its source hash."""
    observed_sha256 = _sha256(trainer_path)
    if observed_sha256 != expected_sha256:
        raise RuntimeError(
            f"trainer hash mismatch: expected {expected_sha256}, got {observed_sha256}"
        )
    spec = importlib.util.spec_from_file_location("h230_trace_train_rnn", trainer_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load trainer module from {trainer_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _ProfilerSession:
    """Start before input fetch and advance after each complete trainer batch."""

    def __init__(self, torch_module: ModuleType, trace_output: Path) -> None:
        self.torch = torch_module
        self.trace_output = trace_output
        self.profiler: object | None = None
        self.steps = 0
        self.trace_callbacks = 0

    def start(self) -> None:
        """Open a CPU/CUDA profile immediately before the first batch fetch."""
        activities = [
            self.torch.profiler.ProfilerActivity.CPU,
            self.torch.profiler.ProfilerActivity.CUDA,
        ]
        schedule = self.torch.profiler.schedule(
            wait=TRACE_WAIT_STEPS,
            warmup=TRACE_WARMUP_STEPS,
            active=TRACE_ACTIVE_STEPS,
            repeat=1,
        )
        self.trace_output.parent.mkdir(parents=True, exist_ok=True)
        self.profiler = self.torch.profiler.profile(
            activities=activities,
            schedule=schedule,
            on_trace_ready=self._write_trace,
            record_shapes=False,
            profile_memory=False,
            with_stack=False,
        )
        self.profiler.__enter__()

    def step(self) -> None:
        """Advance the schedule once after a complete trainer batch."""
        if self.profiler is None:
            return
        self.profiler.step()
        self.steps += 1

    def finish(self, error: BaseException | None = None) -> None:
        """Close the profiler context while preserving any trainer exception."""
        if self.profiler is None:
            return
        error_type = type(error) if error is not None else None
        traceback = error.__traceback__ if error is not None else None
        self.profiler.__exit__(error_type, error, traceback)
        self.profiler = None

    def _write_trace(self, profiler: object) -> None:
        """Write the scheduled trace and count the completed capture."""
        profiler.export_chrome_trace(str(self.trace_output))
        self.trace_callbacks += 1


def _install_timer_hooks(trainer: ModuleType, session: _ProfilerSession) -> tuple[object, object]:
    """Start before the first CUDA-prefetched batch and step after StageTimer.stop."""
    timer_class = trainer.StageTimer
    original_stop = timer_class.stop
    prefetcher_class = trainer.CudaPrefetcher
    original_load = prefetcher_class._load

    def hooked_load(prefetcher: object, iterator: object) -> object:
        if session.profiler is None:
            session.start()
        return original_load(prefetcher, iterator)

    def hooked_stop(timer: object) -> None:
        original_stop(timer)
        if timer.enabled and session.profiler is not None:
            session.step()

    prefetcher_class._load = hooked_load
    timer_class.stop = hooked_stop
    return original_load, original_stop


def _validate_trainer_args(arguments: Sequence[str], trainer_root: Path) -> str:
    """Require a CUDA-only short profile and a fresh output run name."""
    name = _value_after(arguments, "--name")
    if name is None:
        raise ValueError("trainer arguments must include --name")
    if name != EXPECTED_RUN_NAME:
        raise ValueError(f"trace run name must be {EXPECTED_RUN_NAME}")
    if _value_after(arguments, "--device") != "cuda":
        raise ValueError("GPU trace requires the frozen trainer's --device cuda path")
    if _value_after(arguments, "--init") != EXPECTED_INIT:
        raise ValueError(f"trace must initialize from {EXPECTED_INIT}")
    if _value_after(arguments, "--profile-steps") != str(EXPECTED_PROFILE_STEPS):
        raise ValueError("trainer arguments must set --profile-steps 3")
    if _value_after(arguments, "--profile-warmup") != str(EXPECTED_PROFILE_WARMUP):
        raise ValueError("trainer arguments must set --profile-warmup 3")
    run_directory = trainer_root / "artifacts" / "runs" / name
    if run_directory.exists():
        raise FileExistsError(f"trace profile run directory already exists: {run_directory}")
    return name


def main(argv: Sequence[str] | None = None) -> int:
    """Run a six-batch trace profile without changing the trainer or saving weights."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _parse_args(argv)
    trainer_root = args.trainer_root.resolve(strict=True)
    trainer_path = trainer_root / "scripts" / "train_rnn.py"
    helper_path = Path(__file__).resolve()
    helper_sha256 = _sha256(helper_path)
    if helper_sha256 != args.helper_sha256:
        raise RuntimeError(
            f"helper hash mismatch: expected {args.helper_sha256}, got {helper_sha256}"
        )
    trace_output = args.trace_output.resolve()
    receipt_output = args.receipt_output.resolve()
    if trace_output == receipt_output:
        raise ValueError("trace and receipt outputs must use distinct paths")
    if trace_output.exists() or receipt_output.exists():
        raise FileExistsError("trace and receipt outputs must be new paths")
    if not args.trainer_args:
        raise ValueError("append the frozen trainer arguments after --")

    run_name = _validate_trainer_args(args.trainer_args, trainer_root)
    trainer_sha256 = _sha256(trainer_path)
    if trainer_sha256 != args.trainer_sha256:
        raise RuntimeError(
            f"trainer hash mismatch: expected {args.trainer_sha256}, got {trainer_sha256}"
        )

    trainer = _load_trainer(trainer_path, args.trainer_sha256)
    session = _ProfilerSession(trainer.torch, trace_output)
    original_load, original_stop = _install_timer_hooks(trainer, session)
    error: BaseException | None = None
    sys.argv = [str(trainer_path), *args.trainer_args]
    try:
        trainer.main()
    except BaseException as caught:
        error = caught
        raise
    finally:
        trainer.CudaPrefetcher._load = original_load
        trainer.StageTimer.stop = original_stop
        session.finish(error)

    trace_exists = trace_output.is_file()
    profile_run_directory = trainer_root / "artifacts" / "runs" / run_name
    saved_weight_files = sorted(str(path) for path in profile_run_directory.glob("*.pt"))
    receipt = {
        "schema": "h230-gpu-trace-result-v1",
        "captured_at_utc": datetime.now(UTC).isoformat(),
        "status": (
            "passed"
            if trace_exists
            and session.trace_callbacks == 1
            and session.steps == TRACE_WARMUP_STEPS + TRACE_ACTIVE_STEPS
            and not saved_weight_files
            else "incomplete"
        ),
        "helper_path": str(helper_path),
        "helper_sha256": helper_sha256,
        "trainer_path": str(trainer_path),
        "trainer_sha256": trainer_sha256,
        "trainer_args": args.trainer_args,
        "profile_run_name": run_name,
        "profile_run_directory": str(profile_run_directory),
        "saved_weight_files": saved_weight_files,
        "trace": {
            "path": str(trace_output),
            "exists": trace_exists,
            "bytes": trace_output.stat().st_size if trace_exists else 0,
            "sha256": _sha256(trace_output) if trace_exists else None,
        },
        "schedule": {
            "activities": ["CPU", "CUDA"],
            "wait_steps": TRACE_WAIT_STEPS,
            "warmup_steps": TRACE_WARMUP_STEPS,
            "active_steps": TRACE_ACTIVE_STEPS,
            "stage_timer_stops": session.steps,
            "trace_callbacks": session.trace_callbacks,
            "record_shapes": False,
            "profile_memory": False,
            "with_stack": False,
        },
        "weights_saved": False,
        "fit_time_evaluation": False,
    }
    receipt_output.parent.mkdir(parents=True, exist_ok=True)
    receipt_output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    LOGGER.info("GPU trace receipt: %s", receipt_output)
    if receipt["status"] != "passed":
        raise RuntimeError("the scheduled three-step CUDA trace did not complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
