from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

from wave_checks import clock, wave_limits

from wnn.multistage_search import read_json
from wnn.multistage_search_owned_command import run_owned_command
from wnn.multistage_search_runtime import is_same_process, terminate_owned_process

SOURCE = Path(
    "/data2/lc-epanov_s/tabdata/runtime/architecture-execution-20261107-r1/source-gru-monitor-resume-r1"
)


def environment(source: Path, uuid: str) -> dict[str, str]:
    """Match qualified S4 strict-FP32 identity and bound native worker CPU."""
    result = dict(os.environ)
    for key in (
        "RANK",
        "WORLD_SIZE",
        "LOCAL_RANK",
        "CUBLAS_WORKSPACE_CONFIG",
        "CUBLAS_EMULATE_SINGLE_PRECISION",
    ):
        result.pop(key, None)
    result.update(
        PYTHONPATH=str(source / "src"),
        CUDA_VISIBLE_DEVICES=uuid,
        OMP_NUM_THREADS="8",
        MKL_NUM_THREADS="8",
        OPENBLAS_NUM_THREADS="8",
        PYTHONDONTWRITEBYTECODE="1",
        NVIDIA_TF32_OVERRIDE="0",
        TORCH_ALLOW_TF32_CUBLAS_OVERRIDE="0",
    )
    return result


def execute_owned(
    wave: Any,
    cell: dict[str, object],
    stage: str,
    argv: list[str],
    output: Path,
    source: Path = SOURCE,
    cap_seconds: int | None = None,
) -> None:
    """Use the existing executor with a latched, nonthrowing exact-owner stop callback.

    The packet Wave object supplies shared source/authority/status fields. Keeping
    this thin adapter outside the controller avoids copying the trainer or executor.
    """
    bound = clock(wave.runtime) if stage == "profile" else wave_limits(wave.runtime)
    observational = stage.startswith("readout") or stage == "swa"
    seconds = int(
        bound["wave_deadline_epoch" if observational else "fit_deadline_epoch"] - time.time()
    )
    if cap_seconds is not None:
        seconds = min(seconds, cap_seconds)
    if seconds < 1:
        raise TimeoutError("the original wave reserve/deadline has started")
    command = argv
    if stage in ("profile", "fit"):
        command = [
            wave.args.python,
            str(Path(__file__).with_name("observed_worker.py")),
            "--",
            wave.args.python,
            str(SOURCE / "scripts/observe_sequence.py"),
            "--output",
            str(output),
            "--interval",
            "10",
            "--max-seconds",
            str(seconds),
            "--",
            *argv,
        ]
    owner = {"controller": wave.owner, "cell_id": cell["name"], "panel_sha256": wave.plan_sha}
    receipt = wave.control / "owned" / f"{stage}-{cell['name']}.json"
    latched: list[str] = []

    def progress() -> None:
        """Never enter the legacy executor's remaining-deadline exception grace path."""
        try:
            if latched:
                return
            if wave.cancelled.is_set():
                raise RuntimeError("ROOT cancellation received")
            wave.guard.check("readout" if observational else stage)
        except Exception as error:
            latched.append(f"{type(error).__name__}: {error}")
            try:
                record = read_json(receipt)
                if (
                    record.get("owner") != owner
                    or record.get("command") != command
                    or record.get("mode") != stage
                ):
                    raise ValueError("resource-stop receipt differs from this exact owned command")
                child = record.get("child")
                if isinstance(child, dict) and is_same_process(child):
                    terminate_owned_process(child, grace_seconds=0.0)
            except Exception as stop_error:
                latched.append(f"owned cleanup refused: {type(stop_error).__name__}: {stop_error}")

    wave.status(cell, stage, output=str(output))
    code, timed_out, _ = run_owned_command(
        command,
        cwd=source,
        env=environment(source, cell["gpu_uuid"]),
        log_path=wave.control / f"{stage}-{cell['name']}.log",
        receipt_path=receipt,
        owner=owner,
        stage=1,
        trial_number=cell["gpu"],
        mode=stage,
        gpu_key=str(cell["gpu"]),
        output_dir=output,
        timeout_seconds=seconds,
        progress=progress,
    )
    if latched:
        raise RuntimeError("; ".join(latched))
    if code != 0 or timed_out:
        raise RuntimeError(f"{stage} failed exit={code}, timed_out={timed_out}; no automatic retry")
