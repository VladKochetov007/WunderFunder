"""Receipt-bound R6 entrypoint, explicit GPU caps and absolute completion guards.

Any represents the controller's heterogeneous JSON receipt, never scientific state.
"""

from __future__ import annotations

import math
import os
import signal
import time
from pathlib import Path
from typing import Any

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
DEADLINE = 1791057600  # 2026-10-03 23:00 Europe/Kyiv, fixed absolute epoch
ALLOCATOR_BYTES = 4_800_000_000
OBSERVED_BYTES = 5_500_000_000
PAYLOAD = "research/campaign_2026-10-03/continuation/round6/controller_state_drift_payload.sh"
RUNTIME_FILES = (
    "scripts/campaign_job.py",
    "scripts/limited",
    "research/campaign_2026-10-03/continuation/run_gpu.sh",
    PAYLOAD,
)
LIMITS = {
    "deadline_epoch": DEADLINE,
    "minimum_remaining_at_payload_entry_seconds": 325,
    "work_timeout_seconds": 300,
    "kill_grace_seconds": 5,
    "outer_runtime_seconds": 320,
    "allocator_limit_bytes": ALLOCATOR_BYTES,
    "observed_device_limit_bytes": OBSERVED_BYTES,
    "cpu_cores": 8,
    "host_memory_bytes": 24 * 1024**3,
    "root_cpu_cores": 12,
    "root_host_memory_bytes": 36 * 1024**3,
}


def check_time(now: float, stage: str = "completion") -> None:
    """Reject nonfinite times or completion at/after the absolute deadline."""
    if not math.isfinite(now) or now >= DEADLINE:
        raise RuntimeError("absolute 23:00 Kyiv completion deadline: " + stage)


def check_memory(observed: int) -> None:
    """Use observed TOTAL device usage, not just this PyTorch allocator."""
    if type(observed) is not int or observed < 0 or observed > OBSERVED_BYTES:
        raise RuntimeError("observed total GPU memory exceeds 5.5 decimal GB")


def allocator_fraction(total: int) -> float:
    """Exact original min(.70,4.8decimalGB/device_total) ceiling."""
    if type(total) is not int or total <= 0:
        raise ValueError("invalid total device memory")
    return min(0.7, ALLOCATOR_BYTES / total)


def verify_contract(receipt: dict[str, Any]) -> None:
    """Bind the new actual R6 controller wrapper, runtime chain and hard limits."""
    # Import after path setup in protocol; sha is a pure streaming file identity.
    from protocol import cap

    expected = {name: cap.sha(ROOT / name) for name in RUNTIME_FILES}
    if receipt.get("runtime_sha256") != expected or receipt.get("runtime_limits") != LIMITS:
        raise ValueError("UNAPPROVED new R6 runtime/caps/completion contract")


def scope_guard(receipt: dict[str, Any]) -> None:
    """Require controller chain and a handshake from the hash-bound NEW payload."""
    verify_contract(receipt)
    if (
        os.environ.get("WNN_IN_CAMPAIGN_JOB") != "1"
        or os.environ.get("WNN_SLICE") != "wnn-campaign-gpu.slice"
        or os.environ.get("WNN_STATE_DRIFT_PAYLOAD_SHA256") != receipt["runtime_sha256"][PAYLOAD]
        or os.environ.get("WNN_RESEARCH_DEADLINE_UTC") != str(DEADLINE)
        or os.environ.get("WNN_CAMPAIGN_VRAM_BYTES") != str(OBSERVED_BYTES)
    ):
        raise ValueError("UNAPPROVED new controller_state_drift_payload chain")
    check_time(time.time(), "runtime")


def guard_gpu(receipt: dict[str, Any] | None) -> None:
    """Check deadline and observed memory before/after each bounded GPU phase."""
    if receipt is None:
        return  # CPU synthetic-only caller performs no CUDA query.
    scope_guard(receipt)
    free, total = torch.cuda.mem_get_info()
    check_memory(total - free)


def configure_gpu(module: Any, receipt: dict[str, Any]) -> None:
    """Preserve source settings; explicitly cap allocator before any model/RNG restore."""
    scope_guard(receipt)
    module.configure_torch()
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(allocator_fraction(total))
    guard_gpu(receipt)


def arm_completion_alarm() -> None:
    """Stop Python work with deadline slack; reviewed wrapper must provide hard kill.

    Signals alone cannot interrupt a stalled CUDA/native call. The separately
    hash-bound controller wrapper MUST require >=325s at entry and run timeout300
    plus5s hard-kill grace, leaving >=20s before the absolute deadline. This actor
    never creates/approves that wrapper. Independent runtime review is required.
    """
    now = time.time()
    check_time(now, "alarm entry")
    budget = min(300.0, DEADLINE - now - 10.0)
    if budget <= 0:
        raise RuntimeError("insufficient completion slack")

    def expired(signum: int, frame: Any) -> None:
        raise RuntimeError("state-drift bounded completion alarm")

    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, budget)
