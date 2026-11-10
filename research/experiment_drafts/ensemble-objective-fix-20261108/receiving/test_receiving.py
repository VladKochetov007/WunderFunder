from __future__ import annotations

import importlib
import json
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

FOLDER = Path(__file__).parent
ACCEPTED = FOLDER.parents[1] / "tl3-transformer-complement-execution-20261107/receiving"
sys.path[:0] = [str(FOLDER), str(ACCEPTED)]

admission = importlib.import_module("admission")
SPLIT_SHA, approved, authority, retirement = (
    admission.SPLIT_SHA,
    admission.approved,
    admission.authority,
    admission.retirement,
)
progress = importlib.import_module("lifecycle").progress
Receiver = importlib.import_module("receiver").Receiver


def write(path: Path, value: dict) -> None:
    """Create independent native/ROOT object fixtures."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def profile(tmp_path: Path) -> tuple[dict, dict]:
    """Construct an exact qualified native R2 receipt and separate ROOT approval."""
    cell = {
        "cell": "labels_s101101",
        "normalized_config_sha256": "config",
        "profile_receipt_path": str(tmp_path / "profiles-r2/cell/profile_receipt.json"),
    }
    native = {
        "status": "profile_complete",
        "updates": 50,
        "errors": [],
        "scientific_checkpoint_written": False,
        "identity": {
            "world_size": 2,
            "source_sha256": "source",
            "config_sha256": "config",
            "data": {"split_sha256": SPLIT_SHA},
        },
        "engineering_reload": {
            "status": "passed",
            "by_rank": [
                {"rank": rank, "status": "passed", "model_optimizer_scheduler_rng_equal": True}
                for rank in (0, 1)
            ],
        },
    }
    write(Path(cell["profile_receipt_path"]), native)
    write(
        tmp_path / "approvals/labels_s101101.json", native | {"approved": True, "reviewer": "ROOT"}
    )
    return cell, native


def test_profile_r2_override(tmp_path: Path) -> None:
    """A preserved failed R1 path cannot replace the independently approved R2 evidence."""
    cell, _ = profile(tmp_path)
    assert approved(tmp_path, cell, "source")
    with pytest.raises(ValueError):
        approved(tmp_path, cell, "other-source")


def test_profile_exact_copy_and_both_ranks(tmp_path: Path) -> None:
    """Both actual ranks and the exact copied native fields are mandatory."""
    cell, native = profile(tmp_path)
    native["engineering_reload"]["by_rank"][1]["model_optimizer_scheduler_rng_equal"] = False
    write(Path(cell["profile_receipt_path"]), native)
    write(
        tmp_path / "approvals/labels_s101101.json", native | {"approved": True, "reviewer": "ROOT"}
    )
    with pytest.raises(ValueError):
        approved(tmp_path, cell, "source")


def test_clock_never_resets() -> None:
    """Fit/collection deadlines remain derived from the earliest actual native qualification."""
    value = {
        "approved_by": "ROOT",
        "fit_cap": 4,
        "profile_cap": 8,
        "first_profile_epoch": 1794091436,
        "fit_deadline_epoch": 1794102236,
        "deadline_epoch": 1794105836,
        "collection_reserve_seconds": 3600,
    }
    authority(value)
    with pytest.raises(ValueError):
        authority(value | {"deadline_epoch": value["deadline_epoch"] + 1})


def test_progress_latches_cleanup_error() -> None:
    """Even a cleanup refusal returns normally to the canonical process poll/reap loop."""

    def fail() -> None:
        raise RuntimeError("guard")

    def cleanup() -> None:
        raise OSError("receipt read failed")

    receiver = SimpleNamespace(guard=fail, terminate=cleanup, failures={})
    assert progress(receiver, "new-fit") is None
    assert "cleanup refusal" in receiver.failures["new-fit"]


def test_executor_reaps_before_callback_failure(tmp_path: Path) -> None:
    """A fake native executor reaches its return after guard AND cleanup fail in its callback."""
    calls = []
    receiver = Receiver.__new__(Receiver)
    receiver.args = SimpleNamespace(
        python="python",
        plan=tmp_path / "plan.json",
        accepted_receiver=tmp_path,
        support_src=tmp_path,
    )
    receiver.source = tmp_path
    receiver.control = tmp_path
    receiver.run_root = tmp_path / "runs"
    receiver.lock = threading.RLock()
    receiver.records, receiver.failures = {}, {}
    receiver.owner = {"pid": 1, "birth": 2}
    receiver.state = {"fit_attempts": []}
    receiver.authority = lambda: {
        "fit_deadline_epoch": time.time() + 30,
        "deadline_epoch": time.time() + 60,
    }
    receiver.environment = lambda pair: {}

    def guard(**kwargs: object) -> None:
        if not kwargs.get("admission"):
            raise RuntimeError("stop")

    def cleanup() -> None:
        calls.append("new-only-cleanup")
        raise OSError("refusal")

    def run(*args: object, **kwargs: object) -> tuple[int, bool, dict]:
        kwargs["progress"]()
        calls.append("canonical-reaped-and-returned")
        return 0, False, {}

    receiver.guard, receiver.terminate = guard, cleanup
    receiver.h = SimpleNamespace(write=lambda *a, **k: None, run=run)
    with pytest.raises(RuntimeError, match="no retry"):
        receiver.execute("fit", "labels_s101101", "0,1")
    assert calls == ["new-only-cleanup", "canonical-reaped-and-returned"]


def test_retirement_never_signals(tmp_path: Path) -> None:
    """Retirement admission reads exact old checkpoints and checks absence without any signal."""
    from admission import digest

    checkpoint = tmp_path / "old.pt"
    checkpoint.write_bytes(b"unchanged old checkpoint")
    cell = {
        "next_epoch": 9,
        "global_updates": 7380,
        "fixed30E_complete": False,
        "checkpoint": {"path": str(checkpoint), "sha256": digest(checkpoint)},
        "stopped_trainers": [{"pid": 10, "birth": 20}],
    }
    write(
        tmp_path / "control/old-four-retired.json",
        {
            "approved_by": "ROOT",
            "status": "retired_at_safe_epoch",
            "cells": [cell] * 4,
        },
    )
    assert retirement(tmp_path, lambda _: False)
    with pytest.raises(ValueError):
        retirement(tmp_path, lambda _: True)
