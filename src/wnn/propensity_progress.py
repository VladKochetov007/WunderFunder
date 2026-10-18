from __future__ import annotations

import json
import os
import stat
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

MAX_JOURNAL_BYTES = 16_000_000


@dataclass
class UpdateProgress:
    """Count attempted, accepted, and rejected strict propensity updates."""

    optimizer_attempts: int = 0
    optimizer_step_calls: int = 0
    optimizer_completed_updates: int = 0
    nonfinite_updates: int = 0
    nonfinite_inputs: int = 0
    nonfinite_logits: int = 0
    nonfinite_losses: int = 0
    nonfinite_gradients: int = 0
    nonfinite_gradient_norms: int = 0
    nonfinite_weights: int = 0
    nonfinite_optimizer_states: int = 0
    loss_sum: float = 0.0
    loss_steps: int = 0
    preclip_grad_norm_sum: float = 0.0
    preclip_grad_norm_max: float = 0.0
    preclip_grad_norm_steps: int = 0
    learning_rate_first: float | None = None
    learning_rate_last: float | None = None
    learning_rate_min: float | None = None
    learning_rate_max: float | None = None

    def start(self, learning_rate: float) -> None:
        """Count one update slot and track its effective learning rate."""
        self.optimizer_attempts += 1
        self.learning_rate_first = (
            learning_rate if self.learning_rate_first is None else self.learning_rate_first
        )
        self.learning_rate_last = learning_rate
        self.learning_rate_min = (
            learning_rate
            if self.learning_rate_min is None
            else min(learning_rate, self.learning_rate_min)
        )
        self.learning_rate_max = (
            learning_rate
            if self.learning_rate_max is None
            else max(learning_rate, self.learning_rate_max)
        )

    def complete(self, loss: float, preclip_grad_norm: float) -> None:
        """Record a finite optimizer update and its pre-clip diagnostics."""
        self.optimizer_completed_updates += 1
        self.loss_sum += loss
        self.loss_steps += 1
        self.preclip_grad_norm_sum += preclip_grad_norm
        self.preclip_grad_norm_max = max(self.preclip_grad_norm_max, preclip_grad_norm)
        self.preclip_grad_norm_steps += 1

    def step_called(self) -> None:
        """Record that the optimizer step method ran before state validation."""
        self.optimizer_step_calls += 1

    def reject_nonfinite(self, stage: str) -> None:
        """Count a strict update rejected for a non-finite tensor."""
        self.nonfinite_updates += 1
        key = {
            "inputs": "nonfinite_inputs",
            "logits": "nonfinite_logits",
            "loss": "nonfinite_losses",
            "gradients": "nonfinite_gradients",
            "gradient_norm": "nonfinite_gradient_norms",
            "weights": "nonfinite_weights",
            "optimizer_state": "nonfinite_optimizer_states",
        }.get(stage)
        if key is None:
            raise ValueError(f"unknown nonfinite stage: {stage}")
        setattr(self, key, getattr(self, key) + 1)

    def snapshot(self) -> dict[str, int | float | None]:
        """Return JSON-safe counters and aggregate optimizer diagnostics."""
        values = asdict(self)
        values["optimizer_rejected_nonfinite"] = self.nonfinite_updates
        values["loss_mean"] = self.loss_sum / self.loss_steps if self.loss_steps else None
        values["preclip_grad_norm_mean"] = (
            self.preclip_grad_norm_sum / self.preclip_grad_norm_steps
            if self.preclip_grad_norm_steps
            else None
        )
        return values


class ProgressJournal:
    """Append bounded, sequenced JSONL progress without following symlinks."""

    def __init__(self, path: Path, *, resume_after: int | None = None) -> None:
        self.path = path
        self._events: list[dict[str, Any]] = []
        if resume_after is None:
            self._sequence = 0
            self._create()
            return
        self._events = self._read()
        self._sequence = self._events[-1]["sequence"] if self._events else 0
        if resume_after < 0 or resume_after > self._sequence:
            raise ValueError("resume progress cursor is outside the journal")

    def _create(self) -> None:
        flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(self.path, flags, 0o600)
        os.close(descriptor)

    def _read(self) -> list[dict[str, Any]]:
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(self.path, flags)
        try:
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode) or before.st_size > MAX_JOURNAL_BYTES:
                raise ValueError("progress journal is not a bounded regular file")
            chunks = []
            remaining = before.st_size
            while remaining:
                chunk = os.read(descriptor, min(remaining, 1 << 20))
                if not chunk:
                    raise ValueError("progress journal shortened during read")
                chunks.append(chunk)
                remaining -= len(chunk)
            after = os.fstat(descriptor)
            path_stat = self.path.lstat()
            if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
            ) or (path_stat.st_dev, path_stat.st_ino) != (after.st_dev, after.st_ino):
                raise ValueError("progress journal changed during read")
        finally:
            os.close(descriptor)
        raw = b"".join(chunks)
        events = [json.loads(line) for line in raw.splitlines()]
        if any(
            not isinstance(event, dict) or event.get("sequence") != index
            for index, event in enumerate(events, 1)
        ):
            raise ValueError("progress journal sequence is invalid")
        return events

    def append(self, attempt_id: str, event: dict[str, object], *, durable: bool = False) -> int:
        """Append a sequenced event, syncing periodically and at recovery boundaries."""
        self._sequence += 1
        payload = (
            json.dumps(
                {"sequence": self._sequence, "attempt_id": attempt_id, **event},
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
            + b"\n"
        )
        if self.path.stat().st_size + len(payload) > MAX_JOURNAL_BYTES:
            raise ValueError("progress journal exceeded its bounded size")
        flags = os.O_APPEND | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(self.path, flags)
        try:
            current = os.fstat(descriptor)
            if not stat.S_ISREG(current.st_mode):
                raise ValueError("progress journal is not a regular file")
            if os.write(descriptor, payload) != len(payload):
                raise OSError("short write to progress journal")
            if durable or self._sequence % 64 == 0:
                os.fsync(descriptor)
        finally:
            os.close(descriptor)
        self._events.append(json.loads(payload))
        return self._sequence

    def recovery_summary(self, resume_after: int) -> dict[str, object]:
        """Report update work after the durable epoch checkpoint that will roll back."""
        orphaned = [event for event in self._events if event["sequence"] > resume_after]
        updates = {
            (
                event.get("attempt_id"),
                event.get("epoch"),
                event.get("batch"),
                event.get("chunk_start"),
            ): event
            for event in orphaned
            if event.get("event") in {"optimizer_update_completed", "optimizer_update_rejected"}
        }
        starts = [event for event in orphaned if event.get("event") == "optimizer_update_started"]
        unknown_step_calls = sum(
            (
                event.get("attempt_id"),
                event.get("epoch"),
                event.get("batch"),
                event.get("chunk_start"),
            )
            not in updates
            for event in starts
        )
        return {
            "events_after_checkpoint": len(orphaned),
            "rolled_back_optimizer_attempts": sum(
                event.get("event") == "optimizer_update_started" for event in orphaned
            ),
            "rolled_back_completed_updates": sum(
                event.get("event") == "optimizer_update_completed" for event in orphaned
            ),
            "rolled_back_nonfinite_updates": sum(
                event.get("event") == "optimizer_update_rejected" for event in orphaned
            ),
            "known_rolled_back_optimizer_step_calls": sum(
                event.get("event") == "optimizer_update_completed"
                or (
                    event.get("event") == "optimizer_update_rejected"
                    and event.get("optimizer_step_called") is True
                )
                for event in orphaned
            ),
            "unknown_inflight_optimizer_step_calls": unknown_step_calls,
            "failed_attempts": [
                {key: value for key, value in event.items() if key not in {"sequence", "event"}}
                for event in orphaned
                if event.get("event") == "run_failed"
            ],
        }

    def attempt_outcome(self, attempt_id: str) -> str | None:
        """Return an attempt's last terminal marker, if the journal contains one."""
        terminal = [
            event["event"]
            for event in self._events
            if event.get("attempt_id") == attempt_id
            and event.get("event") in {"run_completed", "run_failed"}
        ]
        return terminal[-1] if terminal else None

    @property
    def sequence(self) -> int:
        """Return the last event sequence appended to this journal."""
        return self._sequence
