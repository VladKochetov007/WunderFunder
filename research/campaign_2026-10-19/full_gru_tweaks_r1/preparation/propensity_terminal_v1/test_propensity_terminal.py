"""Synthetic integrity checks for the F0 propensity terminal verifier."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from propensity_terminal_checks import _expect_progress, _read_jsonl, _summarize_events
from propensity_terminal_support import _group_identity, _safe_relative, run_loader_self_test


def _event(sequence: int, attempt: str, kind: str, **fields: object) -> dict[str, object]:
    return {"sequence": sequence, "attempt_id": attempt, "event": kind, **fields}


def _small_journal() -> tuple[list[dict[str, object]], int, str]:
    attempt = "synthetic-attempt"
    events = [_event(1, attempt, "run_started", next_epoch=0)]
    events.append(
        _event(2, attempt, "initial_resume_checkpoint_saved", next_epoch=0, sha256="initial")
    )
    cursor = 2
    sequence = 2
    for epoch in range(2):
        for batch in range(2):
            sequence += 1
            update = {"epoch": epoch, "batch": batch, "chunk_start": 0, "learning_rates": [0.002]}
            events.append(_event(sequence, attempt, "optimizer_update_started", **update))
            sequence += 1
            events.append(
                _event(
                    sequence,
                    attempt,
                    "optimizer_update_completed",
                    **update,
                    loss=0.25,
                    preclip_grad_norm=1.5,
                )
            )
        sequence += 1
        events.append(_event(sequence, attempt, "epoch_completed", epoch=epoch))
        cursor = sequence
        sequence += 1
        events.append(
            _event(
                sequence,
                attempt,
                "resume_checkpoint_saved",
                next_epoch=epoch + 1,
                sha256=f"epoch-{epoch + 1}",
            )
        )
    sequence += 1
    events.append(_event(sequence, attempt, "run_completed", next_epoch=2))
    return events, cursor, f"epoch-{2}"


def test_group_identity_is_ordered_and_rejects_duplicates() -> None:
    first = _group_identity([4, 8, 12], np)
    assert first == _group_identity(np.asarray([4, 8, 12], dtype=np.int32), np)
    assert (
        first["ordered_int64_le_sha256"]
        != _group_identity([12, 8, 4], np)["ordered_int64_le_sha256"]
    )
    with pytest.raises(ValueError, match="unique"):
        _group_identity([4, 8, 4], np)


def test_journal_accepts_initial_and_epoch_checkpoint_boundaries() -> None:
    events, cursor, digest = _small_journal()
    summary = _summarize_events(events, "synthetic-attempt", cursor, digest, epochs=2, per_epoch=2)
    assert summary["event_counts"]["initial_resume_checkpoint_saved"] == 1
    assert summary["event_counts"]["resume_checkpoint_saved"] == 2
    assert summary["event_counts"]["optimizer_update_completed"] == 4


def test_journal_rejects_bad_terminal_cursor_and_missing_initial_boundary() -> None:
    events, cursor, digest = _small_journal()
    events[-1]["next_epoch"] = 1
    with pytest.raises(ValueError, match="epoch cursor"):
        _summarize_events(events, "synthetic-attempt", cursor, digest, epochs=2, per_epoch=2)
    events, cursor, digest = _small_journal()
    del events[1]
    for index, event in enumerate(events, 1):
        event["sequence"] = index
    with pytest.raises(ValueError, match="checkpoint boundary"):
        _summarize_events(events, "synthetic-attempt", cursor, digest, epochs=2, per_epoch=2)


def test_jsonl_requires_complete_newline_terminated_events(tmp_path: Path) -> None:
    path = tmp_path / "progress.jsonl"
    path.write_text(json.dumps({"sequence": 1, "attempt_id": "a", "event": "run_started"}))
    with pytest.raises(ValueError, match="incomplete line"):
        _read_jsonl(path)
    path.write_text(json.dumps({"sequence": 1, "attempt_id": "a", "event": "run_started"}) + "\n")
    events, _, size = _read_jsonl(path)
    assert len(events) == 1 and size == path.stat().st_size


def test_progress_requires_exact_updates_finite_values_and_fixed_lr() -> None:
    progress: dict[str, object] = {
        key: 4
        for key in (
            "optimizer_attempts",
            "optimizer_step_calls",
            "optimizer_completed_updates",
        )
    }
    progress.update(
        {
            "nonfinite_updates": 0,
            "optimizer_rejected_nonfinite": 0,
            "loss_mean": 0.2,
            "preclip_grad_norm_mean": 0.3,
            "preclip_grad_norm_max": 0.5,
            "learning_rate_first": 0.002,
            "learning_rate_last": 0.002,
            "learning_rate_min": 0.002,
            "learning_rate_max": 0.002,
        }
    )
    assert _expect_progress(progress, 4, "synthetic")["optimizer_completed_updates"] == 4
    progress["nonfinite_gradients"] = 1
    with pytest.raises(ValueError, match="non-finite update"):
        _expect_progress(progress, 4, "synthetic")


def test_source_paths_reject_traversal_and_allow_in_root_file(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    source = tmp_path / "src" / "module.py"
    source.write_text("pass\n")
    assert _safe_relative(tmp_path, "src/module.py") == source.resolve()
    with pytest.raises(ValueError, match="unsafe"):
        _safe_relative(tmp_path, "../outside.py")


def test_restricted_numpy_rng_loader_accepts_fixture_and_blocks_unknown_global() -> None:
    assert len(run_loader_self_test(torch, np)) == 5
