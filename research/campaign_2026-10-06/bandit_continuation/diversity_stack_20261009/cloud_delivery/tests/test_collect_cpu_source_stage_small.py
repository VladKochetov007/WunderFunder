"""Offline tests for the bounded CPU source-stage terminal collector."""

from __future__ import annotations

import json
from pathlib import Path

import collect_cpu_source_stage_small as collector
import pytest


def test_terminal_status_reader_ignores_later_poll_errors(tmp_path: Path) -> None:
    """A transient status API error must not erase observed terminal evidence."""
    log_path = tmp_path / "poll.jsonl"
    rows = [
        {
            "observed_at_utc": "2026-10-08T22:10:00+00:00",
            "status": [
                {
                    "kernel": collector.KERNEL,
                    "status": "KernelWorkerStatus.COMPLETE",
                    "monitoring_state": "polling_until_terminal_evidence_downloaded",
                }
            ],
        },
        {
            "observed_at_utc": "2026-10-08T22:15:00+00:00",
            "status": [
                {
                    "kernel": collector.KERNEL,
                    "monitoring_state": "poll_error_retry_next_timer_tick",
                }
            ],
        },
    ]
    log_path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    observation = collector.latest_terminal_observation(log_path)

    assert observation == {
        "observed_at_utc": "2026-10-08T22:10:00+00:00",
        "status": "KernelWorkerStatus.COMPLETE",
        "failure_message": "",
    }


def test_output_filter_keeps_small_receipts_and_rejects_cache_arrays(tmp_path: Path) -> None:
    """Only named JSON/log telemetry can enter the terminal artifact inventory."""
    provider_dir = tmp_path / "provider"
    receipt = provider_dir / "out" / "parsed_cache" / "manifest.json"
    receipt.parent.mkdir(parents=True)
    receipt.write_text("{}\n")

    inventory, total_bytes = collector.file_inventory(provider_dir)

    assert list(inventory) == ["out/parsed_cache/manifest.json"]
    assert total_bytes == receipt.stat().st_size
    (receipt.parent / "train_features.npy").write_bytes(b"never collect")
    with pytest.raises(ValueError, match="escaped the small-file filter"):
        collector.file_inventory(provider_dir)


def test_receipt_hash_recheck_rejects_changed_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A collector receipt is reusable only while every recorded byte still matches."""
    output_root = tmp_path / "outputs"
    output_root.mkdir()
    payload = output_root / "attempt_001" / "provider" / "boot_marker.json"
    payload.parent.mkdir(parents=True)
    payload.write_text("{}\n")
    monkeypatch.setattr(collector, "OUTPUT_ROOT", output_root)
    receipt_path = tmp_path / "receipt.json"
    monkeypatch.setattr(collector, "RECEIPT_PATH", receipt_path)
    digest = collector.sha256_file(payload)
    receipt_path.write_text(
        json.dumps(
            {
                "terminal_status": "KernelWorkerStatus.COMPLETE",
                "terminal_collection": {
                    "collection_complete": True,
                    "output_exit_code": 0,
                    "log_exit_code": 0,
                    "files": {
                        "attempt_001/provider/boot_marker.json": {
                            "size_bytes": payload.stat().st_size,
                            "sha256": digest,
                        }
                    },
                },
            }
        )
    )

    assert collector.already_collected()
    payload.write_text("changed\n")
    assert not collector.already_collected()


@pytest.mark.parametrize("status", ["KernelWorkerStatus.ERROR", "COMPLETED"])
def test_terminal_status_spellings(status: str) -> None:
    """Kaggle enum prefixes and plain terminal values both remain recognized."""
    assert collector.is_terminal_status(status)


def test_nonterminal_status_is_not_collected() -> None:
    """A running job cannot trigger output download."""
    assert not collector.is_terminal_status("KernelWorkerStatus.RUNNING")
