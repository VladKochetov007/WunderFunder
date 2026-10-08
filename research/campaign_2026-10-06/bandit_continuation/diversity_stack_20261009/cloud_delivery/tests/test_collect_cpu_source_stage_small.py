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


def test_listing_preflight_sums_only_allowed_files_and_accepts_cli_headers() -> None:
    """Metadata inventory accepts Kaggle CSV headers and excludes cache arrays."""
    listing = (
        "File Name,Total Bytes,Date\n"
        "out/boot_marker.json,120,2026-10-09\n"
        "out/parsed_cache/manifest.json,450,2026-10-09\n"
        "out/parsed_cache/train_features.npy,7732440000,2026-10-09\n"
    )

    files, total = collector.parse_selected_file_listing(listing)

    assert files == [
        {"name": "out/boot_marker.json", "size_bytes": 120},
        {"name": "out/parsed_cache/manifest.json", "size_bytes": 450},
    ]
    assert total == 570


def test_listing_preflight_rejects_oversized_selected_outputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The provider metadata cap must reject before output-download is invoked."""
    monkeypatch.setattr(collector, "MAX_OUTPUT_BYTES", 100)
    monkeypatch.setattr(
        collector.subprocess,
        "run",
        lambda *args, **kwargs: type(
            "Completed",
            (),
            {"returncode": 0, "stdout": "name,size\nout/boot_marker.json,101\n", "stderr": ""},
        )(),
    )

    with pytest.raises(RuntimeError, match="above .*byte cap"):
        collector.preflight_selected_output_bytes()


def test_listing_preflight_fails_closed_on_truncated_page() -> None:
    """A full metadata page cannot silently understate selected download size."""
    rows = ["name,size"] + [
        f"unrelated-{index}.txt,1" for index in range(collector.FILE_LISTING_PAGE_SIZE)
    ]

    with pytest.raises(ValueError, match="possibly truncated"):
        collector.parse_selected_file_listing("\n".join(rows))


def test_live_download_terminates_when_provider_directory_crosses_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The subprocess is stopped while writing, before post-download validation."""
    attempt = tmp_path / "attempt"
    attempt.mkdir()
    provider_dir = attempt / "provider"
    provider_dir.mkdir()
    monkeypatch.setattr(collector, "MAX_OUTPUT_BYTES", 1)

    class FakeProcess:
        returncode: int | None = None

        def poll(self) -> int | None:
            if self.returncode is not None:
                return self.returncode
            (provider_dir / "boot_marker.json").write_bytes(b"oversize")
            return None

        def terminate(self) -> None:
            self.returncode = -15

        def wait(self, timeout: float | None = None) -> int:
            assert timeout is not None
            return int(self.returncode or 0)

        def kill(self) -> None:
            self.returncode = -9

    fake_process = FakeProcess()
    monkeypatch.setattr(collector.subprocess, "Popen", lambda *args, **kwargs: fake_process)
    monkeypatch.setattr(
        collector.time,
        "sleep",
        lambda _: pytest.fail("live byte limit failed to stop the download subprocess"),
    )

    exit_code, tail = collector.run_capped_download(["kaggle"], provider_dir, attempt)

    assert exit_code == 75
    assert "live byte cap" in tail
    assert fake_process.returncode == -15


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
