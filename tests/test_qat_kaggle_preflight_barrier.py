from __future__ import annotations

import pytest

from scripts.kaggle_qat_kernel_r1 import advise_hashed_inputs_dontneed, execute_preflight_barrier


def test_candidate_phase_runs_only_after_preflight_verification() -> None:
    """Keep profile/fit dispatch behind the separate runner and receipt barrier."""
    events: list[str] = []

    def run_preflight() -> dict[str, str]:
        events.append("preflight_runner_complete")
        return {"terminal": "complete"}

    def verify_preflight(result: dict[str, str]) -> None:
        assert result["terminal"] == "complete"
        events.append("input_receipt_pass")

    def run_candidates() -> dict[str, str]:
        events.append("candidate_runner_start")
        return {"terminal": "complete"}

    result = execute_preflight_barrier(run_preflight, verify_preflight, run_candidates)

    assert result == {"terminal": "complete"}
    assert events == ["preflight_runner_complete", "input_receipt_pass", "candidate_runner_start"]


def test_candidate_phase_does_not_run_when_preflight_fails() -> None:
    """Preserve a fail-closed barrier if either terminal check rejects preflight."""
    events: list[str] = []

    def run_preflight() -> dict[str, str]:
        events.append("preflight_runner_complete")
        return {"terminal": "complete"}

    def reject_receipt(result: dict[str, str]) -> None:
        assert result["terminal"] == "complete"
        events.append("input_receipt_failed")
        raise ValueError("input receipt did not pass")

    def run_candidates() -> dict[str, str]:
        events.append("candidate_runner_start")
        return {"terminal": "complete"}

    with pytest.raises(ValueError, match="input receipt did not pass"):
        execute_preflight_barrier(run_preflight, reject_receipt, run_candidates)

    assert events == ["preflight_runner_complete", "input_receipt_failed"]


def test_targeted_cache_handoff_advises_only_verified_readonly_fixture(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Use only the actual preflight identity and never modify its payload."""
    import hashlib

    root = tmp_path / "datasets"
    root.mkdir()
    payload = root / "fixture.parquet"
    content = b"small actual-producer-shaped fixture"
    payload.write_bytes(content)
    digest = hashlib.sha256(content).hexdigest()
    calls: list[tuple[int, int, int, int]] = []
    monkeypatch.setattr(
        "scripts.kaggle_qat_kernel_r1.os.posix_fadvise",
        lambda fd, offset, length, advice: calls.append((fd, offset, length, advice)),
    )
    receipt = {
        "status": "pass",
        "bytes_hashed": len(content),
        "identity_manifest_sha256": "a" * 64,
        "inputs": [
            {
                "dataset": "private-fixture",
                "path": "fixture.parquet",
                "resolved_path": str(payload.resolve()),
                "expected_size": len(content),
                "size": len(content),
                "expected_sha256": digest,
                "sha256": digest,
            }
        ],
    }

    result = advise_hashed_inputs_dontneed(
        receipt, tmp_path / "handoff.json", allowed_root=root
    )

    assert result["status"] == "pass"
    assert result["files_expected"] == 1
    assert result["files"][0]["path"] == "fixture.parquet"
    assert len(calls) == 1
    assert calls[0][1:] == (0, 0, 4)
    assert hashlib.sha256(payload.read_bytes()).hexdigest() == digest


def test_targeted_cache_handoff_rejects_unverified_or_outside_mount_input(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reject any entry not sealed by the successful preflight identity receipt."""
    import hashlib

    allowed = tmp_path / "datasets"
    allowed.mkdir()
    outside = tmp_path / "outside.parquet"
    outside.write_bytes(b"fixture")
    digest = hashlib.sha256(b"fixture").hexdigest()
    calls: list[int] = []
    monkeypatch.setattr(
        "scripts.kaggle_qat_kernel_r1.os.posix_fadvise",
        lambda *args: calls.append(1),
    )
    receipt = {
        "status": "pass",
        "bytes_hashed": 7,
        "inputs": [
            {
                "resolved_path": str(outside),
                "expected_size": 7,
                "size": 7,
                "expected_sha256": digest,
                "sha256": digest,
            }
        ],
    }

    with pytest.raises(ValueError, match="outside the authorized dataset mount"):
        advise_hashed_inputs_dontneed(
            receipt, tmp_path / "handoff.json", allowed_root=allowed
        )
    receipt["inputs"][0]["resolved_path"] = str(allowed / "missing.parquet")
    with pytest.raises(FileNotFoundError):
        advise_hashed_inputs_dontneed(
            receipt, tmp_path / "handoff.json", allowed_root=allowed
        )
    assert calls == []


def test_targeted_cache_handoff_accepts_collected_actual_preflight_receipt(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercise the exact 34-input Kaggle producer receipt without rereading payloads."""
    import json
    import os
    import stat
    from pathlib import Path

    fixture_path = (
        Path(__file__).resolve().parents[1]
        / "research/campaign_2026-10-13/qat_renewable_r1/kaggle_f0_s4_r2/fixtures/actual_input_preflight_receipt_r1.json"
    )
    receipt = json.loads(fixture_path.read_text(encoding="utf-8"))
    assert receipt["bytes_hashed"] == 40_075_446_871
    root = tmp_path / "datasets"
    root.mkdir()
    sizes: dict[Path, int] = {}
    for index, item in enumerate(receipt["inputs"]):
        payload = root / f"input-{index}.bin"
        payload.write_bytes(b"x")
        resolved = payload.resolve()
        item["resolved_path"] = str(resolved)
        sizes[resolved] = item["expected_size"]

    original_stat = Path.stat
    original_open = Path.open
    original_fstat = os.fstat
    open_sizes: dict[int, int] = {}
    advice: list[int] = []

    def stat_result(size: int) -> os.stat_result:
        return os.stat_result((stat.S_IFREG, 0, 0, 1, 0, 0, size, 0, 0, 0))

    def fixture_stat(path: Path, *args, **kwargs):
        if path in sizes:
            return stat_result(sizes[path])
        return original_stat(path, *args, **kwargs)

    def fixture_open(path: Path, *args, **kwargs):
        stream = original_open(path, *args, **kwargs)
        if path in sizes:
            open_sizes[stream.fileno()] = sizes[path]
        return stream

    def fixture_fstat(fd: int) -> os.stat_result:
        if fd in open_sizes:
            return stat_result(open_sizes[fd])
        return original_fstat(fd)

    monkeypatch.setattr(Path, "stat", fixture_stat)
    monkeypatch.setattr(Path, "open", fixture_open)
    monkeypatch.setattr(os, "fstat", fixture_fstat)
    monkeypatch.setattr(
        "scripts.kaggle_qat_kernel_r1.os.posix_fadvise",
        lambda fd, offset, length, flag: advice.append(fd),
    )

    result = advise_hashed_inputs_dontneed(
        receipt, tmp_path / "handoff.json", allowed_root=root
    )

    assert result["status"] == "pass"
    assert result["files_expected"] == 34
    assert result["bytes_expected"] == 40_075_446_871
    assert len(advice) == 34


def test_cache_handoff_advice_error_records_failure_and_blocks_candidates(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An OS cache-advice failure is durable and fail-closed at the phase barrier."""
    import hashlib
    import json

    root = tmp_path / "datasets"
    root.mkdir()
    payload = root / "fixture.parquet"
    payload.write_bytes(b"fixture")
    digest = hashlib.sha256(b"fixture").hexdigest()
    receipt = {
        "status": "pass",
        "bytes_hashed": 7,
        "inputs": [
            {
                "dataset": "private-fixture",
                "path": "fixture.parquet",
                "resolved_path": str(payload.resolve()),
                "expected_size": 7,
                "size": 7,
                "expected_sha256": digest,
                "sha256": digest,
            }
        ],
    }

    def fail_advice(*args: int) -> None:
        raise OSError("fixture cache advice failure")

    monkeypatch.setattr(
        "scripts.kaggle_qat_kernel_r1.os.posix_fadvise", fail_advice
    )
    events: list[str] = []

    def verify(_: dict[str, str]) -> None:
        events.append("preflight_passed")
        advise_hashed_inputs_dontneed(
            receipt,
            tmp_path / "cache_handoff.json",
            allowed_root=root,
        )

    with pytest.raises(RuntimeError, match="candidate admission blocked"):
        execute_preflight_barrier(
            lambda: {"terminal": "complete"},
            verify,
            lambda: events.append("candidate_started") or {"terminal": "complete"},
        )

    failed = json.loads((tmp_path / "cache_handoff.json").read_text())
    assert failed["status"] == "failed"
    assert "fixture cache advice failure" in failed["error"]
    assert events == ["preflight_passed"]
