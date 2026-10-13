"""Check the actual R2 producer outputs before R4 reuses its profile."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PACKET = Path("research/campaign_2026-10-13/strategy_execution_r1/first_cohort_f3_teachers_r4")
sys.path.insert(0, str(ROOT / PACKET / "kernel_support"))

from profile_reuse import _verify_profile_split_log, validate_profile_reuse  # noqa: E402


def test_actual_r2_profile_outputs_validate_for_exact_r1_recipe() -> None:
    receipt = validate_profile_reuse(ROOT)

    assert receipt["status"] == "passed"
    assert receipt["profile_device"]["name"] == "Tesla T4"
    assert receipt["peak_worker_ram_bytes"] == 4_490_899_456
    assert receipt["profile_batches"] == 10
    assert receipt["optimizer_completed_updates"] == 200
    assert receipt["split_log_evidence"]["group_counts"] == {
        "fit_train_groups": 10539,
        "fit_valid_groups": 1368,
        "evaluation_groups": 505,
    }
    assert receipt["split_log_evidence"]["portable_hash_seconds_total"] == pytest.approx(1125.241)
    assert receipt["scientific_result"] is None


def test_profile_reuse_rejects_changed_collected_worker_log(tmp_path: Path) -> None:
    source_packet = ROOT / PACKET
    target_packet = tmp_path / PACKET
    shutil.copytree(source_packet / "profile_reuse_r2", target_packet / "profile_reuse_r2")
    source_manifest = (
        ROOT
        / "research/campaign_2026-10-13/strategy_execution_r1/first_cohort_f3_teachers_r1/"
        / "source_snapshot/SOURCE_MANIFEST_H390.json"
    )
    target_manifest = (
        tmp_path
        / "research/campaign_2026-10-13/strategy_execution_r1/first_cohort_f3_teachers_r1/"
        / "source_snapshot/SOURCE_MANIFEST_H390.json"
    )
    target_manifest.parent.mkdir(parents=True)
    shutil.copyfile(source_manifest, target_manifest)
    log = target_packet / "profile_reuse_r2/profile_worker_r2.log"
    log.write_bytes(log.read_bytes() + b"tampered\n")

    with pytest.raises(ValueError, match="differs from terminal inventory"):
        validate_profile_reuse(tmp_path)


def test_profile_split_log_rejects_hash_or_membership_drift(tmp_path: Path) -> None:
    source_evidence = ROOT / PACKET / "profile_reuse_r2"
    target_evidence = tmp_path / PACKET / "profile_reuse_r2"
    target_evidence.mkdir(parents=True)
    for filename in ("profile_worker_r2.log", "profile_input_identity_r2.json"):
        shutil.copyfile(source_evidence / filename, target_evidence / filename)

    log = target_evidence / "profile_worker_r2.log"
    lines = log.read_text().splitlines()
    for index, line in enumerate(lines):
        if '"file": "valid.parquet"' in line and '"event": "portable_sha256_complete"' in line:
            lines[index] = line.replace(
                "2861ee30914065702626a3ac033d80d0563cdacaea50c667fc828c2eeb418782",
                "0" * 64,
            )
            break
    log.write_text("\n".join(lines) + "\n")

    with pytest.raises(ValueError, match="does not match the frozen valid.parquet input"):
        _verify_profile_split_log(tmp_path)
