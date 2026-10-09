"""Focused fail-closed tests for the H230 source-seal boundary."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / "seal_h230_source_bundle_v2.py"
SPEC = importlib.util.spec_from_file_location("h230_source_sealer_tested", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
seal_h230_source_bundle = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(seal_h230_source_bundle)


def test_pending_hash_may_be_filled_but_frozen_hash_must_match() -> None:
    """V1 nulls may be completed after producer success; known hashes stay fixed."""
    seal_h230_source_bundle._verify_expected("pending", None, "actual")
    seal_h230_source_bundle._verify_expected("frozen", "same", "same")

    with pytest.raises(seal_h230_source_bundle.SourceSealError, match="changed"):
        seal_h230_source_bundle._verify_expected("frozen", "expected", "observed")


def test_legacy_seed_label_mismatch_is_reported_not_rewritten() -> None:
    """Keep run labels intact while exposing the seed recorded by training."""
    diagnostic = seal_h230_source_bundle._seed_diagnostic(
        {"seed": 4}, {"args": {"seed": 3}}, {"seed": 3}
    )

    assert diagnostic == {
        "source_bundle_seed_label": 4,
        "provenance_trainer_seed": 3,
        "checkpoint_trainer_seed": 3,
        "label_matches_trainer_seed": False,
        "provenance_matches_checkpoint_seed": True,
    }


def test_e521_terminal_sequence_is_a_hard_source_seal_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unfinished producer sequence cannot authorize sealing its outputs."""
    sequence_dir = (
        tmp_path / "research/campaign_2026-10-06/bandit_continuation/local_wide_teacher_E521"
    )
    sequence_dir.mkdir(parents=True)
    (sequence_dir / "TRAIN_EXPORT_PAIR_SEQUENCE_V3.json").write_text("{}\n")
    receipt_path = sequence_dir / "TRAIN_EXPORT_PAIR_SEQUENCE_RECEIPT_V3.json"
    receipt_path.write_text(
        json.dumps(
            {
                "status": "finished",
                "worker_status": "finished",
                "launch_returncode": 0,
                "plan_sha256": seal_h230_source_bundle.EXPECTED_E521_PLAN_SHA256,
            }
        )
    )
    monkeypatch.setattr(
        seal_h230_source_bundle,
        "sha256_file",
        lambda path: seal_h230_source_bundle.EXPECTED_E521_PLAN_SHA256,
    )
    seal_h230_source_bundle._verify_sequence(tmp_path)

    receipt_path.write_text('{"status":"running"}\n')
    with pytest.raises(seal_h230_source_bundle.SourceSealError, match="terminal success"):
        seal_h230_source_bundle._verify_sequence(tmp_path)
