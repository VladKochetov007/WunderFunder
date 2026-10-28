from __future__ import annotations

import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

PACKET = (
    Path(__file__).resolve().parents[1]
    / "research/experiment_drafts/source-holdout-transformer-kd-20261028"
)


@pytest.fixture
def packet(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Load only stdlib packet gates; no model execution is part of these cases."""
    monkeypatch.syspath_prepend(str(PACKET))
    return SimpleNamespace(
        contract=importlib.import_module("packet_contract"),
        lineage=importlib.import_module("packet_lineage"),
        acceptance=importlib.import_module("packet_acceptance"),
    )


def test_same_seed_student_changes_only_kd_weight(packet: SimpleNamespace) -> None:
    """Prevent optimizer, data or architecture drift in the paired causal contrast."""
    for seed in (1001, 1002):
        labels = packet.contract.normalized_config(f"labels_s{seed}")
        kd = packet.contract.normalized_config(f"kd_s{seed}")
        assert labels | {"kd_weight": 1.0} == kd
        assert kd["kd_loss_normalization"] == "weighted_mean"
        assert kd["checkpoint_retention"] == "final_only"


def test_source_members_forbid_valid_fitting(packet: SimpleNamespace, tmp_path: Path) -> None:
    """A fit/evaluation overlap must fail before a trainer subprocess is built."""
    path = tmp_path / "split.json"
    split = {
        "schema": "wnn.sequence-source-holdout.v1",
        "fit": {"train": [0], "valid": []},
        "evaluation": {"valid": [0, 1]},
    }
    path.write_text(json.dumps(split))
    assert packet.contract.members(tmp_path) == ([["train", 0]], [["valid", 0], ["valid", 1]])
    split["fit"]["valid"] = [0]
    path.write_text(json.dumps(split))
    with pytest.raises(ValueError, match="train only"):
        packet.contract.members(tmp_path)


def test_cache_rejects_another_teacher_even_with_same_members(
    packet: SimpleNamespace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Full coverage alone cannot admit a foreign checkpoint into the KD arm."""
    split = {
        "schema": "wnn.sequence-source-holdout.v1",
        "fit": {"train": [0], "valid": []},
        "evaluation": {"valid": [0]},
    }
    (tmp_path / "split.json").write_text(json.dumps(split))
    directory = tmp_path / "teacher_s1001/cache"
    directory.mkdir(parents=True)
    terminal = {"model_sha256": "a" * 64, "identity": {"sentinel": "nominated"}}
    monkeypatch.setattr(packet.lineage, "validate_terminal", lambda root, cell: terminal)
    meta = {
        "schema": "sequence-teacher-cache-v1",
        "complete": True,
        "membership": "fit",
        "input_schema": "raw112-v1",
        "hint_dim": 0,
        "sequence_length": 20_000,
        "split_sha256": packet.contract.file_hash(tmp_path / "split.json"),
        "world_size": 2,
        "producer": {"checkpoint_sha256": "b" * 64},
    }
    (directory / "manifest.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="Foreign teacher weights"):
        packet.lineage.validate_cache(tmp_path, "kd_s1001")


def test_cache_rejects_foreign_seed_config(
    packet: SimpleNamespace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Even an alleged matching weight digest cannot bypass the nominated seed recipe."""
    (tmp_path / "split.json").write_text(
        json.dumps(
            {
                "schema": "wnn.sequence-source-holdout.v1",
                "fit": {"train": [0], "valid": []},
                "evaluation": {"valid": [0]},
            }
        )
    )
    directory = tmp_path / "teacher_s1001/cache"
    directory.mkdir(parents=True)
    monkeypatch.setattr(
        packet.lineage, "validate_terminal", lambda root, cell: {"model_sha256": "a" * 64}
    )
    meta = {
        "schema": "sequence-teacher-cache-v1",
        "complete": True,
        "membership": "fit",
        "input_schema": "raw112-v1",
        "hint_dim": 0,
        "sequence_length": 20_000,
        "split_sha256": packet.contract.file_hash(tmp_path / "split.json"),
        "world_size": 2,
        "producer": {
            "checkpoint_sha256": "a" * 64,
            "config": packet.contract.normalized_config("teacher_s1002"),
        },
    }
    (directory / "manifest.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="Foreign teacher recipe/seed"):
        packet.lineage.validate_cache(tmp_path, "kd_s1001")


def test_acceptance_rejects_missing_module(packet: SimpleNamespace, tmp_path: Path) -> None:
    """A green but partial pytest report must not unlock the six scientific fits."""
    directory = tmp_path / "acceptance"
    directory.mkdir()
    (directory / "pytest.xml").write_text(
        '<testsuite><testcase classname="tests.test_source_kd_packet" name="one"/></testsuite>'
    )
    with pytest.raises(ValueError, match="Missing receiver module"):
        packet.acceptance.seal_acceptance(tmp_path)


def test_evidence_is_never_overwritten(packet: SimpleNamespace, tmp_path: Path) -> None:
    """Repeated review cannot erase an earlier receipt or silently bless another run."""
    path = tmp_path / "approved_profile.json"
    packet.contract.write_new(path, {"approved": True, "identity": "original"})
    with pytest.raises(FileExistsError):
        packet.contract.write_new(path, {"approved": True, "identity": "replacement"})
    assert packet.contract.read(path)["identity"] == "original"
