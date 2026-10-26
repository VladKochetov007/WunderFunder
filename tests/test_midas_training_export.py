"""Receiver-only checks for portable metadata export without model loading."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

PACKET = (
    Path(__file__).resolve().parents[1] / "research/experiment_drafts/"
    "b300-transformer-distillation-runtime/recovery_r5"
)
sys.path.insert(0, str(PACKET))
_spec = importlib.util.spec_from_file_location("recovery_export", PACKET / "export_midas.py")
assert _spec is not None and _spec.loader is not None
exporter = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(exporter)
runtime = importlib.import_module("recovery_runtime")


def fixture_run(root: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Provide synthetic metadata; this is not a trained model or quality evidence."""
    fit = root / "runs/cell/fit"
    fit.mkdir(parents=True)
    config = {
        "epochs": 30,
        "expected_fold": 0,
        "seed": 201001,
        "model": {},
        "precision_mode": "bf16_bf16x9",
    }
    (root / "config.json").write_text(json.dumps(config))
    split = {"fold": 0, "schema": "wnn.sequence-inner-split.v1"}
    split_raw = exporter.encoded(split)
    split_sha = exporter.hashlib.sha256(split_raw).hexdigest()
    (root / "split.json").write_bytes(split_raw)
    normalized = json.loads(json.dumps(config))
    normalized["model"]["precision_mode"] = config["precision_mode"]
    metadata = {
        "source": {"sha256": "a" * 64},
        "config": normalized,
        "provenance": {"data": {"split_sha256": split_sha, "fold": 0}},
    }
    (fit / "run_metadata.json").write_bytes(exporter.encoded(metadata))
    campaign = {"campaign_sha256": "b" * 64, "runtime_source_sha256": "a" * 64}
    cell = {
        "run_dir": "runs/cell",
        "config": "config.json",
        "split_sha256": split_sha,
        "split_manifest": str(root / "split.json"),
        "id": "cell",
        "variant": "lr_1e-4",
        "fold": 0,
        "seed": 201001,
    }
    monkeypatch.setattr(exporter, "load_recovery", lambda _: (campaign, {}))
    monkeypatch.setattr(exporter, "get_recovery_cell", lambda *_: cell)
    return fit


def test_missing_terminal_does_not_claim_live_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A metadata snapshot cannot prove that a GPU process is running or completed."""
    fixture_run(tmp_path, monkeypatch)
    path = exporter.export_bundle(tmp_path, "cell", tmp_path / "bundle")
    receipt = json.loads(path.read_text())
    assert receipt["execution"] == {"status": "partial", "started_at": None, "finished_at": None}
    assert receipt["source"]["commit"] is None
    assert receipt["metrics"] == []
    assert receipt["checkpoints"] == []


def test_source_mismatch_blocks_export(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An artifact from a different runtime cannot be relabeled as the planned cell."""
    fit = fixture_run(tmp_path, monkeypatch)
    metadata = json.loads((fit / "run_metadata.json").read_text())
    metadata["source"]["sha256"] = "c" * 64
    (fit / "run_metadata.json").write_bytes(exporter.encoded(metadata))
    with pytest.raises(ValueError, match="source/split"):
        exporter.export_bundle(tmp_path, "cell", tmp_path / "bundle")


def test_bundle_is_never_overwritten(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An existing human-transferred bundle remains byte-identical on accidental replay."""
    fixture_run(tmp_path, monkeypatch)
    path = exporter.export_bundle(tmp_path, "cell", tmp_path / "bundle")
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        exporter.export_bundle(tmp_path, "cell", tmp_path / "bundle")
    assert path.read_bytes() == before


def test_retention_intent_blocks_fresh_export(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Compaction cannot be exported as a full retained-checkpoint archive."""
    fit = fixture_run(tmp_path, monkeypatch)
    (fit / "retention_intent.json").write_text("{}")
    output = tmp_path / "bundle"

    with pytest.raises(ValueError, match="cleanup has begun"):
        exporter.export_bundle(tmp_path, "cell", output)

    assert not output.exists()


@pytest.mark.parametrize("action", ["fit", "resume"])
def test_retention_intent_blocks_fit_and_resume(
    action: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A durable compaction intent permanently closes fit and resume admission."""
    fit = fixture_run(tmp_path, monkeypatch)
    (fit / "retention_intent.json").write_text("{}")
    campaign = {"campaign_sha256": "b" * 64, "runtime_source_sha256": "a" * 64}
    cell = {"run_dir": "runs/cell"}
    monkeypatch.setattr(runtime, "load_recovery", lambda _: (campaign, {}))
    monkeypatch.setattr(runtime, "get_recovery_cell", lambda *_: cell)
    monkeypatch.setattr(runtime, "receiver_acceptance", lambda *_: tmp_path / "acceptance.json")
    monkeypatch.setattr(runtime, "storage_admission", lambda *_: None)
    monkeypatch.setattr(runtime, "execute_cell", lambda *_: pytest.fail("fit was dispatched"))
    monkeypatch.setenv("CUBLAS_EMULATE_SINGLE_PRECISION", "1")
    monkeypatch.setattr(
        sys,
        "argv",
        ["run.py", action, "--campaign-root", str(tmp_path), "--cell", "cell"],
    )

    with pytest.raises(ValueError, match="cleanup has begun"):
        runtime.main()


def _install_complete_override(
    root: Path,
    fit: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    foreign_fit: bool = False,
) -> Path:
    """Stub the full-sweep reader at its boundary while binding real fixture bytes."""
    _, config_raw = exporter.read(root / "config.json")
    metadata, _ = exporter.read(fit / "run_metadata.json")
    config = metadata["config"]
    split_path = root / "split.json"
    split, split_raw = exporter.read(split_path)
    data = metadata["provenance"]["data"]
    identity = {
        "config_sha256": exporter.canonical(config),
        "source_sha256": metadata["source"]["sha256"],
        "data": data,
    }
    unsigned_index = {
        "schema": "wnn.sequence-epoch-index.v1",
        "identity": identity,
        "completed_epochs": 0,
        "entries": [],
    }
    index = unsigned_index | {"index_sha256": exporter.canonical(unsigned_index)}
    index_raw = exporter.encoded(index)
    (fit / "epoch_index.json").write_bytes(index_raw)
    sweep_path = root / "verified-sweep"
    sweep_path.mkdir()
    foreign_dir = root / "foreign-fit"
    if foreign_fit:
        foreign_dir.mkdir()
    verified_fit = {
        "fit_dir": str(foreign_dir if foreign_fit else fit),
        "identity": identity,
        "config_file_sha256": exporter.hashlib.sha256(config_raw).hexdigest(),
        "index_file_sha256": exporter.hashlib.sha256(index_raw).hexdigest(),
        "index_sha256": index["index_sha256"],
    }
    verified = {
        "fit": verified_fit,
        "plan": {"config_path": str((root / "config.json").resolve())},
        "split_binding": {
            "inner_path": str(split_path.resolve()),
            "inner_sha256": exporter.hashlib.sha256(split_raw).hexdigest(),
            "fold": split["fold"],
        },
        "sweep": {"status": "complete"},
        "observed": [{} for _ in range(30)],
        "missing": [],
    }

    def load_fold(path: Path, fold: int) -> tuple[dict, list]:
        assert path == sweep_path.resolve()
        assert fold == 0
        return verified, []

    monkeypatch.setattr(exporter, "_load_fold", load_fold)
    return sweep_path


def test_incomplete_evaluation_override_is_rejected_before_bundle_creation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An incomplete replacement sweep cannot be exported as canonical evidence."""
    fit = fixture_run(tmp_path, monkeypatch)
    sweep = tmp_path / "partial-sweep"
    sweep.mkdir()
    monkeypatch.setattr(
        exporter,
        "_load_fold",
        lambda *_: (
            {"sweep": None, "observed": [], "missing": [{"artifact": "sweep.json"}]},
            [{"artifact": "sweep.json"}],
        ),
    )
    output = tmp_path / "bundle"

    with pytest.raises(ValueError, match="complete 30-epoch sweep"):
        exporter.export_bundle(tmp_path, "cell", output, evaluation_output=sweep)

    assert not output.exists()
    assert fit.is_dir()


def test_symlink_evaluation_override_is_rejected_before_bundle_creation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An override cannot redirect validation through a symlinked sweep root."""
    fixture_run(tmp_path, monkeypatch)
    actual = tmp_path / "actual-sweep"
    actual.mkdir()
    alias = tmp_path / "sweep-alias"
    alias.symlink_to(actual, target_is_directory=True)
    output = tmp_path / "bundle"

    with pytest.raises(ValueError, match="must not be a symlink"):
        exporter.export_bundle(tmp_path, "cell", output, evaluation_output=alias)

    assert not output.exists()


def test_foreign_evaluation_override_is_rejected_before_bundle_creation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A complete sweep for another fit cannot be attached to this cell's export."""
    fit = fixture_run(tmp_path, monkeypatch)
    sweep = _install_complete_override(tmp_path, fit, monkeypatch, foreign_fit=True)
    output = tmp_path / "bundle"

    with pytest.raises(ValueError, match="different fit, config, index, or split"):
        exporter.export_bundle(tmp_path, "cell", output, evaluation_output=sweep)

    assert not output.exists()


def test_matching_evaluation_override_exports_after_full_sweep_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A complete sweep for the declared fit can be selected explicitly for export."""
    fit = fixture_run(tmp_path, monkeypatch)
    sweep = _install_complete_override(tmp_path, fit, monkeypatch)

    receipt_path = exporter.export_bundle(
        tmp_path, "cell", tmp_path / "bundle", evaluation_output=sweep
    )

    receipt = json.loads(receipt_path.read_text())
    assert receipt["execution"]["status"] == "partial"
    assert receipt["checkpoints"] == []
