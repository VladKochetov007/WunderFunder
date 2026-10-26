"""Prepared receiver checks for R5 checkpoint compaction; never run on a workstation."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
PACKET = (
    Path(__file__).resolve().parents[1]
    / "research/experiment_drafts/b300-transformer-distillation-runtime/recovery_r5"
)
sys.path.insert(0, str(PACKET))

from epoch_contract import file_sha256  # noqa: E402
from epoch_selection import select_shared_epoch  # noqa: E402
from recovery_epoch_sweep_fixtures import _make_fit, _make_sweep  # noqa: E402
from retention_apply import apply_retention_plan, verify_retention  # noqa: E402
from retention_plan import create_retention_plan  # noqa: E402
from retention_receipts import verify_pair_retention  # noqa: E402


def _complete_pair(root: Path) -> tuple[Path, Path, Path, Path, Path]:
    """Build complete synthetic fit/sweep outputs and one immutable paired decision."""
    f0, f2 = _make_fit(root / "f0", 0), _make_fit(root / "f2", 2)
    for fit in (f0, f2):
        (fit["fit_dir"] / "training_state.pt").write_bytes(b"resume-state")
    out0 = _make_sweep(root, f0, [0.9] + [0.0] * 29)
    out2 = _make_sweep(root, f2, [0.7] + [0.0] * 29)
    selection_path = root / "selection.json"
    select_shared_epoch(out0, out2, selection_path)
    return f0["fit_dir"], f2["fit_dir"], out0, out2, selection_path


def _plan(root: Path) -> tuple[dict, str, Path, Path, Path, Path]:
    f0, f2, out0, out2, selection = _complete_pair(root)
    plan_path = root / "retention-plan.json"
    plan, raw_sha256 = create_retention_plan(out0, out2, selection, plan_path)
    return plan, raw_sha256, plan_path, f0, f2, selection


def _assert_all_checkpoints_exist(plan: dict) -> None:
    for fold in plan["folds"]:
        assert all(
            (Path(fold["fit_dir"]) / entry["checkpoint_path"]).is_file()
            for entry in fold["index_entries"]
        )


def test_plan_binds_exact_selected_snapshots_and_only_lists_other_epochs(tmp_path: Path) -> None:
    plan, raw_hash, _, f0, f2, _ = _plan(tmp_path)
    assert plan["selected_completed_epochs"] == 1
    assert len(plan["implementation_sha256"]) >= 5
    _assert_all_checkpoints_exist(plan)
    for fold in plan["folds"]:
        assert len(fold["remove"]) == 29
        assert fold["selected_checkpoint"]["path"] == fold["index_entries"][0]["checkpoint_path"]
        assert {item["path"] for item in fold["remove"]} == {
            item["checkpoint_path"] for item in fold["index_entries"][1:]
        }
        assert not (Path(fold["fit_dir"]) / "retention_intent.json").exists()
    assert raw_hash == file_sha256(tmp_path / "retention-plan.json")
    assert f0.is_dir() and f2.is_dir()


def test_apply_keeps_selected_and_fixed_final_artifacts_and_is_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, raw_hash, plan_path, f0, f2, _ = _plan(tmp_path)
    monkeypatch.setattr("retention_apply.platform.system", lambda: "Linux")
    result = apply_retention_plan(plan_path, raw_hash)
    assert result["status"] == "completed"
    for fold, fit_dir in zip(plan["folds"], (f0, f2), strict=True):
        observed = sorted(
            path.relative_to(fit_dir).as_posix()
            for path in (fit_dir / "epoch_checkpoints").iterdir()
        )
        assert observed == [fold["selected_checkpoint"]["path"]]
        for retained in (
            "model.pt",
            "training_state.pt",
            "epoch_index.json",
            "terminal_status.json",
        ):
            assert (fit_dir / retained).is_file()
        assert (fit_dir / "retention_intent.json").is_file()
        assert (fit_dir / "retention_complete.json").is_file()
        assert verify_retention(fit_dir)["status"] == "completed"
    assert verify_pair_retention(plan_path, raw_hash)["status"] == "verified"
    assert apply_retention_plan(plan_path, raw_hash)["status"] == "completed"


def test_wrong_hash_and_changed_score_fail_before_any_intent_or_deletion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, raw_hash, plan_path, _, _, _ = _plan(tmp_path)
    monkeypatch.setattr("retention_apply.platform.system", lambda: "Linux")
    with pytest.raises(ValueError, match="hash"):
        apply_retention_plan(plan_path, "0" * 64)
    score = Path(plan["folds"][0]["sweep_dir"]) / "epochs/epoch-000001/score.json"
    score.write_bytes(score.read_bytes() + b" ")
    with pytest.raises(ValueError):
        apply_retention_plan(plan_path, raw_hash)
    _assert_all_checkpoints_exist(plan)
    for fold in plan["folds"]:
        assert not (Path(fold["fit_dir"]) / "retention_intent.json").exists()


def test_crash_after_first_unlink_resumes_only_with_both_durable_intents(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, raw_hash, plan_path, _, _, _ = _plan(tmp_path)
    monkeypatch.setattr("retention_apply.platform.system", lambda: "Linux")
    original_unlink = Path.unlink
    state = {"removed": False}

    def unlink_once(path: Path, *args: object, **kwargs: object) -> None:
        if path.parent.name == "epoch_checkpoints" and not state["removed"]:
            state["removed"] = True
            original_unlink(path, *args, **kwargs)
            raise OSError("simulated process interruption after durable unlink")
        original_unlink(path, *args, **kwargs)

    with monkeypatch.context() as patcher:
        patcher.setattr(Path, "unlink", unlink_once)
        with pytest.raises(OSError, match="simulated process interruption"):
            apply_retention_plan(plan_path, raw_hash)
    assert state["removed"]
    assert all(
        (Path(fold["fit_dir"]) / "retention_intent.json").is_file() for fold in plan["folds"]
    )
    assert all(
        not (Path(fold["fit_dir"]) / "retention_complete.json").exists() for fold in plan["folds"]
    )
    assert apply_retention_plan(plan_path, raw_hash)["status"] == "completed"
    assert verify_pair_retention(plan_path, raw_hash)["status"] == "verified"


def test_stale_atomic_receipt_temp_is_recovered_after_pair_intent_is_persisted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, raw_hash, plan_path, f0, _, _ = _plan(tmp_path)
    stale = f0 / ".retention_intent.json.abcd1234.partial"
    stale.write_bytes(b"interrupted temporary receipt")
    monkeypatch.setattr("retention_apply.platform.system", lambda: "Linux")
    apply_retention_plan(plan_path, raw_hash)
    assert not stale.exists()
    assert (f0 / "retention_complete.json").is_file()


def test_pair_verification_rejects_a_reintroduced_pruned_epoch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, raw_hash, plan_path, _, _, _ = _plan(tmp_path)
    monkeypatch.setattr("retention_apply.platform.system", lambda: "Linux")
    removed = plan["folds"][0]["remove"][0]
    restored_path = Path(plan["folds"][0]["fit_dir"]) / removed["path"]
    original_bytes = restored_path.read_bytes()
    apply_retention_plan(plan_path, raw_hash)
    restored_path.write_bytes(original_bytes)
    with pytest.raises(ValueError, match="unexpectedly exists"):
        verify_pair_retention(plan_path, raw_hash)
