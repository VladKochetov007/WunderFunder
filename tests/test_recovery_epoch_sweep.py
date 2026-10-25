"""Prepared receiver checks for R5 retained-epoch selection; never run on a workstation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
PACKET = (
    Path(__file__).resolve().parents[1]
    / "research/experiment_drafts/b300-transformer-distillation-runtime/recovery_r5"
)
sys.path.insert(0, str(PACKET))

from epoch_contract import validate_epoch_index, validate_split_pair  # noqa: E402
from epoch_selection import select_shared_epoch  # noqa: E402
from recovery_epoch_sweep_fixtures import (  # noqa: E402
    _make_fit,
    _make_splits,
    _make_sweep,
    _write_json,
)


def test_inner_split_is_component_bound_and_outer_manifest_is_rejected(tmp_path: Path) -> None:
    outer, inner, _ = _make_splits(tmp_path, 0)
    with pytest.raises(ValueError, match="inner split"):
        validate_split_pair(outer, outer, 0)
    malformed = json.loads(inner.read_text())
    malformed["evaluation"]["valid"] = malformed["fit"]["valid"][:1]
    malformed["fit"]["valid"] = malformed["fit"]["valid"][1:]
    _write_json(inner, malformed)
    with pytest.raises(ValueError):
        validate_split_pair(outer, inner, 0)


def test_full_epoch_index_binds_config_split_and_all_checkpoint_hashes(tmp_path: Path) -> None:
    fit = _make_fit(tmp_path, 0)
    validated = validate_epoch_index(
        fit["fit_dir"],
        fit["index_path"],
        fit["index"]["index_sha256"],
        fit["config_path"],
        fit["binding"],
    )
    assert len(validated["entries"]) == 30
    assert validated["index_sha256"] == fit["index"]["index_sha256"]


def test_full_epoch_index_accepts_matching_immutable_snapshot_and_rejects_stale_pointer(
    tmp_path: Path,
) -> None:
    fit = _make_fit(tmp_path, 0)
    snapshot = fit["fit_dir"] / "epoch_indexes/epoch-000030.json"
    validated = validate_epoch_index(
        fit["fit_dir"],
        snapshot,
        fit["index"]["index_sha256"],
        fit["config_path"],
        fit["binding"],
    )
    assert validated["index_sha256"] == fit["index"]["index_sha256"]

    (fit["fit_dir"] / "epoch_index.json").write_bytes(b"detached latest pointer\n")
    with pytest.raises(ValueError, match="latest and immutable"):
        validate_epoch_index(
            fit["fit_dir"],
            snapshot,
            fit["index"]["index_sha256"],
            fit["config_path"],
            fit["binding"],
        )


def test_epoch_index_rejects_missing_prefix_and_changed_checkpoint(tmp_path: Path) -> None:
    fit = _make_fit(tmp_path, 0)
    (fit["fit_dir"] / "epoch_indexes/epoch-000017.json").unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        validate_epoch_index(
            fit["fit_dir"],
            fit["index_path"],
            fit["index"]["index_sha256"],
            fit["config_path"],
            fit["binding"],
        )
    fit = _make_fit(tmp_path / "other", 0)
    (fit["fit_dir"] / fit["entries"][0]["checkpoint_path"]).write_bytes(b"drift")
    with pytest.raises(ValueError, match="checkpoint hash"):
        validate_epoch_index(
            fit["fit_dir"],
            fit["index_path"],
            fit["index"]["index_sha256"],
            fit["config_path"],
            fit["binding"],
        )


def test_shared_epoch_uses_equal_fold_mean_and_earliest_tie(tmp_path: Path) -> None:
    f0 = _make_fit(tmp_path / "f0", 0)
    f2 = _make_fit(tmp_path / "f2", 2)
    scores0 = [0.9, 0.2, 0.1] + [0.0] * 27
    scores2 = [0.0, 0.8, 0.9] + [0.0] * 27
    out0 = _make_sweep(tmp_path, f0, scores0)
    out2 = _make_sweep(tmp_path, f2, scores2)
    decision = select_shared_epoch(out0, out2, tmp_path / "selected.json")
    assert decision["status"] == "selected"
    assert decision["selected_completed_epochs"] == 2
    assert decision["selected_mean_inner_wp"] == pytest.approx(0.5)
    assert len(decision["curve_score_hashes"]["F0"]) == 30
    tie0 = [0.25] * 30
    tie_out0 = _make_sweep(tmp_path, _make_fit(tmp_path / "tf0", 0), tie0)
    tie_out2 = _make_sweep(tmp_path, _make_fit(tmp_path / "tf2", 2), tie0)
    tie = select_shared_epoch(tie_out0, tie_out2, tmp_path / "tie.json")
    assert tie["selected_completed_epochs"] == 1


def test_selector_publishes_incomplete_summary_without_decision(tmp_path: Path) -> None:
    f0 = _make_fit(tmp_path / "f0", 0)
    f2 = _make_fit(tmp_path / "f2", 2)
    out0 = _make_sweep(tmp_path, f0, [0.1] * 30)
    out2 = _make_sweep(tmp_path, f2, [0.1] * 30)
    (out2 / "epochs/epoch-000011/epoch.json").unlink()
    result = select_shared_epoch(out0, out2, tmp_path / "incomplete.json")
    assert result["status"] == "incomplete"
    assert result["decision"] is None
    assert any(item.get("completed_epochs") == 11 for item in result["missing"])


def test_selector_rejects_recipe_mismatch(tmp_path: Path) -> None:
    f0 = _make_fit(tmp_path / "f0", 0, lr=1e-4)
    f2 = _make_fit(tmp_path / "f2", 2, lr=3e-4)
    out0 = _make_sweep(tmp_path, f0, [0.1] * 30)
    out2 = _make_sweep(tmp_path, f2, [0.2] * 30)
    with pytest.raises(ValueError, match="recipes differ"):
        select_shared_epoch(out0, out2, tmp_path / "mismatch.json")
