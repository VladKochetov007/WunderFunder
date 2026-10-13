from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import predict_run

from wnn.splits import CampaignSplit


@pytest.fixture
def selector(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, dict]:
    manifest = CampaignSplit(tmp_path / "fold.json", "bound", 0, 4, (5, 2), (1, 3), (0, 4), ())
    monkeypatch.setattr(predict_run, "load_campaign_split", lambda path, root: manifest)
    monkeypatch.setattr(predict_run, "ROOT", tmp_path)
    (tmp_path / "cache").mkdir()
    np.savez(tmp_path / "cache/valid_targets.npz", row_group=np.array([4, 3, 2, 1, 0]))
    return manifest.path, {"args": {"split_manifest_sha256": "bound"}}


def test_fit_and_evaluation_membership_stay_separate(selector: tuple[Path, dict]) -> None:
    path, checkpoint = selector
    np.testing.assert_array_equal(
        predict_run.campaign_prediction_groups(path, checkpoint, "train"), [5, 2]
    )
    np.testing.assert_array_equal(
        predict_run.campaign_prediction_groups(path, checkpoint, "valid"), [1, 3]
    )
    np.testing.assert_array_equal(
        predict_run.campaign_prediction_groups(path, checkpoint, "valid", evaluation_only=True),
        [0, 4],
    )


def test_full_valid_keeps_actual_target_order(selector: tuple[Path, dict]) -> None:
    path, checkpoint = selector
    np.testing.assert_array_equal(
        predict_run.campaign_prediction_groups(path, checkpoint, "valid", full_valid=True),
        [4, 3, 2, 1, 0],
    )


def test_other_fold_checkpoint_is_rejected(selector: tuple[Path, dict]) -> None:
    path, checkpoint = selector
    checkpoint["args"]["split_manifest_sha256"] = "other-fold"
    with pytest.raises(ValueError, match="differs"):
        predict_run.campaign_prediction_groups(path, checkpoint, "valid")


@pytest.mark.parametrize(
    "options",
    [
        {"evaluation_only": True, "full_valid": True},
        {"split": "train", "evaluation_only": True},
        {"split": "train", "full_valid": True},
    ],
)
def test_incompatible_selectors_are_rejected(
    selector: tuple[Path, dict],
    options: dict,
) -> None:
    path, checkpoint = selector
    with pytest.raises(ValueError):
        predict_run.campaign_prediction_groups(path, checkpoint, **{"split": "valid", **options})
