from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from scripts import predict_run  # noqa: E402


def test_campaign_export_requires_checkpoint_split_and_sorted_membership(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = tmp_path / "fold0.json"
    manifest.write_text("{}", encoding="utf-8")
    split = SimpleNamespace(sha256="frozen-split", evaluation_groups=(2, 5, 8))
    monkeypatch.setattr(predict_run, "load_campaign_split", lambda path, root: split)

    assert predict_run.load_campaign_evaluation_groups(
        manifest, {"args": {"split_manifest_sha256": "frozen-split"}}
    ) == [2, 5, 8]
    with pytest.raises(ValueError, match="differs from the checkpoint"):
        predict_run.load_campaign_evaluation_groups(
            manifest, {"args": {"split_manifest_sha256": "other-split"}}
        )

    split.evaluation_groups = (5, 2, 8)
    with pytest.raises(ValueError, match="unique and sorted"):
        predict_run.load_campaign_evaluation_groups(
            manifest, {"args": {"split_manifest_sha256": "frozen-split"}}
        )


def test_feature_only_dataset_never_requests_targets_or_mask(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[int, bool, bool]] = []

    class FakeSequenceFile:
        def __init__(self, path: Path) -> None:
            assert path.name == "valid.parquet"

        def read(self, group: int, *, targets: bool, mask: bool):
            calls.append((group, targets, mask))
            return SimpleNamespace(features=np.zeros((2, 3), dtype=np.float32))

    monkeypatch.setattr(predict_run, "SequenceFile", FakeSequenceFile)
    sample = predict_run.FeatureOnlySequences("valid", [17])[0]

    assert sample.shape == (2, 3)
    assert calls == [(17, False, False)]
