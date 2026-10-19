from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts.fit_input_representation import _project_path
from wnn import splits
from wnn.training import environment, input_representation, input_representation_fit
from wnn.training.input_representation_job import run_preparation_job


def test_job_hashes_source_root_and_accepts_external_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Keep source identity separate from data paths and allow an external split manifest."""
    data_root, source_root = tmp_path / "data", tmp_path / "source"
    data_root.mkdir()
    (tmp_path / "splits").mkdir()
    manifest = tmp_path / "splits" / "fold0.json"
    manifest.write_text("pinned fixture", encoding="utf-8")
    source_files = (
        "src/wnn/data.py",
        "src/wnn/splits.py",
        "src/wnn/training/environment.py",
        "src/wnn/training/input_representation.py",
        "src/wnn/training/input_representation_dictionary.py",
        "src/wnn/training/input_representation_fit.py",
        "src/wnn/training/input_representation_job.py",
        "scripts/fit_input_representation.py",
    )
    expected_hashes = {}
    for name in source_files:
        path = source_root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name, encoding="utf-8")
        expected_hashes[name] = splits.file_sha256(path)
    expected_split = SimpleNamespace(train_groups=(3, 7), sha256="split-sha")

    def fake_load_split(path: Path, root: Path) -> SimpleNamespace:
        assert path == manifest.resolve() and root == data_root.resolve()
        return expected_split

    monkeypatch.setattr(splits, "load_campaign_split", fake_load_split)
    monkeypatch.setattr(
        environment,
        "TrainingEnvironment",
        lambda root: SimpleNamespace(root=root, source_root=source_root),
    )
    monkeypatch.setattr(torch, "set_num_threads", lambda count: None)
    monkeypatch.setattr(torch, "set_num_interop_threads", lambda count: None)

    def fake_fit(
        groups: np.ndarray,
        split: SimpleNamespace,
        env: SimpleNamespace,
        device: torch.device,
        *,
        cache_path: Path,
        output_path: Path,
    ) -> SimpleNamespace:
        assert groups.tolist() == [3, 7]
        assert split is expected_split and env.root == data_root.resolve()
        assert str(device) == "cpu" and output_path == cache_path.parent
        cache_path.write_bytes(b"synthetic dictionary")
        return SimpleNamespace(
            fit_metadata={
                "fit_members": [{"source": "train", "group": 3, "component": 4}],
                "fit_members_sha256": "members-sha",
                "fit_row_count": 497,
                "dictionary_training": {"attempted_and_successful_updates": 12},
            }
        )

    monkeypatch.setattr(input_representation_fit, "prepare_input_representation", fake_fit)
    monkeypatch.setattr(
        input_representation, "input_representation_sha256", lambda fitted: "state-sha"
    )
    receipt = run_preparation_job(data_root, manifest, data_root / "cache" / "dict.pt")

    assert receipt["data_root"] == str(data_root.resolve())
    assert receipt["source_root"] == str(source_root.resolve())
    assert receipt["source_sha256"] == expected_hashes
    assert receipt["split_manifest_sha256"] == "split-sha"
    assert receipt["membership"]["sha256"] == "members-sha"
    assert receipt["counts"] == {"fit_sequences": 1, "fit_rows": 497, "successful_updates": 12}
    assert receipt["cache"]["bytes"] == len(b"synthetic dictionary")
    assert _project_path(data_root.resolve(), str(manifest), confined=False) == manifest.resolve()
    with pytest.raises(ValueError, match="remain under WNN_PROJECT_ROOT"):
        _project_path(data_root.resolve(), str(manifest))
