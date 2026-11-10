from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from wnn.gmetric import GlobalAccumulator
from wnn.training.sequence_distributed import DistributedContext
from wnn.training.sequence_monitor_evaluation import evaluate_monitoring_sets
from wnn.training.sequence_monitor_events import MONITOR_EVENT_FILENAME, MonitorEventWriter
from wnn.training.sequence_predict import verify_prediction_reference
from wnn.training.sequence_source_capsule import verify_source_capsule


def test_capsule_rejects_changed_bytes_or_extra_python(tmp_path: Path) -> None:
    """A copied source tree passes only with its trusted original manifest and exact inventory."""
    files = {"scripts/predict_run.py": b"pass\n", "src/wnn/__init__.py": b""}
    for name, payload in files.items():
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
    manifest = tmp_path / "capsule.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "wnn.git-source-capsule.v1",
                "commit": "a" * 40,
                "files": {
                    name: hashlib.sha256(payload).hexdigest() for name, payload in files.items()
                },
            }
        )
    )
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert verify_source_capsule(tmp_path, manifest, digest, "a" * 40) == "a" * 40
    extra = tmp_path / "src/wnn/extra.py"
    extra.write_bytes(b"pass\n")
    with pytest.raises(ValueError, match="additional or missing"):
        verify_source_capsule(tmp_path, manifest, digest, "a" * 40)
    extra.unlink()
    (tmp_path / "scripts/predict_run.py").write_bytes(b"changed\n")
    with pytest.raises(ValueError, match="file changed"):
        verify_source_capsule(tmp_path, manifest, digest, "a" * 40)
    with pytest.raises(ValueError, match="digest or size"):
        verify_source_capsule(tmp_path, manifest, "0" * 64, "a" * 40)


def test_reference_export_integrity_success_is_not_none() -> None:
    """Both primary and nonprimary ranks return a coordination sentinel after composed export."""
    calls = []
    reference = SimpleNamespace(verify_unchanged=lambda: calls.append(True))
    assert verify_prediction_reference(reference, 1) is True
    assert calls == []
    assert verify_prediction_reference(reference, 0) is True
    assert calls == [True]


class _TinyDataset:
    def __init__(self) -> None:
        x = torch.zeros(2, 8, 112)
        x[:, :, 0] = torch.arange(16).reshape(2, 8) / 5
        x[:, :, 1] = torch.arange(16).reshape(2, 8).sin()
        self.items = [
            {
                "x": row,
                "y": row[:, :2] * 0.7 + 0.2,
                "mask": torch.arange(8) >= 2,
                "teacher": row[:, :2] * -0.4,
                "hints": torch.empty(8, 0),
            }
            for row in x
        ]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.items[index]


class _TinyModel(nn.Module):
    def forward(
        self,
        x: torch.Tensor,
        state: object = None,
    ) -> tuple[torch.Tensor, torch.Tensor, None]:
        torch.rand(1)
        random.random()
        np.random.random()  # noqa: NPY002 - deliberately exercise legacy RNG restoration
        return x[:, :, :2], x[:, :, :0], None


def test_report_only_monitor_composes_before_clipping_and_restores_rng() -> None:
    """Per-epoch metrics pool the complete composed cohort without perturbing fitting RNG."""
    dataset = _TinyDataset()
    model = _TinyModel().train()
    before_torch = torch.get_rng_state().clone()
    before_python = random.getstate()
    before_numpy = np.random.get_state()  # noqa: NPY002 - compare the restored stream
    context = DistributedContext(0, 0, 1, torch.device("cpu"), False)
    train, valid = evaluate_monitoring_sets(
        model,
        dataset,
        dataset,
        3,
        "fp32",
        context,
        2,
        blend_alpha=0.1,
    )
    accumulator = GlobalAccumulator()
    for row in dataset.items:
        predicted = 0.9 * row["teacher"] + 0.1 * row["x"][:, :2]
        accumulator.add(row["y"].numpy(), predicted.numpy(), row["mask"].numpy())
    assert valid["valid/wp"] == pytest.approx(accumulator.result()["wp"], abs=1e-14)
    assert valid["valid/sequence_count"] == train["train_eval/sequence_count"] == 2
    assert model.training is True
    assert torch.equal(before_torch, torch.get_rng_state())
    assert before_python == random.getstate()
    after_numpy = np.random.get_state()  # noqa: NPY002 - compare captured global RNG
    assert before_numpy[0] == after_numpy[0]
    assert np.array_equal(before_numpy[1], after_numpy[1])
    assert before_numpy[2:] == after_numpy[2:]


def test_monitor_resume_preserves_orphans_and_replays_safe_epoch(tmp_path: Path) -> None:
    """Report logs resume at a saved boundary while retaining later diagnostic evidence."""
    path = tmp_path / MONITOR_EVENT_FILENAME
    writer = MonitorEventWriter(path, resume=False)
    writer.append("evaluation", 2, 1, {"valid/wp": 0.5}, {"report_only": True})
    offsets = writer.offsets()
    writer.append("evaluation", 4, 2, {"valid/wp": 0.6}, {"report_only": True})
    resumed = MonitorEventWriter(path, resume=True)
    resumed.restore_offsets(offsets, 2)
    resumed.append("evaluation", 4, 2, {"valid/wp": 0.6}, {"report_only": True})
    assert len((tmp_path / "monitor_evaluations.jsonl").read_text().splitlines()) == 2
    assert list(tmp_path.glob("*.replay_tail-*"))
