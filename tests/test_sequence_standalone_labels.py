from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn

from wnn.training.sequence_complement_monitor import ComplementMonitor
from wnn.training.sequence_config import load_config
from wnn.training.sequence_distributed import DistributedContext
from wnn.training.sequence_ensemble_config import (
    STANDALONE_LABELS_MODE,
    objective_contract,
)
from wnn.training.sequence_fit_admission import validate_fit_reference_inputs
from wnn.training.sequence_monitor_dataset import evaluate_dataset
from wnn.training.sequence_monitor_events import MONITOR_EVENT_FILENAME
from wnn.training.sequence_objectives import (
    objective_numerators,
    scaled_objective,
    window_denominators,
)
from wnn.training.sequence_provenance import canonical_sha256
from wnn.training.sequence_reference_pins import PRIMARY_SPLIT_SHA256

_ROOT = Path(__file__).parents[1]
_PRESET = (
    _ROOT
    / "research/experiment_drafts/training-repair-20261108/configs/standalone_labels_wmse_v2.json"
)
_LEGACY_CONFIGS = {
    "labels_s101101.json": "d3ef8df541ec5ce9ac4f68b816e5ac61a59e7c1d0a181954f0fb6bdd4f8409ef",
    "mixture_corr_s101101.json": "2d9e589d65335e8231a7949902abac67d437f64886e4f2b75b22cf610076b879",
}


def _write_variant(tmp_path: Path, changes: dict[str, object]) -> Path:
    raw = json.loads(_PRESET.read_text(encoding="utf-8"))
    raw.update(changes)
    path = tmp_path / "recipe.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    return path


def test_new_preset_has_explicit_contract_and_flexible_generic_controls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CUBLAS_EMULATE_SINGLE_PRECISION", "1")
    config = load_config(_PRESET)
    assert config["objective_mode"] == STANDALONE_LABELS_MODE
    assert config["checkpoint_retention"] == "all_epochs"
    assert config["checkpoint_endpoint_policy"] == "configured_final_epoch"
    assert config["monitoring"] == {
        "train_eval_sequences": 128,
        "eval_batch_size": 4,
        "train_eval_selection": "hash_rank_v1",
    }
    assert objective_contract(config) == {
        "kind": "standalone_label_weighted_mse",
        "mode": "standalone_labels_wmse_v2",
        "target_clip": [-2.0, 2.0],
        "target_weight": "abs(clipped_target)",
        "aggregation": "global_ddp_weighted_wmse_per_actual_optimizer_window",
        "pearson_term": "none",
        "reference_training_ancestry": "none",
    }

    edited = _write_variant(
        tmp_path,
        {
            "epochs": 5,
            "seed": 42,
            "optimizer": "adamw",
            "batch_size": 64,
            "accumulate_chunks": 2,
            "required_world_size": None,
            "required_device": None,
            "model": {
                "architecture": "gru",
                "hidden": 128,
                "layers": 2,
                "heads": 4,
                "context_rows": 512,
                "checkpoint_blocks": False,
                "attention_backend": "math",
                "compile_blocks": False,
            },
        },
    )
    changed = load_config(edited)
    assert (changed["epochs"], changed["seed"], changed["optimizer"]) == (5, 42, "adamw")
    assert changed["required_world_size"] is None and changed["required_device"] is None
    assert changed["model"]["context_rows"] == 512


@pytest.mark.parametrize(
    "changes",
    [
        {"kd_weight": 0.1},
        {"hint_weight": 0.1, "model": {"hint_dim": 8, "hint_projection": "learned"}},
        {"model": {"hint_dim": 8}},
        {"blend_objective_alpha": 0.1},
        {"correlation_prediction_std_floor": 0.01},
        {"normalization_scope": "prefix"},
        {"training_mask": "is_scored"},
        {"monitoring": None},
    ],
)
def test_standalone_mode_rejects_nonlabel_or_nonfit_contracts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, changes: dict[str, object]
) -> None:
    monkeypatch.setenv("CUBLAS_EMULATE_SINGLE_PRECISION", "1")
    with pytest.raises(ValueError):
        load_config(_write_variant(tmp_path, changes))


def test_legacy_config_identities_match_completed_producer_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUBLAS_EMULATE_SINGLE_PRECISION", "1")
    source = _ROOT / "research/experiment_drafts/ensemble-objective-fix-20261108/configs"
    for filename, expected in _LEGACY_CONFIGS.items():
        assert canonical_sha256(load_config(source / filename)) == expected


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("reference_cache", "standalone labels reject --reference-cache"),
        ("monitor_reference_root", "standalone labels reject --monitor-reference-root"),
        ("teacher_cache", "standalone labels reject --teacher-cache"),
    ],
)
def test_new_mode_rejects_reference_flags_before_distributed_or_data_setup(
    monkeypatch: pytest.MonkeyPatch, field: str, message: str
) -> None:
    monkeypatch.setenv("CUBLAS_EMULATE_SINGLE_PRECISION", "1")
    config = load_config(_PRESET)
    with pytest.raises(ValueError, match=message):
        validate_fit_reference_inputs(
            config,
            Path("unused-reference") if field == "teacher_cache" else None,
            Path("unused-reference") if field == "reference_cache" else None,
            Path("unused-reference") if field == "monitor_reference_root" else None,
        )


class _TinyDataset:
    def __init__(self) -> None:
        self.items = []
        for _index in range(2):
            x = torch.zeros(6, 112)
            x[:, :2] = torch.tensor([[0.2, -0.1], [0.3, 0.5], [0.7, -0.8]] * 2)
            self.items.append(
                {
                    "x": x,
                    "y": torch.tensor([[0.0, 0.0], [0.4, 0.5], [0.7, -0.2]] * 2),
                    "mask": torch.tensor([False, True, True, False, True, True]),
                    "teacher": torch.zeros(6, 2),
                    "hints": torch.empty(6, 0),
                }
            )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.items[index]


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.forward_calls = 0

    def forward(
        self, x: torch.Tensor, state: object = None
    ) -> tuple[torch.Tensor, torch.Tensor, None]:
        self.forward_calls += 1
        return x[:, :, :2] * self.scale, x[:, :, :0], None


def test_raw_z_monitoring_adds_no_forward_or_reference_metrics() -> None:
    dataset = _TinyDataset()
    model = _TinyModel()
    context = DistributedContext(0, 0, 1, torch.device("cpu"), False)
    baseline = evaluate_dataset(model, dataset, "valid", 3, "fp32", context, 2)
    baseline_calls = model.forward_calls
    model.forward_calls = 0
    observed = evaluate_dataset(
        model,
        dataset,
        "valid",
        3,
        "fp32",
        context,
        2,
        blend_alpha=None,
        report_standalone=True,
    )
    assert model.forward_calls == baseline_calls == 2
    assert observed["valid/wp"] == baseline["valid/wp"] == observed["valid/z/wp"]
    assert not any("/reference/" in key or "/mixture/" in key for key in observed)


def _monitor_data(root: Path) -> SimpleNamespace:
    fit_members = [("train", index) for index in range(10_325)]
    valid_members = [("valid", index) for index in range(1_873)]
    fit = SimpleNamespace(
        members=fit_members,
        sources={"train": SimpleNamespace(environment=SimpleNamespace(root=root))},
    )
    return SimpleNamespace(
        fit=fit,
        evaluation=SimpleNamespace(members=valid_members),
        split=SimpleNamespace(sha256=PRIMARY_SPLIT_SHA256, valid_groups=()),
        provenance={
            "objective_contract": objective_contract({"objective_mode": STANDALONE_LABELS_MODE})
        },
    )


def test_monitor_constructs_without_tl3_and_keeps_raw_z_primary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import wnn.training.sequence_complement_monitor as monitor_module

    records: list[tuple[int, object, str]] = []

    class FakeRecords:
        def __init__(
            self,
            root: Path,
            members: list[tuple[str, int]],
            hint_dim: int,
            cache: object = None,
            hint_members: object = None,
            training_mask: str = "is_scored",
        ) -> None:
            records.append((len(members), cache, training_mask))

        def __len__(self) -> int:
            return 1

    observed_call: dict[str, object] = {}

    def fake_evaluate(*args: Any, **kwargs: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        observed_call.update(kwargs)
        return (
            {"train_eval/wp": 0.4, "train_eval/z/wp": 0.4, "train_eval/sequence_count": 128},
            {
                "valid/wp": 0.7,
                "valid/z/wp": 0.7,
                "valid/sequence_count": 1_873,
                "valid/selected_rows": 3_528_273,
            },
        )

    monkeypatch.setattr(monitor_module, "SequenceRecords", FakeRecords)
    monkeypatch.setattr(monitor_module, "evaluate_monitoring_sets", fake_evaluate)
    config = {
        "objective_mode": STANDALONE_LABELS_MODE,
        "epochs": 4,
        "chunk_rows": 256,
        "precision_mode": "fp32",
        "kd_weight": 0,
        "hint_weight": 0,
        "monitoring": {
            "train_eval_sequences": 128,
            "eval_batch_size": 4,
            "train_eval_selection": "hash_rank_v1",
        },
    }
    data = _monitor_data(tmp_path)
    monitor = ComplementMonitor(data, None, config, tmp_path, resume=False, primary=True)
    assert records == [(128, None, "all_after_warmup"), (1_873, None, "is_scored")]
    assert "reference_sha256" not in monitor.context
    context = DistributedContext(0, 0, 1, torch.device("cpu"), False)
    monitor.evaluate(nn.Linear(1, 1), config, context, 0, 1)
    assert observed_call["blend_alpha"] is None
    assert observed_call["report_standalone"] is True
    assert monitor.context["endpoint"] == "raw standalone z; no reference or blend"
    with (tmp_path / MONITOR_EVENT_FILENAME).open(encoding="utf-8") as stream:
        event = json.loads(stream.readline())
    assert event["metrics"]["valid/wp"] == event["metrics"]["valid/z/wp"] == 0.7

    with pytest.raises(ValueError, match="offline fixed readout"):
        ComplementMonitor(
            data, tmp_path / "unused-reference", config, tmp_path, resume=True, primary=False
        )


def test_new_label_gradient_matches_manual_weighted_window_oracle() -> None:
    target = torch.tensor([[[3.0, -0.5], [0.2, -0.8], [-1.5, 2.5]]])
    prediction = torch.tensor([[[0.2, -0.3], [0.7, 0.1], [-0.6, 0.5]]], requires_grad=True)
    mask = torch.tensor([[True, True, False]])
    empty = torch.empty(1, 3, 0)
    context = DistributedContext(0, 0, 1, torch.device("cpu"), False)
    denominators = window_denominators(target, mask, empty)
    numerators = objective_numerators(
        prediction,
        empty,
        target,
        torch.zeros_like(target),
        empty,
        mask,
        objective_mode=STANDALONE_LABELS_MODE,
        context=context,
    )
    loss = scaled_objective(numerators, denominators, 1, 0, 0)
    clipped = target[:, :2].clamp(-2, 2)
    weights = clipped.abs()
    oracle = (weights * (prediction[:, :2] - clipped).square()).sum() / weights.sum()
    actual_gradient = torch.autograd.grad(loss, prediction, retain_graph=True)[0]
    oracle_gradient = torch.autograd.grad(oracle, prediction)[0]
    torch.testing.assert_close(loss, oracle, rtol=0, atol=0)
    torch.testing.assert_close(actual_gradient, oracle_gradient, rtol=0, atol=0)
