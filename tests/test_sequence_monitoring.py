from __future__ import annotations

import json
import random
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from wnn.data import SEQUENCE_LENGTH, WARMUP
from wnn.galam import connectome_112_to_2_config
from wnn.gmetric import GlobalAccumulator, global_wp
from wnn.splits import CHRONOLOGICAL_PRIMARY_SPLIT_SCHEMA, CampaignSplit, file_sha256
from wnn.training import sequence_checkpoint, sequence_profile, sequence_score
from wnn.training.sequence_config import load_config
from wnn.training.sequence_distributed import DistributedContext
from wnn.training.sequence_epoch_prediction import EPOCH_KIND, digest
from wnn.training.sequence_loop import EpochResult
from wnn.training.sequence_models import build_sequence_model
from wnn.training.sequence_monitor import (
    MonitorEventWriter,
    SequenceMonitor,
    _gpu_identifier,
    resource_metrics,
)
from wnn.training.sequence_monitor_diagnostics import DiagnosticProbe
from wnn.training.sequence_monitor_evaluation import evaluate_dataset, evaluate_monitoring_sets
from wnn.training.sequence_provenance import canonical_sha256


def _settings() -> dict[str, object]:
    config = replace(
        connectome_112_to_2_config(),
        width=8,
        attention_heads=2,
        attention_history=4,
        conv_kernel_size=3,
        conv_hidden_width=8,
        relative_bias_width=4,
        feed_forward_width=16,
    )
    return {"architecture": "galam", "galam": asdict(config)}


def _json_default(value: object) -> object:
    if isinstance(value, frozenset):
        return sorted(value)
    raise TypeError(type(value).__name__)


class _SequenceDataset:
    def __init__(self, groups: list[int], rows: int = 8) -> None:
        self.records = []
        for group in groups:
            index = np.arange(rows, dtype=np.float32)
            x = np.zeros((rows, 112), dtype=np.float32)
            x[:, 0] = np.sin(index * 0.4 + group * 0.2)
            x[:, 1] = np.cos(index * 0.3 - group * 0.1)
            y = np.stack(
                (np.sin(index * 0.7 + group), np.cos(index * 0.5 - group)), axis=-1
            ).astype(np.float32)
            mask = np.ones(rows, dtype=bool)
            mask[:2] = False
            mask[-1] = False
            self.records.append(
                {
                    "x": torch.from_numpy(x),
                    "y": torch.from_numpy(y),
                    "mask": torch.from_numpy(mask),
                    "teacher": torch.zeros(rows, 2),
                    "hints": torch.zeros(rows, 0),
                }
            )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.records[index]


@pytest.mark.parametrize("scope", ["train_eval", "valid"])
def test_monitor_rejects_nonofficial_reporting_masks(tmp_path: Path, scope: str) -> None:
    """Do not label all-after-warmup evaluation as the official scored-row endpoint."""
    train_eval = _SequenceDataset([9])
    valid = _SequenceDataset([12])
    selected = train_eval if scope == "train_eval" else valid
    selected.training_mask = "all_after_warmup"
    monitor = SequenceMonitor(
        tmp_path,
        {"diagnostics_every_steps": 2},
        [("train", 9)],
        [("valid", 12)],
        "c" * 64,
        "d" * 64,
        resume=False,
    )
    with pytest.raises(ValueError, match="official is_scored"):
        monitor.evaluate_epoch(
            _RandomBatchNormModel(),
            train_eval,
            valid,
            3,
            2,
            "fp32",
            DistributedContext(0, 0, 1, torch.device("cpu"), False),
            0,
            1,
        )
    assert (tmp_path / "monitor_evaluations.jsonl").stat().st_size == 0


class _RandomBatchNormModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.norm = nn.BatchNorm1d(2)

    def forward(
        self, values: torch.Tensor, state: object = None
    ) -> tuple[torch.Tensor, torch.Tensor, object]:
        torch.rand((), device=values.device)
        raw = values[..., :2].reshape(-1, 2)
        prediction = self.norm(raw).reshape(*values.shape[:2], 2)
        return prediction, prediction.new_empty((*prediction.shape[:2], 0)), state


class _GRUProbeModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gru = nn.GRU(112, 4, batch_first=True)
        self.head = nn.Linear(4, 2)

    def forward(
        self, values: torch.Tensor, state: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden, next_state = self.gru(values, state)
        return self.head(hidden), values.new_empty((*values.shape[:2], 0)), next_state


class _StatefulMonitorModel(nn.Module):
    def forward(
        self, values: torch.Tensor, state: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = state if state is not None else values.new_zeros(values.shape[0], 2)
        outputs = []
        for index in range(values.shape[1]):
            hidden = 0.25 * hidden + values[:, index, :2]
            outputs.append(hidden)
        prediction = torch.stack(outputs, dim=1)
        return prediction, values.new_empty((*prediction.shape[:2], 0)), hidden


def _numpy_rng_equal(left: tuple[object, ...], right: tuple[object, ...]) -> bool:
    return left[0] == right[0] and np.array_equal(left[1], right[1]) and left[2:] == right[2:]


@pytest.fixture(autouse=True)
def _single_threaded_torch() -> object:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _expected_eval(
    dataset: _SequenceDataset, model: _RandomBatchNormModel
) -> dict[str, float | int]:
    targets, predictions, masks = [], [], []
    for record in dataset.records:
        x = record["x"].numpy()[None, :, :2]
        prediction = (x - model.norm.running_mean.numpy()) / np.sqrt(
            model.norm.running_var.numpy() + model.norm.eps
        )
        prediction = (
            prediction * model.norm.weight.detach().numpy() + model.norm.bias.detach().numpy()
        )
        targets.append(record["y"].numpy()[None])
        predictions.append(prediction.astype(np.float32))
        masks.append(record["mask"].numpy()[None])
    y, p, mask = np.concatenate(targets), np.concatenate(predictions), np.concatenate(masks)
    clipped = np.clip(y, -2, 2)
    weight = np.abs(clipped) * mask[..., None]
    loss = float((weight * np.square(p - clipped)).sum() / weight.sum())
    return {
        "loss": loss,
        "wp": global_wp(y, p, mask)["wp"],
        "selected_rows": int(mask.sum()),
        "sequence_count": len(dataset),
    }


def test_monitor_evaluation_uses_full_masks_pooled_wp_and_restores_state() -> None:
    model = _RandomBatchNormModel().train()
    train_eval, valid = _SequenceDataset([2]), _SequenceDataset([11, 17])
    batch_norm = model.norm
    before_buffers = [
        batch_norm.running_mean.clone(),
        batch_norm.running_var.clone(),
        batch_norm.num_batches_tracked.clone(),
    ]
    random.seed(29)
    np.random.seed(31)  # noqa: NPY002 - exercise preservation of the process-global RNG
    torch.manual_seed(37)
    python_rng = random.getstate()
    numpy_rng = np.random.get_state()  # noqa: NPY002 - compare process-global RNG restoration
    torch_rng = torch.get_rng_state().clone()
    context = DistributedContext(0, 0, 1, torch.device("cpu"), False)

    train_metrics, valid_metrics = evaluate_monitoring_sets(
        model, train_eval, valid, 3, "fp32", context
    )

    train_expected = _expected_eval(train_eval, model)
    valid_expected = _expected_eval(valid, model)
    assert train_metrics["train_eval/loss"] == pytest.approx(train_expected["loss"])
    assert train_metrics["train_eval/wp"] == pytest.approx(train_expected["wp"])
    assert valid_metrics["valid/loss"] == pytest.approx(valid_expected["loss"])
    assert valid_metrics["valid/wp"] == pytest.approx(valid_expected["wp"])
    assert valid_metrics["valid/selected_rows"] == valid_expected["selected_rows"]
    assert valid_metrics["valid/sequence_count"] == 2
    assert model.training and batch_norm.training
    assert torch.equal(torch_rng, torch.get_rng_state())
    assert python_rng == random.getstate()
    assert _numpy_rng_equal(
        numpy_rng,
        np.random.get_state(),  # noqa: NPY002 - compare process-global RNG restoration
    )
    assert all(
        torch.equal(before, after)
        for before, after in zip(
            before_buffers,
            (
                batch_norm.running_mean,
                batch_norm.running_var,
                batch_norm.num_batches_tracked,
            ),
            strict=True,
        )
    )


def test_batched_monitor_evaluation_matches_one_sequence_batches() -> None:
    model = _StatefulMonitorModel().eval()
    dataset = _SequenceDataset([4, 9, 12])
    context = DistributedContext(0, 0, 1, torch.device("cpu"), False)
    one_at_a_time = evaluate_dataset(model, dataset, "valid", 3, "fp32", context)
    batched = evaluate_dataset(model, dataset, "valid", 3, "fp32", context, 2)
    assert one_at_a_time == pytest.approx(batched)


def test_zero_target_weight_uses_zero_loss_and_official_zero_wp() -> None:
    dataset = _SequenceDataset([5])
    dataset.records[0]["y"].zero_()
    model = _RandomBatchNormModel().eval()
    context = DistributedContext(0, 0, 1, torch.device("cpu"), False)
    metrics = evaluate_dataset(model, dataset, "valid", 4, "fp32", context)
    assert metrics["valid/loss"] == 0.0
    assert metrics["valid/wp"] == 0.0


def test_cuda_sampler_uses_only_the_physical_property_uuid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = torch.device("cuda:1")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda selected: SimpleNamespace(uuid="12345678-abcd-4321-9876-1234567890ab"),
    )
    assert _gpu_identifier(device) == "GPU-12345678-abcd-4321-9876-1234567890ab"


def test_event_writer_schema_resume_and_missing_resource_metrics(tmp_path: Path) -> None:
    events = tmp_path / "monitor_events.jsonl"
    monitor = SequenceMonitor(
        tmp_path,
        {"diagnostics_every_steps": 2},
        [("train", 9), ("train", 3)],
        [("valid", 12)],
        "c" * 64,
        "d" * 64,
        resume=False,
    )
    monitor.status("running", 0, 0, monitor.fit_status_context(False))
    monitor.train_step(
        {
            "step": 1,
            "total_loss": 0.5,
            "label_loss": 0.4,
            "kd_loss": 0.1,
            "hint_loss": 0.0,
            "gradient_norm_preclip": 2.0,
            "gradient_clip_threshold": 1.0,
            "gradient_clipped": 1,
            "window_seconds": 0.25,
            "optimizer_seconds": 0.1,
            "valid_rows_per_second": 12.0,
            "forwarded_rows_per_second": 20.0,
            "valid_rows": 3,
            "forwarded_rows": 5,
            "real_rows": 5,
            "learning_rates": {"adamw": 0.001},
        },
        0,
        0.5,
        torch.device("cpu"),
    )
    monitor.resources(1, 0, 0.5, torch.device("cpu"))
    monitor.diagnostics(1, 0, 0.5, {"local/activation_rms/0": 0.5})
    monitor.evaluate_epoch(
        _RandomBatchNormModel().eval(),
        _SequenceDataset([9, 3]),
        _SequenceDataset([12]),
        3,
        2,
        "fp32",
        DistributedContext(0, 0, 1, torch.device("cpu"), False),
        0,
        1,
    )
    first_bytes = events.read_bytes()
    resumed = MonitorEventWriter(events, resume=True)
    resumed.restore_offsets(resumed.offsets(), global_step=1)
    resumed.append("status", 1, 1, {}, {"status": "running"})
    rows = [json.loads(line) for line in events.read_text().splitlines()]
    assert events.read_bytes().startswith(first_bytes)
    assert rows[0]["schema"] == "wnn.monitor.v1"
    assert set(rows[0]) == {"schema", "kind", "step", "epoch", "wall_time", "metrics", "context"}
    assert rows[0]["wall_time"].endswith("Z")
    train_eval = next(row for row in rows if row["kind"] == "status")
    assert train_eval["context"]["train_eval_members"] == [["train", 9], ["train", 3]]
    train_step = next(row for row in rows if row["kind"] == "train_step")
    assert train_step["epoch"] == 0.5
    assert train_step["context"]["epoch_index"] == 1
    evaluations = [
        json.loads(line)
        for line in (tmp_path / "monitor_evaluations.jsonl").read_text().splitlines()
    ]
    assert len(evaluations) == 1
    evaluation = evaluations[0]
    assert evaluation["kind"] == "evaluation"
    assert evaluation["metrics"]["train_eval/score"] == evaluation["metrics"]["train_eval/wp"]
    assert evaluation["metrics"]["train_eval/score/t0"] == evaluation["metrics"]["train_eval/wp/t0"]
    assert evaluation["metrics"]["train_eval/score/t1"] == evaluation["metrics"]["train_eval/wp/t1"]
    assert evaluation["metrics"]["valid/score"] == evaluation["metrics"]["valid/wp"]
    assert evaluation["metrics"]["valid/score/t0"] == evaluation["metrics"]["valid/wp/t0"]
    assert evaluation["metrics"]["valid/score/t1"] == evaluation["metrics"]["valid/wp/t1"]
    assert evaluation["context"]["train_eval"]["member_count"] == 2
    assert evaluation["context"]["valid"]["coverage_complete"] is True
    # Replay ignores wall time and evaluation duration, retaining the original report line.
    before_replay = events.read_bytes()
    replay_metrics = evaluation["metrics"].copy()
    replay_metrics["timing/evaluation_seconds"] += 100
    resumed.append("evaluation", 1, 1, replay_metrics, evaluation["context"])
    assert events.read_bytes() == before_replay
    assert len((tmp_path / "monitor_evaluations.jsonl").read_text().splitlines()) == 1
    conflicting_metrics = replay_metrics | {"valid/score": replay_metrics["valid/score"] + 0.1}
    with pytest.raises(ValueError, match="conflicts"):
        resumed.append("evaluation", 1, 1, conflicting_metrics, evaluation["context"])
    cpu_metrics = resource_metrics(torch.device("cpu"))
    assert "cpu/process_time_seconds" in cpu_metrics
    assert not any(name.startswith("gpu/") for name in cpu_metrics)
    with pytest.raises(ValueError, match="finite"):
        resumed.append("resources", 1, 1, {"cpu/value": float("nan")}, {})


def test_monitor_resume_rejects_partial_jsonl_tail(tmp_path: Path) -> None:
    events = tmp_path / "monitor_events.jsonl"
    writer = MonitorEventWriter(events, resume=False)
    writer.append("status", 0, 0, {}, {"status": "running"})
    safe_offsets = writer.offsets()
    with (tmp_path / "monitor_evaluations.jsonl").open("ab") as stream:
        stream.write(b'{"schema":"wnn.monitor.v1"')
    resumed = MonitorEventWriter(events, resume=True)
    with pytest.raises(RuntimeError, match="restore monitor offsets"):
        resumed.append("status", 1, 1, {}, {"status": "running"})
    resumed.restore_offsets(safe_offsets, global_step=0)
    assert resumed.evaluation_path.read_bytes() == b""


def test_monitor_safe_offsets_preserve_orphans_and_replay_without_duplicates(
    tmp_path: Path,
) -> None:
    events = tmp_path / "monitor_events.jsonl"
    writer = MonitorEventWriter(events, resume=False)
    writer.append("status", 0, 0, {}, {"status": "running"})
    safe_offsets = writer.offsets()
    writer.append("train_step", 1, 1, {"train/loss": 0.4}, {})
    metrics = {"valid/loss": 0.2, "valid/score": 0.5}
    context = {"valid": {"member_sha256": "a" * 64}}
    writer.append("evaluation", 1, 1, metrics, context)
    orphan_event_bytes = events.read_bytes()[safe_offsets["events"] :]
    orphan_evaluation_bytes = writer.evaluation_path.read_bytes()

    writer.restore_offsets(safe_offsets, global_step=0)
    assert [json.loads(line)["kind"] for line in events.read_text().splitlines()] == ["status"]
    assert writer.evaluation_path.read_bytes() == b""
    evidence = list(tmp_path.glob("*.replay_tail-*.jsonl"))
    assert len(evidence) == 2
    evidence_bytes = b"".join(path.read_bytes() for path in evidence)
    assert orphan_event_bytes in evidence_bytes
    assert orphan_evaluation_bytes in evidence_bytes

    writer.append("train_step", 1, 1, {"train/loss": 0.4}, {})
    writer.append("evaluation", 1, 1, metrics, context)
    assert [json.loads(line)["kind"] for line in events.read_text().splitlines()] == [
        "status",
        "train_step",
        "evaluation",
    ]
    writer.append("evaluation", 1, 1, metrics, context)
    assert len(writer.evaluation_path.read_text().splitlines()) == 1


def test_monitored_resume_checkpoint_requires_and_loads_monitor_offsets(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    identity: dict[str, object] = {"data": {"monitoring_plan": {"event_files": []}}}
    state: dict[str, object] = {
        "checkpoint_schema": "wnn.sequence-resume.v1",
        "contract": identity,
        "world_size": 1,
        "next_epoch": 0,
        "global_step": 0,
        "metrics_offset": 0,
        "model": {},
        "optimizer": {},
        "scheduler": {},
        "history": [],
        "rank_rng_states": [{"python": None, "numpy": None, "torch_cpu": None, "torch_cuda": None}],
        "rank_log_offsets": [0],
        "monitor_offsets": {"events": 21, "evaluations": 0},
    }
    monkeypatch.setattr(sequence_checkpoint.torch, "load", lambda *args, **kwargs: state)
    monkeypatch.setattr(sequence_checkpoint, "all_ranks_true", lambda value, context: value)
    context = DistributedContext(0, 0, 1, torch.device("cpu"), False)
    loaded = sequence_checkpoint.load_training_state(tmp_path / "state.pt", identity, context)
    assert loaded["monitor_offsets"] == {"events": 21, "evaluations": 0}

    state.pop("monitor_offsets")
    with pytest.raises(RuntimeError, match="safe monitor log offsets"):
        sequence_checkpoint.load_training_state(tmp_path / "state.pt", identity, context)


def test_profile_routes_monitor_telemetry_without_validation_scoring(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class _ProfileMonitor:
        def __init__(self) -> None:
            self.statuses: list[tuple[str, dict[str, object]]] = []
            self.train_steps = 0
            self.diagnostic_calls = 0
            self.resource_calls = 0
            self.evaluations = 0

        def profile_status_context(self) -> dict[str, object]:
            return {
                "no_eval_during_fit": True,
                "validation_reporting": "not evaluated during bounded engineering profile",
            }

        def status(self, state: str, step: int, epoch: int, context: dict[str, object]) -> None:
            self.statuses.append((state, context))

        def train_step(self, *_args: object) -> None:
            self.train_steps += 1

        def diagnostics(self, *_args: object) -> None:
            self.diagnostic_calls += 1

        def resources(self, *_args: object) -> None:
            self.resource_calls += 1

        def evaluate_epoch(self, *_args: object) -> None:
            self.evaluations += 1

    monitor = _ProfileMonitor()
    trainer_calls: list[dict[str, object]] = []
    totals = torch.tensor([1, 0, 0, 1, 1, 1, 2, 0], dtype=torch.float64)

    def fake_train_epoch(
        dataset: object,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LambdaLR,
        config: dict[str, object],
        context: DistributedContext,
        rank_log: object,
        epoch: int,
        global_step: int,
        max_updates: int | None = None,
        monitor: _ProfileMonitor | None = None,
        monitor_enabled: bool = False,
    ) -> EpochResult:
        trainer_calls.append({"monitor": monitor, "monitor_enabled": monitor_enabled})
        assert monitor is not None and monitor_enabled
        record = {
            "step": 1,
            "total_loss": 0.5,
            "label_loss": 0.5,
            "kd_loss": 0.0,
            "hint_loss": 0.0,
            "gradient_norm_preclip": 0.4,
            "gradient_clip_threshold": 1.0,
            "gradient_clipped": 0,
            "window_seconds": 0.1,
            "optimizer_seconds": 0.01,
            "valid_rows_per_second": 10.0,
            "forwarded_rows_per_second": 20.0,
            "valid_rows": 1,
            "forwarded_rows": 2,
            "real_rows": 1,
            "learning_rates": {"adamw": 0.001},
        }
        monitor.train_step(record, epoch, 0.5, context.device)
        monitor.diagnostics(1, epoch, 0.5, {"local/activation_rms/0": 0.1})
        monitor.resources(1, epoch, 0.5, context.device)
        return EpochResult(1, True, totals, 0.1)

    monkeypatch.setattr(sequence_profile, "train_epoch", fake_train_epoch)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda device: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda device: 10)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda device: 20)
    monkeypatch.setattr(
        torch.cuda, "get_device_properties", lambda device: SimpleNamespace(total_memory=100)
    )
    monkeypatch.setattr(sequence_profile, "_host_peak_rss_bytes", lambda: 30)
    receipts: list[dict[str, object]] = []
    monkeypatch.setattr(
        sequence_profile,
        "_write_profile_receipt",
        lambda receipt, path: receipts.append(receipt) or True,
    )
    context = DistributedContext(0, 0, 1, torch.device("cpu"), False)
    model = nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
    plan = {"validation_reporting": "full official valid after each epoch; report-only"}
    sequence_profile.run_profile(
        dataset=object(),
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        config={
            "epochs": 1,
            "kd_weight": 0.0,
            "hint_weight": 0.0,
            "monitoring": {"diagnostics_every_steps": 1},
        },
        context=context,
        rank_log=None,
        recorder=None,
        identity={"model_parameters": {}},
        output=tmp_path,
        profile_updates=1,
        max_updates=1,
        monitor=monitor,
        monitor_enabled=True,
        monitoring_plan=plan,
    )
    assert trainer_calls == [{"monitor": monitor, "monitor_enabled": True}]
    assert (
        monitor.train_steps,
        monitor.diagnostic_calls,
        monitor.resource_calls,
        monitor.evaluations,
    ) == (
        1,
        1,
        1,
        0,
    )
    assert [status for status, _ in monitor.statuses] == ["running", "profile_complete"]
    assert all(context["no_eval_during_fit"] is True for _, context in monitor.statuses)
    annotation = receipts[0]["evaluation_annotation"]
    assert annotation["no_eval_during_fit"] is True
    assert annotation["validation_reporting"] == "not evaluated during bounded engineering profile"
    assert annotation["intended_validation_reporting"] == plan["validation_reporting"]


def test_monitoring_config_is_opt_in_and_identity_preserving(tmp_path: Path) -> None:
    recipe_path = tmp_path / "recipe.json"
    raw = {"model": _settings(), "batch_size": 2}
    recipe_path.write_text(json.dumps(raw, default=_json_default))
    legacy = load_config(recipe_path)
    assert "monitoring" not in legacy
    legacy_hash = canonical_sha256(legacy)
    assert canonical_sha256(load_config(recipe_path)) == legacy_hash

    raw["monitoring"] = {"train_eval_sequences": 4, "diagnostics_every_steps": 10}
    recipe_path.write_text(json.dumps(raw, default=_json_default))
    monitored = load_config(recipe_path)
    assert monitored["monitoring"] == raw["monitoring"]
    assert canonical_sha256(monitored) != legacy_hash
    raw["monitoring"]["diagnostics_every_steps"] = 0
    recipe_path.write_text(json.dumps(raw, default=_json_default))
    with pytest.raises(ValueError, match="positive integer"):
        load_config(recipe_path)
    raw["monitoring"]["diagnostics_every_steps"] = 10
    raw["epochs"] = 10_001
    recipe_path.write_text(json.dumps(raw, default=_json_default))
    with pytest.raises(ValueError, match=r"epochs in \[1, 10000\]"):
        load_config(recipe_path)


def test_diagnostic_probe_is_math_and_rng_neutral_and_reports_galam_branches() -> None:
    torch.manual_seed(71)
    settings = _settings()
    baseline = build_sequence_model(settings).train()
    monitored = build_sequence_model(settings).train()
    monitored.load_state_dict(baseline.state_dict())
    x = torch.randn(2, 4, 112)
    starting_rng = torch.get_rng_state().clone()

    base_prediction, _, _ = baseline(x)
    base_prediction.square().mean().backward()
    baseline_rng = torch.get_rng_state().clone()
    torch.set_rng_state(starting_rng)
    with DiagnosticProbe(monitored) as probe:
        prediction, _, _ = monitored(x)
        probe.observe_prediction(prediction, torch.ones(2, 4, dtype=torch.bool))
        prediction.square().mean().backward()
    observed_rng = torch.get_rng_state().clone()
    metrics = probe.metrics()

    torch.testing.assert_close(prediction, base_prediction, rtol=0, atol=0)
    assert torch.equal(baseline_rng, observed_rng)
    for name, parameter in baseline.named_parameters():
        observed_gradient = dict(monitored.named_parameters())[name].grad
        assert (parameter.grad is None) == (observed_gradient is None)
        if parameter.grad is not None and observed_gradient is not None:
            torch.testing.assert_close(parameter.grad, observed_gradient)
    for name in (
        "local/cell_state_rms",
        "local/gate/i/layer_0/mean",
        "local/gate/i/layer_0/saturated_fraction",
        "local/gradient_rms/0",
        "global/attention_activation_rms/0",
        "global/mixer_activation_rms/0",
        "fusion/output_gradient_rms",
        "fusion/gate/mean",
        "fusion/gate/saturated_fraction",
        "attention/local/entropy_mean",
        "attention/global/lag_abs_mean",
        "relative_bias/local/std",
        "relative_bias/global/abs_max",
        "train/prediction/std",
        "train/prediction/clipped_fraction",
    ):
        assert name in metrics
        assert np.isfinite(metrics[name])
    assert all(getattr(module, "_monitor_observer", None) is None for module in monitored.modules())


def test_diagnostics_omit_absent_gru_cell_and_global_branches() -> None:
    model = _GRUProbeModel().train()
    with DiagnosticProbe(model) as probe:
        prediction, _, _ = model(torch.randn(2, 3, 112))
        prediction.square().mean().backward()
    metrics = probe.metrics()
    assert "local/hidden_state_rms" in metrics
    assert "local/gate/r/layer_0/mean" in metrics
    assert "local/cell_state_rms" not in metrics
    assert not any(name.startswith("global/") for name in metrics)


def _primary_epoch_producer(split_hash: str) -> dict[str, object]:
    fit_members = [["train", 1], ["train", 2]]
    data = {
        "split_sha256": split_hash,
        "fit_members": fit_members,
        "teacher": {"fit_only": True},
    }
    config = {"kd_weight": 0.5, "hint_weight": 0.0, "epochs": 1}
    source_files = {"sequence_fit.py": "source"}
    identity = {
        "data": data,
        "data_sha256": digest(data),
        "config_sha256": digest(config),
        "source_files": source_files,
        "source_sha256": digest(source_files),
    }
    entry = {
        "epoch": 0,
        "completed_epochs": 1,
        "global_step": 4,
        "checkpoint_path": f"epoch_checkpoints/epoch-000001-{'a' * 64}.pt",
        "checkpoint_sha256": "a" * 64,
    }
    index: dict[str, object] = {
        "schema": "wnn.sequence-epoch-index.v1",
        "identity": identity,
        "completed_epochs": 1,
        "global_step": 4,
        "previous_index_sha256": None,
        "entries": [entry],
    }
    index["index_sha256"] = digest(index)
    return {
        "kind": EPOCH_KIND,
        "epoch_admission": {"index": index, "index_sha256": index["index_sha256"], "entry": entry},
        "checkpoint_sha256": entry["checkpoint_sha256"],
        "split_sha256": split_hash,
        "fit_members": fit_members,
        "teacher": data["teacher"],
        "config": config,
        "provenance": {"identity": identity, "completed_epochs": 1},
        "primary_valid_report_only": True,
    }


def test_retained_epoch_can_be_scored_only_as_explicit_full_primary_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    groups = (10, 11)
    rows = np.arange(SEQUENCE_LENGTH, dtype=np.float32)
    targets = {
        group: np.stack(
            (np.sin(rows * 0.001 + group), np.cos(rows * 0.0007 - group)), axis=-1
        ).astype(np.float32)
        for group in groups
    }
    masks = {group: np.ones(SEQUENCE_LENGTH, dtype=bool) for group in groups}
    predictions_dir = tmp_path / "predictions"
    predictions_dir.mkdir()
    sequences = []
    for group in groups:
        prediction = targets[group] * (1 if group == 10 else -0.3)
        path = predictions_dir / f"valid-{group}.npy"
        np.save(path, prediction.astype(np.float32), allow_pickle=False)
        sequences.append(
            {
                "split": "valid",
                "row_group": group,
                "file": path.name,
                "sha256": file_sha256(path),
            }
        )
    split_hash = "b" * 64
    manifest = {
        "schema": "sequence-predictions-v1",
        "complete": True,
        "input_schema": "raw112-v1",
        "membership": "evaluation",
        "split_sha256": split_hash,
        "hint_dim": 0,
        "sequence_length": SEQUENCE_LENGTH,
        "producer": _primary_epoch_producer(split_hash),
        "world_size": 1,
        "sequences": sequences,
    }
    (predictions_dir / "manifest.json").write_text(json.dumps(manifest))
    split = CampaignSplit(
        tmp_path / "primary.json",
        split_hash,
        0,
        1,
        (1, 2),
        (),
        groups,
        (("train", 1, 0), ("train", 2, 1), ("valid", 10, 2), ("valid", 11, 3)),
        CHRONOLOGICAL_PRIMARY_SPLIT_SCHEMA,
    )

    class _File:
        def __init__(self, path: Path) -> None:
            self.path = path

        def read(self, group: int) -> SimpleNamespace:
            return SimpleNamespace(targets=targets[group], scored=masks[group])

    monkeypatch.setattr(sequence_score, "load_campaign_split", lambda path, root: split)
    monkeypatch.setattr(sequence_score, "SequenceFile", _File)
    with pytest.raises(ValueError, match="explicit report-only"):
        sequence_score.score_export(
            tmp_path,
            split.path,
            predictions_dir,
            tmp_path / "not-explicit.json",
        )

    receipt_path = tmp_path / "score.json"
    sequence_score.score_export(
        tmp_path,
        split.path,
        predictions_dir,
        receipt_path,
        report_only_primary_valid=True,
    )
    receipt = json.loads(receipt_path.read_text())
    expected = GlobalAccumulator()
    for group in groups:
        mask = masks[group].copy()
        mask[:WARMUP] = False
        expected.add(targets[group], targets[group] * (1 if group == 10 else -0.3), mask)
    assert receipt["score"]["wp"] == pytest.approx(expected.result()["wp"])
    assert receipt["score"]["selected_rows"] == 2 * (SEQUENCE_LENGTH - WARMUP)
    assert receipt["report_only_primary_valid"] is True
    assert "no checkpoint selection" in receipt["interpretation"]

    manifest["sequences"] = sequences[:1]
    (predictions_dir / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="coverage is incomplete"):
        sequence_score.score_export(
            tmp_path,
            split.path,
            predictions_dir,
            tmp_path / "incomplete.json",
            report_only_primary_valid=True,
        )
