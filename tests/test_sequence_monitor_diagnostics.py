from __future__ import annotations

import random

import numpy as np
import pytest
import torch
from torch import nn

from wnn.data import I0
from wnn.training.sequence_distributed import DistributedContext
from wnn.training.sequence_monitor_evaluation import evaluate_dataset, evaluate_monitoring_sets
from wnn.training.sequence_monitor_head_metrics import HeadMetrics, TargetMoments, add_score_aliases


class _DiagnosticDataset:
    def __init__(self) -> None:
        target = np.array(
            [
                [
                    [5.0, -4.0],
                    [1.0, -1.0],
                    [2.2, 0.5],
                    [0.0, 2.0],
                    [-3.0, 0.2],
                    [1.5, 1.2],
                    [0.7, -2.5],
                    [2.0, 3.0],
                ],
                [
                    [-0.5, 1.1],
                    [3.5, -2.2],
                    [1.8, 2.4],
                    [-1.3, 0.8],
                    [2.7, -0.3],
                    [-2.6, 1.9],
                    [0.4, -1.7],
                    [1.2, 2.8],
                ],
            ],
            dtype=np.float32,
        )
        z = np.array(
            [
                [
                    [-3.2, 0.1],
                    [0.5, -1.2],
                    [2.8, 0.3],
                    [-0.8, 2.4],
                    [-1.7, 0.9],
                    [2.6, 1.3],
                    [0.4, -3.4],
                    [1.7, 2.1],
                ],
                [
                    [0.6, 1.7],
                    [3.1, -0.4],
                    [1.1, 2.9],
                    [-2.1, 0.6],
                    [2.5, -1.0],
                    [-0.9, 2.2],
                    [0.2, -1.8],
                    [1.4, 3.3],
                ],
            ],
            dtype=np.float32,
        )
        reference = np.array(
            [
                [
                    [1.8, -2.7],
                    [1.1, -0.4],
                    [-1.3, 2.2],
                    [2.6, -1.5],
                    [0.9, -0.2],
                    [-1.8, 2.4],
                    [2.3, -1.1],
                    [0.2, 1.8],
                ],
                [
                    [-1.0, 1.6],
                    [-2.9, 1.4],
                    [0.7, -1.8],
                    [1.8, 0.3],
                    [-0.6, 2.7],
                    [1.9, -2.4],
                    [-1.2, 1.0],
                    [2.5, -0.8],
                ],
            ],
            dtype=np.float32,
        )
        mask = np.array(
            [
                [False, False, True, True, True, False, True, True],
                [False, True, True, False, True, True, True, True],
            ],
            dtype=bool,
        )
        self.items = [
            {
                "x": torch.from_numpy(np.pad(z[index], ((0, 0), (0, 110)))),
                "y": torch.from_numpy(target[index]),
                "mask": torch.from_numpy(mask[index]),
                "teacher": torch.from_numpy(reference[index]),
                "hints": torch.empty(8, 0),
            }
            for index in range(2)
        ]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.items[index]


class _CountingModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.forward_calls = 0

    def forward(
        self, x: torch.Tensor, state: object = None
    ) -> tuple[torch.Tensor, torch.Tensor, None]:
        self.forward_calls += 1
        torch.rand(1)
        random.random()
        np.random.random()  # noqa: NPY002 - exercise exact RNG restoration
        return x[:, :, :2] * self.scale, x[:, :, :0], None


def _weighted_wp(target: np.ndarray, prediction: np.ndarray, mask: np.ndarray) -> float:
    """Compute the published weighted Pearson formula independently on each target."""
    values = []
    for index in range(2):
        y = np.clip(target[..., index][mask].astype(np.float64), -2.0, 2.0)
        p = np.clip(prediction[..., index][mask].astype(np.float64), -2.0, 2.0)
        weights = np.abs(y)
        total = weights.sum()
        if total < 1e-8:
            values.append(0.0)
            continue
        mean_y = np.sum(weights * y) / total
        mean_p = np.sum(weights * p) / total
        covariance = np.sum(weights * (y - mean_y) * (p - mean_p)) / total
        std_y = np.sqrt(np.sum(weights * np.square(y - mean_y)) / total)
        std_p = np.sqrt(np.sum(weights * np.square(p - mean_p)) / total)
        values.append(0.0 if std_y <= 1e-8 or std_p <= 1e-8 else covariance / (std_y * std_p))
    return float(np.mean(values))


def _wmse_by_target(target: np.ndarray, prediction: np.ndarray, mask: np.ndarray) -> np.ndarray:
    clipped = np.clip(target[mask].astype(np.float64), -2.0, 2.0)
    predicted = prediction[mask].astype(np.float64)
    weights = np.abs(clipped)
    return np.sum(weights * np.square(predicted - clipped), axis=0) / weights.sum(axis=0)


def _same_numpy_state(left: tuple, right: tuple) -> bool:
    return left[0] == right[0] and np.array_equal(left[1], right[1]) and left[2:] == right[2:]


def test_same_forward_diagnostics_preserve_primary_metrics_and_training_state() -> None:
    """Extra component statistics reuse forwards and leave the official mixture path unchanged."""
    dataset = _DiagnosticDataset()
    model = _CountingModel().train()
    model.scale.grad = torch.tensor(7.0)
    context = DistributedContext(0, 0, 1, torch.device("cpu"), False)
    torch.manual_seed(101)
    random.seed(202)
    np.random.seed(303)  # noqa: NPY002 - seed the exercised legacy global stream
    before_torch = torch.get_rng_state().clone()
    before_python = random.getstate()
    before_numpy = np.random.get_state()  # noqa: NPY002 - compare the captured legacy stream

    primary_train, primary_valid = evaluate_monitoring_sets(
        model, dataset, dataset, 3, "fp32", context, 2, blend_alpha=0.1
    )
    primary_calls = model.forward_calls
    model.forward_calls = 0
    diagnostic_train, diagnostic_valid = evaluate_monitoring_sets(
        model,
        dataset,
        dataset,
        3,
        "fp32",
        context,
        2,
        blend_alpha=0.1,
        report_standalone=True,
    )

    assert model.forward_calls == primary_calls == 6
    assert model.training is True
    torch.testing.assert_close(model.scale.grad, torch.tensor(7.0), rtol=0, atol=0)
    assert torch.equal(before_torch, torch.get_rng_state())
    assert before_python == random.getstate()
    assert _same_numpy_state(
        before_numpy,
        np.random.get_state(),  # noqa: NPY002 - compare the captured legacy stream
    )
    for base, extended in (
        (primary_train, diagnostic_train),
        (primary_valid, diagnostic_valid),
    ):
        for key, value in base.items():
            if not key.startswith("timing/"):
                assert extended[key] == value

    target = np.stack([item["y"].numpy() for item in dataset.items])
    mask = np.stack([item["mask"].numpy() for item in dataset.items])
    z = np.stack([item["x"][:, :2].numpy() for item in dataset.items])
    reference = np.stack([item["teacher"].numpy() for item in dataset.items])
    mixture = (0.9 * torch.from_numpy(reference) + 0.1 * torch.from_numpy(z)).numpy()
    assert diagnostic_valid["valid/wp"] == pytest.approx(
        _weighted_wp(target, mixture, mask), abs=2e-15
    )
    assert diagnostic_valid["valid/wp/t0"] == pytest.approx(
        _weighted_wp(target[..., :1].repeat(2, axis=-1), mixture[..., :1].repeat(2, axis=-1), mask),
        abs=2e-15,
    )
    assert diagnostic_valid["valid/wp/t1"] == pytest.approx(
        _weighted_wp(target[..., 1:].repeat(2, axis=-1), mixture[..., 1:].repeat(2, axis=-1), mask),
        abs=2e-15,
    )
    assert diagnostic_valid["valid/mixture/wp"] == diagnostic_valid["valid/wp"]
    assert diagnostic_valid["valid/z/wp"] == pytest.approx(_weighted_wp(target, z, mask))
    assert diagnostic_valid["valid/reference/wp"] == pytest.approx(
        _weighted_wp(target, reference, mask)
    )
    np.testing.assert_allclose(
        diagnostic_valid["valid/z/wmse/t0"], _wmse_by_target(target, z, mask)[0]
    )
    clipped_target_full = np.clip(target[mask].astype(np.float64), -2.0, 2.0)
    mixture_selected = mixture[mask].astype(np.float64)
    mixture_weights = np.abs(clipped_target_full)
    assert diagnostic_valid["valid/mixture/wmse"] == pytest.approx(
        np.sum(mixture_weights * np.square(mixture_selected - clipped_target_full))
        / mixture_weights.sum(),
        abs=1e-15,
    )
    train_alias, valid_alias = diagnostic_train.copy(), diagnostic_valid.copy()
    add_score_aliases(train_alias, valid_alias, "valid")
    assert train_alias["train_eval/score"] == train_alias["train_eval/wp"]
    assert valid_alias["valid/score"] == valid_alias["valid/wp"]

    weights = np.abs(np.clip(target[mask].astype(np.float64), -2.0, 2.0))
    for name, prediction in (("z", z), ("reference", reference), ("mixture", mixture)):
        values = prediction[mask].astype(np.float64)
        np.testing.assert_allclose(
            [diagnostic_valid[f"valid/{name}/raw_mean/t{i}"] for i in range(2)],
            values.mean(axis=0),
        )
        np.testing.assert_allclose(
            [diagnostic_valid[f"valid/{name}/raw_std/t{i}"] for i in range(2)],
            values.std(axis=0),
        )
        np.testing.assert_allclose(
            [diagnostic_valid[f"valid/{name}/weighted_raw_mean/t{i}"] for i in range(2)],
            (weights * values).sum(axis=0) / weights.sum(axis=0),
        )
        weighted_mean = (weights * values).sum(axis=0) / weights.sum(axis=0)
        np.testing.assert_allclose(
            [diagnostic_valid[f"valid/{name}/weighted_raw_std/t{i}"] for i in range(2)],
            np.sqrt(
                (weights * np.square(values - weighted_mean)).sum(axis=0) / weights.sum(axis=0)
            ),
        )
        clipped = np.abs(values) > 2.0
        np.testing.assert_array_equal(
            [diagnostic_valid[f"valid/{name}/prediction_clipped_count/t{i}"] for i in range(2)],
            clipped.sum(axis=0),
        )
        np.testing.assert_allclose(
            [
                diagnostic_valid[f"valid/{name}/prediction_clipped_weight_mass/t{i}"]
                for i in range(2)
            ],
            (weights * clipped).sum(axis=0),
        )
        np.testing.assert_allclose(
            [diagnostic_valid[f"valid/{name}/prediction_clipped_fraction/t{i}"] for i in range(2)],
            clipped.mean(axis=0),
        )

    selected_target = target[mask].astype(np.float64)
    clipped_target = np.clip(selected_target, -2.0, 2.0)
    np.testing.assert_allclose(
        [diagnostic_valid[f"valid/target/raw_mean/t{i}"] for i in range(2)],
        selected_target.mean(axis=0),
    )
    np.testing.assert_allclose(
        [diagnostic_valid[f"valid/target/raw_std/t{i}"] for i in range(2)],
        selected_target.std(axis=0),
    )
    target_weighted_mean = (weights * selected_target).sum(axis=0) / weights.sum(axis=0)
    np.testing.assert_allclose(
        [diagnostic_valid[f"valid/target/weighted_raw_mean/t{i}"] for i in range(2)],
        target_weighted_mean,
    )
    np.testing.assert_allclose(
        [diagnostic_valid[f"valid/target/weighted_raw_std/t{i}"] for i in range(2)],
        np.sqrt(
            (weights * np.square(selected_target - target_weighted_mean)).sum(axis=0)
            / weights.sum(axis=0)
        ),
    )
    np.testing.assert_allclose(
        [diagnostic_valid[f"valid/target/clipped_mean/t{i}"] for i in range(2)],
        clipped_target.mean(axis=0),
    )
    np.testing.assert_allclose(
        [diagnostic_valid[f"valid/target/clipped_std/t{i}"] for i in range(2)],
        clipped_target.std(axis=0),
    )
    clipped_weighted_mean = (weights * clipped_target).sum(axis=0) / weights.sum(axis=0)
    np.testing.assert_allclose(
        [diagnostic_valid[f"valid/target/weighted_clipped_mean/t{i}"] for i in range(2)],
        clipped_weighted_mean,
    )
    np.testing.assert_allclose(
        [diagnostic_valid[f"valid/target/weighted_clipped_std/t{i}"] for i in range(2)],
        np.sqrt(
            (weights * np.square(clipped_target - clipped_weighted_mean)).sum(axis=0)
            / weights.sum(axis=0)
        ),
    )
    np.testing.assert_allclose(
        [diagnostic_valid[f"valid/target/official_weight_mass/t{i}"] for i in range(2)],
        np.abs(clipped_target).sum(axis=0),
    )
    np.testing.assert_allclose(
        [diagnostic_valid[f"valid/target/clipped_weight_mass_delta/t{i}"] for i in range(2)],
        np.abs(selected_target).sum(axis=0) - np.abs(clipped_target).sum(axis=0),
    )
    target_clipped = np.abs(selected_target) > 2.0
    np.testing.assert_array_equal(
        [diagnostic_valid[f"valid/target/clipped_count/t{i}"] for i in range(2)],
        target_clipped.sum(axis=0),
    )
    np.testing.assert_allclose(
        [diagnostic_valid[f"valid/target/clipped_weight_mass/t{i}"] for i in range(2)],
        (weights * target_clipped).sum(axis=0),
    )


def test_standalone_without_blend_reports_no_reference_or_mixture() -> None:
    """A labels-only standalone diagnostic never names z as a reference mixture."""
    context = DistributedContext(0, 0, 1, torch.device("cpu"), False)
    metrics = evaluate_dataset(
        _CountingModel(),
        _DiagnosticDataset(),
        "train_eval",
        3,
        "fp32",
        context,
        2,
        report_standalone=True,
    )
    assert "train_eval/z/wp" in metrics
    assert not any("/reference/" in key or "/mixture/" in key for key in metrics)


def test_since_diagnostics_report_full_valid_age_strata_without_changing_primary_wp() -> None:
    class Dataset:
        def __init__(self) -> None:
            inputs = torch.zeros(40, 112)
            inputs[2:, I0.bid_px.start] = 1.0
            inputs[2:, I0.ask_px.start] = 1.0
            target = torch.stack(
                (torch.arange(40, dtype=torch.float32) / 20, torch.arange(40).float().sin()),
                dim=-1,
            )
            self.items = [
                {
                    "x": inputs,
                    "y": target,
                    "mask": torch.ones(40, dtype=torch.bool),
                    "teacher": torch.zeros_like(target),
                    "hints": torch.zeros(40, 0),
                }
            ]

        def __len__(self) -> int:
            return len(self.items)

        def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
            return self.items[index]

    dataset = Dataset()
    context = DistributedContext(0, 0, 1, torch.device("cpu"), False)
    model = _CountingModel()
    primary = evaluate_dataset(model, dataset, "valid", 8, "fp32", context, 1)
    diagnostics = evaluate_dataset(
        model, dataset, "valid", 8, "fp32", context, 1, since_diagnostics=True
    )

    assert diagnostics["valid/wp"] == primary["valid/wp"]
    assert diagnostics["valid/wp/since_lt32/selected_rows"] == 34
    assert diagnostics["valid/wp/since_gte32/selected_rows"] == 6
    assert "valid/wp/since_lt32/t0" in diagnostics
    assert "valid/wp/since_gte32/t1" in diagnostics


def test_since_diagnostics_emit_finite_unavailable_metrics_for_empty_age_bucket() -> None:
    class Dataset:
        def __init__(self) -> None:
            inputs = torch.zeros(40, 112)
            inputs[:, I0.bid_px.start] = torch.arange(40, dtype=torch.float32)
            inputs[:, I0.ask_px.start] = torch.arange(40, dtype=torch.float32) + 1.0
            target = torch.stack(
                (torch.arange(40, dtype=torch.float32), torch.arange(40).float().sin()), dim=-1
            )
            self.item = {
                "x": inputs,
                "y": target,
                "mask": torch.ones(40, dtype=torch.bool),
                "teacher": torch.zeros_like(target),
                "hints": torch.zeros(40, 0),
            }

        def __len__(self) -> int:
            return 1

        def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
            assert index == 0
            return self.item

    context = DistributedContext(0, 0, 1, torch.device("cpu"), False)
    metrics = evaluate_dataset(
        _CountingModel(), Dataset(), "valid", 8, "fp32", context, 1, since_diagnostics=True
    )

    assert metrics["valid/wp/since_lt32/available"] == 1
    assert metrics["valid/wp/since_gte32/available"] == 0
    assert metrics["valid/wp/since_gte32/selected_rows"] == 0
    assert "valid/wp/since_gte32" not in metrics
    assert "valid/wp/since_gte32/t0" not in metrics
    assert "valid/wp/since_gte32/t1" not in metrics
    assert metrics["valid/wp"] == pytest.approx(metrics["valid/wp/since_lt32"])
    assert all(np.isfinite(value) for value in metrics.values())


def test_streaming_diagnostics_ignore_unselected_nans_and_reject_selected_nans() -> None:
    """Sparse masks ignore unselected nonfinites while selected values fail closed."""
    dataset = _DiagnosticDataset()
    target = np.stack([item["y"].numpy() for item in dataset.items])
    mask = np.stack([item["mask"].numpy() for item in dataset.items])
    z = np.stack([item["x"][:, :2].numpy() for item in dataset.items])
    reference = np.stack([item["teacher"].numpy() for item in dataset.items])
    mixture = (0.9 * torch.from_numpy(reference) + 0.1 * torch.from_numpy(z)).numpy()
    bad_target = target.copy()
    bad_target[~mask] = np.nan
    bad_predictions = [array.copy() for array in (z, reference, mixture)]
    for prediction in bad_predictions:
        prediction[~mask] = np.nan

    for component, clean_prediction, bad_prediction in zip(
        ("z", "reference", "mixture"), (z, reference, mixture), bad_predictions, strict=True
    ):
        clean = HeadMetrics()
        clean.add(target, clean_prediction, mask)
        masked = HeadMetrics()
        masked.add(bad_target, bad_prediction, mask)
        assert clean.metrics("valid", component) == masked.metrics("valid", component)

    clean_targets, masked_targets = TargetMoments(), TargetMoments()
    clean_targets.add(target, mask)
    masked_targets.add(bad_target, mask)
    assert clean_targets.metrics("valid") == masked_targets.metrics("valid")

    row, column = np.argwhere(mask)[0]
    selected_bad_target = target.copy()
    selected_bad_target[row, column, 0] = np.nan
    with pytest.raises(ValueError, match="nonfinite selected"):
        HeadMetrics().add(selected_bad_target, z, mask)
    with pytest.raises(ValueError, match="finite selected"):
        TargetMoments().add(selected_bad_target, mask)

    selected_bad_prediction = z.copy()
    selected_bad_prediction[row, column, 0] = np.nan
    with pytest.raises(ValueError, match="nonfinite selected"):
        HeadMetrics().add(target, selected_bad_prediction, mask)
