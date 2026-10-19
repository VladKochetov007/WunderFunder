"""Synthetic invariants for the fixed recurrent carry diagnostic."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from idea_screens_r6.model_contract import state_fingerprint
from idea_screens_r6.sample_panel import select_eight_fit_records
from torch import nn

from idea_screens_r9.history_math import PredictionDifference, state_distance
from idea_screens_r9.state_replay import advance
from idea_screens_r9.window_probe import compare_window

R9_DIR = Path(__file__).resolve().parent
R3_DIR = R9_DIR.parent / "idea_screens_r3"


class TinyGRU(nn.Module):
    """Small deterministic recurrent model for no-update invariants."""

    def __init__(self) -> None:
        super().__init__()
        self.gru = nn.GRU(3, 4, batch_first=True)
        self.readout = nn.Linear(4, 2)

    def forward(
        self, values: torch.Tensor, state: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden, next_state = self.gru(values, state)
        return self.readout(hidden), next_state


def test_r3_metadata_selects_fixed_fit_prefix_only() -> None:
    """The real producer fixture resolves the fixed first eight fit components."""
    sample = json.loads(
        (R3_DIR / "records/sample_v2/terminal_capture/REMOTE_SAMPLE.json").read_bytes()
    )
    membership = json.loads(
        (R3_DIR / "records/sample_v2/terminal_capture/REMOTE_MEMBERSHIP.json").read_bytes()
    )
    selected = select_eight_fit_records(sample, membership)
    component_order = sample["selection"]["fit_full64_component_order"]
    assert [record["component"] for record in selected] == component_order[:8]
    assert all(record["role"] == "fit" and record["source"] == "train" for record in selected)
    assert all(set(record["files"]) == {"x", "y", "scored"} for record in selected)


def test_causal_chunk_segmentation_preserves_carried_state() -> None:
    """Splitting the full prefix keeps the same hidden state within FP32 tolerance."""
    torch.manual_seed(91)
    model = TinyGRU().eval()
    inputs = torch.randn(1, 800, 3)
    whole_counts = {"full_prefix_rows": 0, "model_forward_calls": 0}
    whole = advance(model, inputs, None, 0, 800, "full_prefix", whole_counts)
    split_counts = {"full_prefix_rows": 0, "model_forward_calls": 0}
    first = advance(model, inputs, None, 0, 333, "full_prefix", split_counts)
    split = advance(model, inputs, first, 333, 800, "full_prefix", split_counts)
    torch.testing.assert_close(whole, split, rtol=2e-5, atol=2e-6)
    assert whole_counts["full_prefix_rows"] == split_counts["full_prefix_rows"] == 800


def test_window_branches_are_independent_and_do_not_update_model() -> None:
    """Recent/zero rollouts leave starting states and all model bytes unchanged."""
    torch.manual_seed(92)
    model = TinyGRU().eval()
    inputs = torch.randn(1, 130, 3)
    full = torch.randn(1, 1, 4)
    recent = torch.randn(1, 1, 4)
    full_before, recent_before = full.clone(), recent.clone()
    zero = torch.zeros_like(full)
    counters = {"rollout_rows": 0, "model_forward_calls": 0}
    fingerprint = state_fingerprint(model)
    output = compare_window(model, inputs, 2, full, recent, zero, counters)
    assert torch.equal(full, full_before) and torch.equal(recent, recent_before)
    assert output.full_state_after.shape == (1, 1, 4)
    assert counters["rollout_rows"] == 3 * 128
    assert state_fingerprint(model) == fingerprint
    assert all(parameter.grad is None for parameter in model.parameters())
    assert len(output.record["endpoint_predictions"]) == 4
    snapshots = output.record["state_snapshots"]
    assert set(snapshots) == {"0", "1", "16", "64", "128"}
    assert snapshots["1"]["zero"] != snapshots["128"]["zero"]


def test_prediction_sufficient_statistics_pool_exactly() -> None:
    """Segmented scalar statistics reproduce a direct full-window contrast."""
    full = np.array([[1.0, -0.5], [0.0, 0.25], [2.0, 1.5]], dtype=np.float32)
    alternative = full + np.array([[0.1, -0.2], [0.3, 0.4], [-0.7, 0.2]], dtype=np.float32)
    direct = PredictionDifference()
    direct.add(full, alternative)
    pooled = PredictionDifference()
    for start, stop in ((0, 1), (1, 3)):
        part = PredictionDifference()
        part.add(full[start:stop], alternative[start:stop])
        pooled.merge(part)
    assert pooled.summary() == direct.summary()
    summary = pooled.summary()
    assert summary["pooled"]["count"] == 6
    assert summary["target_0"]["rms_difference"] > 0.0


def test_zero_state_distance_is_finite_and_zero() -> None:
    """Null reference state is retained without division-by-zero or nonfinite JSON."""
    zero = np.zeros((1, 1, 4), dtype=np.float32)
    summary = state_distance(zero, zero.copy())
    assert summary["reference_l2"] == 0.0
    assert summary["difference_l2"] == 0.0
    assert summary["relative_difference_l2"] == 0.0
