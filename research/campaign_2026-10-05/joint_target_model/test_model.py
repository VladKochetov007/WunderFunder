"""Small CPU invariants for the isolated paired-target proposal."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from codebook import fit_codebooks, nearest_clipped_labels, nearest_labels
from e2e_model import JointTargetContinuation
from model import JointTargetHead, require_finite_training_state, tilted_bayes_readout


class _TinyBackbone(torch.nn.Module):
    """Minimal causal GRU matching the parent checkpoint's module interface."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("mean", torch.zeros(5))
        self.register_buffer("std", torch.ones(5))
        self.inp = torch.nn.Linear(5, 4)
        self.gru = torch.nn.GRU(4, 4, batch_first=True)
        self.head = torch.nn.Sequential(
            torch.nn.Linear(8, 4), torch.nn.GELU(), torch.nn.Linear(4, 2)
        )

    def encode(self, values: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.gelu(self.inp(values))


def test_tilted_readout_and_zero_denominator_fallback() -> None:
    """Each margin uses its own |c| denominator and a safe zero fallback."""
    atoms = torch.tensor([[-1.0, 0.0], [2.0, 1.0]])
    q = torch.tensor([[0.5, 0.5], [1.0, 0.0]])
    fallback = torch.tensor([[0.25, -0.5], [0.3, 0.4]])
    decoded = tilted_bayes_readout(q, atoms, fallback)
    assert torch.allclose(decoded[0], torch.tensor([1.0, 1.0]))
    assert torch.equal(decoded[1, 1], fallback[1, 1])


def test_train_only_codebook_and_shuffled_marginal_control() -> None:
    """The shuffled negative control preserves atom marginals, not pairings."""
    target = np.asarray([[-1, 0.5], [-0.5, 1], [0.25, -1], [2.5, 3]], dtype=np.float32)
    paired, marginal, shuffled, used = fit_codebooks(target, 4, 4, seed=3)
    assert used == len(target)
    assert paired.min() >= -2 and paired.max() <= 2
    for k in range(2):
        assert np.array_equal(np.sort(paired[:, k]), np.sort(shuffled[:, k]))
        assert len(nearest_labels(target[:, k : k + 1].clip(-2, 2), marginal[k, :, None])) == len(
            target
        )
    assert not np.array_equal(paired, shuffled)


def test_metric_outliers_are_clipped_before_nearest_atom_assignment() -> None:
    """Raw labels can pick a different atom than the official clipped target."""
    atoms = np.asarray([[2.0, 0.5], [1.8, 0.0]], dtype=np.float32)
    outlier = np.asarray([[3.0, 0.0]], dtype=np.float32)
    raw_assignment = nearest_labels(outlier, atoms)
    metric_assignment = nearest_clipped_labels(outlier, atoms)
    assert raw_assignment.tolist() == [0]
    assert metric_assignment.tolist() == [1]


def test_all_arms_initialize_at_the_clipped_parent_mean() -> None:
    """Zero residual and gate make every decoder an exact parent anchor."""
    torch.manual_seed(5)
    paired = torch.tensor([[-1.0, 0.5], [0.5, -1.0], [2.0, 2.0]])
    independent = torch.tensor([[-1.0, 0.5, 2.0], [-1.0, 0.5, 2.0]])
    head = JointTargetHead(4, paired, independent, paired.flip(0))
    states = torch.randn(7, 2, 4)
    parent_members = torch.rand(7, 2, 2) * 4 - 2
    expected = parent_members.mean(dim=1).clamp(-2, 2)
    for arm in (
        "joint_tilted",
        "joint_ordinary",
        "shuffled_tilted",
        "independent_tilted",
        "regression_anchor",
    ):
        prediction, _, _ = head(states, parent_members, arm)
        assert torch.equal(prediction, expected)


def test_end_to_end_wrapper_is_causal_and_returns_two_outputs() -> None:
    """The wrapper preserves the causal recurrent state and two-value callback."""
    torch.manual_seed(12)
    codebook = torch.tensor([[-1.0, 0.5], [0.5, -1.0], [2.0, 2.0]])
    marginal = torch.tensor([[-1.0, 0.5, 2.0], [-1.0, 0.5, 2.0]])
    backbone = _TinyBackbone().eval()
    head = JointTargetHead(4, codebook, marginal, codebook.roll(1, dims=0))
    model = JointTargetContinuation(backbone, head, "joint_tilted").eval()
    inputs = torch.randn(2, 11, 5)
    changed = inputs.clone()
    changed[:, 7:] += 50
    first, state = model(inputs, None)
    second, _ = model(changed, None)
    assert first.shape == (2, 11, 2)
    assert state is not None and state.shape == (1, 2, 4)
    assert torch.equal(first[:, :7], second[:, :7])
    anchored, _, _, _, parent = model.forward_aux(inputs, None)
    assert torch.equal(anchored, parent.clamp(-2, 2))


def test_nonfinite_training_step_fails_closed() -> None:
    """A NaN gradient cannot pass the fixed-step guard."""
    parameter = torch.nn.Parameter(torch.ones(1))
    loss = (parameter * torch.tensor(float("nan"))).sum()
    loss.backward()
    with pytest.raises(FloatingPointError, match="nonfinite"):
        require_finite_training_state(loss, (), [parameter], "gradients")
