from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
from feature_branch import run  # noqa: E402
from feature_branch_head import (  # noqa: E402
    ARMS,
    PARAMETER_BUDGET,
    Head,
    empirical_moments,
    factorized_log_probabilities,
    rational_readout,
)
from feature_screen import digest_file  # noqa: E402


def fixture(path: Path, first: int) -> None:
    rng = np.random.default_rng(51 + first)
    rows = 128
    y = rng.normal(scale=0.4, size=(rows, 2)).astype(np.float32)
    np.savez(
        path,
        state=rng.normal(size=(rows, 64)).astype(np.float32),
        y=y,
        pred=0.5 * y + rng.normal(scale=0.1, size=(rows, 2)).astype(np.float32),
        fit_weight=np.ones(rows),
        sequence=np.repeat(np.arange(first, first + 4), 32),
        component=np.repeat(np.arange(first, first + 4), 32),
        row=np.tile(np.arange(250, 282), 4),
        model_id=np.full(rows, "a" * 64),
    )
    path.with_suffix(".json").write_text(
        json.dumps(
            dict(
                sha256=digest_file(path),
                rows=rows,
                state_width=64,
                outer_fold=0,
                component_graph_sha256="b" * 64,
                members={"a" * 64: dict(checkpoint_sha256="a" * 64, fit_components=[1000])},
            )
        )
    )


def test_factorized_probabilities_and_sparse_readout_are_finite_normalized_and_differentiable() -> (
    None
):
    torch.manual_seed(12)
    logits = torch.randn(13, 32, requires_grad=True)
    logp = factorized_log_probabilities(logits)
    assert logp.shape == (13, 504)
    torch.testing.assert_close(logp.exp().sum(-1), torch.ones(13))
    # Only two codes seen: all others must have explicit finite fit-only backoff.
    codes = np.array([0] * 8 + [18] * 8)
    target = np.tile(np.array([[-1.0, 0.2], [-0.5, 0.3]]), (8, 1))
    m1, m0, report = empirical_moments(codes, target, np.ones(16))
    assert report["unknown_codes"] == 502 and np.isfinite(m1).all() and np.isfinite(m0).all()
    for tilted in (False, True):
        result = rational_readout(logp, torch.from_numpy(m1), torch.from_numpy(m0), tilted)
        assert torch.isfinite(result).all() and result.abs().max() <= 2
    (-logp[:, 0].mean() + result.square().mean()).backward()
    assert torch.isfinite(logits.grad).all() and logits.grad.abs().sum() > 0


def test_parameter_budgets_match_on_current_trunk_widths() -> None:
    for width in (256, 512, 768, 1024):
        for arm in ARMS:
            head = Head(width, arm, torch.zeros(width), torch.ones(width))
            count = sum(value.numel() for value in head.parameters())
            assert 0.98 * PARAMETER_BUDGET <= count <= PARAMETER_BUDGET


def test_actual_head_fits_share_budget_and_predictions_ignore_evaluation_labels(
    tmp_path: Path,
) -> None:
    fit_path, evaluation_path = tmp_path / "fit.npz", tmp_path / "eval.npz"
    fixture(fit_path, 0)
    fixture(evaluation_path, 4)
    labels = tmp_path / "labels.npz"
    np.savez(labels, seq_ix=np.arange(8), market_class=np.arange(8) % 4)
    output = tmp_path / "first"
    run(fit_path, evaluation_path, labels, output, updates=2, bootstrap=100)
    first = dict(np.load(output / "predictions.npz"))
    report = json.loads((output / "report.json").read_text())
    assert report["training"]["updates"] == 2
    assert set(report["training"]["history"]) == set(ARMS)
    assert all(len(value) == 2 for value in report["training"]["history"].values())
    assert {"unchanged", "global_affine", *ARMS} <= first.keys()
    assert "3" in report["class_wp"] and report["selection"].startswith("none")
    # Mutate only evaluation class strata and target labels; predictions must be invariant.
    np.savez(labels, seq_ix=np.arange(8), market_class=np.r_[np.arange(4), [3, 3, 3, 3]])
    data = dict(np.load(evaluation_path))
    data["y"] *= -2
    np.savez(evaluation_path, **data)
    metadata = json.loads(evaluation_path.with_suffix(".json").read_text())
    metadata["sha256"] = digest_file(evaluation_path)
    evaluation_path.with_suffix(".json").write_text(json.dumps(metadata))
    run(fit_path, evaluation_path, labels, tmp_path / "second", updates=2, bootstrap=100)
    with np.load(tmp_path / "second/predictions.npz") as second:
        for name in ("unchanged", "global_affine", *ARMS):
            np.testing.assert_array_equal(first[name], second[name])
    with pytest.raises(ValueError, match="overlap"):
        run(fit_path, fit_path, labels, tmp_path / "bad", updates=2, bootstrap=100)
