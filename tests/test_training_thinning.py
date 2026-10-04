from __future__ import annotations

import pytest
import torch

from tests.test_trainer_resume_integration import training


@pytest.mark.parametrize("probability", [0.0, 1.0])
def test_retained_mass_matches_support_without_changing_observations(
    monkeypatch: pytest.MonkeyPatch, probability: float
) -> None:
    """Pair the real transformation across inputs, teachers, propensity and score masks."""
    monkeypatch.setattr(training, "WARMUP", 3)
    row = torch.arange(30)
    raw = row[:, None].float().repeat(1, 112)
    raw[:, :22] = (row // 3)[:, None]
    labels_and_teacher = row[:, None].float().repeat(1, 4)
    scored = row % 2 == 0
    propensity = row.float() / 30
    spread = row.to(torch.uint8) % 12
    original = (raw, labels_and_teacher, scored, propensity, spread)
    torch.manual_seed(81080)
    start = torch.get_rng_state()
    compressed = training.thin_rows(original, (probability, 0.5, 0.5))
    compressed_rng = torch.get_rng_state()
    torch.set_rng_state(start)
    retained = training.thin_rows(original, (probability, 0.5, 0.5, "retained-mass"))
    assert torch.equal(compressed_rng, torch.get_rng_state())
    keep = retained[-1]
    length = int(compressed[-1])
    assert keep.dtype == torch.bool
    assert int(keep.sum()) == length
    assert keep[:3].all()
    assert keep[3::3].all()  # every changed quote survives
    for actual, source in zip(retained[:-1], original, strict=True):
        assert actual is source  # no input, supervision or cache rewrite
    for actual, source in zip(compressed[:-1], original, strict=True):
        assert torch.equal(actual[:length], source[keep])
    assert not compressed[2][length:].any()
    if probability:
        assert length < len(row)
        valid = (row >= 3) & keep
        assert torch.equal(propensity[valid].sum(), compressed[3][3:length].sum())
        assert torch.equal(
            labels_and_teacher[valid].abs().sum(), compressed[1][3:length].abs().sum()
        )
    else:
        assert length == len(row)


@pytest.mark.parametrize("mode", ["compress", "retained-mass"])
def test_thinning_loader_schema_budget(monkeypatch: pytest.MonkeyPatch, mode: str) -> None:
    """Account for a per-row bool mask instead of a scalar compressed length."""
    monkeypatch.setattr(training, "SEQUENCE_LENGTH", 30)
    plain = training.Sequences("valid", [0])
    thinned = training.Sequences("valid", [0], thin=(1.0, 0.1, 0.4, mode))
    expected_extra = 30 if mode == "retained-mass" else 8
    assert thinned.batch_tensor_bytes(2) - plain.batch_tensor_bytes(2) == 2 * expected_extra
