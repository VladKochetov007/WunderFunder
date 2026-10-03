from __future__ import annotations

import copy

import pytest
import torch
from live_capture import (
    Capture,
    compare,
    get_rng,
    restore,
    set_rng,
    trainer_module,
    validate,
    validate_rng,
    validate_torch_rng,
)
from runner import capture_step, synthetic_environment


@pytest.fixture(scope="module")
def captured() -> Capture:
    """Genuine warm student/laggedEMA/optimizer prestate from actual source."""
    saved = get_rng("cpu")
    torch.manual_seed(734019)
    result = capture_step(synthetic_environment())
    set_rng(saved, "cpu")
    return result


@pytest.mark.parametrize("offset", [20, 28, 36, 44, 5048])
@pytest.mark.parametrize("boundary", ["validate_rng", "validate", "restore", "set_rng"])
def test_padding_normalization_rejected_before_global_mutation(
    captured: Capture, offset: int, boundary: str
) -> None:
    """Same metadata and accepted setter bytes must still fail exact roundtrip."""
    invalid = copy.deepcopy(captured.pre)
    state = invalid["rng"]["torch_cpu"]
    state[offset] ^= 1
    engine = torch.Generator(device="cpu")
    engine.set_state(state)
    assert not torch.equal(state, engine.get_state())
    before = get_rng("cpu")
    try:
        with pytest.raises(ValueError, match="normalized state; exact roundtrip"):
            if boundary == "validate_rng":
                validate_rng(invalid["rng"], "cpu")
            elif boundary == "validate":
                validate(invalid)
            elif boundary == "restore":
                restore(invalid, trainer_module())
            else:
                set_rng(invalid["rng"], "cpu")
    finally:
        compare(before, get_rng("cpu"))


@pytest.mark.parametrize("seed", [0, 19, 734019])
@pytest.mark.parametrize("advance", ["initial", "uniform", "odd_gaussian"])
def test_legitimate_private_engine_states_are_exact(seed: int, advance: str) -> None:
    """Actual seeded/advanced/cached engine states need no invented byte layout."""
    before = get_rng("cpu")
    source = torch.Generator(device="cpu").manual_seed(seed)
    if advance == "uniform":
        torch.rand(17, generator=source)
    elif advance == "odd_gaussian":
        torch.randn(1, generator=source)
    declared = source.get_state().clone()
    target = torch.Generator(device="cpu")
    validate_torch_rng(declared, target, "torch_cpu")
    compare(declared, target.get_state())
    compare(declared, source.get_state())
    bundle = copy.deepcopy(before)
    bundle["torch_cpu"] = declared
    validate_rng(bundle, "cpu")
    compare(before, get_rng("cpu"))
    try:
        set_rng(bundle, "cpu")
        compare(bundle, get_rng("cpu"))
    finally:
        set_rng(before, "cpu")


def test_pristine_warm_prestate_preserves_rng(captured: Capture) -> None:
    """Validation performs no process-global draws or state mutation."""
    before = get_rng("cpu")
    validate(captured.pre)
    compare(before, get_rng("cpu"))
