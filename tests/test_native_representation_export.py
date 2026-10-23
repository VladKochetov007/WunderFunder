from __future__ import annotations

import numpy as np
import pytest

from wnn.native_representation import N_BASE, N_NATIVE_EMA_INPUT, N_RAW, native_input_metadata


def state_for_width(width: int) -> dict[str, np.ndarray]:
    return {
        "mean": np.zeros(width, dtype=np.float32),
        "std": np.ones(width, dtype=np.float32),
        "inp.weight": np.zeros((8, width), dtype=np.float32),
    }


@pytest.mark.parametrize("mode", ["base", "ema"])
def test_467_input_metadata_binds_the_supported_producer_contract(mode: str) -> None:
    metadata = native_input_metadata(
        {"representation_mode": mode}, state_for_width(N_NATIVE_EMA_INPUT), N_NATIVE_EMA_INPUT
    )

    assert metadata["representation_mode"] == mode
    assert metadata["input_width"] == N_NATIVE_EMA_INPUT
    assert metadata["layout"][:2] == ["raw112", "event99"]
    assert metadata["layout"][2:] == [
        "zero128",
        "causal_signed_raw_delta_ema112_hl32" if mode == "ema" else "zero112",
        "zero16",
    ]
    assert metadata["producer_contract"]["git_commit"] == (
        "68ff682977b822f38351aac0f621146f5f13b440"
    )


@pytest.mark.parametrize("mode", ["base", "ema"])
def test_467_input_rejects_non_shared_appended_normalizers(mode: str) -> None:
    state = state_for_width(N_NATIVE_EMA_INPUT)
    state["mean"][N_BASE + 3] = 0.25

    with pytest.raises(ValueError, match="appended means must all be zero"):
        native_input_metadata({"representation_mode": mode}, state, N_NATIVE_EMA_INPUT)

    state = state_for_width(N_NATIVE_EMA_INPUT)
    state["std"][N_BASE + 3] = 0.75
    with pytest.raises(ValueError, match="appended scales must all be one"):
        native_input_metadata({"representation_mode": mode}, state, N_NATIVE_EMA_INPUT)


@pytest.mark.parametrize("width", [N_BASE, N_BASE + 112])
def test_legacy_native_input_widths_remain_available(width: int) -> None:
    metadata = native_input_metadata(
        {"online_norm": width == N_BASE + N_RAW}, state_for_width(width), width
    )

    assert metadata["representation_mode"] == "none"
    assert metadata["input_width"] == width


@pytest.mark.parametrize("mode", ["sparse", "ema_sparse"])
def test_sparse_representation_modes_fail_closed(mode: str) -> None:
    with pytest.raises(ValueError, match="does not support representation mode"):
        native_input_metadata(
            {"representation_mode": mode}, state_for_width(N_NATIVE_EMA_INPUT), N_NATIVE_EMA_INPUT
        )


def test_467_input_requires_mode_and_matching_checkpoint_width() -> None:
    state = state_for_width(N_NATIVE_EMA_INPUT)

    with pytest.raises(ValueError, match="has no supported representation metadata"):
        native_input_metadata({}, state, N_NATIVE_EMA_INPUT)
    with pytest.raises(ValueError, match="does not match"):
        native_input_metadata({"representation_mode": "ema"}, state, N_BASE)
