from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from typing import cast

from f3_checkpoint_common import mapping, require


def validate_adam_state_descriptors(
    parameter_shapes: list[tuple[int, ...]],
    states: list[Mapping[str, object]],
    expected_step_value: int,
) -> None:
    """Reject malformed Adam moments or step counters using CPU descriptors."""
    expected_shapes = Counter(parameter_shapes)
    observed_shapes: Counter[tuple[int, ...]] = Counter()
    require(len(states) == len(parameter_shapes), "Adam state parameter count")
    for index, state in enumerate(states):
        require(
            state["state_fields"] == ("exp_avg", "exp_avg_sq", "step"),
            f"Adam state {index} fields",
        )
        require(
            state["step_numel"] == 1 and state["step_finite"] is True, f"Adam state {index} step"
        )
        require(
            float(state["step_value"]) == expected_step_value,
            f"Adam state {index} step value",
        )
        require(state["exp_avg_dtype"] == "torch.float32", f"Adam state {index} exp_avg dtype")
        require(
            state["exp_avg_sq_dtype"] == "torch.float32", f"Adam state {index} exp_avg_sq dtype"
        )
        require(state["moments_finite"] is True, f"Adam state {index} moments finite")
        first = cast(tuple[int, ...], state["exp_avg_shape"])
        second = cast(tuple[int, ...], state["exp_avg_sq_shape"])
        require(first == second, f"Adam state {index} moment shapes")
        observed_shapes[first] += 1
    require(observed_shapes == expected_shapes, "Adam moment/model parameter shapes")


def validate_pair_contracts(left: Mapping[str, object], right: Mapping[str, object]) -> None:
    """Require identical F3 data/runtime recipe except declared architecture fields."""
    left_options = mapping(left["options"], "left options")
    right_options = mapping(right["options"], "right options")
    differences = {
        key
        for key in set(left_options) | set(right_options)
        if left_options.get(key) != right_options.get(key)
    }
    require(differences == {"hidden", "layers"}, "paired option differences")
    require(
        {"hidden": left_options["hidden"], "layers": left_options["layers"]}
        == {"hidden": 512, "layers": 1},
        "wide architecture",
    )
    require(
        {"hidden": right_options["hidden"], "layers": right_options["layers"]}
        == {"hidden": 384, "layers": 2},
        "deep architecture",
    )
    left_rest, right_rest = dict(left), dict(right)
    left_rest.pop("options")
    right_rest.pop("options")
    require(left_rest == right_rest, "paired non-option contract")
