from __future__ import annotations

import numpy as np
import pytest
from predict_e424_recurrent import infer_output_width


def test_specialist_output_width_uses_all_architecture_heads() -> None:
    state = {
        "universal_head.2.weight": np.zeros((23, 96), dtype=np.float32),
        "expert_heads.0.weight": np.zeros((23, 32), dtype=np.float32),
        "expert_heads.1.weight": np.zeros((23, 32), dtype=np.float32),
    }
    assert infer_output_width("specialist", state) == 23


def test_specialist_rejects_missing_or_inconsistent_expert_heads() -> None:
    with pytest.raises(ValueError, match="no expert output heads"):
        infer_output_width(
            "specialist",
            {"universal_head.2.weight": np.zeros((23, 96), dtype=np.float32)},
        )
    state = {
        "universal_head.2.weight": np.zeros((23, 96), dtype=np.float32),
        "expert_heads.0.weight": np.zeros((2, 32), dtype=np.float32),
    }
    with pytest.raises(ValueError, match="widths disagree"):
        infer_output_width("specialist", state)


def test_prefix_adapter_infers_output_width_from_its_head() -> None:
    state = {"head.2.weight": np.zeros((23, 256), dtype=np.float32)}
    assert infer_output_width("prefix-adapter", state) == 23


def test_inference_width_rejects_unknown_architecture() -> None:
    with pytest.raises(ValueError, match="unsupported E424"):
        infer_output_width("gru", {})
