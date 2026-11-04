"""CPU verification for strict-FP32 setup in the synthetic shape probe."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

import pytest
import torch

from wnn.galam import connectome_112_to_2_config
from wnn.training.sequence_shape_probe import run_probe
from wnn.training.sequence_shape_probe_support import _validate_shape


def _json_default(value: object) -> list[str]:
    """Serialize GALAM fidelity sets for the canonical JSON recipe reader."""
    if isinstance(value, frozenset):
        return sorted(value)
    raise TypeError(f"unsupported recipe value: {type(value).__name__}")


def test_probe_batch_bound_accepts_384_and_rejects_larger() -> None:
    """The bounded model-width proxy permits the approved 384 x 256 x 64 shape."""
    args = argparse.Namespace(
        batch_size=384,
        warmup_updates=2,
        measure_updates=3,
        device="cpu",
    )
    config = {
        "batch_size": 1,
        "chunk_rows": 256,
        "accumulate_chunks": 1,
        "precision_mode": "fp32",
        "model": {
            "architecture": "gru",
            "hidden": 64,
            "layers": 1,
            "input_feature_mode": "raw112",
        },
    }
    _, batch_size, rows, chunks, width = _validate_shape(args, config)
    assert (batch_size, rows, chunks, width) == (384, 256, 1, 112)
    args.batch_size = 385
    with pytest.raises(ValueError, match="batch size must be between 1 and 384"):
        _validate_shape(args, config)


def test_galam_probe_resets_permissive_tf32_and_receipts_strict_fp32(tmp_path: Path) -> None:
    """A CPU GALAM Muon step starts from permissive flags and records their reset."""
    galam = replace(
        connectome_112_to_2_config(),
        width=64,
        attention_heads=4,
        lstm_layers=1,
        global_blocks=1,
        local_attention_blocks=1,
        attention_history=2,
        conv_kernel_size=3,
        conv_hidden_width=8,
        relative_bias_width=4,
        feed_forward_width=32,
    )
    recipe = {
        "epochs": 1,
        "batch_size": 2,
        "chunk_rows": 2,
        "accumulate_chunks": 1,
        "optimizer": "muon",
        "adamw_beta1": 0.9,
        "adamw_beta2": 0.999,
        "model": {
            "architecture": "galam",
            "galam": asdict(galam) | {"variant": "schematic_lstm_full"},
        },
    }
    config_path = tmp_path / "galam64-muon.json"
    config_path.write_text(json.dumps(recipe, default=_json_default), encoding="utf-8")
    args = argparse.Namespace(
        config=config_path,
        batch_size=2,
        warmup_updates=0,
        measure_updates=1,
        device="cpu",
        output_json=None,
        profiler_trace=None,
    )
    old_precision = torch.get_float32_matmul_precision()
    old_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    old_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    try:
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        result = run_probe(args)
    finally:
        torch.set_float32_matmul_precision(old_precision)
        torch.backends.cuda.matmul.allow_tf32 = old_matmul_tf32
        torch.backends.cudnn.allow_tf32 = old_cudnn_tf32

    assert result["optimizer"] == "muon"
    assert result["predictions_and_losses_finite"] is True
    assert result["optimizer_fp32_state_finite"] is True
    assert result["precision"] == {
        "mode": "fp32",
        "float32_matmul_precision": "highest",
        "cuda_matmul_tf32": False,
        "cudnn_tf32": False,
        "cuda_autocast_enabled": False,
        "cpu_autocast_enabled": False,
        "strict_fp32": True,
    }
