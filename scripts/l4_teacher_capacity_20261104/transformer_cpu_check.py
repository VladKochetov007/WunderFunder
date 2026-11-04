from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from wnn.gmetric import global_wp, reference_global_wp
from wnn.training.sequence_config import load_config
from wnn.training.sequence_distributed import configure_strict_fp32
from wnn.training.sequence_features import EVENT_INPUT_MODE, sequence_inputs_from_raw
from wnn.training.sequence_fit_utils import SEQUENCE_ROWS, sequence_lr_multiplier, total_updates
from wnn.training.sequence_models import build_sequence_model

LOGGER = logging.getLogger(__name__)
EXPECTED_SEEDS = (1001, 1002)
EXPECTED_CONFIG_PREFIX = "causal_transformer_d384_l8_h8_s"
EXPECTED_TRAIN_SEQUENCES = 10_325


def _load_configs(config_dir: Path) -> list[dict[str, Any]]:
    """Load and validate the paired Transformer recipes through the production parser."""
    configs = [
        load_config(config_dir / f"{EXPECTED_CONFIG_PREFIX}{seed}.json") for seed in EXPECTED_SEEDS
    ]
    for config, seed in zip(configs, EXPECTED_SEEDS, strict=True):
        model = config["model"]
        if not isinstance(model, dict):
            raise TypeError("validated model config must remain a mapping")
        expected = {
            "seed": seed,
            "epochs": 4,
            "batch_size": 8,
            "accumulate_chunks": 4,
            "chunk_rows": 1024,
            "lr": 0.0003,
            "weight_decay": 0.01,
            "optimizer": "adamw",
            "grad_clip": 1.0,
            "warmup_fraction": 0.05,
            "schedule": "warmup_cosine_v2",
            "min_lr_ratio": 0.0,
            "normalization_scope": "all_fit",
            "training_mask": "all_after_warmup",
            "precision_mode": "fp32",
            "required_world_size": 1,
        }
        if any(config.get(key) != value for key, value in expected.items()):
            raise ValueError(f"seed {seed} recipe differs from the frozen Transformer contract")
        model_expected = {
            "architecture": "transformer",
            "input_feature_mode": EVENT_INPUT_MODE,
            "hidden": 384,
            "layers": 8,
            "heads": 8,
            "context_rows": 1024,
            "attention_backend": "math",
            "checkpoint_blocks": True,
            "compile_blocks": False,
        }
        if any(model.get(key) != value for key, value in model_expected.items()):
            raise ValueError(f"seed {seed} Transformer dimensions or input mode changed")
        ff_width = int(384 * float(model["ff_multiplier"]))
        if ff_width != 1536:
            raise ValueError(f"seed {seed} feed-forward width must be 1536, received {ff_width}")
        if config.get("checkpoint_retention") != "all_epochs":
            raise ValueError("all four fixed epoch snapshots must be retained")
        monitoring = config.get("monitoring")
        if (
            not isinstance(monitoring, dict)
            or monitoring.get("train_eval_mask") != "all_after_warmup"
        ):
            raise ValueError("training diagnostics must share the all-after-warmup fitting mask")
    return configs


def _check_schedule(config: dict[str, Any]) -> int:
    """Check the exact endpoint-aligned warmup/cosine schedule and update count."""
    updates = total_updates(config, EXPECTED_TRAIN_SEQUENCES, world_size=1)
    if updates != 25_820:
        raise ValueError(f"expected 25820 scheduled updates, got {updates}")
    chunk_rows = int(config["chunk_rows"])
    intervals = [
        (start, min(start + chunk_rows, SEQUENCE_ROWS))
        for start in range(0, SEQUENCE_ROWS, chunk_rows)
    ]
    if (
        intervals[0][0] != 0
        or intervals[-1][1] != SEQUENCE_ROWS
        or any(left[1] != right[0] for left, right in zip(intervals, intervals[1:], strict=False))
    ):
        raise ValueError("chunk schedule must cover every row of each complete 20,000-row sequence")
    multiplier = sequence_lr_multiplier(
        "warmup_cosine_v2", updates, warmup_fraction=0.05, min_lr_ratio=0.0
    )
    warmup_updates = math.ceil(updates * 0.05)
    if warmup_updates != 1_291 or multiplier(0) != 1.0 / warmup_updates:
        raise ValueError("warmup must rise linearly from 1/1291 of peak LR")
    if multiplier(warmup_updates - 1) != 1.0 or multiplier(updates - 1) != 0.0:
        raise ValueError("cosine schedule must peak after warmup and end at zero on update 25820")
    return updates


def _check_transformer_causality_and_reset(config: dict[str, Any]) -> None:
    """Exercise 211-feature causal prediction, chunk continuation, and reset on CPU."""
    torch.manual_seed(int(config["seed"]))
    model_config = config["model"]
    if not isinstance(model_config, dict):
        raise TypeError("validated model config must remain a mapping")
    model = build_sequence_model(model_config).eval()
    if model.input_projection.in_features != 211 or model.mean.numel() != 211:
        raise ValueError("Transformer projection and train-only normalizer must cover event211")
    if model.blocks[0].feedforward_in.out_features != 1536:
        raise ValueError("Transformer FFN must have the frozen width 1536")
    if any(parameter.dtype is not torch.float32 for parameter in model.parameters()):
        raise TypeError("Transformer parameters must remain strict FP32")

    generator = np.random.default_rng(int(config["seed"]))
    raw = generator.normal(size=(1, 19, 112)).astype(np.float32)
    future_changed = raw.copy()
    future_changed[:, 11:] += np.float32(7.0)
    values = sequence_inputs_from_raw(torch.from_numpy(raw), EVENT_INPUT_MODE)
    changed = sequence_inputs_from_raw(torch.from_numpy(future_changed), EVENT_INPUT_MODE)
    if not isinstance(values, torch.Tensor) or not isinstance(changed, torch.Tensor):
        raise TypeError("event211 assembly must return tensors for the Transformer")
    torch.testing.assert_close(values[:, :11], changed[:, :11], rtol=0.0, atol=0.0)

    configure_strict_fp32()
    with torch.inference_mode():
        prediction, _, first_state = model(values)
        replay, _, _ = model(values)
        altered_prediction, _, _ = model(changed)
        prefix, _, carried_state = model(values[:, :7])
        suffix, _, final_state = model(values[:, 7:], carried_state)
        reset_replay, _, _ = model(values, None)
    torch.testing.assert_close(prediction, replay, rtol=0.0, atol=0.0)
    torch.testing.assert_close(prediction[:, :11], altered_prediction[:, :11], rtol=0.0, atol=1e-6)
    torch.testing.assert_close(torch.cat((prefix, suffix), dim=1), prediction, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(reset_replay, prediction, rtol=0.0, atol=0.0)
    if first_state.next_position != values.shape[1] or final_state.next_position != values.shape[1]:
        raise ValueError("sequence state did not advance by the exact number of rows")

    second_sequence = sequence_inputs_from_raw(torch.from_numpy(raw[:, 4:]), EVENT_INPUT_MODE)
    if not isinstance(second_sequence, torch.Tensor):
        raise TypeError("second sequence event assembly must return a tensor")
    torch.testing.assert_close(
        second_sequence[:, 0, 112:], torch.zeros_like(second_sequence[:, 0, 112:]), rtol=0, atol=0
    )


def _check_rolling_window() -> None:
    """Check that 211-wide Transformer state carries then trims its rolling KV window."""
    torch.manual_seed(882)
    model = build_sequence_model(
        {
            "architecture": "transformer",
            "input_feature_mode": EVENT_INPUT_MODE,
            "hidden": 16,
            "layers": 2,
            "heads": 4,
            "ff_multiplier": 2,
            "context_rows": 8,
            "dropout": 0.0,
            "checkpoint_blocks": False,
            "hint_dim": 0,
            "hint_projection": "learned",
            "precision_mode": "fp32",
            "attention_backend": "math",
            "compile_blocks": False,
        }
    ).eval()
    values = torch.randn(1, 20, 211)
    changed = values.clone()
    changed[:, 0] += 3.0
    with torch.inference_mode():
        complete, _, complete_state = model(values)
        altered, _, _ = model(changed)
        first, _, first_state = model(values[:, :5])
        middle, _, middle_state = model(values[:, 5:13], first_state)
        final, _, final_state = model(values[:, 13:], middle_state)
    torch.testing.assert_close(torch.cat((first, middle, final), dim=1), complete)
    if not torch.equal(complete[:, 15:], altered[:, 15:]):
        raise ValueError("rolling context allowed a row older than the layered window to leak")
    if (
        complete_state.next_position != 20
        or first_state.next_position != 5
        or middle_state.next_position != 13
        or any(key.shape[-2] != 7 for key in final_state.keys)
    ):
        raise ValueError("rolling context state did not retain exactly seven preceding rows")


def _check_masks_and_official_metric() -> int:
    """Verify warmup/scored-row masks and the stable official pooled WP implementation."""
    rows = 512
    row_index = np.arange(rows)
    scored = row_index % 5 != 0
    train_mask = row_index >= 99
    valid_mask = scored & (row_index >= 99)
    if train_mask[:99].any() or not train_mask[99:].all():
        raise ValueError("fitting objective must include every row after warmup index 99")
    if not np.array_equal(valid_mask, scored & train_mask):
        raise ValueError("primary valid mask must combine official is_scored and row>=99")

    generator = np.random.default_rng(517)
    targets = generator.normal(size=(1, rows, 2)).astype(np.float32)
    predictions = (0.7 * targets + generator.normal(scale=0.4, size=targets.shape)).astype(
        np.float32
    )
    score = global_wp(targets, predictions, valid_mask[None, :])
    reference = reference_global_wp(targets, predictions, valid_mask[None, :])
    if score["selected_rows"] != int(valid_mask.sum()):
        raise ValueError("weighted metric selected-row count differs from the exact valid mask")
    if not math.isclose(float(score["wp"]), reference, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("streaming official weighted metric differs from its reference formula")
    return int(score["selected_rows"])


def main() -> None:
    """Run bounded CPU contract checks without data access, fitting, or GPU use."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", type=Path, default=None)
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    config_dir = args.config_dir or (
        repo_root / "research/experiment_drafts/l4-teacher-capacity-20261104/transformer/configs"
    )
    configs = _load_configs(config_dir)
    update_counts = [_check_schedule(config) for config in configs]
    _check_transformer_causality_and_reset(configs[0])
    _check_rolling_window()
    selected_rows = _check_masks_and_official_metric()
    LOGGER.info(
        "Transformer CPU checks passed: seeds=%s, updates=%s, synthetic_valid_rows=%s",
        list(EXPECTED_SEEDS),
        update_counts,
        selected_rows,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
