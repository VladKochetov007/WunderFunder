"""Small CPU invariants for the i1 causal-history screen."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from wnn.i1_history import (
    I1_HISTORY_MODE,
    I1_HISTORY_WIDTH,
    I1_INPUT_WIDTH,
    I1_RAW_START,
    I1_RAW_STOP,
    fit_i1_history_prior,
    i1_history_input_mask_indices,
)
from wnn.i1_history_scan import i1_history_features_numpy, i1_history_features_torch_chunk
from wnn.i1_history_streaming import StreamingI1History
from wnn.training.cli import build_parser
from wnn.training.models import make_model
from wnn.training.validation_inputs import validate_distillation, validate_inputs
from wnn.training.validation_training import validate_models


def _raw(rows: int = 19, offset: float = 0.0) -> np.ndarray:
    """Build a tiny reproducible sequence with changing i1 channels."""
    generator = np.random.default_rng(20261021)
    values = np.zeros((rows, 112), dtype=np.float32)
    values[:, I1_RAW_START:I1_RAW_STOP] = offset + np.cumsum(
        generator.normal(size=(rows, I1_RAW_STOP - I1_RAW_START)).astype(np.float32), axis=0
    )
    return values


def _prior() -> dict[str, object]:
    """Fit a valid test prior using only synthetic sequence rows."""
    return fit_i1_history_prior([_raw(11), _raw(8, 2.0)])


def _reference(raw: np.ndarray, prior: dict[str, object]) -> np.ndarray:
    """Build history with a scalar causal recurrence independent of the scan code."""
    scale = np.asarray(prior["scale"], dtype=np.float32)
    output = np.zeros((len(raw), I1_HISTORY_WIDTH), dtype=np.float32)
    decay32 = np.float32(2.0 ** (-1.0 / 32.0))
    decay512 = np.float32(2.0 ** (-1.0 / 512.0))
    mean32 = raw[0, I1_RAW_START:I1_RAW_STOP].copy()
    mean512 = mean32.copy()
    energy512 = np.zeros_like(mean32)
    previous = mean32.copy()
    for row in range(1, len(raw)):
        current = raw[row, I1_RAW_START:I1_RAW_STOP]
        delta = current - previous
        energy512 = decay512 * energy512 + (1.0 - decay512) * np.square(delta)
        output[row] = np.concatenate(
            (
                current - mean32,
                current - mean512,
                np.log1p(np.sqrt(energy512) / scale),
            )
        )
        mean32 = decay32 * mean32 + (1.0 - decay32) * current
        mean512 = decay512 * mean512 + (1.0 - decay512) * current
        previous = current.copy()
    return output


def test_i1_history_is_prefix_causal_and_matches_scalar_reference() -> None:
    """Future mutations leave the prefix unchanged and all channels match the recurrence."""
    prior = _prior()
    raw = _raw()
    changed_future = raw.copy()
    changed_future[8:, I1_RAW_START:I1_RAW_STOP] += np.float32(7.0)

    actual = i1_history_features_numpy(raw, prior)
    altered = i1_history_features_numpy(changed_future, prior)

    np.testing.assert_allclose(actual, _reference(raw, prior), rtol=2e-5, atol=2e-5)
    np.testing.assert_array_equal(actual[:8], altered[:8])
    np.testing.assert_array_equal(actual[0], np.zeros(I1_HISTORY_WIDTH, dtype=np.float32))


def test_i1_history_chunk_state_carries_and_resets_between_sequences() -> None:
    """Torch chunks preserve state, while a new sequence starts with zero history."""
    prior = _prior()
    raw = torch.from_numpy(_raw())
    complete, _ = i1_history_features_torch_chunk(raw[None], prior)
    first, state = i1_history_features_torch_chunk(raw[None, :6], prior)
    second, state = i1_history_features_torch_chunk(raw[None, 6:13], prior, state)
    third, _ = i1_history_features_torch_chunk(raw[None, 13:], prior, state)
    chunked = torch.cat((first, second, third), dim=1)

    torch.testing.assert_close(chunked, complete, rtol=2e-5, atol=2e-5)
    restarted, _ = i1_history_features_torch_chunk((raw + 3.0)[None], prior)
    assert torch.count_nonzero(restarted[:, 0]).item() == 0
    stream = StreamingI1History(prior)
    stream.update_chunk(raw[:6].numpy())
    stream.reset()
    assert torch.count_nonzero(torch.from_numpy(stream.update(raw[0].numpy()))).item() == 0


def test_i1_controls_keep_367_inputs_and_apply_semantic_masks_after_normalization(
    tmp_path: Path,
) -> None:
    """A and B mask the declared columns after normalization; C keeps history active."""
    arguments = {
        "arch": "gru",
        "hidden": 8,
        "layers": 1,
        "dropout": 0.0,
        "research_features": I1_HISTORY_MODE,
        "research_feature_control": "all",
        "drop_i1": True,
    }
    model = make_model(arguments, I1_INPUT_WIDTH, 2)
    model.mean.copy_(torch.linspace(-2.0, 2.0, I1_INPUT_WIDTH))
    model.std.copy_(torch.linspace(0.5, 2.0, I1_INPUT_WIDTH))
    values = torch.full((1, 2, I1_INPUT_WIDTH), 4.0)
    expected_a = i1_history_input_mask_indices("all", True)
    normalized_a = model._normalized_inputs(values)
    assert normalized_a[..., list(expected_a)].count_nonzero().item() == 0
    assert normalized_a[..., 0].count_nonzero().item() > 0
    assert I1_INPUT_WIDTH == 367

    b_arguments = {**arguments, "drop_i1": False}
    b_model = make_model(b_arguments, I1_INPUT_WIDTH, 2)
    b_model.mean.copy_(model.mean)
    b_model.std.copy_(model.std)
    normalized_b = b_model._normalized_inputs(values)
    history_indices = i1_history_input_mask_indices("all", False)
    assert normalized_b[..., list(history_indices)].count_nonzero().item() == 0
    assert normalized_b[..., I1_RAW_START].count_nonzero().item() > 0
    assert normalized_b[..., 112].count_nonzero().item() > 0

    c_arguments = {**arguments, "research_feature_control": "none", "drop_i1": False}
    c_model = make_model(c_arguments, I1_INPUT_WIDTH, 2)
    c_model.mean.copy_(model.mean)
    c_model.std.copy_(model.std)
    normalized_c = c_model._normalized_inputs(values)
    assert normalized_c[..., list(history_indices)].count_nonzero().item() > 0

    checkpoint = tmp_path / "i1_checkpoint.pt"
    torch.save({"args": arguments, "model": model.state_dict()}, checkpoint)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    restored = make_model(saved["args"], I1_INPUT_WIDTH, 2)
    restored.load_state_dict(saved["model"])
    torch.testing.assert_close(restored._normalized_inputs(values), normalized_a)


def test_equal_width_legacy_initialization_matches_explicit_widths() -> None:
    """The omitted width defaults produce the same native 211-input GRU tensors."""
    default_arguments = {
        "arch": "gru",
        "hidden": 8,
        "layers": 1,
        "dropout": 0.0,
        "research_features": "none",
    }
    explicit_arguments = {
        **default_arguments,
        "encoder_width": 8,
        "recurrent_width": 8,
        "readout_width": 8,
        "head_layers": 1,
    }
    torch.manual_seed(901)
    default_model = make_model(default_arguments, 211, 2)
    torch.manual_seed(901)
    explicit_model = make_model(explicit_arguments, 211, 2)

    assert default_model.state_dict().keys() == explicit_model.state_dict().keys()
    for name, default_value in default_model.state_dict().items():
        torch.testing.assert_close(default_value, explicit_model.state_dict()[name], rtol=0, atol=0)


def test_sustained_profile_recipe_parses_and_passes_static_cli_validation() -> None:
    """The exact profile argv validates before any campaign data is opened."""
    repository = Path(__file__).resolve().parents[1]
    plan_path = repository / (
        "research/campaign_2026-10-21/next_cycle_r1/i1/profile_c_f0_s1001_r1.json"
    )
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    argv = plan["argv"]
    assert argv.count("scripts/train_rnn.py") == 1
    command_at = argv.index("scripts/train_rnn.py")
    parser = build_parser()
    arguments = parser.parse_args(argv[command_at + 1 :])
    run = SimpleNamespace(
        options=SimpleNamespace(
            ap=parser,
            args=arguments,
            distillation_requested=False,
            init_checkpoint=None,
            continuation=False,
        ),
        environment=SimpleNamespace(argv=[]),
    )

    validate_inputs(run)
    validate_distillation(run)
    validate_models(run)

    assert arguments.max_batches == plan["profile_warmup"] + plan["profile_steps"]
    assert arguments.research_features == I1_HISTORY_MODE
    assert arguments.encoder_width is None
    assert arguments.recurrent_width is None
    assert arguments.readout_width is None
    assert not arguments.save_training_state


def test_all_sustained_fit_recipes_parse_and_pass_static_cli_validation() -> None:
    """All six literal fit commands validate and use one trainer entrypoint."""
    repository = Path(__file__).resolve().parents[1]
    recipe_dir = repository / "research/campaign_2026-10-21/next_cycle_r1/i1/recipes"
    parser = build_parser()
    for plan_path in sorted(recipe_dir.glob("i1_*_r1.json")):
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        fit = plan["steps"][0]
        argv = fit["argv"]
        assert argv.count("scripts/train_rnn.py") == 1
        assert argv[:3] == ["{python}", "-u", "scripts/train_rnn.py"]
        command_at = argv.index("scripts/train_rnn.py")
        arguments = parser.parse_args(argv[command_at + 1 :])
        run = SimpleNamespace(
            options=SimpleNamespace(
                ap=parser,
                args=arguments,
                distillation_requested=False,
                init_checkpoint=None,
                continuation=False,
            ),
            environment=SimpleNamespace(argv=[]),
        )

        validate_inputs(run)
        validate_distillation(run)
        validate_models(run)

        assert arguments.name == plan_path.stem
        assert arguments.research_features == I1_HISTORY_MODE
        assert arguments.encoder_width is None
        assert arguments.recurrent_width is None
        assert arguments.readout_width is None
        assert arguments.save_training_state


def test_panel_grader_resolves_repo_and_checks_npy_header_bytes(tmp_path: Path) -> None:
    """The fixed panel grader anchors to this checkout and rejects truncated NPYs."""
    repository = Path(__file__).resolve().parents[1]
    grader_path = repository / ("research/campaign_2026-10-21/next_cycle_r1/i1/grade_i1_panel.py")
    spec = importlib.util.spec_from_file_location("i1_panel_grader", grader_path)
    assert spec is not None and spec.loader is not None
    grader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(grader)
    assert repository == grader.ROOT

    prediction_path = tmp_path / "prediction.npy"
    np.save(prediction_path, np.zeros((2, 3, 2), dtype=np.float32), allow_pickle=False)
    sizes = grader._npy_header(prediction_path)
    assert sizes["payload_bytes"] == 2 * 3 * 2 * np.dtype(np.float32).itemsize
    assert sizes["file_bytes"] == prediction_path.stat().st_size

    prediction_path.write_bytes(prediction_path.read_bytes()[:-1])
    try:
        grader._npy_header(prediction_path)
    except ValueError as error:
        assert "byte count" in str(error)
    else:
        raise AssertionError("truncated prediction NPY was accepted")


def test_panel_grader_rejects_unpaired_initialization_or_normalization() -> None:
    """The grader refuses A/B/C fit cells that do not share seeded setup."""
    repository = Path(__file__).resolve().parents[1]
    grader_path = repository / ("research/campaign_2026-10-21/next_cycle_r1/i1/grade_i1_panel.py")
    spec = importlib.util.spec_from_file_location("i1_panel_grader_pairing", grader_path)
    assert spec is not None and spec.loader is not None
    grader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(grader)
    identities = {
        arm: {
            "initial_parameter_sha256": "same-seeded-initialization",
            "normalization_identity": {"mean": "same", "std": "same"},
        }
        for arm in "ABC"
    }
    grader._verify_paired_fit_identities(identities, 0)
    for identity in ("initial_parameter_sha256", "normalization_identity"):
        mismatch = {arm: dict(values) for arm, values in identities.items()}
        mismatch["C"][identity] = "different"
        with pytest.raises(ValueError, match=identity):
            grader._verify_paired_fit_identities(mismatch, 0)
