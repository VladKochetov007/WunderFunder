from __future__ import annotations

import numpy as np
import pytest
import torch

from scripts.evaluate_qat_deployment import (
    checkpoint_quantization,
    endpoint_semantics,
    export_case_archives,
)
from scripts.export_qat_control import export_control
from wnn.nn import RNNModel
from wnn.quantization import quantize_model
from wnn.quantization_export import read_packed_quantized_export
from wnn.training_qat import (
    configure_qat,
    effective_qat_state,
    inference_state,
    scale_learning_metadata,
)


@pytest.mark.parametrize(
    ("preset", "matching_case"),
    [
        ("recurrent_head", "packed_q8"),
        ("recurrent_w4", "recurrent_w4"),
        ("input_w4_recurrent_w8", "input_w4_recurrent_w8"),
    ],
)
def test_postfit_checkpoint_codes_and_scale_bits_reach_matching_archive(
    tmp_path, preset: str, matching_case: str
) -> None:
    torch.manual_seed(20261013)
    latent = RNNModel(3, hidden=4, layers=1).state_dict()
    effective, metadata = effective_qat_state(latent, preset, "ema")
    checkpoint = {
        "args": {"qat": preset},
        "model": effective,
        "quantization": metadata,
        "n_in": 3,
    }
    weights = {name: value.detach().cpu().numpy().copy() for name, value in effective.items()}

    validated = checkpoint_quantization(checkpoint, weights)
    _, native_codes, _ = quantize_model(weights, "recurrent_head")
    semantics = export_case_archives(tmp_path, weights, native_codes, validated)

    archive, codes, details = read_packed_quantized_export(tmp_path / f"{matching_case}.npz")
    assert details["preset"] == preset
    assert semantics[matching_case] == "exact_checkpoint_codes_scales"
    for name, value in validated["encoded"].items():
        np.testing.assert_array_equal(codes[name].view(np.uint8), value.view(np.uint8))
    for name, value in weights.items():
        np.testing.assert_array_equal(archive[name], value)

    # An inference checkpoint stores effective reconstructed floats. The FP32
    # diagnostic must never be described as the unavailable latent model.
    assert endpoint_semantics(validated, semantics)["fp32"] == (
        "effective_reconstructed_diagnostic_not_latent_weights"
    )


def test_plain_float_checkpoint_retains_ptq_case_semantics(tmp_path) -> None:
    model = RNNModel(3, hidden=4, layers=1)
    weights = {name: value.detach().cpu().numpy().copy() for name, value in model.state_dict().items()}
    validated = checkpoint_quantization({"args": {"qat": "none"}, "model": model.state_dict()}, weights)
    _, native_codes, _ = quantize_model(weights, "recurrent_head")
    semantics = export_case_archives(tmp_path, weights, native_codes, validated)
    labels = endpoint_semantics(validated, semantics)

    assert validated["kind"] == "float_checkpoint"
    assert labels["fp32"] == "stored_float_checkpoint_weights"
    assert labels["recurrent_w4"] == "ptq_from_effective_checkpoint"
    _, _, archive_metadata = read_packed_quantized_export(tmp_path / "recurrent_w4.npz")
    assert archive_metadata["preset"] == "recurrent_w4"


def test_no_finetune_quantization_control_requires_and_preserves_its_provenance(tmp_path) -> None:
    model = RNNModel(3, hidden=4, layers=1)
    source = tmp_path / "source.pt"
    control_path = tmp_path / "control.pt"
    torch.save(
        {"args": {"qat": "none"}, "model": model.state_dict(), "n_in": 3},
        source,
    )
    export_control(source, control_path, "recurrent_head")
    checkpoint = torch.load(control_path, map_location="cpu", weights_only=False)
    weights = {
        name: value.detach().cpu().numpy().copy()
        for name, value in checkpoint["model"].items()
    }
    validated = checkpoint_quantization(checkpoint, weights)

    assert validated["kind"] == "post_training_quantization_control"
    assert endpoint_semantics(validated, {})["fp32"] == (
        "effective_reconstructed_diagnostic_not_latent_weights"
    )


def test_rejects_qat_checkpoint_with_missing_or_incompatible_metadata() -> None:
    model = RNNModel(3, hidden=4, layers=1)
    weights = {name: value.detach().cpu().numpy().copy() for name, value in model.state_dict().items()}
    with pytest.raises(ValueError, match="missing its quantization metadata"):
        checkpoint_quantization({"args": {"qat": "recurrent_w4"}}, weights)

    effective, metadata = effective_qat_state(model.state_dict(), "recurrent_w4", "ema")
    checkpoint = {"args": {"qat": "input_w4_recurrent_w8"}, "model": effective, "quantization": metadata}
    with pytest.raises(ValueError, match="args disagree"):
        checkpoint_quantization(checkpoint, weights)
    checkpoint["args"] = {"qat": "none"}
    with pytest.raises(ValueError, match="no-finetune control provenance"):
        checkpoint_quantization(checkpoint, weights)


def test_learned_scale_checkpoint_roundtrips_exact_archive_and_validates_rule(tmp_path) -> None:
    torch.manual_seed(20261014)
    model = RNNModel(3, hidden=4, layers=1)
    configure_qat(model, "recurrent_w4", learn_scales=True)
    with torch.no_grad():
        model._qat_scale_params[0].mul_(1.125)
    effective, metadata = inference_state(model, "raw")
    checkpoint = {
        "args": {
            "qat": "recurrent_w4",
            "qat_learn_scales": True,
            "optim": "muon",
            "wd": 0.015,
        },
        "model": effective,
        "quantization": metadata,
        "n_in": 3,
        "provenance": {
            "qat": {"scale_learning": scale_learning_metadata("muon", 0.015)}
        },
    }
    weights = {name: value.detach().cpu().numpy().copy() for name, value in effective.items()}

    validated = checkpoint_quantization(checkpoint, weights)
    _, native_codes, _ = quantize_model(weights, "recurrent_head")
    semantics = export_case_archives(tmp_path, weights, native_codes, validated)
    archive, codes, _ = read_packed_quantized_export(tmp_path / "recurrent_w4.npz")
    assert semantics["recurrent_w4"] == "exact_checkpoint_codes_scales"
    assert validated["metadata"]["scale_learning"] == metadata["scale_learning"]
    for name, value in metadata["encoded"].items():
        np.testing.assert_array_equal(codes[name].view(np.uint8), value.numpy().view(np.uint8))
    for name, value in weights.items():
        np.testing.assert_array_equal(archive[name], value)

    bad = dict(checkpoint)
    bad_metadata = dict(metadata)
    bad_metadata["scale_learning"] = {**metadata["scale_learning"], "version": 999}
    bad["quantization"] = bad_metadata
    with pytest.raises(ValueError, match="incompatible learned-scale semantics"):
        checkpoint_quantization(bad, weights)

    mismatched_args = {**checkpoint, "args": {"qat": "recurrent_w4", "qat_learn_scales": False}}
    with pytest.raises(ValueError, match="disagree with checkpoint quantization semantics"):
        checkpoint_quantization(mismatched_args, weights)

    mismatched_decay = dict(checkpoint)
    mismatched_decay["provenance"] = {
        "qat": {"scale_learning": scale_learning_metadata("muon", 0.0)}
    }
    with pytest.raises(ValueError, match="optimizer-group provenance"):
        checkpoint_quantization(mismatched_decay, weights)

