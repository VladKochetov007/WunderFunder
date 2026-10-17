"""GPU-host trainer/export ABI checks; intentionally not executed on the local Mac."""

from __future__ import annotations

import io
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import train_rnn as training  # noqa: E402

from wnn.architecture_wave import rejection  # noqa: E402

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires target CUDA host"),
]


@pytest.mark.parametrize("arch", ["gru", "causal-transformer", "mamba2"])
def test_trainer_tbptt_and_checkpoint_prediction_abi(arch: str) -> None:
    """Exercise the real factory, chunk adapter, detach and prediction-loader head ABI."""
    torch.manual_seed(17)
    options = {
        "arch": arch,
        "hidden": 64,
        "layers": 2,
        "dropout": 0.0,
        "cell": "gru",
        "enc": "linear",
        "precision": "fp32",
        "device": "cuda",
        "attention_heads": 4,
        "attention_window": 16,
        "attention_ffn_mult": 4,
        "mamba_state_size": 16,
        "mamba_head_dim": 16,
        "mamba_expand": 2,
        "mamba_conv_width": 4,
        "mamba_chunk_size": 16,
    }
    n_in = training.N_FEATURES + training.N_EVENT_FEATURES
    model = training.make_model(options, n_in, 2).cuda().train()
    raw = torch.randn(2, 37, training.N_FEATURES)
    training.fit_normalization(model, [(raw,), (raw * 0.7,)], torch.device("cuda"))
    values = training.build_inputs(raw.cuda())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    state = None
    before = model.head[2].weight.detach().clone()
    for start in range(0, values.shape[1], 13):
        optimizer.zero_grad(set_to_none=True)
        predictions, state = training.forward_chunk(model, values, slice(start, start + 13), state)
        state = training.detach_state(state)
        loss = predictions.square().mean()
        assert torch.isfinite(loss)
        loss.backward()
        gradients = [
            parameter.grad for parameter in model.parameters() if parameter.grad is not None
        ]
        assert gradients and all(bool(torch.isfinite(gradient).all()) for gradient in gradients)
        optimizer.step()
    assert not torch.equal(before, model.head[2].weight)
    model.eval()
    with torch.no_grad():
        expected, _ = model(values)
    stream = io.BytesIO()
    torch.save(
        {
            "args": options,
            "n_in": n_in,
            **training.inference_checkpoint_fields(model, "raw"),
        },
        stream,
    )
    stream.seek(0)
    checkpoint = torch.load(stream, map_location="cuda", weights_only=False)
    restored = (
        training.make_model(
            checkpoint["args"],
            checkpoint["n_in"],
            checkpoint["model"]["head.2.weight"].shape[0],
        )
        .cuda()
        .eval()
    )
    restored.load_state_dict(checkpoint["model"], strict=True)
    restored._inference_fp32 = True
    actual = training.predict(restored, raw, 13, torch.device("cuda"))
    torch.testing.assert_close(torch.from_numpy(actual), expected.cpu(), rtol=5e-4, atol=5e-4)
    assert torch.equal(restored.mean, model.mean)
    assert torch.equal(restored.std, model.std)


@pytest.mark.parametrize("arch", ["causal-transformer", "mamba2"])
def test_scratch_backend_rejects_ignored_recurrent_modifiers(arch: str) -> None:
    """An incompatible initializer/optimizer must not create a mislabeled scratch arm."""
    assert rejection({"arch": arch}) is None
    for option, value in (("init", "parent"), ("optim", "muon"), ("dropout", 0.2)):
        assert rejection({"arch": arch, option: value}) is not None
