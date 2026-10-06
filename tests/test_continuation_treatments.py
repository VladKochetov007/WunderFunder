from __future__ import annotations

import argparse
import importlib.util
import io
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch
import torch.nn.functional as F

from wnn.nn import FixedReadoutMixture, RNNModel
from wnn.training_controls import blended_target

ROOT = Path(__file__).resolve().parents[1]
NIGHT = ROOT / "research/campaign_2026-10-05/continuation_night"
sys.path.insert(0, str(ROOT / "scripts"))
import export_onnx  # noqa: E402
import package_submission  # noqa: E402
import train_rnn as training  # noqa: E402


def _original(relative: str) -> ModuleType:
    specification = importlib.util.spec_from_file_location("treatment_reference", NIGHT / relative)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _arguments(activation: str = "gelu") -> dict[str, object]:
    return {"hidden": 4, "layers": 2, "dropout": 0.0, "readout_activation": activation}


def test_default_model_is_bitwise_historical_through_gradient_and_update() -> None:
    historical = _original("activation_transfer/original_nn.py")
    torch.manual_seed(31)
    reference = historical.RNNModel(3, hidden=4, layers=2)
    reference_rng = torch.get_rng_state().clone()
    torch.manual_seed(31)
    current = RNNModel(3, hidden=4, layers=2)
    assert torch.equal(torch.get_rng_state(), reference_rng)
    assert reference.state_dict().keys() == current.state_dict().keys()
    for key, tensor in reference.state_dict().items():
        assert torch.equal(tensor, current.state_dict()[key])
    inputs = torch.randn(2, 9, 3)
    target = torch.randn(2, 9, 2)
    for model in (reference, current):
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        output, hidden = model(inputs)
        (output - target).square().sum().backward()
        model.observed = (output.detach(), hidden.detach())
        optimizer.step()
    for expected, actual in zip(reference.observed, current.observed, strict=True):
        assert torch.equal(expected, actual)
    for expected, actual in zip(reference.parameters(), current.parameters(), strict=True):
        assert torch.equal(expected.grad, actual.grad)
        assert torch.equal(expected, actual)


def test_mixture_adds_no_parameters_or_rng_and_changes_only_readout() -> None:
    torch.manual_seed(41)
    baseline = RNNModel(3, hidden=4, layers=2)
    rng = torch.get_rng_state().clone()
    torch.manual_seed(41)
    candidate = RNNModel(3, hidden=4, layers=2, readout_activation="fixed_mix")
    assert torch.equal(rng, torch.get_rng_state())
    assert candidate.state_dict().keys() == baseline.state_dict().keys()
    for key, tensor in baseline.state_dict().items():
        assert torch.equal(tensor, candidate.state_dict()[key])
    inputs = torch.randn(2, 9, 3)
    projected = baseline.encode(inputs)
    recurrent, expected_state = baseline.gru(projected)
    preactivation = baseline.head[0](torch.cat([recurrent, projected], dim=-1))
    negative = torch.where(preactivation >= 0, preactivation, 0.01 * preactivation)
    expected = baseline.head[2](0.5 * F.gelu(preactivation) + 0.5 * negative)
    actual, state = candidate(inputs)
    assert torch.equal(expected, actual)
    assert torch.equal(expected_state, state)
    first, first_state = candidate(inputs[:, :4])
    last, _ = candidate(inputs[:, 4:], first_state)
    torch.testing.assert_close(torch.cat([first, last], dim=1), actual)
    perturbed = inputs.clone()
    perturbed[:, 4:] += 10
    torch.testing.assert_close(candidate(perturbed)[0][:, :4], actual[:, :4])


def test_checkpoint_roundtrip_and_parent_head_inheritance() -> None:
    arguments = _arguments("fixed_mix")
    model = training.make_model(arguments, 3, 2)
    checkpoint = {"args": arguments, "model": model.state_dict()}
    stream = io.BytesIO()
    torch.save(checkpoint, stream)
    stream.seek(0)
    restored = torch.load(stream, weights_only=False)
    inherited = argparse.Namespace(**_arguments())
    training.inherit_init_arguments(inherited, restored, [])
    assert inherited.readout_activation == "fixed_mix"
    copied = training.make_model(vars(inherited), 3, 2)
    copied.load_state_dict(restored["model"])
    assert isinstance(copied.head[1], FixedReadoutMixture)
    inputs = torch.randn(1, 7, 3)
    assert torch.equal(model(inputs)[0], copied(inputs)[0])
    with pytest.raises(ValueError, match="disagrees with the parent"):
        training.inherit_init_arguments(
            argparse.Namespace(**_arguments()), restored, ["--readout-activation", "gelu"]
        )
    old_args = _arguments()
    del old_args["readout_activation"]
    assert isinstance(training.make_model(old_args, 3, 2).head[1], torch.nn.GELU)


def test_reanchor_boundary_algebra_clipping_and_origin_weights() -> None:
    labels = torch.tensor([[[4.0, -4.0], [0.3, -0.4], [0.6, -0.7], [0.9, -1.0]]])
    teacher = torch.tensor([[[3.0, -3.0], [0.1, 0.2], [0.5, 0.6], [0.7, 0.8]]])
    changes = torch.tensor([[3, 2, 4, 4]])
    untouched = labels.clone()
    target = blended_target(labels, teacher, changes, slice(0, 4), 0.5, 1, 0.1)
    clipped = labels.clamp(-2, 2)
    expected = torch.empty_like(target)
    for row in range(4):
        if row in (0, 2):
            expected[:, row] = (
                0.45 * teacher[:, row].clamp(-2, 2)
                + 0.45 * teacher[:, row + 1].clamp(-2, 2)
                + 0.1 * clipped[:, row]
            )
        else:
            expected[:, row] = 0.5 * clipped[:, row] + 0.5 * teacher[:, row].clamp(-2, 2)
    torch.testing.assert_close(target, expected)
    assert torch.equal(labels, untouched)
    baseline = blended_target(labels, teacher, changes, slice(0, 4), 0.5, 1)
    assert torch.equal(target[:, [1, 3]], baseline[:, [1, 3]])
    historical = _original("supervision/original_training_controls.py")
    assert torch.equal(
        baseline, historical.blended_target(labels, teacher, changes, slice(0, 4), 0.5, 1)
    )
    assert torch.equal(
        blended_target(labels, teacher, None, slice(1, 4), 0.5, 0),
        historical.blended_target(labels, teacher, None, slice(1, 4), 0.5, 0),
    )


@pytest.mark.parametrize("coefficient", [-0.1, 1.1, float("nan"), float("inf")])
def test_reanchor_rejects_invalid_coefficients(coefficient: float) -> None:
    tensor = torch.ones(1, 4, 2)
    with pytest.raises(ValueError, match="reanchor must"):
        blended_target(tensor, tensor, None, slice(0, 4), 0.5, 1, coefficient)


@pytest.mark.parametrize(
    "options",
    [
        ["--distill-reanchor", "nan"],
        ["--distill-reanchor", "0.1"],
        ["--distill-reanchor", "0.1", "--distill", "teacher", "--distill-boot-h", "0"],
        [
            "--distill-reanchor",
            "0.1",
            "--distill",
            "teacher",
            "--distill-boot-h",
            "8",
            "--distill-alpha-far",
            "0.8",
        ],
        ["--readout-activation", "fixed_mix", "--arch", "tcn"],
    ],
)
def test_invalid_treatment_cli_exits_before_data(
    options: list[str], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "argv", ["train_rnn.py", "--name", "invalid_treatment", *options])
    with pytest.raises(SystemExit) as failure:
        training.main()
    assert failure.value.code == 2
    assert "error:" in capsys.readouterr().err


def test_unsupported_readout_cannot_silently_export_or_replace_package(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = {"args": {"readout_activation": "fixed_mix"}}
    monkeypatch.setattr(torch, "load", lambda *_args, **_kwargs: checkpoint)
    monkeypatch.setattr(sys, "argv", ["export_onnx.py", "candidate"])
    with pytest.raises(SystemExit, match="non-GELU"):
        export_onnx.main()
    output = tmp_path / "weights.npz"
    for run in ("candidate", "candidate@ep3", "candidate@ep3|head@ep0"):
        with pytest.raises(ValueError, match="non-GELU"):
            package_submission.export(run, output)
        assert not output.exists()
    monkeypatch.setattr(package_submission, "ROOT", tmp_path)
    retained = tmp_path / "submissions/candidate/pkg/retained.txt"
    retained.parent.mkdir(parents=True)
    retained.write_text("old package remains intact")
    monkeypatch.setattr(sys, "argv", ["package_submission.py", "candidate", "candidate:1"])
    with pytest.raises(ValueError, match="non-GELU"):
        package_submission.main()
    assert retained.read_text() == "old package remains intact"
