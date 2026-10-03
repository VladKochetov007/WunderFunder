from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

HERE = Path(__file__).resolve().parent


def trainer_module() -> ModuleType:
    """Load candidate for read-only CLI checks; importing does not fit/load data."""
    spec = importlib.util.spec_from_file_location(
        "opt2_trainer_candidate", HERE / "trainer_candidate.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("candidate module unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_antithetic_normalization_matches_seeded_control() -> None:
    """Exercise real normalization with identical arm membership and ordering."""
    from sampling import EpochOrderSampler

    from wnn.nn import RNNModel

    tree = ast.parse((HERE.parents[2] / "scripts/train_rnn.py").read_text())
    node = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "fit_normalization"
    )
    namespace = {"torch": torch, "build_inputs": lambda value: value}
    exec(
        compile(ast.Module(body=[node], type_ignores=[]), "normalization_regression", "exec"),
        namespace,
    )
    data = TensorDataset(torch.arange(64.0).reshape(64, 1, 1).expand(-1, 7, 1))
    models = [RNNModel(1, 4, 1), RNNModel(1, 4, 1)]
    for model, reverse in zip(models, (False, True), strict=True):
        sampler = EpochOrderSampler(64, 3, reverse)
        sampler.set_epoch(-1)
        loader = DataLoader(data, batch_size=8, sampler=sampler, drop_last=True)
        namespace["fit_normalization"](model, loader, torch.device("cpu"))
    assert torch.equal(models[0].mean, models[1].mean)
    assert torch.equal(models[0].std, models[1].std)


@pytest.mark.parametrize(
    "flags,message",
    [
        (["--zoneout", "0.1"], "exact zoneout is unsupported"),
        (["--batch-order", "qmc"], "integration is deferred"),
        (["--antithetic"], "requires --batch-order seeded"),
        (["--init-qmc", "--arch", "tcn"], "fresh plain GRU/LSTM"),
    ],
)
def test_candidate_rejects_unsupported_before_data(
    flags: list[str],
    message: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = trainer_module()
    monkeypatch.setattr(sys, "argv", ["trainer_candidate.py", "--name", "tiny", *flags])
    with pytest.raises(SystemExit) as error:
        module.main()
    assert error.value.code == 2
    assert message in capsys.readouterr().err


def test_candidate_help_lists_supported_flags(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    module = trainer_module()
    monkeypatch.setattr(sys, "argv", ["trainer_candidate.py", "--help"])
    with pytest.raises(SystemExit) as error:
        module.main()
    assert error.value.code == 0
    help_text = capsys.readouterr().out
    for flag in (
        "--optim-switch",
        "--warmup-frac",
        "--head-lr-mult",
        "--clip-per-module",
        "--grad-noise",
        "--init-qmc",
        "--antithetic",
    ):
        assert flag in help_text
    assert "sgdr" in help_text


def test_loader_budget_arguments_preserved() -> None:
    before = ast.parse((HERE / "baseline/train_rnn.py").read_text())
    after = ast.parse((HERE / "trainer_candidate.py").read_text())

    def calls(tree: ast.AST) -> list[str]:
        return [
            ast.dump(node)
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in {"dataloader_budget", "DataLoader"}
        ]

    assert calls(before) == calls(after)
