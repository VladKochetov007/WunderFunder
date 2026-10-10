from __future__ import annotations

import hashlib
import json
import os
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest
import torch

SOURCE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE_ROOT / "scripts"))

from train_rnn import (  # noqa: E402
    N_FEATURES,
    Sequences,
    family_output_width,
    has_distillation_family_head,
    make_model,
)

from wnn.data import Sequence as DataSequence  # noqa: E402
from wnn.distillation import (  # noqa: E402
    append_family_output_head,
    extend_family_output_head,
    four_family_weighted_mse,
)

PROJECT_ROOT = Path(os.environ["WNN_PROJECT_ROOT"])
CACHE_ROOT = PROJECT_ROOT / "cache" / "bd26_h230_w384_family_v1"
CACHE_METADATA_SHA256 = "398d7753a127054f3e38cd34c4730099843d60663541787e09ab38c9f45e47ad"
INITIALIZER_PATH = PROJECT_ROOT / "artifacts/runs/c20261003_combo_s3_f0/model_ep3.pt"
INITIALIZER_SHA256 = "d4a1e50b1892b4d9edd7c255908b68073a760c059ca80223e98a63c7e023371e"


def family_args(weight: float) -> Namespace:
    """Create the matched H230 family-head arguments at one loss coefficient."""
    return Namespace(distill_family_head=True, distill_family_weight=weight)


def read_first_cache_row(path: Path) -> np.ndarray:
    """Read one real cache row without mapping its multi-gigabyte array."""
    with path.open("rb") as source:
        version = np.lib.format.read_magic(source)
        if version == (1, 0):
            shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(source)
        elif version == (2, 0):
            shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(source)
        else:
            raise ValueError(f"unsupported cache array version: {version}")
        if len(shape) != 3 or fortran_order or dtype != np.dtype(np.float32):
            raise ValueError("H230 cache row must be C-order three-dimensional float32")
        row_bytes = int(np.prod(shape[1:])) * dtype.itemsize
        payload = source.read(row_bytes)
        if len(payload) != row_bytes:
            raise ValueError("H230 cache ended before its first complete row")
    return np.frombuffer(payload, dtype=dtype).reshape((1, *shape[1:])).copy()


class StaticSequenceFile:
    """Supply a small source row while keeping the actual H230 cache arrays."""

    def __init__(self, group: int, sequence: DataSequence) -> None:
        self.group = group
        self.sequence = sequence

    def read(self, index: int) -> DataSequence:
        assert index == self.group
        return self.sequence


def test_zero_weight_preserves_family_head_and_initial_state() -> None:
    candidate_args = family_args(0.1)
    control_args = family_args(0.0)
    assert has_distillation_family_head(candidate_args)
    assert has_distillation_family_head(control_args)
    assert family_output_width(23, has_distillation_family_head(candidate_args)) == 31
    assert family_output_width(23, has_distillation_family_head(control_args)) == 31

    initializer_digest = hashlib.sha256(INITIALIZER_PATH.read_bytes()).hexdigest()
    assert initializer_digest == INITIALIZER_SHA256
    initializer = torch.load(INITIALIZER_PATH, map_location="cpu", weights_only=False)
    parent_state = initializer["model"]
    assert parent_state["head.2.weight"].shape[0] == 23
    model_args = {
        **initializer["args"],
        "arch": "gru",
        "cell": "gru",
        "enc": "linear",
        "hidden": 256,
        "layers": 2,
        "dropout": 0.0,
    }
    input_width = int(initializer["n_in"])

    states = []
    outputs = []
    for args in (candidate_args, control_args):
        torch.manual_seed(3)
        model = make_model(model_args, n_in=input_width, n_out=23)
        if has_distillation_family_head(args):
            model = append_family_output_head(model)
            model.load_state_dict(extend_family_output_head(parent_state, model.state_dict()))
        states.append(model.state_dict())
        outputs.append(model(torch.randn(2, 7, input_width))[0])

    assert states[0].keys() == states[1].keys()
    for name in states[0]:
        torch.testing.assert_close(states[0][name], states[1][name], rtol=0, atol=0)
    torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
    assert states[0]["head.2.weight"].shape[0] == 31


def test_real_source_cache_reader_preserves_main_and_family_columns(tmp_path: Path) -> None:
    metadata_path = CACHE_ROOT / "metadata.json"
    digest = hashlib.sha256(metadata_path.read_bytes()).hexdigest()
    assert digest == CACHE_METADATA_SHA256
    metadata = json.loads(metadata_path.read_text())
    assert metadata["family_order"] == [
        "strong",
        "full_finetuned",
        "raw_wmse",
        "samefold_baseline",
    ]

    for split in ("train", "valid"):
        candidate_args = family_args(0.1)
        control_args = family_args(0.0)
        row_groups = metadata["files"][f"main_source_{split}"]["row_groups"]
        group = int(row_groups[0])
        main_row = read_first_cache_row(CACHE_ROOT / f"main_source_{split}.npy")
        family_row = read_first_cache_row(CACHE_ROOT / f"families_{split}.npy")
        np.save(tmp_path / f"main_source_{split}.npy", main_row)
        np.save(tmp_path / f"families_{split}.npy", family_row)
        cache_metadata = {"metadata": {"files": {f"main_source_{split}": {"row_groups": [group]}}}}
        datasets = []
        for args in (candidate_args, control_args):
            dataset = Sequences(
                split,
                [group],
                distill_cache_root=tmp_path,
                distill_cache_variant="source",
                distill_cache_metadata=cache_metadata,
                include_family_targets=has_distillation_family_head(args),
            )
            sequence = DataSequence(
                group,
                np.zeros((20_000, N_FEATURES), dtype=np.float32),
                np.full((20_000, 2), 0.25, dtype=np.float32),
                np.ones(20_000, dtype=np.bool_),
            )
            dataset._file = StaticSequenceFile(group, sequence)
            datasets.append(dataset)
        _, candidate_targets, _ = datasets[0][0]
        _, control_targets, _ = datasets[1][0]
        assert candidate_targets.shape == (20_000, 12)
        torch.testing.assert_close(candidate_targets, control_targets, rtol=0, atol=0)
        np.testing.assert_array_equal(candidate_targets[:, 2:4].numpy(), main_row[0])
        np.testing.assert_array_equal(candidate_targets[:, 4:12].numpy(), family_row[0])


def test_zero_family_weight_removes_only_family_loss_gradient() -> None:
    main_prediction = torch.tensor([[[0.2, -0.1], [0.4, 0.3]]], requires_grad=True)
    main_target = torch.zeros_like(main_prediction)
    family_prediction = torch.ones((1, 2, 4, 2), requires_grad=True)
    family_target = torch.zeros_like(family_prediction)
    row_weights = torch.ones_like(main_prediction)
    main_loss = (row_weights * (main_prediction - main_target).square()).sum() / row_weights.sum()
    family_loss = four_family_weighted_mse(family_prediction, family_target, row_weights)

    candidate_loss = main_loss + 0.1 * family_loss
    control_loss = main_loss + 0.0 * family_loss
    assert candidate_loss.item() - control_loss.item() == pytest.approx(0.1 * family_loss.item())

    candidate_gradient = torch.autograd.grad(candidate_loss, family_prediction, retain_graph=True)[
        0
    ]
    control_gradient = torch.autograd.grad(control_loss, family_prediction)[0]
    assert candidate_gradient.abs().sum() > 0
    torch.testing.assert_close(control_gradient, torch.zeros_like(control_gradient), rtol=0, atol=0)
