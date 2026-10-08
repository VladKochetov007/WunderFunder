from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

PACKET_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = PACKET_ROOT / "code"
sys.path.insert(0, str(CODE_ROOT / "scripts"))
sys.path.insert(0, str(CODE_ROOT / "src"))

import loss_pair_gpu_worker as worker  # noqa: E402
import train_rnn as trainer  # noqa: E402

from wnn.nn import RNNModel  # noqa: E402
from wnn.training_sampling import EpochOrderSampler  # noqa: E402


def _cache_fixture() -> SimpleNamespace:
    fit_rows = [
        {
            "source": "datasets/train.parquet",
            "row_group": index,
            "sequence_id": f"train:{index:04d}",
        }
        for index in range(350)
    ]
    eval_rows = [
        {
            "source": "datasets/valid.parquet",
            "row_group": index,
            "sequence_id": f"valid:{index:04d}",
        }
        for index in range(468)
    ]
    return SimpleNamespace(
        membership={"memberships": {"fit_train": fit_rows, "evaluation_valid": eval_rows}},
        manifest={"membership_artifact": {"sha256": "a" * 64}},
    )


def test_actual_trainer_producers_write_tpu_contract_fixture(tmp_path, monkeypatch) -> None:
    arm_dir = tmp_path / "wmse"
    arm_dir.mkdir()
    monkeypatch.setenv("WNN_LOSS_PAIR_ARM_DIR", str(arm_dir))
    monkeypatch.setenv("WNN_LOSS_PAIR_RUN_NAME", "fixture_wmse")
    loader = DataLoader(
        TensorDataset(torch.zeros(350, 1)),
        batch_size=32,
        sampler=EpochOrderSampler(350, seed=20261009, batch_size=32, drop_last=False),
        drop_last=False,
    )
    model = RNNModel(211, hidden=8, layers=1, cell="gru", enc="linear")
    trainer.export_loss_pair_prefit_artifacts(model, loader, np.arange(350), _cache_fixture())
    prefit = arm_dir / "prefit"
    assert {path.name for path in prefit.iterdir()} == {
        "initial_state.npz",
        "normalization.npz",
        "normalization.json",
        "sample_orders.json",
        "prefit_receipt.json",
    }

    row_mask = torch.arange(2000)[None, :] >= 99
    trainer.export_loss_pair_profile_inputs(
        torch.zeros(32, 2000, 211),
        torch.zeros(32, 2000, 2),
        row_mask.expand(32, -1),
        arm_dir,
    )
    with np.load(prefit / "profile_inputs.npz", allow_pickle=False) as arrays:
        assert set(arrays.files) == {"inputs", "targets", "row_mask"}
        assert arrays["inputs"].shape == (32, 2000, 211)
        assert arrays["inputs"].dtype == np.dtype("<f4")
        assert arrays["targets"].shape == (32, 2000, 2)
        assert arrays["targets"].dtype == np.dtype("<f4")
        assert arrays["row_mask"].shape == (32, 2000)
        assert arrays["row_mask"].dtype == np.dtype("|b1")
        assert np.array_equal(
            arrays["row_mask"], np.broadcast_to(np.arange(2000) >= 99, (32, 2000))
        )
    orders = json.loads((prefit / "sample_orders.json").read_text(encoding="utf-8"))
    assert orders["schema"] == "wnn.loss-pair.sample-orders.v1"
    assert set(orders["epochs"]) == {"-1", "0", "1"}
    assert all(len(order) == 350 for order in orders["epochs"].values())
    assert len(orders["evaluation_ids"]) == 468


def test_gpu_arms_differ_only_by_loss_and_use_matched_fp32(tmp_path) -> None:
    validation = tmp_path / "validation.json"
    args = worker._common_train_argv(
        "pair", "wmse", tmp_path / "manifest.json", validation, "b" * 64
    )
    huber = worker._common_train_argv(
        "pair", "huber", tmp_path / "manifest.json", validation, "b" * 64
    )
    loss_index = args.index("--loss")
    assert args[loss_index + 1] == "wmse"
    assert huber[loss_index + 1] == "huber"
    args[loss_index + 1] = huber[loss_index + 1]
    assert args == huber
    assert args[args.index("--precision") + 1] == "fp32"
