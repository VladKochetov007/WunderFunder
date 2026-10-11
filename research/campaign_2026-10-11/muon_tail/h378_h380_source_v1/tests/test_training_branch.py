from __future__ import annotations

import copy
import json
import random
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset
from wnn.training_branch import (
    TailCosineScheduler,
    contract_source_digest,
    restore_branch_resume_options,
    tail_auxiliary_coefficients,
    validate_branch_contracts,
    validate_parent_source_binding,
    validate_parent_state_schema,
)

from wnn.nn import RNNModel
from wnn.optim import MuonAdamW, muon_param_groups
from wnn.training_state import (
    load_training_state,
    loader_state,
    restore_rng,
    restore_training_state,
)

SOURCE_ROOT = Path(__file__).resolve().parents[1]
PARENT_STATE = Path(
    "/home/vlad/development/wunder/wnn_connectome_starterpack/artifacts/runs/"
    "r3_muonall256prop0_f1_s3_e3/training_state.pt"
)


def _branch_contracts() -> tuple[dict, dict]:
    parent = {
        "epochs": 3,
        "total_steps": 30,
        "options": {"epochs": 3, "checkpoint_epoch": 2, "precision": "fp16", "aux_struct": 0.3},
        "sources": {"trainer": "parent"},
        "fit_membership": {"train": "fixed"},
    }
    child = copy.deepcopy(parent)
    child["epochs"] = 6
    child["total_steps"] = 60
    child["sources"] = {"trainer": "child"}
    child["options"]["epochs"] = 6
    child["options"]["checkpoint_epoch"] = 5
    return parent, child


def test_branch_contract_accepts_only_declared_schedule_horizon() -> None:
    parent, child = _branch_contracts()
    validate_branch_contracts(parent, child, tail_epochs=3, tail_steps=30, parent_next_epoch=3)

    changed = copy.deepcopy(child)
    changed["options"]["precision"] = "bf16"
    with pytest.raises(ValueError, match="precision"):
        validate_branch_contracts(
            parent, changed, tail_epochs=3, tail_steps=30, parent_next_epoch=3
        )

    changed = copy.deepcopy(child)
    changed["fit_membership"]["train"] = "different"
    with pytest.raises(ValueError, match="fit_membership"):
        validate_branch_contracts(
            parent, changed, tail_epochs=3, tail_steps=30, parent_next_epoch=3
        )

    changed = copy.deepcopy(child)
    changed["options"]["aux_struct"] = 0
    with pytest.raises(ValueError, match="aux_struct"):
        validate_branch_contracts(
            parent, changed, tail_epochs=3, tail_steps=30, parent_next_epoch=3
        )

    changed = copy.deepcopy(child)
    changed["total_steps"] += 1
    with pytest.raises(ValueError, match="append exactly"):
        validate_branch_contracts(
            parent, changed, tail_epochs=3, tail_steps=30, parent_next_epoch=3
        )


def test_tail_scheduler_cosine_and_fixed_restored_beta1() -> None:
    optimizer = torch.optim.SGD(
        [torch.nn.Parameter(torch.ones(1))],
        lr=0.004,
        momentum=0.95,
        weight_decay=0.01,
    )
    optimizer.param_groups[0]["betas"] = (0.95, 0.999)
    optimizer.param_groups[0]["lr_scale"] = 0.5
    scheduler = TailCosineScheduler(optimizer, total_steps=3, peak_lr=3e-5, final_lr=3e-7)
    assert scheduler.get_last_lr() == pytest.approx([1.5e-5])
    for _ in range(2):
        scheduler.step()
    assert scheduler.get_last_lr() == pytest.approx([1.5e-7])
    assert optimizer.param_groups[0]["betas"] == (0.95, 0.999)
    assert optimizer.param_groups[0]["weight_decay"] == 0.01

    saved = scheduler.state_dict()
    restored = TailCosineScheduler(optimizer, total_steps=3, peak_lr=3e-5, final_lr=3e-7)
    restored.load_state_dict(saved)
    assert restored.position == 2
    assert restored.get_last_lr() == pytest.approx([1.5e-7])


def test_auxiliary_tail_decays_without_mutating_declared_base_coefficients() -> None:
    aux_struct = aux_next = 0.3
    assert tail_auxiliary_coefficients(
        aux_struct=aux_struct, aux_next=aux_next, mode="linear-to-zero", position=0, total_steps=3
    ) == pytest.approx((0.3, 0.3))
    assert tail_auxiliary_coefficients(
        aux_struct=aux_struct, aux_next=aux_next, mode="linear-to-zero", position=2, total_steps=3
    ) == pytest.approx((0.0, 0.0))
    assert (aux_struct, aux_next) == (0.3, 0.3)


def test_branch_resume_rehydrates_full_child_horizon_and_rejects_drift() -> None:
    child_contract = {
        "options": {
            "epochs": 6,
            "checkpoint_policy": "fixed",
            "checkpoint_epoch": 5,
            "precision": "fp16",
            "split_manifest_sha256": "derived-by-manifest-load",
            "fit_groups_manifest_sha256": "derived-by-inner-manifest-load",
        }
    }
    args = Namespace(
        epochs=2,
        checkpoint_policy="best",
        checkpoint_epoch=None,
        precision="fp16",
        save_training_state=False,
    )
    restore_branch_resume_options(args, child_contract, explicit_fields=set())
    assert (args.epochs, args.checkpoint_policy, args.checkpoint_epoch) == (6, "fixed", 5)
    assert args.save_training_state

    drifted = Namespace(
        epochs=7,
        checkpoint_policy="best",
        checkpoint_epoch=None,
        precision="fp16",
        save_training_state=False,
    )
    with pytest.raises(ValueError, match="epochs"):
        restore_branch_resume_options(drifted, child_contract, explicit_fields={"epochs"})


@pytest.mark.skipif(not PARENT_STATE.is_file(), reason="campaign parent artifact is not mounted")
def test_actual_parent_state_binds_producer_and_restores_full_prefix_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = load_training_state(PARENT_STATE)
    contract, next_epoch = validate_parent_state_schema(state)
    binding_path = SOURCE_ROOT / "PARENT_SOURCE_BINDING_V1.json"
    binding = json.loads(binding_path.read_text())
    source_digest = validate_parent_source_binding(contract["sources"], binding)
    assert source_digest == contract_source_digest(binding["contract_sources"])
    assert next_epoch == 3
    assert contract["epochs"] == 3
    assert contract["total_steps"] == 22_380
    assert contract["options"]["optim"] == "muon"
    assert contract["options"]["muon_scope"] == "all"
    assert contract["options"]["precision"] == "fp16"
    assert contract["options"]["wd"] == pytest.approx(0.01)
    assert contract["options"]["aux_struct"] == pytest.approx(0.3)
    assert contract["options"]["aux_next"] == pytest.approx(0.3)
    assert state["loader"]["num_workers"] == 1
    assert state["loader"]["dataset_length"] == 11_949
    assert len(state["rng"]["cuda"]) == 1

    options = contract["options"]
    model = RNNModel(
        state["model"]["mean"].numel(),
        options["hidden"],
        options["layers"],
        options["dropout"],
        n_out=state["model"]["head.2.weight"].shape[0],
        cell=options["cell"],
        enc=options["enc"],
        tok_dim=options["tok_dim"],
        att_layers=options["att_layers"],
        conv_k=options["conv_k"],
        readout_activation=options["readout_activation"],
    )
    optimizer = MuonAdamW(
        muon_param_groups(model, options["muon_scope"]),
        lr=options["lr"],
        betas=(0.9, options["beta2"]),
        weight_decay=options["wd"],
    )
    scaler = torch.amp.GradScaler("cpu", enabled=True)
    loader = DataLoader(
        TensorDataset(torch.zeros(state["loader"]["dataset_length"], dtype=torch.uint8)),
        batch_size=state["loader"]["batch_size"],
        shuffle=True,
        drop_last=state["loader"]["drop_last"],
        num_workers=state["loader"]["num_workers"],
    )

    class TrackingScheduler:
        loaded = None

        def load_state_dict(self, saved: dict) -> None:
            self.loaded = saved

    scheduler = TrackingScheduler()
    restore_training_state(
        state,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        loader=loader,
    )
    assert scheduler.loaded == state["scheduler"]
    assert len(optimizer.param_groups) == len(state["optimizer"]["param_groups"]) == 3
    assert [len(group["params"]) for group in optimizer.param_groups] == [7, 4, 3]
    assert [group["betas"][0] for group in optimizer.param_groups] == pytest.approx([0.95] * 3)
    assert [group["weight_decay"] for group in optimizer.param_groups] == pytest.approx([0.01] * 3)
    assert scaler.get_scale() == state["scaler"]["scale"]
    assert scaler.state_dict() == state["scaler"]
    assert all(torch.equal(model.state_dict()[key], value) for key, value in state["model"].items())

    restored_optimizer_state = optimizer.state_dict()
    saved_optimizer_state = state["optimizer"]
    assert len(restored_optimizer_state["param_groups"]) == len(
        saved_optimizer_state["param_groups"]
    )
    for saved_group, restored_group in zip(
        saved_optimizer_state["param_groups"], restored_optimizer_state["param_groups"], strict=True
    ):
        assert saved_group["params"] == restored_group["params"]
        for parameter_id in saved_group["params"]:
            saved_slot = saved_optimizer_state["state"].get(parameter_id, {})
            restored_slot = restored_optimizer_state["state"].get(parameter_id, {})
            assert saved_slot.keys() == restored_slot.keys()
            for field, expected in saved_slot.items():
                actual = restored_slot[field]
                if isinstance(expected, torch.Tensor):
                    assert torch.equal(actual, expected), (parameter_id, field)
                else:
                    assert actual == expected, (parameter_id, field)

    restored_loader_state = loader_state(loader)
    for field in (
        "sampler_type",
        "batch_size",
        "drop_last",
        "num_workers",
        "dataset_length",
        "sampler_state",
        "sampler_epoch",
    ):
        assert restored_loader_state[field] == state["loader"][field]
    for field in ("loader_generator", "sampler_generator"):
        expected = state["loader"][field]
        actual = restored_loader_state[field]
        if expected is None:
            assert actual is None
        else:
            assert torch.equal(actual, expected)

    class ForbiddenScheduler:
        def load_state_dict(self, saved: dict) -> None:
            raise AssertionError("branch restore must discard the parent scheduler")

    restore_training_state(
        state,
        model=model,
        optimizer=optimizer,
        scheduler=ForbiddenScheduler(),
        scaler=scaler,
        loader=loader,
        restore_scheduler=False,
    )
    assert [group["betas"][0] for group in optimizer.param_groups] == pytest.approx([0.95] * 3)

    cpu_rng = dict(state["rng"])
    cpu_rng["cuda"] = []  # keep this CPU compatibility test from touching accelerator RNG state
    restore_rng(cpu_rng)
    actual_torch = torch.rand(4)
    torch.set_rng_state(cpu_rng["torch"])
    assert torch.equal(actual_torch, torch.rand(4))
    observed_rng = {}
    monkeypatch.setattr(random, "setstate", lambda value: observed_rng.setdefault("python", value))
    monkeypatch.setattr(
        np.random, "set_state", lambda value: observed_rng.setdefault("numpy", value)
    )
    restore_rng(cpu_rng)
    assert observed_rng["python"] == cpu_rng["python"]
    assert observed_rng["numpy"][0] == cpu_rng["numpy"][0]
    assert np.array_equal(observed_rng["numpy"][1], cpu_rng["numpy"][1])
