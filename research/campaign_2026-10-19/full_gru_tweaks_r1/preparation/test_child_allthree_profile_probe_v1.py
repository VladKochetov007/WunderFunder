from __future__ import annotations

import hashlib
import json
from argparse import Namespace
from pathlib import Path

import pytest
import torch
from child_allthree_profile_contract_v1 import (
    PARENT_EP3_SHA256,
    PARENT_INVOCATION,
    PARENT_UNIT,
    require_sha256,
    validate_exact_stop,
    validate_inherited_preprocessing,
    validate_parent_acceptance,
    validate_profile_no_short_circuit,
    verify_parent_extension,
)
from child_allthree_profile_probe_v1 import _load_parent_state


def _state() -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    parent = {
        "inp.weight": torch.ones((256, 211)),
        "inp.bias": torch.ones(256),
        "gru.weight": torch.full((768, 256), 2.0),
        "head.2.weight": torch.full((23, 256), 3.0),
        "head.2.bias": torch.full((23,), 4.0),
        "mean": torch.full((211,), 5.0),
        "std": torch.full((211,), 6.0),
    }
    child = {name: value.clone() for name, value in parent.items()}
    child["inp.weight"] = torch.cat((parent["inp.weight"], torch.zeros((256, 256))), dim=1)
    child["head.2.weight"] = torch.cat((parent["head.2.weight"], torch.zeros((2, 256))), dim=0)
    child["head.2.bias"] = torch.cat((parent["head.2.bias"], torch.zeros(2)))
    child["mean"] = torch.cat((parent["mean"], torch.zeros(256)))
    child["std"] = torch.cat((parent["std"], torch.ones(256)))
    return parent, child


def _write_json(path: Path, value: dict[str, object]) -> str:
    data = json.dumps(value, sort_keys=True).encode()
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def test_parent_extension_preserves_trunk_prefix_and_zero_suffixes() -> None:
    parent, child = _state()

    result = verify_parent_extension(parent, child)

    assert result["parent_trunk_and_prefix_equal"] is True
    assert result["new_input_columns_zero"] is True
    assert result["new_logvariance_rows_zero"] is True
    assert result["normalization_suffix_mean_zero_std_one"] is True


@pytest.mark.parametrize(
    ("name", "index", "value"),
    [
        ("inp.weight", (0, 211), 1.0),
        ("gru.weight", (0, 0), 9.0),
        ("head.2.bias", (23,), 1.0),
        ("mean", (211,), 1.0),
        ("std", (211,), 0.0),
    ],
)
def test_parent_extension_rejects_changed_or_nonzero_suffix(
    name: str, index: tuple[int, ...], value: float
) -> None:
    parent, child = _state()
    child[name][index] = value

    with pytest.raises((ValueError, FloatingPointError)):
        verify_parent_extension(parent, child)


def test_parent_acceptance_requires_exact_parent_and_checkpoint(tmp_path: Path) -> None:
    acceptance = {
        "status": "ACCEPTED",
        "selected_epoch": 3,
        "successful_updates": 29840,
        "nonfinite_updates": 0,
        "strict_fp32_finite_model_optimizer_verified": True,
        "main_pid": 0,
        "exec_main_code": 1,
        "exec_main_status": 0,
        "native_unit": {
            "unit": PARENT_UNIT,
            "InvocationID": PARENT_INVOCATION,
            "ActiveState": "active",
            "SubState": "exited",
            "MainPID": "0",
            "ExecMainCode": "1",
            "ExecMainStatus": "0",
        },
        "files": {"model_ep3.pt": {"sha256": PARENT_EP3_SHA256}},
    }
    path = tmp_path / "acceptance.json"
    receipt_sha = _write_json(path, acceptance)

    assert validate_parent_acceptance(path, receipt_sha, PARENT_EP3_SHA256)["selected_epoch"] == 3
    acceptance["selected_epoch"] = 2
    _write_json(path, acceptance)
    with pytest.raises(ValueError, match="fixed-epoch-3"):
        validate_parent_acceptance(
            path, hashlib.sha256(path.read_bytes()).hexdigest(), PARENT_EP3_SHA256
        )


def test_parent_acceptance_rejects_bool_for_typed_exit_integer(tmp_path: Path) -> None:
    acceptance = {
        "status": "ACCEPTED",
        "selected_epoch": 3,
        "successful_updates": 29840,
        "nonfinite_updates": 0,
        "strict_fp32_finite_model_optimizer_verified": True,
        "main_pid": False,
        "exec_main_code": 1,
        "exec_main_status": 0,
        "native_unit": {
            "unit": PARENT_UNIT,
            "InvocationID": PARENT_INVOCATION,
            "ActiveState": "active",
            "SubState": "exited",
            "MainPID": "0",
            "ExecMainCode": "1",
            "ExecMainStatus": "0",
        },
        "files": {"model_ep3.pt": {"sha256": PARENT_EP3_SHA256}},
    }
    path = tmp_path / "acceptance.json"
    receipt_sha = _write_json(path, acceptance)

    with pytest.raises(ValueError, match="typed successful native terminal fields"):
        validate_parent_acceptance(path, receipt_sha, PARENT_EP3_SHA256)


def test_exact_stop_requires_dead_parent_and_empty_cgroup(tmp_path: Path) -> None:
    stop = {
        "unit": PARENT_UNIT,
        "invocation_id": PARENT_INVOCATION,
        "before": {
            "InvocationID": PARENT_INVOCATION,
            "ActiveState": "active",
            "SubState": "exited",
            "MainPID": "0",
        },
        "stop": {"returncode": 0},
        "after": {
            "LoadState": "not-found",
            "ActiveState": "inactive",
            "SubState": "dead",
            "MainPID": "0",
        },
        "populated_after": {},
    }
    path = tmp_path / "stop.json"
    receipt_sha = _write_json(path, stop)

    assert validate_exact_stop(path, receipt_sha)["populated_after"] == {}
    stop["populated_after"] = {"pid": "123"}
    _write_json(path, stop)
    with pytest.raises(ValueError, match="does not prove"):
        validate_exact_stop(path, hashlib.sha256(path.read_bytes()).hexdigest())


@pytest.mark.parametrize("value", ["", "0" * 64, "PENDING", "a" * 63, "g" * 64])
def test_sha_gate_rejects_missing_or_placeholder_identity(value: str) -> None:
    with pytest.raises(ValueError):
        require_sha256(value, "fixture")


@pytest.mark.parametrize("flag", ["student_frontier_preflight_only", "calibrate_distill_reference"])
def test_profile_rejects_canonical_zero_update_short_circuits(flag: str) -> None:
    args = Namespace(student_frontier_preflight_only=False, calibrate_distill_reference=False)
    setattr(args, flag, True)

    with pytest.raises(ValueError, match="short-circuits"):
        validate_profile_no_short_circuit(args)


def test_profile_requires_parent_preprocessing_inheritance() -> None:
    args = Namespace(inherit_init_preprocessing=True)
    validate_inherited_preprocessing(args)
    args.inherit_init_preprocessing = False

    with pytest.raises(ValueError, match="inherited parent preprocessing"):
        validate_inherited_preprocessing(args)


def test_parent_state_uses_pinned_weights_only_reader(tmp_path: Path) -> None:
    state = {"inp.weight": torch.zeros((256, 211)), "mean": torch.zeros(211)}
    path = tmp_path / "model_ep3.pt"
    torch.save(
        {
            "args": {
                "seed": 1001,
                "hidden": 256,
                "layers": 2,
                "representation_mode": "none",
                "mean_variance_head": 0,
            },
            "model": state,
        },
        path,
    )
    expected_sha = hashlib.sha256(path.read_bytes()).hexdigest()

    loaded = _load_parent_state(path, expected_sha)

    assert loaded.keys() == state.keys()
    assert all(torch.equal(loaded[name], state[name]) for name in state)
