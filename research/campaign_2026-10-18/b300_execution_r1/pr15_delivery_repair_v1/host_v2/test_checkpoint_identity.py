"""CPU-only Torch tests for the isolated PR15 delivery validator repair."""

from __future__ import annotations

import copy
from collections import OrderedDict
from pathlib import Path

import pytest
from checkpoint_identity import (
    sha256_file,
    validate_prediction_pin,
    validate_provenance_file_hashes,
    verify_checkpoint_pair,
)


def _torch():
    return pytest.importorskip("torch", reason="run this validator test on CPU Torch")


def _checkpoint(torch, *, seed: int = 1001, budget: int = 44_760):
    state = OrderedDict(
        (
            ("mean", torch.zeros(271, dtype=torch.float32)),
            ("std", torch.ones(271, dtype=torch.float32)),
            ("gru.weight", torch.arange(12, dtype=torch.float32).reshape(3, 4)),
        )
    )
    state._metadata = OrderedDict((("", {"version": 1}),))
    return {
        "model": state,
        "epoch": 2,
        "n_in": 271,
        "args": {
            "name": "pr15_baseline_f0_s1001",
            "seed": seed,
            "valid_fold": 0,
            "checkpoint_epoch": 2,
            "research_feature_control": "all",
            "precision": "fp32",
            "split_manifest_sha256": "a" * 64,
            "source_manifest_sha256": "b" * 64,
        },
        "research_feature_prior": {
            "schema": "causal_geometry_discrete_v1",
            "fit_rows": 42,
            "atoms": [1.0, 2.0, 3.0, 4.0],
        },
        "provenance": {
            "split_manifest_sha256": "a" * 64,
            "source_manifest_sha256": "b" * 64,
            "cumulative_updates": {
                "optimizer_attempts": budget,
                "optimizer_completed_updates": budget,
                "amp_skipped_updates": 0,
            },
            "trainer_implementation": {"sha256": "c" * 64},
        },
    }


def _save_pair(tmp_path: Path, torch, left, right=None):
    fixed = tmp_path / "model_ep2.pt"
    selected = tmp_path / "model.pt"
    torch.save(left, fixed)
    torch.save(left if right is None else right, selected)
    return fixed, selected


def test_separate_torch_archives_accept_exact_full_payload_and_keep_raw_hashes(tmp_path):
    torch = _torch()
    checkpoint = _checkpoint(torch)
    fixed, selected = _save_pair(tmp_path, torch, checkpoint)

    # The real trainer saves the same object in separate calls under two names.
    # Torch archive container bytes can differ even when the loaded payload is exact.
    identity = verify_checkpoint_pair(fixed, selected)
    assert identity.fixed_file_sha256 == sha256_file(fixed)
    assert identity.selected_file_sha256 == sha256_file(selected)
    assert identity.payload_sha256
    assert identity.model_state_sha256
    assert identity.fixed_file_sha256 != identity.selected_file_sha256

    provenance = {
        "checkpoint_sha256": {
            "model_ep2.pt": identity.fixed_file_sha256,
            "model.pt": identity.selected_file_sha256,
        }
    }
    validate_provenance_file_hashes(provenance, identity)


@pytest.mark.parametrize(
    ("path", "mutate"),
    [
        ("model", lambda value: value["model"]["gru.weight"].add_(1.0)),
        (
            "tensor_stride",
            lambda value: value["model"].__setitem__(
                "gru.weight", value["model"]["gru.weight"].T.contiguous().T
            ),
        ),
        ("seed", lambda value: value["args"].__setitem__("seed", 1002)),
        (
            "source_pin",
            lambda value: value["args"].__setitem__("source_manifest_sha256", "d" * 64),
        ),
        (
            "fit_budget",
            lambda value: value["provenance"]["cumulative_updates"].__setitem__(
                "optimizer_attempts", 44_759
            ),
        ),
        (
            "feature_prior",
            lambda value: value["research_feature_prior"].__setitem__("fit_rows", 43),
        ),
    ],
)
def test_any_tensor_or_full_metadata_drift_rejects_pair(tmp_path, path, mutate):
    torch = _torch()
    fixed_value = _checkpoint(torch)
    selected_value = copy.deepcopy(fixed_value)
    mutate(selected_value)
    fixed, selected = _save_pair(tmp_path, torch, fixed_value, selected_value)
    with pytest.raises(ValueError, match="payload differs"):
        verify_checkpoint_pair(fixed, selected)


def test_wrong_fixed_epoch_or_input_width_is_rejected(tmp_path):
    torch = _torch()
    checkpoint = _checkpoint(torch)
    checkpoint["epoch"] = 1
    fixed, selected = _save_pair(tmp_path, torch, checkpoint)
    with pytest.raises(ValueError, match="fixed epoch 2"):
        verify_checkpoint_pair(fixed, selected)

    checkpoint = _checkpoint(torch)
    checkpoint["n_in"] = 270
    fixed, selected = _save_pair(tmp_path, torch, checkpoint)
    with pytest.raises(ValueError, match="input width"):
        verify_checkpoint_pair(fixed, selected)


def test_each_archive_hash_is_independently_bound_in_trainer_provenance(tmp_path):
    torch = _torch()
    checkpoint = _checkpoint(torch)
    fixed, selected = _save_pair(tmp_path, torch, checkpoint)
    identity = verify_checkpoint_pair(fixed, selected)
    provenance = {
        "checkpoint_sha256": {
            "model_ep2.pt": identity.fixed_file_sha256,
            "model.pt": "0" * 64,
        }
    }
    with pytest.raises(ValueError, match="per-file provenance"):
        validate_provenance_file_hashes(provenance, identity)


def test_prediction_pin_keeps_fixed_checkpoint_hash_and_evaluation_groups(tmp_path):
    torch = _torch()
    checkpoint = _checkpoint(torch)
    fixed, selected = _save_pair(tmp_path, torch, checkpoint)
    identity = verify_checkpoint_pair(fixed, selected)
    prediction = tmp_path / "pred.npy"
    prediction.write_bytes(b"finite prediction artifact fixture")
    split_hash = "e" * 64
    groups = [3, 7]
    sidecar = {
        "run": "pr15_baseline_f0_s1001",
        "epoch": 2,
        "campaign_export_scope": "evaluation",
        "campaign_split_manifest_sha256": split_hash,
        "split_manifest_sha256": split_hash,
        "checkpoint_sha256": identity.fixed_file_sha256,
        "row_groups": groups,
        "prediction_sha256": sha256_file(prediction),
    }
    binding = validate_prediction_pin(
        sidecar,
        prediction,
        identity,
        name="pr15_baseline_f0_s1001",
        split_manifest_sha256=split_hash,
        evaluation_groups=groups,
    )
    assert binding["checkpoint_file_sha256"] == identity.fixed_file_sha256

    bad = dict(sidecar, checkpoint_sha256=identity.selected_file_sha256)
    with pytest.raises(ValueError, match="prediction sidecar"):
        validate_prediction_pin(
            bad,
            prediction,
            identity,
            name="pr15_baseline_f0_s1001",
            split_manifest_sha256=split_hash,
            evaluation_groups=groups,
        )
