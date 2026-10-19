from __future__ import annotations

import json
import pickle
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from model_contract import _load_restricted_checkpoint, _require_plain_r01_args
from model_probe import (
    PreparedSequence,
    _advance_without_grad,
    parameter_inventory,
    run_diagnostic,
    state_fingerprint,
)
from torch.torch_version import TorchVersion

from wnn.nn import RNNModel
from wnn.training.forward import forward_chunk


def test_chunked_prefix_uses_full_causal_history_and_ignores_suffix() -> None:
    """Chunked no-grad replay equals the full prefix state and cannot see future rows."""
    torch.manual_seed(31)
    model = RNNModel(n_in=3, hidden=4, layers=1, dropout=0.0, n_out=2).eval()
    inputs = torch.randn(1, 1_300, 3)
    with torch.no_grad():
        _, full_state = model(inputs[:, :1_024], None)
    replayed_state = _advance_without_grad(model, inputs, None, 0, 1_024)
    torch.testing.assert_close(replayed_state, full_state, rtol=1e-5, atol=1e-6)
    changed_suffix = inputs.clone()
    changed_suffix[:, 1_024:] = changed_suffix[:, 1_024:] * -17.0
    suffix_state = _advance_without_grad(model, changed_suffix, None, 0, 1_024)
    torch.testing.assert_close(suffix_state, replayed_state, rtol=0.0, atol=0.0)
    with torch.no_grad():
        window_prediction, _ = forward_chunk(model, inputs, slice(1_024, 1_152), replayed_state)
        direct_window_prediction, _ = model(inputs[:, 1_024:1_152], full_state)
    torch.testing.assert_close(window_prediction, direct_window_prediction, rtol=1e-5, atol=1e-6)


def test_inventory_separates_shared_blocks_and_excludes_final_projection() -> None:
    """R01 parameter counts and target-output exclusion are explicit and stable."""
    model = RNNModel(n_in=211, hidden=512, layers=1, dropout=0.0, n_out=2)
    parameters, blocks, inventory = parameter_inventory(model)
    counts = {name: 0 for name in ("trunk_inp_gru", "shared_head_hidden", "target_output_excluded")}
    for item in inventory:
        counts[item["block"]] += item["numel"]
    assert counts == {
        "trunk_inp_gru": 1_684_480,
        "shared_head_hidden": 524_800,
        "target_output_excluded": 1_026,
    }
    shared_names = {parameters[index][0] for indices in blocks.values() for index in indices}
    assert any(name.startswith("inp.") for name in shared_names)
    assert any(name.startswith("gru.") for name in shared_names)
    assert any(name.startswith("head.0.") for name in shared_names)
    assert not any(name.startswith("head.2.") for name in shared_names)


def test_synthetic_sixteen_window_probe_keeps_model_immutable() -> None:
    """The real loop emits all fixed windows, finite scalars and no update or .grad state."""
    torch.manual_seed(43)
    model = RNNModel(n_in=3, hidden=4, layers=1, dropout=0.0, n_out=2).eval()
    inputs = torch.randn(1, 4_224, 3)
    windows = tuple(
        (start, torch.randn(1, 128, 2), torch.ones(1, 128, dtype=torch.bool))
        for start in (1_024, 4_096)
    )
    prepared = [
        PreparedSequence(
            {"component": index, "source": "train", "group": index, "seq_ix": index + 10},
            inputs,
            windows,
        )
        for index in range(8)
    ]
    before = state_fingerprint(model)
    progress: list[dict[str, object]] = []
    result = run_diagnostic(
        model,
        prepared,
        lambda stage: {"stage": stage, "process_rss_bytes": 1, "own_cgroup_current_bytes": 1},
        progress.append,
    )
    assert result["window_count"] == 16
    assert len(progress) == 16
    assert result["model_state_unchanged"] is True
    assert (
        result["model_state_fingerprint_before"]
        == before
        == result["model_state_fingerprint_after"]
    )
    assert result["parameter_grad_fields_empty"] is True
    assert result["cosine_counts"]["trunk_inp_gru"]["window_count_denominator"] == 16
    assert all(record["joint_weight_mass"] > 0.0 for record in result["windows"])
    assert all(parameter.grad is None for parameter in model.parameters())


def test_restricted_loader_allows_only_torch_version_metadata(tmp_path) -> None:
    """Narrow metadata allowlist loads synthetic tensors and rejects other globals."""
    supported_path = tmp_path / "supported.pt"
    torch.save(
        {"tensor": torch.tensor([1.0]), "version": TorchVersion("2.14.synthetic")},
        supported_path,
    )
    loaded = _load_restricted_checkpoint(supported_path)
    assert isinstance(loaded, dict)
    assert torch.equal(loaded["tensor"], torch.tensor([1.0]))
    assert str(loaded["version"]) == "2.14.synthetic"

    unsupported_path = tmp_path / "unsupported.pt"
    torch.save({"value": SimpleNamespace(answer=42)}, unsupported_path)
    with pytest.raises(pickle.UnpicklingError):
        _load_restricted_checkpoint(unsupported_path)


def test_observed_checkpoint_aux_weight_default_is_inert_only_when_disabled() -> None:
    """The observed 0.5 training weight is accepted only with all aux objectives off."""
    fixture_path = Path(__file__).parent / "records" / "CURRENT_MODEL_ARGS_OBSERVATION.json"
    observation = json.loads(fixture_path.read_text())
    assert (
        observation["sha256"] == "a90e53b82c2247b344001c2eb17f6f5a0416edb579a2566875339514709f2a19"
    )
    fields = observation["resolved_args_fields"][0]
    args = {**fields["model_fields"], **fields["all_aux_fields"]}
    _require_plain_r01_args(args, n_in=211)

    active_aux = dict(args, aux_struct=0.1)
    with pytest.raises(ValueError, match="aux_struct"):
        _require_plain_r01_args(active_aux, n_in=211)

    active_next_heads = dict(args, aux_next_k="20")
    with pytest.raises(ValueError, match="next-step"):
        _require_plain_r01_args(active_next_heads, n_in=211)

    changed_default = dict(args, aux_weight=0.25)
    with pytest.raises(ValueError, match="aux_weight"):
        _require_plain_r01_args(changed_default, n_in=211)
