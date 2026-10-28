from __future__ import annotations

import importlib
import json
from dataclasses import replace
from pathlib import Path
from types import ModuleType

import pytest
import torch

from wnn.galam.config import connectome_112_to_2_config
from wnn.galam.variants import build_galam_variant

PACKET = Path(__file__).resolve().parents[1] / "research/experiment_drafts/galam-block-bn-20261029"


@pytest.fixture
def verification(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Import receiver helpers without running their command entrypoints."""
    monkeypatch.syspath_prepend(str(PACKET))
    return importlib.import_module("block_verification")


@pytest.fixture
def receipts(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Load the fail-closed receipt validator for negative tests."""
    monkeypatch.syspath_prepend(str(PACKET))
    return importlib.import_module("block_receipts")


@pytest.fixture
def device(receipts: ModuleType) -> torch.device:
    """Skip on non-receivers; do not substitute CPU for CUDA parity evidence."""
    if not torch.cuda.is_available():
        pytest.skip("Receiver CUDA numerical check")
    return receipts.configure_cuda("cuda:0")


@pytest.mark.parametrize("training", [False, True])
def test_chunk_grad_state_reload(
    verification: ModuleType, device: torch.device, training: bool
) -> None:
    """Unequal chunks straddle history and convolution boundaries with all-state comparison."""
    config = replace(connectome_112_to_2_config(), width=72, attention_history=9)
    verification.check_chunks(config, device, [0, 1, 7, 13, 2], 4, training)


def test_per_time_batchnorm(verification: ModuleType, device: torch.device) -> None:
    """Cover EMA/CMA, no tracking, batch-one eval and parameter/input gradients."""
    assert verification.check_batch_norm(device) == 16


@pytest.mark.parametrize("training", [False, True])
def test_future_prefix(verification: ModuleType, device: torch.device, training: bool) -> None:
    """Block normalization and attention must not expose a changed future suffix."""
    verification.check_causality(connectome_112_to_2_config(), device, training)


def test_diagnostics_keep_step_semantics(verification: ModuleType, device: torch.device) -> None:
    """The observer fallback retains every sampled attention distribution."""
    verification.check_observers(connectome_112_to_2_config(), device)


def test_unsupported_lstm_block() -> None:
    """No silent substitution of a different recurrent graph is permitted."""
    with pytest.raises(ValueError):
        build_galam_variant(
            connectome_112_to_2_config(), "schematic_lstm_full", execution_backend="block"
        )


def _acceptance(current: dict[str, object]) -> dict[str, object]:
    """Build a complete positive receipt so mutation tests cannot pass vacuously."""
    cases = []
    for index in range(7):
        cases.append(
            {
                "batch": 384 if index < 2 else 4,
                "chunks": [64, 64] if index < 2 else [1, 7],
                "training": bool(index % 2),
                "max_absolute_errors": {
                    name: 0.0
                    for name in ("output", "state", "input_gradient", "gradient", "buffer")
                },
            }
        )
    return {
        "schema": "wnn.galam-block-acceptance.v1",
        "status": "passed",
        "identity": current.copy(),
        "representative_shape": [384, 64, 112],
        "suite": "state-gradient-bn-causal-update-v1",
        "results": {
            "cases": cases,
            "batch_norm_cases": 16,
            "causality": "passed",
            "reload": "passed",
            "diagnostic_observers": "passed",
        },
    }


def test_profile_accepts_complete_receipt(receipts: ModuleType, tmp_path: Path) -> None:
    """Accept complete matching evidence before testing every rejection reason."""
    current = {"source_commit": "a" * 40, "config_sha256": "x", "device_uuid": "gpu0"}
    path = tmp_path / "acceptance.json"
    path.write_text(json.dumps(_acceptance(current)))
    receipts.validate_acceptance(path, current, [384, 64, 112])


@pytest.mark.parametrize(
    "mutation", ["failed", "identity", "shape", "suite", "schema", "results", "representative"]
)
def test_profile_rejects_stale_acceptance(
    receipts: ModuleType, tmp_path: Path, mutation: str
) -> None:
    """A successful check from another source/config/device cannot admit this profile."""
    current = {"source_commit": "a" * 40, "config_sha256": "x", "device_uuid": "gpu0"}
    value = _acceptance(current)
    if mutation == "failed":
        value["status"] = "failed"
    elif mutation == "identity":
        value["identity"] = {**current, "device_uuid": "gpu1"}
    elif mutation == "shape":
        value["representative_shape"] = [32, 64, 112]
    elif mutation == "results":
        value["results"] = {}
    elif mutation == "representative":
        value["representative_shape"] = [32, 64, 112]
        path = tmp_path / "acceptance.json"
        path.write_text(json.dumps(value))
        with pytest.raises(ValueError, match="representative"):
            receipts.validate_acceptance(path, current, [32, 64, 112])
        return
    else:
        value[mutation] = "wrong"
    path = tmp_path / "acceptance.json"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        receipts.validate_acceptance(path, current, [384, 64, 112])
