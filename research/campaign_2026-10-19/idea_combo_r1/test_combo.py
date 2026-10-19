"""Small synthetic invariants for the fixed combo pipeline."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
import torch
from idea_screens_r3.probe_helpers import raw_sums_by_group
from idea_screens_r7.model_math import bounded_gaussian_loss
from idea_screens_r7.model_training import _optimizer
from idea_screens_r15.representation import DictionaryAutoencoder, build_sequence_features

from idea_combo_r1.panel import _require_equal_normalizer
from idea_combo_r1.reference import read_reference_result
from idea_combo_r1.run_probe import _cli_module
from idea_combo_r1.scoring import _label_sums_match
from idea_combo_r1.training import initialize_candidate
from wnn.gmetric import GlobalAccumulator

PACKAGE_DIR = Path(__file__).resolve().parent
CAMPAIGN_DIR = PACKAGE_DIR.parent
REPO_ROOT = CAMPAIGN_DIR.parents[1]


def test_seeded_candidate_matches_stored_initial_and_zero_variance_head() -> None:
    model, initial = initialize_candidate()
    assert initial == "bce617ce88cb42d0c2cce82f3ec479c90909adc2b9e8adcd4cb6dc1d50963405"
    assert torch.count_nonzero(model.logvar_head.weight).item() == 0
    assert torch.count_nonzero(model.logvar_head.bias).item() == 0
    assert all(parameter.dtype is torch.float32 for parameter in model.parameters())


def test_bounded_gaussian_loss_uses_clipped_weighted_nll() -> None:
    target = torch.tensor([[3.0, -0.5], [0.0, 2.5]], dtype=torch.float32)
    mean = torch.tensor([[1.0, -0.25], [0.1, 2.0]], dtype=torch.float32)
    raw = torch.tensor([[7.0, -7.0], [0.0, 0.25]], dtype=torch.float32)
    clipped_y = target.clamp(-2, 2)
    logvar = raw.clamp(-4, 4)
    weight = clipped_y.abs()
    expected = (weight * (torch.exp(-logvar) * (mean - clipped_y).square() + logvar)).sum()
    expected /= weight.sum().clamp_min(1e-8)
    torch.testing.assert_close(bounded_gaussian_loss(mean, raw, target), expected)


def test_nll_updates_zero_initialized_variance_head() -> None:
    model, _ = initialize_candidate()
    optimizer = _optimizer(model)
    features = torch.tensor([[0.5] * 256, [-0.25] * 256], dtype=torch.float32)
    targets = torch.tensor([[1.0, -0.5], [-0.25, 1.5]], dtype=torch.float32)
    mean, logvar = model.forward_all(features)
    bounded_gaussian_loss(mean, logvar, targets).backward()
    assert model.logvar_head.weight.grad is not None
    assert torch.count_nonzero(model.logvar_head.weight.grad).item() > 0
    optimizer.step()
    assert torch.count_nonzero(model.logvar_head.weight).item() > 0


def test_full_history_feature_prefix_is_causal_and_ema_resets() -> None:
    generator = torch.Generator().manual_seed(41)
    dictionary = DictionaryAutoencoder(sparse=True)
    with torch.no_grad():
        dictionary.encoder.weight.copy_(
            torch.randn(dictionary.encoder.weight.shape, generator=generator)
        )
        dictionary.decoder.weight.copy_(
            torch.randn(dictionary.decoder.weight.shape, generator=generator)
        )
    raw = torch.randn((32, 112), generator=generator).numpy().astype(np.float32)
    changed = raw.copy()
    changed[19:] += 0.75
    mean, scale = np.zeros(112, dtype=np.float64), np.ones(112, dtype=np.float64)
    first, _ = build_sequence_features(raw, mean, scale, dictionary)
    second, _ = build_sequence_features(changed, mean, scale, dictionary)
    np.testing.assert_array_equal(first[:19], second[:19])
    np.testing.assert_array_equal(first[0, 128:], np.zeros(128, dtype=np.float32))


def test_fit_normalizer_parity_is_exact_and_fail_closed() -> None:
    values = np.array([1.0, 2.0], dtype=np.float64)
    _require_equal_normalizer("mean", values, values.copy())
    with pytest.raises(ValueError, match="differs"):
        _require_equal_normalizer("mean", values, values + 1e-12)


def test_candidate_label_sums_match_both_fixed_reference_masks() -> None:
    y = np.array([[1.0, -0.5], [0.0, 2.5], [-3.0, 0.25]], dtype=np.float32)
    p = np.zeros_like(y)
    mask = np.ones(3, dtype=bool)
    accumulator = GlobalAccumulator()
    accumulator.add(y, p, mask, group=17)
    candidate = raw_sums_by_group(accumulator)
    _label_sums_match(candidate, candidate)
    altered = {"17": np.asarray(candidate["17"]).copy()}
    altered["17"][0, 0] += 1.0
    with pytest.raises(ValueError, match="labels/mask differ"):
        _label_sums_match(candidate, altered)


def test_reference_result_is_exact_pinned_compact_fixture() -> None:
    result = read_reference_result(PACKAGE_DIR / "REFERENCE_R15_RESULT.json")
    assert result["evaluation"]["arms"]["sparse"]["official_metric"]["wp"] == 0.4824412038543008
    assert result["evaluation"]["arms"]["dense"]["official_metric"]["wp"] == 0.4798730901246624


def test_python_m_cli_identity_accepts_only_exact_module_path() -> None:
    cli = ModuleType("__main__")
    cli.__file__ = str(PACKAGE_DIR / "run_probe.py")
    cli.__spec__ = type("Spec", (), {"name": "idea_combo_r1.run_probe"})()
    assert _cli_module({"__main__": cli}) is cli
    cli.__spec__ = type("Spec", (), {"name": "other.module"})()
    with pytest.raises(RuntimeError, match="python -m"):
        _cli_module({"__main__": cli})
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join((str(REPO_ROOT / "src"), str(CAMPAIGN_DIR)))
    completed = subprocess.run(
        [sys.executable, "-m", "idea_combo_r1.run_probe", "--help"],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    assert "--expected-preparation-manifest-sha256" in completed.stdout


def test_bshape_cold_import_closure_under_payload_sentinels() -> None:
    manifest = PACKAGE_DIR / "PREPARATION_MANIFEST.json"
    if not manifest.exists():
        pytest.skip("final source manifest is created after code freeze")
    manifest_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
    script = r"""
import json, sys
from pathlib import Path
import numpy as np
import torch
import pyarrow.parquet as pq
import pyarrow.dataset as ds
calls=[]
def forbidden(*args, **kwargs):
    calls.append("payload")
    raise AssertionError("payload read sentinel fired")
np.load=forbidden
np.memmap=forbidden
torch.load=forbidden
pq.ParquetFile=forbidden
pq.read_table=forbidden
pq.ParquetDataset=forbidden
ds.dataset=forbidden
from idea_combo_r1.run_probe import smoke_imports
inventory=smoke_imports(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4])
assert not calls, calls
print(json.dumps({"modules":len(inventory),"sentinel_calls":calls},sort_keys=True))
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join((str(REPO_ROOT / "src"), str(CAMPAIGN_DIR)))
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(CAMPAIGN_DIR),
            str(REPO_ROOT),
            str(manifest),
            manifest_sha,
        ],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    receipt = json.loads(completed.stdout)
    assert receipt["modules"] == 31
    assert receipt["sentinel_calls"] == []
