"""Small CPU-only invariants for the fixed R13 warmup-context screen."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from idea_screens_r1.feature_views import build_view
from idea_screens_r7.model_math import BATCH_SIZE, EPOCHS, epoch_permutation
from idea_screens_r7.model_training import _optimizer
from idea_screens_r7.panel import EVAL_SCORED_ROWS

from idea_screens_r13 import scoring
from idea_screens_r13.input_panel import CONTEXT_START, WARMUP_VIEW, _normalize_raw_block
from idea_screens_r13.model_arms import (
    ARM_NAMES,
    assert_initial_prediction_parity,
    initialize_models,
)
from idea_screens_r13.scoring import _add_component_predictions
from wnn.gmetric import GlobalAccumulator
from wnn.nn import weighted_mse_loss

REPO_ROOT = Path(__file__).resolve().parents[3]
CAMPAIGN_ROOT = Path(__file__).resolve().parents[1]


def test_warmup99_is_reset_local_and_future_independent() -> None:
    first = np.arange(140 * 112, dtype=np.float32).reshape(140, 112)
    second = first + np.float32(50_000)
    first_view, _ = build_view(first, WARMUP_VIEW)
    second_view, _ = build_view(second, WARMUP_VIEW)
    changed = first.copy()
    changed[CONTEXT_START:] += np.float32(1234)
    changed_view, _ = build_view(changed, WARMUP_VIEW)
    np.testing.assert_array_equal(first_view[:CONTEXT_START, 112:], 0.0)
    np.testing.assert_array_equal(
        first_view[CONTEXT_START:, 112:],
        np.broadcast_to(
            first[:CONTEXT_START].mean(axis=0, dtype=np.float64),
            first_view[CONTEXT_START:, 112:].shape,
        ).astype(np.float32),
    )
    np.testing.assert_array_equal(first_view[:, 112:], changed_view[:, 112:])
    np.testing.assert_array_equal(
        second_view[CONTEXT_START, 112:],
        second[:CONTEXT_START].mean(axis=0, dtype=np.float64).astype(np.float32),
    )
    assert not np.array_equal(first_view[CONTEXT_START, 112:], second_view[CONTEXT_START, 112:])


def test_shared_raw_normalizer_and_exact_early_side_zero() -> None:
    values = np.arange(120 * 112, dtype=np.float32).reshape(120, 112)
    mean = np.full(224, 7.0, dtype=np.float32)
    scale = np.full(224, 3.0, dtype=np.float32)
    rows = np.arange(120)
    side = _normalize_raw_block(values, rows, mean, scale)
    np.testing.assert_array_equal(side[:99], 0.0)
    expected = (values[99:] - mean[:112]) / scale[:112]
    np.testing.assert_array_equal(side[99:], expected)
    assert side.dtype == np.float32 and side.flags.c_contiguous


def test_three_arms_share_exact_initial_predictions_and_unused_head() -> None:
    models, fingerprints = initialize_models()
    assert set(models) == set(ARM_NAMES)
    assert len(set(fingerprints.values())) == 1
    assert {
        str(parameter.dtype) for model in models.values() for parameter in model.parameters()
    } == {"torch.float32"}
    assert {
        str(parameter.device) for model in models.values() for parameter in model.parameters()
    } == {"cpu"}
    rng = np.random.default_rng(19)
    base = rng.normal(size=(9, 224)).astype(np.float32)
    sides = {
        "zero": np.zeros((9, 112), dtype=np.float32),
        "current_bypass": rng.normal(size=(9, 112)).astype(np.float32),
        "warmup99": rng.normal(size=(9, 112)).astype(np.float32),
    }
    assert_initial_prediction_parity(models, base, sides)
    model = models["zero"]
    variance_before = {
        name: tensor.detach().clone()
        for name, tensor in model.state_dict().items()
        if name.startswith("logvar_head.")
    }
    x = torch.zeros((8, 336), dtype=torch.float32)
    y = torch.linspace(-1.0, 1.0, 16, dtype=torch.float32).reshape(8, 2)
    optimizer = _optimizer(model)
    optimizer.zero_grad(set_to_none=True)
    loss = weighted_mse_loss(model.forward_mean(x), y, torch.ones(8, dtype=torch.bool))
    loss.backward()
    assert all(
        model.get_parameter(f"logvar_head.{name}").grad is None for name in ("weight", "bias")
    )
    optimizer.step()
    assert torch.count_nonzero(model.trunk.weight[:, 224:]).item() == 0
    for name, before in variance_before.items():
        torch.testing.assert_close(model.state_dict()[name], before, rtol=0.0, atol=0.0)


def test_fixed_common_permutation_budget_arithmetic() -> None:
    first = epoch_permutation(31_808, 0)
    repeat = epoch_permutation(31_808, 0)
    second = epoch_permutation(31_808, 1)
    assert torch.equal(first, repeat)
    assert not torch.equal(first, second)
    assert len(first) == 31_808
    batches = (31_808 + BATCH_SIZE - 1) // BATCH_SIZE
    assert batches == 63
    assert EPOCHS * batches == 756
    assert EPOCHS * 31_808 == 381_696
    assert 3 * EPOCHS * batches == 2_268


def test_canonical_component_scoring_and_drop_one() -> None:
    pooled = GlobalAccumulator()
    parts: dict[int, GlobalAccumulator] = {}
    targets = np.array([[-3.0, 0.5], [1.0, -0.2], [0.3, 1.5]], dtype=np.float32)
    predictions = np.array([[0.7, -0.1], [0.2, 0.4], [-0.3, 1.2]], dtype=np.float32)
    _add_component_predictions(pooled, parts, targets[:2], predictions[:2], 17)
    _add_component_predictions(pooled, parts, targets[2:], predictions[2:], 23)
    result = pooled.result()
    assert np.isfinite(result["wp"])
    assert result["selected_rows"] == 3
    assert set(parts) == {17, 23}
    combined = GlobalAccumulator()
    combined.add(targets[2:], predictions[2:], np.ones(1, dtype=bool))
    assert abs(combined.result()["wp"] - parts[23].result()["wp"]) < 1e-12


def test_three_arm_scorer_accumulates_all_32_components(monkeypatch) -> None:
    models, _ = initialize_models()
    base = np.zeros((3, 224), dtype=np.float32)
    target = np.array([[-0.8, 0.2], [0.4, -0.6], [1.2, 0.9]], dtype=np.float32)
    side = {arm: np.zeros((3, 112), dtype=np.float32) for arm in ARM_NAMES}
    evaluation = [{"component": index} for index in range(32)]

    def prepared(sequence, mean, scale):
        return int(sequence["component"]), base, side, target, len(target)

    monkeypatch.setattr(scoring, "prepare_evaluation_rows", prepared)
    monkeypatch.setattr(scoring, "EVAL_SCORED_ROWS", 96)
    result = scoring.score_models(models, evaluation, np.zeros(224), np.ones(224))
    assert result["eval_scored_rows"] == 96
    assert len(result["eval_components"]) == 32
    assert len(result["component_drop_one_deltas_wp"]) == 32
    for arm in ARM_NAMES:
        assert result["arms"][arm]["official_metric"]["selected_rows"] == 96
        assert sum(result["arms"][arm]["component_rows"].values()) == 96
        assert len(result["arms"][arm]["component_scores"]) == 32
    assert set(result["deltas_wp"]) == {
        "warmup99_minus_current_bypass_wp",
        "warmup99_minus_zero_wp",
        "current_bypass_minus_zero_wp",
    }


def test_actual_r3_metadata_fixture_matches_consumed_panel_shape() -> None:
    folder = CAMPAIGN_ROOT / "idea_screens_r3/records/sample_v2/terminal_capture"
    sample = json.loads((folder / "REMOTE_SAMPLE.json").read_bytes())
    membership = json.loads((folder / "REMOTE_MEMBERSHIP.json").read_bytes())
    assert sample["schema"] == "idea_screens_r3.sample.v2"
    assert membership["schema"] == "idea_screens_r3.membership.v2"
    assert sample["split_sha256"] == membership["split_sha256"]
    assert sample["selection"]["selected_component_counts"] == {
        "fit_full64": 64,
        "fit_nested_low32": 32,
        "eval32": 32,
    }
    records = sample["records"]
    assert len(records) == 96
    assert sum(item["role"] == "fit" for item in records) == 64
    assert sum(item["role"] == "eval" for item in records) == 32
    assert all(set(item["files"]) == {"x", "y", "scored"} for item in records)
    assert len({item["component"] for item in records if item["role"] == "fit"}) == 64
    assert len({item["component"] for item in records if item["role"] == "eval"}) == 32
    assert sum(item["scored_rows"] for item in records if item["role"] == "eval") == 59_436
    assert EVAL_SCORED_ROWS == 59_436


def test_full_import_closure_does_not_open_payloads() -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        (str(CAMPAIGN_ROOT), str(REPO_ROOT / "src"), env.get("PYTHONPATH", ""))
    )
    env["CUDA_VISIBLE_DEVICES"] = ""
    code = """
import importlib
from pathlib import Path
import numpy as np
import torch
import pyarrow.dataset as ds
import pyarrow.parquet as pq

def forbidden(*args, **kwargs):
    raise AssertionError('payload loader called during import smoke')

np.load = forbidden
np.memmap = forbidden
torch.load = forbidden
pq.ParquetFile = forbidden
pq.read_table = forbidden
if hasattr(pq, 'ParquetDataset'):
    pq.ParquetDataset = forbidden
ds.dataset = forbidden
rp = importlib.import_module('idea_screens_r13.run_probe')
support = Path(rp.__file__).resolve().parent.parent
source = Path(rp.__file__).resolve().parents[3]
pins = {
    key: rp.sha256_file(rp._source_path(key, support, source))
    for key in rp.RUNTIME_IMPORT_PATHS.values()
}
inventory = rp.smoke_imports(support, source, pins)
print(len(inventory))
"""
    completed = subprocess.run(
        [sys.executable, "-B", "-c", code],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "30"


def test_python_m_entrypoint_checks_provenance_before_help() -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        (str(CAMPAIGN_ROOT), str(REPO_ROOT / "src"), env.get("PYTHONPATH", ""))
    )
    completed = subprocess.run(
        [sys.executable, "-B", "-m", "idea_screens_r13.run_probe", "--help"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--expected-preparation-manifest-sha256" in completed.stdout
