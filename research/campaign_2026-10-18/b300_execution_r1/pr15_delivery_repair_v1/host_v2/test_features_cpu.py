"""Small CPU-only gates for the immutable PR15 geometry feature package.

Run with ``PR15_SOURCE_ROOT=<full checkout> pytest -q pr15_review_cpu_tests.py``.
The downloaded PR contains changed files only. On such an archive, narrowly
stub two unrelated base imports so the actual PR feature implementation can be
tested without installing torch; a complete checkout uses its real imports.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest

AUDIT_SOURCE = Path(
    r"C:\Users\admin\Documents\Codex\reviews\WunderFunder-audit-20261018"
    r"\context-followup\github-prs\15\source"
)
SOURCE = Path(os.environ.get("PR15_SOURCE_ROOT", str(AUDIT_SOURCE))).resolve()
sys.path.insert(0, str(SOURCE / "src"))
sys.path.insert(0, str(SOURCE / "scripts" / "geometry_b300"))
TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


def _install_archive_only_import_stubs() -> None:
    """Fill only absent base-tree imports omitted from the PR changed-file pack."""
    if importlib.util.find_spec("torch") is None:
        torch_stub = types.ModuleType("torch")
        torch_stub.Tensor = object  # type: ignore[attr-defined]
        sys.modules.setdefault("torch", torch_stub)
    for name, attrs in (
        (
            "wnn.moment_geometry.analysis",
            {"robust_arc": lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("fit_prior is outside this feature-transform test")
            )},
        ),
        ("wnn.moment_geometry.features", {"SELECTED": (0, 3, 52, 55)}),
    ):
        if importlib.util.find_spec(name) is None:
            module = types.ModuleType(name)
            for key, value in attrs.items():
                setattr(module, key, value)
            sys.modules[name] = module


_install_archive_only_import_stubs()
research_features = importlib.import_module("wnn.research_features")
online = importlib.import_module("wnn.moment_geometry.online")
common = importlib.import_module("common")


def _fixture() -> tuple[np.ndarray, dict[str, object]]:
    raw = np.zeros((4, 112), dtype=np.float32)
    raw[:, [0, 3, 52, 55]] = np.asarray(
        [
            [1, 10, 100, 1000],
            [1, 20, 100, 2000],
            [3, 20, 300, 3000],
            [9, 40, 500, 4000],
        ],
        dtype=np.float32,
    )
    prior: dict[str, object] = {
        "schema": "causal_geometry_discrete_v1",
        "selected_raw_columns": [0, 3, 52, 55],
        "feature_order": list(online.FEATURE_NAMES),
        "beta": [0.5, -0.25, 0.1, 0.01],
        "intercept": [0.25, 1.0, -2.0, 3.0],
        "atoms": [3, 20, 300, 2000],
        "vocabulary": [[1, 3, 5], [10, 20, 40], [100, 200, 300], [1000, 2000, 4000]],
        "fit_sequences": 2,
        "fit_rows": 8,
        "frequency_tie_break": "test fixture",
    }
    return raw, prior


def _independent_reference(raw: np.ndarray, prior: dict[str, object]) -> np.ndarray:
    """Direct row-by-row oracle; deliberately avoids production vectorized code."""
    selected = np.asarray(raw[:, [0, 3, 52, 55]], dtype=np.float64)
    beta = np.asarray(prior["beta"], dtype=np.float32).astype(np.float64)
    intercept = np.asarray(prior["intercept"], dtype=np.float32).astype(np.float64)
    atoms = np.asarray(prior["atoms"], dtype=np.float32).astype(np.float64)
    vocabulary = np.asarray(prior["vocabulary"], dtype=np.float32).astype(np.float64)
    output: list[np.ndarray] = []
    seen = np.zeros((4, 4), dtype=np.float64)
    event_totals = np.zeros(4, dtype=np.float64)
    dominant = np.zeros(4, dtype=np.float64)
    last_change = np.zeros(4, dtype=np.int64)
    previous: np.ndarray | None = None
    for row_index, row in enumerate(selected):
        prefix = selected[: row_index + 1]
        mean = prefix.sum(axis=0) / (row_index + 1)
        second = np.square(prefix).sum(axis=0) / (row_index + 1)
        std = np.sqrt(np.maximum(second - mean * mean, 0.0))
        residual = second - beta * mean - intercept
        atom_occupancy = np.asarray(
            [(prefix[:, field] == atoms[field]).sum() / len(prefix) for field in range(4)]
        )
        normalized = np.clip((row - mean) / np.maximum(std, 0.05), -8.0, 8.0)
        category = np.zeros((4, 4), dtype=np.float64)
        changed = np.ones(4, dtype=bool) if previous is None else row != previous
        for field in range(4):
            matches = np.flatnonzero(vocabulary[field] == row[field])
            category[field, int(matches[0]) if len(matches) else 3] = 1.0
            seen[field] += category[field]
            event_totals[field] += float(changed[field])
            dominant[field] += float(changed[field] and category[field, 0] == 1.0)
            if changed[field]:
                last_change[field] = row_index
        occupancy = seen / (row_index + 1)
        dominant_events = dominant / np.maximum(event_totals, 1.0)
        dwell = np.log1p(np.minimum(row_index - last_change, 10_000))
        output.append(
            np.concatenate(
                (
                    mean,
                    std,
                    residual,
                    atom_occupancy,
                    normalized,
                    category.ravel(),
                    occupancy.ravel(),
                    dominant_events,
                    dwell,
                )
            )
        )
        previous = row.copy()
    return np.asarray(output, dtype=np.float32)


def test_numpy_feature_values_match_independent_row_oracle() -> None:
    raw, prior = _fixture()
    actual = research_features.research_features_numpy(
        raw, "causal_geometry_discrete", prior
    )
    expected = _independent_reference(raw, prior)
    assert actual.shape == (4, 60)
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_suffix_mutation_cannot_change_prefix_and_sequences_reset() -> None:
    raw, prior = _fixture()
    baseline = online.features_numpy(raw, prior)
    changed = raw.copy()
    changed[2:, [0, 3, 52, 55]] += np.asarray([77, 88, 99, 111], dtype=np.float32)
    mutated = online.features_numpy(changed, prior)
    np.testing.assert_array_equal(mutated[:2], baseline[:2])
    assert not np.array_equal(mutated[2], baseline[2])  # current row is available

    batch = np.stack((raw, raw + 1234.0))
    actual_batch = research_features.research_features_numpy(
        batch, "causal_geometry_discrete", prior
    )
    np.testing.assert_array_equal(actual_batch[0], baseline)
    np.testing.assert_array_equal(
        actual_batch[1], online.features_numpy(batch[1], prior)
    )


def test_matched_controls_keep_width_and_zero_exact_partition() -> None:
    raw, prior = _fixture()
    mode = "causal_geometry_discrete"
    full = research_features.research_features_numpy(raw, mode, prior, "none")
    baseline = research_features.research_features_numpy(raw, mode, prior, "all")
    moments = research_features.research_features_numpy(raw, mode, prior, "moments_only")
    dictionary = research_features.research_features_numpy(raw, mode, prior, "dictionary_only")
    assert {value.shape for value in (full, baseline, moments, dictionary)} == {(4, 60)}
    np.testing.assert_array_equal(baseline, np.zeros((4, 60), dtype=np.float32))
    np.testing.assert_array_equal(moments[:, 20:], np.zeros((4, 40), dtype=np.float32))
    np.testing.assert_array_equal(dictionary[:, :20], np.zeros((4, 20), dtype=np.float32))
    np.testing.assert_array_equal(moments[:, :20], full[:, :20])
    np.testing.assert_array_equal(dictionary[:, 20:], full[:, 20:])
    assert research_features.research_feature_control_indices(mode, "all") == tuple(range(60))
    assert research_features.research_feature_control_indices(mode, "moments_only") == tuple(
        range(20, 60)
    )
    assert research_features.research_feature_control_indices(mode, "dictionary_only") == tuple(
        range(20)
    )


def test_torch_cpu_features_match_numpy_for_prefix_and_each_control() -> None:
    if not TORCH_AVAILABLE:
        pytest.skip("PyTorch is absent from the local review venv; run on the B300 host CPU")
    import torch

    raw, prior = _fixture()
    batch = np.stack((raw, raw[::-1].copy()))
    tensor = torch.from_numpy(batch)
    for control in ("all", "moments_only", "dictionary_only", "none"):
        expected = research_features.research_features_numpy(
            batch, "causal_geometry_discrete", prior, control
        )
        actual = research_features.research_features_torch(
            tensor, "causal_geometry_discrete", prior, control=control
        )
        np.testing.assert_allclose(actual.cpu().numpy(), expected, rtol=2e-6, atol=2e-6)

    changed = batch.copy()
    changed[:, 2:, [0, 3, 52, 55]] += np.asarray([17, 23, 31, 41], dtype=np.float32)
    actual = research_features.research_features_torch(
        torch.from_numpy(changed), "causal_geometry_discrete", prior
    )
    baseline = research_features.research_features_torch(
        tensor, "causal_geometry_discrete", prior
    )
    torch.testing.assert_close(actual[:, :2], baseline[:, :2], rtol=0, atol=0)


def _flag(command: list[str], name: str) -> str:
    return command[command.index(name) + 1]


def test_frozen_command_builder_covers_30_paired_cells_and_fixed_recipe() -> None:
    arm_controls = common.CONTROL_BY_ARM
    assert arm_controls == {
        "baseline": "all",
        "moments": "moments_only",
        "dictionary": "dictionary_only",
        "combined": "none",
    }
    cells: set[tuple[str, int, int]] = set()
    for pair_index in range(30):
        fold, seed = common.pair_identity(pair_index)
        for arm, control in arm_controls.items():
            name = f"pr15-{arm}-f{fold}-s{seed}-pair{pair_index:02d}"
            command = common.train_command(name, fold, seed, control)
            cells.add((arm, fold, seed))
            assert _flag(command, "--seed") == str(seed)
            assert _flag(command, "--split-manifest").endswith(f"fold{fold}.json")
            assert _flag(command, "--research-feature-control") == control
            assert _flag(command, "--research-features") == "causal_geometry_discrete"
            assert _flag(command, "--train-source") == "train+valid"
            assert _flag(command, "--train-seqs") == "10607"
            assert "--no-eval-during-fit" in command
            assert _flag(command, "--precision") == "fp32"
            assert _flag(command, "--epochs") == "3"
            assert _flag(command, "--checkpoint-policy") == "fixed"
            assert _flag(command, "--checkpoint-epoch") == "2"
            assert _flag(command, "--hidden") == "512"
            assert _flag(command, "--layers") == "1"
            assert _flag(command, "--batch") == "8"
            assert _flag(command, "--chunk") == "2000"
    assert len(cells) == 120
    for arm in arm_controls:
        assert sum(cell[0] == arm for cell in cells) == 30
    config = common.PROFILE_CONFIG
    assert config["model_input_width"] == 211 + 60 == 271
    assert config["profile_warmup_updates"] == 2
    assert config["profile_measured_updates"] == 8


def test_original_profile_settings_mean_100_optimizer_attempts() -> None:
    """Profile flags count batches; each 20k-row B8 batch has ten TBPTT chunks."""
    config = common.PROFILE_CONFIG
    batches = config["profile_warmup_updates"] + config["profile_measured_updates"]
    chunks_per_sequence = 20_000 // config["chunk_rows"]
    assert batches == 10
    assert chunks_per_sequence == 10
    # One optimizer attempt is made per chunked B8 batch, not per sequence.
    assert batches * chunks_per_sequence == 100
    assert config["profile_measured_updates"] * chunks_per_sequence == 80
