"""Synthetic checks for fixed-panel streaming and canonical metric reuse."""

from __future__ import annotations

import os
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))
import first_cell_score_contract as CONTRACT  # noqa: E402
import first_cell_score_core as SCORER  # noqa: E402


class SyntheticReader:
    """Return in-memory groups while recording the exact requested order."""

    def __init__(self, targets: np.ndarray, scored: np.ndarray, groups: list[int]) -> None:
        self.targets = dict(zip(groups, targets, strict=True))
        self.scored = dict(zip(groups, scored, strict=True))
        self.reads: list[int] = []

    def read(self, index: int, *, targets: bool = True, mask: bool = True) -> SimpleNamespace:
        assert targets is True and mask is True
        self.reads.append(index)
        return SimpleNamespace(targets=self.targets[index], scored=self.scored[index])


@pytest.fixture(scope="module")
def metric_helpers():
    """Load only the pinned canonical data/metric module closure from source68."""
    source_root = Path(os.environ["WNN_SOURCE_ROOT"])
    return SCORER._load_helpers(source_root)


def _synthetic_panel() -> tuple[list[int], np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    groups = [17, 5, 29]
    rng = np.random.default_rng(4481)
    targets = rng.normal(0, 0.7, size=(len(groups), SCORER.N_ROWS, 2)).astype(np.float32)
    scored = np.zeros((len(groups), SCORER.N_ROWS), dtype=bool)
    scored[:, :110] = True
    scored[0, 140:146] = True
    scored[1, :] = False
    targets[0, 99, :] = 0.0  # selected zero-weight row remains in the row count
    predictions = {
        arm: np.clip(targets + offset, -2.5, 2.5).astype(np.float32)
        for arm, offset in zip(SCORER.ARM_MODES, (0.0, 0.05, -0.04, 0.08, -0.09), strict=True)
    }
    return groups, targets, scored, predictions


def _synthetic_registered_data_contract() -> tuple[dict, dict]:
    """Build metadata-only producer identity matching the fixed ADMISSION contract."""
    base = "/home/grinchenkoeg/computer-vision-homework-20261018"
    binding = {
        "data_root": f"{base}/data",
        "registered_data": {
            "admission": deepcopy(SCORER.EXPECTED_DATA_ADMISSION),
            "portable_manifest": deepcopy(SCORER.EXPECTED_PORTABLE_MANIFEST),
            "source_identities": deepcopy(SCORER.EXPECTED_REGISTERED_SOURCES),
            "valid_parquet": deepcopy(SCORER.EXPECTED_VALID_PARQUET),
        },
    }
    valid = SCORER.EXPECTED_VALID_PARQUET
    observed = {
        "path": valid["path"],
        "resolved": valid["resolved"],
        "bytes": valid["bytes"],
        "device": valid["device"],
        "inode": valid["inode"],
        "islink": False,
        "mtime_ns": valid["mtime_ns"],
        "parent": {
            "path": f"{base}/data/datasets",
            "resolved": SCORER.EXPECTED_PARENT_RESOLUTIONS["datasets"],
            "device": 64512,
            "inode": 42,
        },
    }
    source_observation_fields = (
        "path",
        "resolved",
        "bytes",
        "device",
        "inode",
        "islink",
        "mtime_ns",
    )
    source_observations = {
        relative: {key: source[key] for key in source_observation_fields}
        for relative, source in SCORER.EXPECTED_REGISTERED_SOURCES.items()
    }
    identity = {
        "schema": "fg19.registered-valid-data-identity.v1",
        "admission": deepcopy(SCORER.EXPECTED_DATA_ADMISSION),
        "portable_manifest": {
            **deepcopy(SCORER.EXPECTED_PORTABLE_MANIFEST),
            "source_names": deepcopy(SCORER.EXPECTED_PORTABLE_SOURCE_NAMES),
            "split_manifest_sha256": SCORER.EXPECTED_SPLIT_SHA256,
            "version": 1,
        },
        "registered_source_identities": deepcopy(SCORER.EXPECTED_REGISTERED_SOURCES),
        "registered_valid_parquet": deepcopy(valid),
        "observed_sources_before_split": deepcopy(source_observations),
        "observed_sources_after_split": deepcopy(source_observations),
        "observed_before_sequence_open": deepcopy(observed),
        "observed_after_export": deepcopy(observed),
    }
    return binding, {"data_identity": identity}


def test_streamed_scores_and_four_contrasts_match_canonical_batch(metric_helpers) -> None:
    groups, targets, scored, predictions = _synthetic_panel()
    reader = SyntheticReader(targets, scored, groups)
    components = {17: 3, 5: 9, 29: 3}
    streamed = SCORER.score_stream(groups, components, predictions, reader, metric_helpers)
    expected_mask = scored & (np.arange(SCORER.N_ROWS) >= metric_helpers.warmup)

    assert reader.reads == groups
    assert streamed["ordered_group_count"] == len(groups)
    assert set(streamed["paired_contrasts_vs_control"]) == {"ema", "sparse", "nll", "allthree"}
    assert streamed["arms"]["control"]["selected_rows"] == int(expected_mask.sum())
    for arm in SCORER.ARM_MODES:
        expected = metric_helpers.global_wp(targets, predictions[arm], expected_mask)
        for key in ("t0", "t1", "wp", "selected_rows"):
            assert streamed["arms"][arm][key] == pytest.approx(expected[key])

    for arm in ("ema", "sparse", "nll", "allthree"):
        candidate, reference = metric_helpers.paired_component_moments(
            targets, predictions[arm], predictions["control"], expected_mask
        )
        expected = metric_helpers.paired_component_bootstrap(
            candidate,
            reference,
            np.asarray([components[group] for group in groups]),
            n_boot=SCORER.BOOTSTRAP_REPLICATES,
            seed=SCORER.BOOTSTRAP_SEED,
            centered=True,
        )
        actual = streamed["paired_contrasts_vs_control"][arm]["candidate_minus_control"]
        assert actual == expected


def test_shuffled_order_is_rejected() -> None:
    SCORER.require_ordered_groups([17, 5, 29], [17, 5, 29])
    with pytest.raises(ValueError, match="order differs"):
        SCORER.require_ordered_groups([17, 5, 29], [5, 17, 29])


def test_nonfinite_selected_prediction_is_rejected(metric_helpers) -> None:
    groups, targets, scored, predictions = _synthetic_panel()
    scored[0, 100] = True
    predictions["control"][0, 100, 0] = np.nan
    reader = SyntheticReader(targets, scored, groups)
    components = {17: 3, 5: 3, 29: 8}
    with pytest.raises(ValueError, match="nonfinite exported prediction"):
        SCORER.score_stream(groups, components, predictions, reader, metric_helpers)


def test_zero_scored_group_stays_in_component_panel(metric_helpers) -> None:
    groups, targets, scored, predictions = _synthetic_panel()
    assert not scored[1].any()
    reader = SyntheticReader(targets, scored, groups)
    components = {17: 3, 5: 9, 29: 3}
    streamed = SCORER.score_stream(groups, components, predictions, reader, metric_helpers)
    assert reader.reads == groups
    assert streamed["distinct_component_count"] == 2
    for arm in ("ema", "sparse", "nll", "allthree"):
        result = streamed["paired_contrasts_vs_control"][arm]["candidate_minus_control"]
        assert result["n_sequences"] == len(groups)
        assert result["n_components"] == 2
    assert streamed["arms"]["control"]["selected_rows"] == int(
        (scored & (np.arange(SCORER.N_ROWS) >= metric_helpers.warmup)).sum()
    )


def test_registered_data_producer_contract_joins_all_six_sources() -> None:
    binding, manifest = _synthetic_registered_data_contract()
    CONTRACT._validate_data_identity(binding, manifest)


def test_registered_data_contract_rejects_source_or_parent_alias_drift() -> None:
    binding, manifest = _synthetic_registered_data_contract()
    binding["registered_data"]["source_identities"]["cache/chains.npz"]["resolved"] = (
        "/tmp/unregistered/chains.npz"
    )
    with pytest.raises(ValueError, match="identity differs from admission"):
        CONTRACT._validate_data_identity(binding, manifest)

    binding, manifest = _synthetic_registered_data_contract()
    manifest["data_identity"]["observed_before_sequence_open"]["parent"]["resolved"] = (
        "/tmp/unregistered/datasets"
    )
    with pytest.raises(ValueError, match="parent alias observation"):
        CONTRACT._validate_data_identity(binding, manifest)

    binding, manifest = _synthetic_registered_data_contract()
    manifest["data_identity"]["observed_sources_after_split"]["datasets/train.parquet"][
        "mtime_ns"
    ] += 1
    with pytest.raises(ValueError, match="differs from admission"):
        CONTRACT._validate_data_identity(binding, manifest)
