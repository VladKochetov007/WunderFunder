"""Receiver receipt checks for the versioned numerical-input panel."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

from scripts import run_numeric_experiment
from wnn.training.sequence_config import load_config
from wnn.training.sequence_numeric_panel import ARMS, _read_cell
from wnn.training.sequence_provenance import source_identity


def _write_cell_receipts(
    root: Path,
    configs: Path,
    *,
    fold: int,
    seed: int,
    completed_epochs: int,
) -> None:
    repository = Path(__file__).resolve().parents[1]
    evaluator = source_identity(repository, repository / "scripts/score_sequence.py")
    for arm in ARMS:
        config = load_config(configs / f"f{fold}_{arm}.json")
        assert config["expected_fold"] == fold and config["seed"] == seed
        data: dict[str, Any] = {
            "fold": fold,
            "split_sha256": "split-hash",
            "input_artifact_sha256": "input-hash",
            "input_preparation": {
                "calibration_seed": 221022,
                "reservoir_rows_limit": 131072,
            },
            "feature_encoding": arm,
        }
        grade = {
            "evaluator": evaluator,
            "score": {"metric_id": "global_wp_clip2_v1", "selected_rows": 7},
            "producer": {
                "config": config,
                "provenance": {
                    "completed_epochs": completed_epochs,
                    "fixed_checkpoint_policy": "configured-final-epoch",
                    "identity": {
                        "data": data,
                        "source_sha256": "source-hash",
                        "runtime": {"torch": "pinned"},
                        "nvidia_driver_version": "test-driver",
                        "world_size": 1,
                        "initial_common_parameters_sha256": "initialization-hash",
                    },
                },
            },
            "split_sha256": "split-hash",
            "component_raw_sums": {"component-1": {}},
            "sequences": 1,
        }
        cell = root / f"f{fold}_{arm}"
        cell.mkdir(parents=True, exist_ok=True)
        (cell / "score.json").write_text(json.dumps(grade), encoding="utf-8")


def test_r2_receiver_panel_accepts_only_configured_three_epoch_endpoint(
    tmp_path: Path,
) -> None:
    """A 3-epoch R2 cell passes; the former universal 12-epoch check does not."""
    packet = (
        Path(__file__).resolve().parents[1]
        / "research/experiment_drafts/typed-bin-feature-encoding"
    )
    configs = packet / "r2_e3/configs"
    runs = tmp_path / "runs"

    _write_cell_receipts(runs, configs, fold=0, seed=1001, completed_epochs=3)
    assert set(_read_cell(runs, configs, fold=0, seed=1001)) == set(ARMS)

    _write_cell_receipts(runs, configs, fold=0, seed=1001, completed_epochs=12)
    with pytest.raises(ValueError, match="completed fixed-final endpoints"):
        _read_cell(runs, configs, fold=0, seed=1001)


def test_r2_profile_and_fit_share_config_and_use_distinct_output_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both R2 stages bind one amended config and cannot reuse R1 output paths."""
    split = tmp_path / "fold0.json"
    split.write_text('{"fold": 0}', encoding="utf-8")
    output = tmp_path / "output"
    commands: list[list[str]] = []
    monkeypatch.setattr(
        run_numeric_experiment.subprocess,
        "run",
        lambda command, cwd, check: commands.append(command),
    )
    common = [
        "--revision",
        "r2_e3",
        "--fold",
        "0",
        "--arm",
        "raw_events",
        "--data-root",
        str(tmp_path / "data"),
        "--split-manifest",
        str(split),
        "--output-root",
        str(output),
    ]

    monkeypatch.setattr(sys, "argv", ["run_numeric_experiment.py", "profile", *common])
    run_numeric_experiment.main()
    profile_config = commands[-1][commands[-1].index("--config") + 1]
    packet = (
        Path(__file__).resolve().parents[1]
        / "research/experiment_drafts/typed-bin-feature-encoding"
    )
    assert Path(profile_config) == packet / "r2_e3/configs/f0_raw_events.json"
    profile_output = Path(commands[-1][commands[-1].index("--output") + 1])
    assert profile_output == output.resolve() / "profiles_r2_e3/f0_raw_events"

    receipt = output / "profiles_r2_e3" / "f0_raw_events" / "approved.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_numeric_experiment.py",
            "fit",
            *common,
            "--profile-receipt",
            str(receipt),
        ],
    )
    run_numeric_experiment.main()
    fit_config = commands[-1][commands[-1].index("--config") + 1]
    assert fit_config == profile_config
    fit_output = Path(commands[-1][commands[-1].index("--output") + 1])
    assert fit_output == output.resolve() / "runs_r2_e3/f0_raw_events"
