from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.plot_sequence_monitor import _cohort_label, _comparison_note, export_plots


def test_plot_labels_identify_masks_and_gap_limits_without_rendering() -> None:
    train = _cohort_label(
        "Train-eval in-sample subset",
        {"member_count": 128, "selection": "hash_rank_v1", "row_mask": "all_after_warmup"},
        include_selection=True,
    )
    valid = _cohort_label("Official valid", {"member_count": 1873, "row_mask": "is_scored"})

    assert train == "Train-eval in-sample subset (n=128, hash_rank_v1, all_after_warmup)"
    assert valid == "Official valid (n=1873, is_scored)"
    assert "not a calibrated generalization gap" in _comparison_note("valid")


def test_local_plot_export_preserves_native_journals_and_role(tmp_path: Path) -> None:
    """Synthetic schema fixtures test rendering only, without any model execution."""
    pytest.importorskip("matplotlib")
    run = tmp_path / "fixture-run"
    run.mkdir()
    evaluation = {
        "schema": "wnn.monitor.v1",
        "kind": "evaluation",
        "step": 50,
        "epoch": 1,
        "wall_time": "2026-11-07T10:00:00Z",
        "metrics": {
            "train_eval/loss": 0.9,
            "valid/loss": 1.1,
            "train_eval/wp": 0.4,
            "valid/wp": 0.3,
            "train_eval/wp/t0": 0.41,
            "valid/wp/t0": 0.31,
            "train_eval/wp/t1": 0.39,
            "valid/wp/t1": 0.29,
            **{
                f"{cohort}/{component}/{statistic}/{target}": 0.2
                for cohort in ("train_eval", "valid")
                for component in ("z", "reference", "mixture")
                for statistic in ("raw_mean", "weighted_raw_mean", "raw_std", "weighted_raw_std")
                for target in ("t0", "t1")
            },
        },
        "context": {
            "train_eval": {
                "row_mask": "all_after_warmup",
                "member_count": 32,
                "selection": "hash_rank_v1",
                "comparison_to_evaluation": {
                    "comparable_to_primary_valid": False,
                    "gap_is_calibrated_generalization_estimate": False,
                },
            },
            "valid": {"row_mask": "is_scored", "member_count": 1873},
        },
    }
    training = {
        **evaluation,
        "kind": "train_step",
        "wall_time": "2026-11-07T09:59:00Z",
        "metrics": {
            "train/loss": 0.95,
            "train/native_label_objective": 0.8,
            "train/objective_wmse_raw": 0.7,
            "train/objective_correlation_loss_raw": 0.2,
            "train/objective_correlation_surrogate/t0": 0.3,
            "train/objective_correlation_surrogate/t1": 0.4,
            "train/kd_loss": 0.1,
            "train/hint_loss": 0.05,
            "train/objective_weight/kd": 0.0,
            "train/objective_weight/hint": 0.0,
            "optimizer/lr/adamw": 0.001,
            "optimizer/gradient_norm_preclip": 0.7,
            "optimizer/was_clipped": 0,
        },
        "context": {"native_objective": {"objective_mode": "mixture_wmse_corr"}},
    }
    observation = {
        **training,
        "kind": "resources",
        "metrics": {"gpu/device_1/utilization_pct": 90, "observer/event_age_seconds": 45},
    }
    for name, event in (
        ("monitor_evaluations.jsonl", evaluation),
        ("monitor_events.jsonl", training),
        ("monitor_observer.jsonl", observation),
        ("monitor_rank0.jsonl", {**training, "context": {"wall_time_is_update_time": False}}),
    ):
        (run / name).write_text(json.dumps(event) + "\n", encoding="utf-8")
    before = {path.name: path.read_bytes() for path in run.iterdir()}
    output = tmp_path / "plots"
    result = export_plots(run, output)
    assert result["status"] == "exported"
    assert result["evaluation_role"] == "valid"
    assert result["model_evaluations_triggered"] == 0
    assert result["evaluation_context"]["valid"]["row_mask"] == "is_scored"
    assert result["train_eval_comparability"]["comparable_to_primary_valid"] is False
    assert (output / "train_eval_vs_evaluation_wmse.png").is_file()
    wp_plot = output / "train_eval_vs_evaluation_wp.svg"
    assert wp_plot.is_file()
    wp_markup = wp_plot.read_text(encoding="utf-8")
    assert "Train-eval in-sample subset (n=32, hash_rank_v1, all_after_warmup)" in wp_markup
    assert "not a calibrated generalization gap" in wp_markup
    assert (output / "gpu_utilization.png").is_file()
    assert (output / "native_rank0_objectives_by_update.png").is_file()
    online_plot = (output / "online_objectives.svg").read_text(encoding="utf-8")
    assert "Raw WMSE component (weight separate)" in online_plot
    assert "Raw correlation loss (weight separate)" in online_plot
    assert "Correlation surrogate t0" in online_plot
    assert "Raw KD term" not in online_plot
    native_updates = (output / "native_rank0_objectives_by_update.svg").read_text(encoding="utf-8")
    assert "Optimizer updates" in native_updates
    for component in ("z", "reference", "mixture"):
        assert (output / f"prediction_{component}_means_by_target.png").is_file()
        assert (output / f"prediction_{component}_spread_by_target.png").is_file()
    for name, content in before.items():
        assert (run / name).read_bytes() == content
        assert result["source_sha256"][name] == hashlib.sha256(content).hexdigest()
