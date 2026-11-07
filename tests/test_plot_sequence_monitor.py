from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts.plot_sequence_monitor import export_plots


def test_local_plot_export_preserves_native_journals_and_role(tmp_path: Path) -> None:
    """Synthetic schema fixtures test rendering only, without any model execution."""
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
        },
        "context": {
            "train_eval": {"row_mask": "all_after_warmup"},
            "valid": {"row_mask": "is_scored"},
        },
    }
    training = {
        **evaluation,
        "kind": "train_step",
        "wall_time": "2026-11-07T09:59:00Z",
        "metrics": {
            "train/loss": 0.95,
            "optimizer/lr/adamw": 0.001,
            "optimizer/gradient_norm_preclip": 0.7,
            "optimizer/was_clipped": 0,
        },
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
    ):
        (run / name).write_text(json.dumps(event) + "\n", encoding="utf-8")
    before = {path.name: path.read_bytes() for path in run.iterdir()}
    output = tmp_path / "plots"
    result = export_plots(run, output)
    assert result["status"] == "exported"
    assert result["evaluation_role"] == "valid"
    assert result["model_evaluations_triggered"] == 0
    assert result["evaluation_context"]["valid"]["row_mask"] == "is_scored"
    assert (output / "train_eval_vs_evaluation_wmse.png").is_file()
    assert (output / "train_eval_vs_evaluation_wp.svg").is_file()
    assert (output / "gpu_utilization.png").is_file()
    for name, content in before.items():
        assert (run / name).read_bytes() == content
        assert result["source_sha256"][name] == hashlib.sha256(content).hexdigest()
