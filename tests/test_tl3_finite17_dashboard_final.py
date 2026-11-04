from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from scripts import tl3_finite17_dashboard_final as final_view
from scripts import tl3_finite17_dashboard_final_validation as final_validation
from scripts.tl3_finite17_dashboard_core import _canonical_sha256, _sha256_file
from scripts.tl3_finite17_github_posts import _terminal_receipt

WEIGHTS = {
    "w18final_distilled_s5": 0.34,
    "fttall_tlead_s5": 0.33,
    "fttall_tlead_s6": 0.33,
}


def _fixture(tmp_path: Path, *, wrong_weight: bool = False) -> tuple[Any, ...]:
    root = tmp_path / "flat"
    state_dir = root / "finite17-state"
    runs_dir = tmp_path / "dashboard-runs"
    scripts_dir = root / "scripts"
    for path in (state_dir, runs_dir, scripts_dir, root / "cache"):
        path.mkdir(parents=True, exist_ok=True)
    recipe_path = scripts_dir / "tl3_chronological_recipes.py"
    recipe_path.write_text(
        "PREFIX = 'tl3c_'\nFINAL_WEIGHTS = " + repr(WEIGHTS) + "\n", encoding="utf-8"
    )
    scorer_path = scripts_dir / "score_tl3_chronological.py"
    scorer_path.write_text("# scorer bytes pinned by plan\n", encoding="utf-8")
    files = {
        "scripts/tl3_chronological_recipes.py": _sha256_file(recipe_path),
        "scripts/score_tl3_chronological.py": _sha256_file(scorer_path),
    }
    split_sha = "b" * 64
    source_sha = "a" * 64
    final_runs = [f"tl3c_{name}" for name in WEIGHTS]
    nodes = []
    status_nodes: dict[str, Any] = {}
    exports: dict[str, Any] = {}
    score_members = []
    valid = list(range(1_873))
    for index, (name, run) in enumerate(zip(WEIGHTS, final_runs, strict=True)):
        checkpoint_sha = str(index + 1) * 64
        prediction_bytes = f"prediction {run}".encode()
        prediction = root / "cache" / f"pred_{run}_valid.npy"
        prediction.write_bytes(prediction_bytes)
        prediction_sha = _sha256_file(prediction)
        sidecar = prediction.with_suffix(".json")
        sidecar.write_text(
            json.dumps(
                {
                    "run": run,
                    "split": "valid",
                    "split_manifest_sha256": split_sha,
                    "campaign_split_manifest_sha256": split_sha,
                    "campaign_export_scope": "evaluation",
                    "feature_only": True,
                    "checkpoint_file": "model.pt",
                    "checkpoint_sha256": checkpoint_sha,
                    "prediction_sha256": prediction_sha,
                    "row_groups": valid,
                }
            ),
            encoding="utf-8",
        )
        nodes.append({"id": name, "run": run, "endpoint": 3, "epochs": 4})
        status_nodes[name] = {"state": "complete", "phase": "fit_complete"}
        exports[run] = {
            "state": "complete",
            "sha256": prediction_sha,
            "sidecar_sha256": _sha256_file(sidecar),
            "checkpoint_sha256": checkpoint_sha,
        }
        score_members.append(
            {
                "run": run,
                "weight": 0.33 if wrong_weight and index == 0 else WEIGHTS[name],
                "checkpoint_sha256": checkpoint_sha,
                "prediction_sha256": prediction_sha,
                "epoch": 3,
            }
        )
    identity = {
        "sha256": _canonical_sha256(files),
        "files": files,
        "training_source_sha256": source_sha,
    }
    plan: dict[str, Any] = {
        "schema": "wnn.tl3-finite17-plan.v1",
        "root": str(root),
        "state_dir": str(state_dir),
        "source_identity": identity,
        "scorer_sha256": files["scripts/score_tl3_chronological.py"],
        "recipe_sha256": files["scripts/tl3_chronological_recipes.py"],
        "split": {"split_manifest_sha256": split_sha},
        "nodes": nodes,
        "final_runs": final_runs,
    }
    plan["sha256"] = _canonical_sha256(plan)
    score = {
        "schema": "wnn.tl3-chronological-score.v1",
        "split_manifest_sha256": split_sha,
        "valid_sequences": 1_873,
        "selected_rows": 3_528_273,
        "members": score_members,
        "scores": {
            "fixed_tl3_blend": {
                "wp": 0.68,
                "t0": 0.67,
                "t1": 0.69,
                "selected_rows": 3_528_273,
                "metric_id": "global_wp_clip2_v1",
            }
        },
        "clean_gru_reference_wp": 0.6778561232160607,
        "blend_delta_to_clean_gru": 0.68 - 0.6778561232160607,
        "exposure": "exposed official valid; fixed coefficients",
    }
    score_path = root / "final-score.json"
    score_path.write_text(json.dumps(score), encoding="utf-8")
    score_state = {"state": "complete", "path": str(score_path), "sha256": _sha256_file(score_path)}
    status = {
        "schema": "wnn.tl3-finite17-state.v1",
        "status": "completed",
        "plan_sha256": plan["sha256"],
        "fits_used": 17,
        "nodes": status_nodes,
        "final_exports": exports,
        "score": score_state,
    }
    status_path = state_dir / "status.json"
    status_path.write_text(json.dumps(status), encoding="utf-8")
    (state_dir / "terminal.json").write_text(
        json.dumps(
            {
                "schema": "wnn.tl3-finite17-terminal.v1",
                "status": "completed",
                "plan_sha256": plan["sha256"],
                "fits_used": 17,
                "state_sha256": _sha256_file(status_path),
                "score": score_state,
            }
        ),
        encoding="utf-8",
    )
    return root, state_dir, runs_dir, plan, status, valid


def test_final_blend_view_is_one_idempotent_official_valid_event(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, state_dir, runs_dir, plan, status, valid = _fixture(tmp_path)
    monkeypatch.setattr(
        final_validation,
        "_verify_run",
        lambda _root, _plan, node, *_args: {
            "model_sha256": str(list(WEIGHTS).index(node["id"]) + 1) * 64
        },
    )

    result = final_view.publish_final_view(
        root, state_dir, runs_dir, plan, status, list(range(10_325)), valid
    )
    assert result is not None
    view = runs_dir / "tl3c_final_fixed_blend"
    events = [json.loads(line) for line in (view / "monitor_events.jsonl").read_text().splitlines()]
    evaluation = next(event for event in events if event["kind"] == "evaluation")
    assert evaluation["metrics"] == {
        "valid/score": 0.68,
        "valid/score/t0": 0.67,
        "valid/score/t1": 0.69,
        "valid/selected_rows": 3_528_273,
    }
    assert not any(key.startswith("train") for key in evaluation["metrics"])
    assert evaluation["context"]["evaluation"]["row_mask"] == "is_scored"

    repeated = final_view.publish_final_view(
        root, state_dir, runs_dir, plan, status, list(range(10_325)), valid
    )
    assert repeated == {
        "run": "tl3c_final_fixed_blend",
        "score_sha256": result["score_sha256"],
        "events_added": 0,
    }


def test_final_blend_rejects_score_member_weight_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, state_dir, runs_dir, plan, status, valid = _fixture(tmp_path, wrong_weight=True)
    monkeypatch.setattr(
        final_validation,
        "_verify_run",
        lambda _root, _plan, node, *_args: {
            "model_sha256": str(list(WEIGHTS).index(node["id"]) + 1) * 64
        },
    )

    with pytest.raises(final_view.BridgeError, match="member is not bound"):
        final_view.publish_final_view(
            root, state_dir, runs_dir, plan, status, list(range(10_325)), valid
        )
    assert not (runs_dir / "tl3c_final_fixed_blend").exists()


def test_missing_terminal_receipt_keeps_final_view_and_watcher_pending(tmp_path: Path) -> None:
    root, state_dir, runs_dir, plan, status, valid = _fixture(tmp_path)
    (state_dir / "terminal.json").unlink()

    assert (
        final_view.publish_final_view(
            root, state_dir, runs_dir, plan, status, list(range(10_325)), valid
        )
        is None
    )
    assert not (runs_dir / "tl3c_final_fixed_blend").exists()
    assert _terminal_receipt(state_dir, status, plan) is None
