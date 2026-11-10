from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

_REPORTING = (
    Path(__file__).parents[1]
    / "research/experiment_drafts/transformer-optimizer-horizon-20261108/reporting"
)
sys.path.insert(0, str(_REPORTING))

import collect_readouts  # noqa: E402
import horizon_progress as progress  # noqa: E402
import report_horizon  # noqa: E402


def _plan(runtime: Path) -> dict:
    cells = []
    for name in collect_readouts.EXPECTED_CELLS:
        arm = name[:2]
        seed = int(name[-6:])
        cells.append(
            {
                "cell": name,
                "arm": arm,
                "seed": seed,
                "optimizer": "adamw" if arm == "C1" else "muon",
                "pair": "2,3" if arm == "C1" else ("4,5" if arm == "C2" else "0,1"),
                "config_sha256": progress.EXPECTED_CONFIGS[name],
                "normalized_config_sha256": hashlib.sha256(name.encode()).hexdigest(),
            }
        )
    return {
        "schema": "wnn.transformer-optimizer-horizon-panel.v1",
        "source_commit": collect_readouts.SOURCE_COMMIT,
        "source_manifest_sha256": collect_readouts.MANIFEST_SHA256,
        "expected_training_source_sha256": progress.SOURCE_SHA256,
        "split_sha256": collect_readouts.SPLIT_SHA256,
        "runtime": str(runtime),
        "authority_sha256": "a" * 64,
        "cells": cells,
    }


def _write_plan(runtime: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, str]:
    plan = _plan(runtime)
    path = runtime / "fit-operator-r1" / "plan.json"
    path.parent.mkdir(parents=True)
    payload = json.dumps(plan, sort_keys=True).encode()
    path.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    monkeypatch.setattr(collect_readouts, "PLAN_SHA256", digest)
    return path, digest


def test_readout_collector_reads_only_six_score_terminal_pairs_and_comparison(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan_path, digest = _write_plan(tmp_path, monkeypatch)
    for name in collect_readouts.EXPECTED_CELLS:
        fit = tmp_path / "runs" / name / "fit"
        fit.mkdir(parents=True)
        for filename in ("terminal_status.json", "score.json"):
            (fit / filename).write_text(json.dumps({"cell": name, "artifact": filename}))
        (fit / "model.pt").write_bytes(b"never collected")
    (tmp_path / "comparison.json").write_text(json.dumps({"complete": True}))
    value = collect_readouts.collect_readouts(tmp_path, plan_path, digest)
    assert value["total_bytes"] < collect_readouts.MAX_TOTAL_BYTES
    assert value["schema"] == "wnn.transformer-optimizer-horizon-readouts.v1"
    assert len(value["files"]) == 13
    assert value["files"]["comparison.json"]["status"] == "read"
    assert all(
        value["files"][f"runs/{name}/fit/{filename}"]["status"] == "read"
        for name in collect_readouts.EXPECTED_CELLS
        for filename in ("terminal_status.json", "score.json")
    )
    assert not any("model.pt" in path for path in value["files"])


def test_readout_collector_rejects_wrong_panel_and_bounds_each_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan_path, digest = _write_plan(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="byte digest"):
        collect_readouts.collect_readouts(tmp_path, plan_path, "0" * 64)
    fit = tmp_path / "runs" / collect_readouts.EXPECTED_CELLS[0] / "fit"
    fit.mkdir(parents=True)
    too_large = fit / "score.json"
    too_large.write_bytes(b" " * (collect_readouts.MAX_FILE_BYTES + 1))
    value = collect_readouts.collect_readouts(tmp_path, plan_path, digest)
    key = f"runs/{collect_readouts.EXPECTED_CELLS[0]}/fit/score.json"
    assert value["files"][key]["status"] == "invalid"
    assert "byte bound" in value["files"][key]["reason"]


def test_readout_collector_accepts_thirteen_valid_files_over_eight_mib(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan_path, digest = _write_plan(tmp_path, monkeypatch)
    payload = json.dumps({"padding": "x" * 850_000}).encode()
    for name in collect_readouts.EXPECTED_CELLS:
        fit = tmp_path / "runs" / name / "fit"
        fit.mkdir(parents=True)
        for filename in ("terminal_status.json", "score.json"):
            (fit / filename).write_bytes(payload)
    (tmp_path / "comparison.json").write_bytes(payload)
    value = collect_readouts.collect_readouts(tmp_path, plan_path, digest)
    assert 8 * 1_048_576 < value["total_bytes"] <= 16 * 1_048_576
    assert all(entry["status"] == "read" for entry in value["files"].values())


def _grade_fixture() -> tuple[dict, dict, dict, dict]:
    ids = {
        "source": progress.SOURCE_SHA256,
        "config": "c" * 64,
        "data": "d" * 64,
        "split": progress.SPLIT_SHA256,
    }
    identity = {
        "source_sha256": ids["source"],
        "config_sha256": ids["config"],
        "data_sha256": ids["data"],
        "world_size": 2,
    }
    config = {"objective_mode": "standalone_labels_wmse_v2"}
    checkpoint = "e" * 64
    terminal = {
        "status": "completed",
        "completed_epochs": 4,
        "global_updates": 3280,
        "model_sha256": checkpoint,
        "identity": identity,
    }
    score = {
        "split_sha256": progress.SPLIT_SHA256,
        "sequences": 1873,
        "score": {"metric_id": "global_wp_clip2_v1", "selected_rows": 3_528_273, "wp": 0.7, "t0": 0.68, "t1": 0.72},
        "producer": {
            "checkpoint_sha256": checkpoint,
            "config": config,
            "provenance": {"identity": identity},
            "teacher": None,
        },
    }
    metadata = {"config": config}
    return ids, terminal, score, metadata


def test_canonical_grade_requires_fixed4e_fullvalid_and_matching_producer_hash() -> None:
    ids, terminal, score, metadata = _grade_fixture()
    grade = progress._canonical_grade(terminal, (score, "f" * 64), ids, metadata)
    assert grade is not None
    assert grade["status"] == "grade_producer_verified_pending_final_byte_audit"
    assert progress._canonical_grade(terminal, (score | {"sequences": 1872}, "f" * 64), ids, metadata) is None
    wrong = score | {"producer": score["producer"] | {"checkpoint_sha256": "0" * 64}}
    assert progress._canonical_grade(terminal, (wrong, "f" * 64), ids, metadata) is None
    assert progress._canonical_grade(terminal | {"completed_epochs": 30}, (score, "f" * 64), ids, metadata) is None


def test_monitor_progress_is_provisional_only_with_full_valid_raw_z_coverage() -> None:
    context = {
        "split_sha256": progress.SPLIT_SHA256,
        "evaluation_member_count": 1873,
        "evaluation_mask": "is_scored after warmup",
    }
    metadata = {"provenance": {"data": {"report_only_monitor": context}}}
    metrics = {
        "valid/sequence_count": 1873,
        "valid/selected_rows": 3_528_273,
        "valid/wp": 0.7,
        "valid/wp/t0": 0.68,
        "valid/wp/t1": 0.72,
        "valid/z/wp": 0.7,
        "valid/wmse": 1.2,
        "valid/z/wmse": 1.2,
    }
    event = {
        "kind": "evaluation",
        "epoch": 1,
        "step": 820,
        "context": context,
        "metrics": metrics,
    }
    assert progress._full_valid_events([event], metadata) == [event]
    incomplete = event | {"metrics": metrics | {"valid/selected_rows": 3_528_272}}
    assert progress._full_valid_events([incomplete], metadata) == []


def test_actual_producer_monitor_event_is_accepted_with_raw_z_wp_alias() -> None:
    # Selected values are from the actual C0/seed203101 E1 mirror event. Its
    # aggregate WP aliases agree exactly; aggregate WMSE differs slightly by
    # reduction order and is not a contract alias.
    context = {
        "split_sha256": progress.SPLIT_SHA256,
        "evaluation_member_count": 1873,
        "evaluation_mask": "is_scored after warmup",
        "metric_id": "global_wp_clip2_v1",
        "generalization_gap_comparable": False,
    }
    metadata = {"provenance": {"data": {"report_only_monitor": context}}}
    metrics = {
        "valid/sequence_count": 1873,
        "valid/selected_rows": 3_528_273,
        "valid/wp": 0.6675113021465455,
        "valid/wp/t0": 0.6585251436296471,
        "valid/wp/t1": 0.6764974606634441,
        "valid/z/wp": 0.6675113021465455,
        "valid/wmse": 1.0054704748780694,
        "valid/z/wmse": 1.0054704889970745,
    }
    event = {
        "schema": "wnn.monitor.v1",
        "kind": "evaluation",
        "epoch": 1,
        "step": 820,
        "context": context,
        "metrics": metrics,
    }
    assert progress._full_valid_events([event], metadata) == [event]


def test_checkpoint_byte_audit_requires_full_comparison_and_score_hashes() -> None:
    rows = []
    cells = {}
    for name in collect_readouts.EXPECTED_CELLS:
        grade = {"wp": 0.7, "checkpoint_sha256": "e" * 64, "score_sha256": "f" * 64}
        rows.append({"cell": name, "grade": grade})
        cells[name] = {
            "checkpoint_sha256": grade["checkpoint_sha256"],
            "score_sha256": grade["score_sha256"],
            "official_raw_z": {"wp": grade["wp"]},
        }
    deltas = [
        {"seed": seed, "arm": arm, "delta_wp": 0.01}
        for seed in (203101, 203102)
        for arm in ("C1", "C2")
    ]
    comparison = {
        "schema": "wnn.transformer-optimizer-horizon-comparison.v1",
        "status": "complete_exposed_fixed4E_screen",
        "metric_id": "global_wp_clip2_v1",
        "split_sha256": progress.SPLIT_SHA256,
        "cells": cells,
        "paired_deltas": deltas,
    }
    assert progress._comparison_valid((comparison, "a" * 64), rows)
    corrupt = comparison | {"cells": cells | {collect_readouts.EXPECTED_CELLS[0]: {"score_sha256": "0" * 64}}}
    assert not progress._comparison_valid((corrupt, "a" * 64), rows)
    assert not progress._comparison_valid((comparison | {"cells": {}}, "a" * 64), rows)


def test_progress_render_distinguishes_provisional_monitor_from_fixed_grade() -> None:
    rows = [
        {
            "cell": name,
            "optimizer": "muon",
            "chunk_rows": 256,
            "seed": 203101,
            "phase": "latest producer phase: training",
                "fit_complete": False,
            "updates": 820,
            "epoch": 1.0,
            "latest_valid_is_provisional": name == collect_readouts.EXPECTED_CELLS[0],
            "latest_valid_epoch": 1 if name == collect_readouts.EXPECTED_CELLS[0] else None,
            "latest_valid": {"valid/wp": 0.7, "valid/wp/t0": 0.68, "valid/wp/t1": 0.72},
            "grade": None,
            "grade_status": "canonical official score pending or unverified",
            "byte_audit": False,
            "source_age_seconds": 10,
            "probe": {},
            "probe_step": None,
            "latest_train": {},
            "gpu_utilization_pct": [],
            "gpu_memory_gib": [],
        }
        for name in collect_readouts.EXPECTED_CELLS
    ]
    body = progress.render(
        {
            "observed_at_utc": "2026-11-08T00:00:00+00:00",
            "panel": {"manifest_sha256": "a" * 64, "source_commit": collect_readouts.SOURCE_COMMIT},
            "cells": rows,
            "comparison_present": False,
            "terminal_signature": "b" * 64,
        }
    )
    assert "Latest full-valid monitor (provisional WP)" in body
    assert "E1 0.700000 (0.680000, 0.720000)" in body
    assert "Fixed E4 official WP | Status" in body
    assert "Canonical grades require" in body
    assert "0/6 producer-verified official grades" in body
    assert "0/6 final checkpoint byte audits" in body


def test_dry_run_never_constructs_github_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    native = tmp_path / "native"
    native.mkdir()
    config = tmp_path / "mirror.json"
    config.write_text(json.dumps({"native_root": str(native)}))
    plan = {"sha256": "a" * 64, "plan": {"authority_sha256": "b" * 64}}
    monkeypatch.setattr(report_horizon, "load_panel", lambda *_: plan)
    monkeypatch.setattr(report_horizon, "_authority", lambda *_: {"deadline_epoch": 100})
    monkeypatch.setattr(
        report_horizon,
        "_shared_helpers",
        lambda *_: (_ for _ in ()).throw(AssertionError("dry run loaded live publication helpers")),
    )
    monkeypatch.setattr(
        report_horizon,
        "snapshot",
        lambda *_: {
            "host": "test",
            "observed_at_utc": "2026-11-08T00:00:00+00:00",
            "panel": {"manifest_sha256": "a" * 64, "source_commit": collect_readouts.SOURCE_COMMIT},
            "cells": [],
            "comparison_present": False,
        },
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "report_horizon.py",
            "--mirror-config",
            str(config),
            "--receiver-plan",
            str(tmp_path / "plan.json"),
            "--authority",
            str(tmp_path / "authority.json"),
            "--pr",
            "72",
            "--state",
            str(tmp_path / "state.json"),
            "--dry-run",
            "--once",
        ],
    )
    (tmp_path / "authority.json").write_text("{}")
    assert report_horizon.main() == 0
    assert not (tmp_path / "state.json").exists()
    assert (tmp_path / "horizon-pr72-dry-run.md").is_file()
