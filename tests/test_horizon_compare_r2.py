from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

_RECEIVING = (
    Path(__file__).parents[1]
    / "research/experiment_drafts/transformer-optimizer-horizon-20261108/receiving"
)
sys.path.insert(0, str(_RECEIVING))

import admission  # noqa: E402
import compare_r2  # noqa: E402


def _cell_fixture(tmp_path: Path, name: str, arm: str, seed: int) -> dict:
    fit = tmp_path / "runs" / name / "fit"
    fit.mkdir(parents=True)
    config = {"chunk_rows": 512 if arm == "C2" else 256, "objective_mode": "standalone_labels_wmse_v2"}
    config_path = tmp_path / "configs" / f"{name}.json"
    config_path.parent.mkdir(exist_ok=True)
    config_path.write_text(json.dumps(config, sort_keys=True))
    config_sha = hashlib.sha256(config_path.read_bytes()).hexdigest()
    model = f"checkpoint-{name}".encode()
    (fit / "model.pt").write_bytes(model)
    checkpoint_sha = hashlib.sha256(model).hexdigest()
    identity = {
        "source_sha256": compare_r2.TRAINING_SOURCE_SHA256,
        "config_sha256": "c" * 64,
        "data_sha256": "d" * 64,
        "world_size": 2,
    }
    terminal = {
        "status": "completed",
        "completed_epochs": 4,
        "global_updates": 3280,
        "model_sha256": checkpoint_sha,
        "identity": identity,
    }
    score = {
        "split_sha256": admission.SPLIT_SHA,
        "sequences": 1873,
        "score": {
            "metric_id": "global_wp_clip2_v1",
            "selected_rows": 3_528_273,
            "wp": 0.66,
            "t0": 0.65,
            "t1": 0.67,
        },
        "producer": {
            "checkpoint_sha256": checkpoint_sha,
            "config": config,
            "provenance": {"identity": identity},
            "teacher": None,
        },
    }
    (fit / "terminal_status.json").write_text(json.dumps(terminal))
    (fit / "score.json").write_text(json.dumps(score))
    monitor = {
        "kind": "evaluation",
        "epoch": 4,
        "step": 3280,
        "metrics": {
            "valid/sequence_count": 1873,
            "valid/selected_rows": 3_528_273,
            "valid/wp": 0.6675113021465455,
            "valid/z/wp": 0.6675113021465455,
            "valid/wp/t0": 0.6585251436296471,
            "valid/wp/t1": 0.6764974606634441,
            "valid/loss": 1.0054704748780694,
            "valid/z/wmse": 1.0054704889970745,
        },
    }
    (fit / "monitor_evaluations.jsonl").write_text(json.dumps(monitor) + "\n")
    return {
        "cell": name,
        "arm": arm,
        "seed": seed,
        "optimizer": "adamw" if arm == "C1" else "muon",
        "config_path": str(config_path),
        "config_sha256": config_sha,
        "normalized_config_sha256": identity["config_sha256"],
    }


def _panel_fixture(tmp_path: Path) -> tuple[dict, object]:
    cells = [
        _cell_fixture(tmp_path, name, arm, seed)
        for name, arm, seed in compare_r2.CELL_IDENTITIES
    ]
    plan = {"runtime": str(tmp_path), "expected_training_source_sha256": compare_r2.TRAINING_SOURCE_SHA256, "cells": cells}
    return plan, lambda path: json.loads(path.read_text())


def test_real_wmse_roundoff_and_raw_z_wp_alias_pass_monitor_check() -> None:
    # Values are from the native C0/seed203101 E1 event; the WMSE difference is
    # about 1.4e-8 while the raw-z WP alias is exact.
    event = {
        "kind": "evaluation",
        "epoch": 4,
        "step": 3280,
        "metrics": {
            "valid/sequence_count": 1873,
            "valid/selected_rows": 3_528_273,
            "valid/wp": 0.6675113021465455,
            "valid/z/wp": 0.6675113021465455,
            "valid/wp/t0": 0.6585251436296471,
            "valid/wp/t1": 0.6764974606634441,
            "valid/loss": 1.0054704748780694,
            "valid/z/wmse": 1.0054704889970745,
        },
    }
    assert compare_r2._validate_monitor(event, "C0") == event["metrics"]


def test_monitor_rejects_out_of_tolerance_or_nonfinite_wmse() -> None:
    event = {
        "kind": "evaluation",
        "epoch": 4,
        "step": 3280,
        "metrics": {
            "valid/sequence_count": 1873,
            "valid/selected_rows": 3_528_273,
            "valid/wp": 0.7,
            "valid/z/wp": 0.7,
            "valid/wp/t0": 0.68,
            "valid/wp/t1": 0.72,
            "valid/loss": 1.0,
            "valid/z/wmse": 1.01,
        },
    }
    with pytest.raises(ValueError, match="raw-z monitor"):
        compare_r2._validate_monitor(event, "C0")
    bad = event | {"metrics": event["metrics"] | {"valid/loss": float("nan")}}
    with pytest.raises(ValueError, match="raw-z monitor"):
        compare_r2._validate_monitor(bad, "C0")


def test_comparison_rechecks_all_six_model_bytes_and_score_hashes(tmp_path: Path) -> None:
    plan, load_config = _panel_fixture(tmp_path)
    value = compare_r2.build_comparison(
        plan,
        load_config,
        script_sha256="a" * 64,
        original_grader_sha256=compare_r2.ORIGINAL_GRADER_SHA256,
        receiver_exit_sha256="b" * 64,
    )
    assert set(value["cells"]) == {name for name, _, _ in compare_r2.CELL_IDENTITIES}
    for cell in plan["cells"]:
        record = value["cells"][cell["cell"]]
        fit = tmp_path / "runs" / cell["cell"] / "fit"
        assert record["checkpoint_sha256"] == admission.digest(fit / "model.pt")
        assert record["score_sha256"] == admission.digest(fit / "score.json")
    assert value["comparison_repair"]["wmse_tolerance"] == {"relative": 1e-6, "absolute": 1e-7}
    observed = value["comparison_repair"]["wmse_roundoff_by_cell"][plan["cells"][0]["cell"]]
    assert observed["absolute_difference"] == pytest.approx(1.4119005076196345e-08)


def test_comparison_rejects_wrong_checkpoint_binding(tmp_path: Path) -> None:
    plan, load_config = _panel_fixture(tmp_path)
    score_path = tmp_path / "runs" / plan["cells"][0]["cell"] / "fit" / "score.json"
    score = json.loads(score_path.read_text())
    score["producer"]["checkpoint_sha256"] = "0" * 64
    score_path.write_text(json.dumps(score))
    with pytest.raises(ValueError, match="ancestry"):
        compare_r2.build_comparison(
            plan,
            load_config,
            script_sha256="a" * 64,
            original_grader_sha256=compare_r2.ORIGINAL_GRADER_SHA256,
            receiver_exit_sha256="b" * 64,
        )


def test_exclusive_comparison_write_never_overwrites_existing_bytes(tmp_path: Path) -> None:
    path = tmp_path / "comparison.json"
    value = {"schema": "test", "status": "complete", "metric_id": "test", "split_sha256": "a", "cells": {}, "paired_deltas": []}
    assert compare_r2.write_comparison_once(path, value)
    before = path.read_bytes()
    assert not compare_r2.write_comparison_once(path, value)
    with pytest.raises(FileExistsError, match="does not match"):
        compare_r2.write_comparison_once(path, value | {"cells": {"wrong": {}}})
    assert path.read_bytes() == before


def test_existing_r2_metadata_must_match_expected_source_and_exit_proof(tmp_path: Path) -> None:
    path = tmp_path / "comparison.json"
    core = {
        "schema": "wnn.transformer-optimizer-horizon-comparison.v1",
        "status": "complete_exposed_fixed4E_screen",
        "metric_id": "global_wp_clip2_v1",
        "split_sha256": admission.SPLIT_SHA,
        "cells": {},
        "paired_deltas": [],
    }
    metadata = {
        "schema": compare_r2.REPAIR_SCHEMA,
        "script_sha256": "a" * 64,
        "original_grader_sha256": compare_r2.ORIGINAL_GRADER_SHA256,
        "receiver_exit_receipt_sha256": "b" * 64,
    }
    proposed = core | {"comparison_repair": metadata}
    path.write_text(json.dumps(core))  # A verified canonical R1 comparison may no-op.
    assert not compare_r2.write_comparison_once(path, proposed)
    path.write_text(json.dumps(core | {"comparison_repair": metadata | {"script_sha256": "0" * 64}}))
    with pytest.raises(FileExistsError, match="does not match"):
        compare_r2.write_comparison_once(path, proposed)


def test_receiver_exit_receipt_accepts_zombie_and_rejects_wrong_birth(tmp_path: Path) -> None:
    path = tmp_path / "receiver-exit.json"
    authority = {"deadline_epoch": compare_r2.FIXED_DEADLINE_EPOCH}
    plan = {"deadline_epoch": compare_r2.FIXED_DEADLINE_EPOCH}
    receipt = {
        "schema": compare_r2.RECEIVER_EXIT_SCHEMA,
        "observed_absent": True,
        "receiver_pid": compare_r2.RECEIVER_PID,
        "receiver_birth": compare_r2.RECEIVER_BIRTH,
        "observed_status": "exited_zombie",
        "plan_sha256": compare_r2.PLAN_SHA256,
        "source_commit": compare_r2.SOURCE_COMMIT,
        "authority_sha256": compare_r2.AUTHORITY_SHA256,
        "observed_at_epoch": compare_r2.FIRST_PROFILE_EPOCH + 1,
    }
    path.write_text(json.dumps(receipt))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert compare_r2._valid_receiver_exit(path, digest, plan, authority) == receipt
    wrong = receipt | {"receiver_birth": compare_r2.RECEIVER_BIRTH + 1}
    path.write_text(json.dumps(wrong))
    with pytest.raises(ValueError, match="receiver-exit receipt"):
        compare_r2._valid_receiver_exit(path, hashlib.sha256(path.read_bytes()).hexdigest(), plan, authority)
