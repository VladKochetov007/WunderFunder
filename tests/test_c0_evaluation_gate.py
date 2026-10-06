"""Synthetic lifecycle and byte-integrity checks for the C0 evaluation gate."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import evaluate_c0_after_exports as gate  # noqa: E402


def _hash(path: Path) -> str:
    """Hash synthetic fixture bytes using the same binding as the runner."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _expected_exports(artifact_root: Path) -> list[dict[str, object]]:
    """Return two synthetic seed-bound export registrations."""
    outputs = []
    for seed in (3, 4):
        run = f"candidate_s{seed}_f0"
        outputs.append(
            {
                "attempt": f"e503-bd26-c0-s{seed}-f0-valid-v1",
                "run": run,
                "split": "valid",
                "checkpoint_epoch": 1,
                "kind": "gpu",
                "parent_slice": "wnn-campaign.slice",
                "command": ["predict", run, "valid", "--epoch", "1"],
                "output": str(artifact_root / "cache" / f"pred_{run}_valid.npy"),
            }
        )
    return outputs


def _prepare(tmp_path: Path, *, finished: bool = True) -> tuple[Path, dict[str, object]]:
    """Create an isolated repository, two cache exports, and durable records."""
    evaluator_root = tmp_path / "frozen-evaluator"
    (evaluator_root / "scripts").mkdir(parents=True)
    grader_script = evaluator_root / "scripts/evaluate_bandit_continuation.py"
    grader_script.write_text(
        "import argparse\n"
        "from pathlib import Path\n"
        "parser = argparse.ArgumentParser()\n"
        "parser.add_argument('--artifact-root')\n"
        "parser.add_argument('--candidate-prefix')\n"
        "parser.add_argument('--candidate-id')\n"
        "parser.add_argument('--repository-root')\n"
        "parser.add_argument('--output-dir')\n"
        "args = parser.parse_args()\n"
        "out = Path(args.output_dir)\n"
        "out.mkdir()\n"
        "(out / 'report.json').write_text('{}\\n')\n"
        "(out / 'report.md').write_text('# synthetic\\n')\n"
        "(out / 'components.json').write_text('{}\\n')\n",
        encoding="utf-8",
    )
    e502_path = evaluator_root / "E502.json"
    e503_path = evaluator_root / "E503.json"
    sources_path = evaluator_root / "SOURCE_FAMILIES.json"
    e502_path.write_text('{"id":"E-502","status":"preregistered"}\n', encoding="utf-8")
    sources_path.write_text('{"families":[]}\n', encoding="utf-8")

    artifact_root = tmp_path / "artifacts"
    cache = artifact_root / "cache"
    cache.mkdir(parents=True)
    (artifact_root / "research/campaign_2026-09-29/artifacts/splits").mkdir(parents=True)
    split_path = artifact_root / "research/campaign_2026-09-29/artifacts/splits/fold0.json"
    split_path.write_text('{"synthetic":true}\n', encoding="utf-8")
    rows = np.array([17, 41], dtype=np.int64)
    np.savez_compressed(cache / "valid_targets.npz", row_group=rows)

    exports = _expected_exports(artifact_root)
    experiments = []
    for export in exports:
        experiments.append(
            {
                key: export[key]
                for key in ("attempt", "run", "split", "checkpoint_epoch", "command", "output")
            }
        )
        run_root = artifact_root / "artifacts/runs" / str(export["run"])
        run_root.mkdir(parents=True)
        checkpoint = run_root / "model_ep1.pt"
        checkpoint.write_bytes(f"checkpoint-{export['run']}".encode())
        prediction_path = Path(str(export["output"]))
        np.save(prediction_path, np.zeros((2, 20_000, 2), dtype=np.float32))
        prediction_sha = _hash(prediction_path)
        sidecar = {
            "run": export["run"],
            "split": "valid",
            "epoch": 1,
            "checkpoint_sha256": _hash(checkpoint),
            "prediction_sha256": prediction_sha,
            "split_manifest_sha256": _hash(split_path),
            "row_groups": rows.tolist(),
        }
        prediction_path.with_suffix(".json").write_text(json.dumps(sidecar), encoding="utf-8")

    e503_path.write_text(
        json.dumps({"id": "E-503", "status": "preregistered", "experiments": experiments}),
        encoding="utf-8",
    )
    subprocess.run(["git", "init", "-q", str(evaluator_root)], check=True)
    subprocess.run(
        ["git", "-C", str(evaluator_root), "config", "user.email", "gate-tests@example.invalid"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(evaluator_root), "config", "user.name", "Gate Tests"], check=True
    )
    subprocess.run(["git", "-C", str(evaluator_root), "add", "."], check=True)
    subprocess.run(
        ["git", "-C", str(evaluator_root), "commit", "-qm", "synthetic frozen evaluator"],
        check=True,
    )
    commit = subprocess.run(
        ["git", "-C", str(evaluator_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    attestation_path = tmp_path / "seed3-attestation.json"
    attestation_path.write_text(
        '{"decision":"accept_as_completed_with_guard_incident"}\n', encoding="utf-8"
    )
    runtime = tmp_path / "runtime"
    attempts_dir = runtime / "attempts"
    attempts_dir.mkdir(parents=True)
    now = datetime.now(UTC).isoformat()
    for export in exports:
        record = {
            "id": export["attempt"],
            "command": export["command"],
            "kind": "gpu",
            "parent_slice": "wnn-campaign.slice",
            "status": "finished" if finished else "running",
            "exit_status": 0 if finished else None,
            "heartbeat_utc": now,
        }
        (attempts_dir / f"{export['attempt']}.json").write_text(
            json.dumps(record), encoding="utf-8"
        )
    frozen_sources = {
        "repository_root": str(evaluator_root),
        "commit": commit,
        "grader": {
            "path": "scripts/evaluate_bandit_continuation.py",
            "sha256": _hash(grader_script),
        },
        "e502_preregistration": {"path": "E502.json", "sha256": _hash(e502_path)},
        "e503_preregistration": {"path": "E503.json", "sha256": _hash(e503_path)},
        "source_families": {"path": "SOURCE_FAMILIES.json", "sha256": _hash(sources_path)},
        "seed3_attestation": {"path": str(attestation_path), "sha256": _hash(attestation_path)},
    }
    plan = {
        "schema_version": 1,
        "poll_seconds": 0.01,
        "runtime": str(runtime),
        "artifact_root": str(artifact_root),
        "receipt": str(runtime / "gate.json"),
        "output_dir": str(tmp_path / "graded"),
        "frozen_sources": frozen_sources,
        "exports": exports,
    }
    plan_path = tmp_path / "gate-plan.json"
    plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    return plan_path, plan


def test_gate_waits_for_exact_exports_and_grades_once(tmp_path: Path) -> None:
    """Finished exact exports are hash-bound before the frozen command runs."""
    plan_path, plan = _prepare(tmp_path, finished=False)

    def finish_exports(_seconds: float) -> None:
        for export in plan["exports"]:
            attempt_path = Path(str(plan["runtime"])) / "attempts" / f"{export['attempt']}.json"
            record = json.loads(attempt_path.read_text())
            record.update(status="finished", exit_status=0)
            attempt_path.write_text(json.dumps(record), encoding="utf-8")

    receipt = gate.run_gate(plan_path, sleep_fn=finish_exports)
    assert receipt["status"] == "finished"
    assert len(receipt["cache_bindings"]) == 2
    assert (Path(str(plan["output_dir"])) / "report.json").is_file()
    with pytest.raises(FileExistsError):
        gate.run_gate(plan_path)
    saved = json.loads(Path(str(plan["receipt"])).read_text())
    assert saved["status"] == "finished"


def test_gate_records_nonzero_export_for_review_without_launching(tmp_path: Path) -> None:
    """A failed export is terminal review evidence, never an automatic retry."""
    plan_path, plan = _prepare(tmp_path)
    failed_path = Path(str(plan["runtime"])) / "attempts" / "e503-bd26-c0-s4-f0-valid-v1.json"
    failed = json.loads(failed_path.read_text())
    failed.update(status="failed", exit_status=1)
    failed_path.write_text(json.dumps(failed), encoding="utf-8")
    with pytest.raises(gate.GateError, match="needs review"):
        gate.run_gate(plan_path)
    receipt = json.loads(Path(str(plan["receipt"])).read_text())
    assert receipt["status"] == "needs_review"
    assert not Path(str(plan["output_dir"])).exists()


def test_gate_drain_waits_for_active_export_then_writes_terminal_receipt(
    tmp_path: Path,
) -> None:
    """A drain closes grading while allowing an active export to finish."""
    plan_path, plan = _prepare(tmp_path, finished=False)
    runtime = Path(str(plan["runtime"]))
    (runtime / "STOP_ADMISSION.json").write_text('{"reason":"test drain"}\n')
    decision, states = gate._inspect_attempts(
        runtime, plan["exports"], datetime.now(UTC), admission_closed=True
    )
    assert decision == "waiting"
    assert all(item["state"] == "active" for item in states)

    for export in plan["exports"]:
        record_path = runtime / "attempts" / f"{export['attempt']}.json"
        record = json.loads(record_path.read_text())
        record.update(status="finished", exit_status=0)
        record_path.write_text(json.dumps(record), encoding="utf-8")
    decision, states = gate._inspect_attempts(
        runtime, plan["exports"], datetime.now(UTC), admission_closed=True
    )
    assert decision == "drained"
    assert all(item["state"] == "success" for item in states)
    receipt = gate.run_gate(plan_path)
    assert receipt["status"] == "drained"
    assert json.loads(Path(str(plan["receipt"])).read_text())["status"] == "drained"
    assert not Path(str(plan["output_dir"])).exists()


def test_stale_export_is_reviewed_without_stopping_worker(tmp_path: Path) -> None:
    """A stale observer stops admission to grading but does not signal the export."""
    runtime = tmp_path / "runtime"
    record_dir = runtime / "attempts"
    record_dir.mkdir(parents=True)
    now = datetime.now(UTC)
    expected = _expected_exports(tmp_path)[0]
    record = {
        "id": expected["attempt"],
        "command": expected["command"],
        "kind": "gpu",
        "parent_slice": "wnn-campaign.slice",
        "status": "running",
        "heartbeat_utc": (now - timedelta(seconds=181)).isoformat(),
    }
    (record_dir / f"{expected['attempt']}.json").write_text(json.dumps(record))
    with pytest.raises(gate.GateError, match="stale worker heartbeat"):
        gate._attempt_state(runtime, expected, now)
    assert (
        json.loads((record_dir / f"{expected['attempt']}.json").read_text())["status"] == "running"
    )


def test_cache_preflight_checks_sidecar_hashes_and_rows_without_loading_predictions(
    tmp_path: Path,
) -> None:
    """The gate reads only the NPY header and byte hash, leaving values to the grader."""
    _plan_path, plan = _prepare(tmp_path)
    bindings = gate._cache_bindings(plan)
    assert len(bindings) == 2
    assert bindings[0]["shape"] == [2, 20_000, 2]
    assert bindings[0]["dtype"] == "float32"
