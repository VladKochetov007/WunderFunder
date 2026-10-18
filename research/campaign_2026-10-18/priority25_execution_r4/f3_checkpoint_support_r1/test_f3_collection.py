from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import collect_f3_checkpoint as collector
import f3_checkpoint_common as common
import f3_checkpoint_evidence as evidence
from f3_checkpoint_common import arm_spec, expected_counts, validate_plan

PLAN = (
    Path(__file__).resolve().parents[1] / "f3_beleriand_preparation_r1" / "F3_BELERIAND_PLAN.json"
)


def _history(epochs: int) -> list[dict[str, object]]:
    rows = []
    for epoch in range(epochs):
        one = expected_counts(1)
        rows.append(
            {
                "epoch": epoch,
                "train_seconds": 1.0,
                "evaluation": "withheld",
                "update_diagnostics": {
                    "epoch": one,
                    "cumulative": expected_counts(epoch + 1),
                    "grad_stats": {"epoch": epoch, "steps": 14_880, "nonfinite_steps": 0},
                },
                "fit_summary": {
                    "objective_mean": 1.0,
                    "optimizer_windows": 14_880,
                    "optimizer_updates": one,
                    "learning_rate_by_group": [
                        {"group": 0, "first": 0.1, "last": 0.01, "mean": 0.05}
                    ],
                },
            }
        )
    return rows


def _write_boundary(base: Path, arm: str, phase: str, parent: Path | None = None) -> Path:
    plan = validate_plan(PLAN)
    spec = arm_spec(plan, arm, phase)
    source = base / "data/artifacts/runs" / str(spec["run"])
    source.mkdir(parents=True)
    checkpoint_names = [
        name for name in spec["artifacts"] if name.startswith("model_") or name == "model.pt"
    ]
    checkpoint_hashes = {}
    for name in spec["artifacts"]:
        payload = f"tiny fixture {arm} {phase} {name}".encode()
        (source / name).write_bytes(payload)
        if name in checkpoint_names:
            checkpoint_hashes[name] = hashlib.sha256(payload).hexdigest()
    epochs = int(spec["completed_epochs"])
    provenance: dict[str, object] = {
        "cumulative_updates": expected_counts(epochs),
        "checkpoint_sha256": checkpoint_hashes,
    }
    if phase == "terminal":
        assert parent is not None
        parent_path = parent / "training_state.pt"
        provenance["resumed_from"] = {
            "path": str(
                base / "data/artifacts/runs" / str(spec["stage1_run"]) / "training_state.pt"
            ),
            "sha256": common.sha256_file(parent_path),
            "next_epoch": 2,
        }
    (source / "provenance.json").write_text(json.dumps(provenance))
    (source / "history.json").write_text(json.dumps(_history(epochs)))
    source_map = {
        str(common.F3_PLAN_REMOTE_PATH): common.PLAN_SHA256,
        "/home/grinchenkoeg/WunderFunder/scripts/bandit_resources.py": "b" * 64,
    }
    source_manifest_sha = hashlib.sha256(
        json.dumps(source_map, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    attempt = {
        "id": spec["attempt"],
        "command": spec["command"],
        "unit": f"wnn-bandit-{spec['attempt']}.service",
        "working_directory": str(base / "source-a266b036"),
        "contract": str(common.F3_PLAN_REMOTE_PATH),
        "contract_sha256": common.PLAN_SHA256,
        "source_sha256": source_map,
        "source_manifest_sha256": source_manifest_sha,
        "status": "finished",
        "exit_status": 0,
    }
    attempt_path = base / "runtime/attempts" / f"{spec['attempt']}.json"
    attempt_path.parent.mkdir(parents=True, exist_ok=True)
    attempt_path.write_text(json.dumps(attempt))
    log_path = base / "runtime/logs" / f"{spec['attempt']}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("bounded fixture\n")
    return source


class F3CollectionTests(unittest.TestCase):
    def test_stage1_and_terminal_copy_bind_exact_parent_and_provenance(self) -> None:
        validate_plan(PLAN)
        envelope = {
            "service": "fixture.service",
            "collector_cpu_max": "25000 100000",
            "collector_memory_max_bytes": 512_000_000,
            "collector_memory_swap_max_bytes": 0,
            "collector_runtime_seconds": 120,
        }
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary).resolve()
            with patch.object(collector, "REMOTE_ROOT", base):
                _write_boundary(base, "wide", "stage1")
                stage1_output = base / "receipts/stage1.json"
                stage1 = collector.collect_snapshot(
                    base, "wide", "stage1", PLAN, stage1_output, resource_envelope=envelope
                )
                self.assertEqual(stage1["schema"], "priority25.f3_checkpoint_collection.v1")
                self.assertEqual(stage1["phase"], "stage1")
                attempt_evidence = stage1["attempt_evidence"]
                self.assertEqual(attempt_evidence["contract_sha256"], common.PLAN_SHA256)
                self.assertEqual(
                    attempt_evidence["source_sha256"][str(common.F3_PLAN_REMOTE_PATH)],
                    common.PLAN_SHA256,
                )
                source_manifest_payload = json.dumps(
                    attempt_evidence["source_sha256"], sort_keys=True, separators=(",", ":")
                ).encode()
                self.assertEqual(
                    attempt_evidence["source_manifest_sha256"],
                    hashlib.sha256(source_manifest_payload).hexdigest(),
                )
                self.assertEqual(set(stage1["artifacts"]), set(common.ARTIFACTS["stage1"]))
                self.assertTrue(stage1["source_snapshot_hash_match"])
                stage_snapshot = Path(stage1["snapshot_directory"])
                self.assertTrue((stage_snapshot / "training_state.pt").is_file())
                with patch.object(evidence, "REMOTE_ROOT", base):
                    verified_artifacts, verified_receipt = evidence._verify_collection(
                        stage_snapshot,
                        stage1_output,
                        arm_spec(validate_plan(PLAN), "wide", "stage1"),
                        PLAN,
                    )
                self.assertEqual(set(verified_artifacts), set(common.ARTIFACTS["stage1"]))
                self.assertEqual(
                    verified_receipt["attempt_evidence"]["source_sha256"],
                    attempt_evidence["source_sha256"],
                )
                with self.assertRaisesRegex(ValueError, "checkpoint archive already exists"):
                    collector.collect_snapshot(
                        base,
                        "wide",
                        "stage1",
                        PLAN,
                        base / "receipts/duplicate.json",
                        resource_envelope=envelope,
                    )

                _write_boundary(base, "wide", "terminal", stage_snapshot)
                terminal_output = base / "receipts/terminal.json"
                terminal = collector.collect_snapshot(
                    base,
                    "wide",
                    "terminal",
                    PLAN,
                    terminal_output,
                    stage1_snapshot=stage_snapshot,
                    resource_envelope=envelope,
                )
                self.assertEqual(terminal["schema"], "priority25.f3_checkpoint_collection.v1")
                self.assertEqual(terminal["phase"], "terminal")
                self.assertEqual(
                    terminal["resumed_from"]["sha256"],
                    common.sha256_file(stage_snapshot / "training_state.pt"),
                )
                self.assertEqual(terminal["resumed_from"]["next_epoch"], 2)
                self.assertEqual(set(terminal["artifacts"]), set(common.ARTIFACTS["terminal"]))
                self.assertEqual(json.loads(terminal_output.read_text()), terminal)

    def test_wrong_f3_attempt_command_is_rejected_before_copy(self) -> None:
        envelope = {"service": "fixture.service", "collector_runtime_seconds": 120}
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary).resolve()
            _write_boundary(base, "deep", "stage1")
            spec = arm_spec(validate_plan(PLAN), "deep", "stage1")
            attempt_path = base / "runtime/attempts" / f"{spec['attempt']}.json"
            record = json.loads(attempt_path.read_text())
            record["command"][record["command"].index("--valid-fold") + 1] = "1"
            attempt_path.write_text(json.dumps(record))
            with (
                patch.object(collector, "REMOTE_ROOT", base),
                self.assertRaisesRegex(ValueError, "attempt command differs"),
            ):
                collector.collect_snapshot(
                    base,
                    "deep",
                    "stage1",
                    PLAN,
                    base / "receipts/fail.json",
                    resource_envelope=envelope,
                )
            self.assertFalse(
                (base / "checkpoint_snapshots" / str(spec["run"]) / "stage1_epoch1").exists()
            )


if __name__ == "__main__":
    unittest.main()
