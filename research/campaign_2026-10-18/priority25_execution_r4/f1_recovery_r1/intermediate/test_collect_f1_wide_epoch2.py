from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

MODULE_PATH = Path(__file__).with_name("collect_f1_wide_epoch2.py")
SPEC = importlib.util.spec_from_file_location("collect_f1_wide_epoch2", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
collector = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(collector)


class F1WideEpoch2CollectorTests(unittest.TestCase):
    """Exercise actual V2 parent receipt fields and synthetic producer boundaries."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.common = collector.load_v3_common()
        cls.repo_root = Path(__file__).resolve().parents[5]
        cls.receipt_path = (
            cls.repo_root / "research/campaign_2026-10-18/priority25_execution_r4/receipts/"
            "RANK01_WIDE_F1_S1001_STAGE1_VERIFICATION.json"
        )
        cls.receipt = collector.read_json(cls.receipt_path)
        cls.plan = cls.common.validate_plan(cls.common.ORIGINAL_PLAN_PATH)
        cls.spec = cls.common.arm_spec(cls.plan, "wide", "terminal")

    def test_actual_v2_stage1_receipt_is_the_exact_resume_parent(self) -> None:
        self.assertEqual(
            collector.sha256_file(self.receipt_path),
            collector.STAGE1_RECEIPT_SHA256,
        )
        collector.validate_stage1_receipt(self.receipt, collector.STAGE1_STATE_SHA256, self.common)

    def test_attempt_requires_exact_recovery_command_and_status(self) -> None:
        record = {
            "id": self.spec["attempt"],
            "unit": f"wnn-bandit-{self.spec['attempt']}.service",
            "command": self.spec["command"],
            "working_directory": "/B/source-a266b036",
            "contract": self.common.ORIGINAL_PLAN_REMOTE_PATH,
            "contract_sha256": self.common.PLAN_SHA256,
            "status": "running",
            "exit_status": None,
        }
        self.assertEqual(
            collector.validate_attempt(record, self.spec, self.common.PLAN_SHA256)["status"],
            "running",
        )
        bad = dict(record, command=[*record["command"], "--wrong"])
        with self.assertRaisesRegex(collector.CollectionMismatch, "frozen trainer command"):
            collector.validate_attempt(bad, self.spec, self.common.PLAN_SHA256)
        bad = dict(record, status="finished", exit_status=1)
        with self.assertRaisesRegex(collector.CollectionMismatch, "finished attempt exit status"):
            collector.validate_attempt(bad, self.spec, self.common.PLAN_SHA256)

    def test_attempt_json_hash_binds_the_exact_bytes_parsed(self) -> None:
        record = {"id": self.spec["attempt"], "status": "running"}
        raw = json.dumps(record, indent=2).encode()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "attempt.json"
            path.write_bytes(raw)
            parsed, captured_sha = collector.read_json_with_sha(path)
            path.write_text('{"id":"updated","status":"finished"}\n')
            self.assertEqual(parsed, record)
            self.assertEqual(captured_sha, hashlib.sha256(raw).hexdigest())
            self.assertNotEqual(captured_sha, collector.sha256_file(path))

    def _provenance(self) -> dict[str, object]:
        parent_path = str(
            collector.REMOTE_ROOT / "data/artifacts/runs/rr18_r01_wide_f1_s1001/training_state.pt"
        )
        diagnostics = []
        for epoch in range(3):
            cumulative = dict(collector.ONE_EPOCH_COUNTS)
            for key in ("optimizer_attempts", "optimizer_completed_updates", "completed_batches"):
                cumulative[key] *= epoch + 1
            diagnostics.append(
                {
                    "epoch_index": epoch,
                    "epoch": dict(collector.ONE_EPOCH_COUNTS),
                    "cumulative": cumulative,
                    "grad_stats": {"nonfinite_steps": 0},
                }
            )
        return {
            "cumulative_updates": dict(collector.EXPECTED_PROVENANCE_COUNTS),
            "split_manifest_sha256": self.common.FOLD1_SHA256,
            "fit_membership_identity": self.receipt["fit_membership_identity"],
            "resumed_from": {
                "path": parent_path,
                "sha256": collector.STAGE1_STATE_SHA256,
                "next_epoch": 2,
            },
            "checkpoint_sha256": {"model_ep2.pt": "a" * 64},
            "update_diagnostics": diagnostics,
        }

    def test_provenance_binds_epoch_counters_model_and_stage1_parent(self) -> None:
        payload = self._provenance()
        collector.validate_provenance(payload, "a" * 64, self.receipt, self.common)
        wrong_parent = dict(payload)
        wrong_parent["resumed_from"] = {**payload["resumed_from"], "sha256": "0" * 64}
        with self.assertRaisesRegex(collector.CollectionMismatch, "stage1 resume parent"):
            collector.validate_provenance(wrong_parent, "a" * 64, self.receipt, self.common)
        wrong_counts = dict(
            payload,
            cumulative_updates={
                **payload["cumulative_updates"],
                "optimizer_completed_updates": 44_789,
            },
        )
        with self.assertRaisesRegex(collector.CollectionMismatch, "cumulative optimizer counters"):
            collector.validate_provenance(wrong_counts, "a" * 64, self.receipt, self.common)
        bad_diag = self._provenance()
        bad_diag["update_diagnostics"][2]["grad_stats"]["nonfinite_steps"] = 1
        with self.assertRaisesRegex(collector.CollectionMismatch, "nonfinite gradient"):
            collector.validate_provenance(bad_diag, "a" * 64, self.receipt, self.common)

    def test_copy_refuses_existing_snapshot_without_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            source.mkdir()
            for name in collector.ARTIFACT_NAMES:
                (source / name).write_bytes(name.encode())
            destination = root / "snapshot"
            destination.mkdir()
            sentinel = destination / "keep.txt"
            sentinel.write_text("prior evidence")
            with self.assertRaises(FileExistsError):
                collector.copy_generation(source, destination)
            self.assertEqual(sentinel.read_text(), "prior evidence")

    def test_copy_detects_source_mutation_and_removes_only_its_partial(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            source.mkdir()
            for name in collector.ARTIFACT_NAMES:
                (source / name).write_bytes(name.encode())
            destination = root / "snapshot"
            original_copy = collector._copy_one
            calls = 0

            def mutate_after_copy(src: Path, dst: Path) -> None:
                nonlocal calls
                original_copy(src, dst)
                calls += 1
                if calls == 1:
                    src.write_bytes(src.read_bytes() + b"mutation")

            with (
                patch.object(collector, "_copy_one", side_effect=mutate_after_copy),
                self.assertRaisesRegex(
                    collector.CollectionMismatch,
                    "source/copy hash|generation changed|bytes changed",
                ),
            ):
                collector.copy_generation(source, destination)
            self.assertFalse(destination.exists())

    def test_copy_rejects_same_size_rewrite_with_restored_mtime(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            source.mkdir()
            for name in collector.ARTIFACT_NAMES:
                (source / name).write_bytes(name.encode())
            expected_stats = {
                name: collector.stats(source / name) for name in collector.ARTIFACT_NAMES
            }
            expected_hashes = {
                name: collector.sha256_file(source / name) for name in collector.ARTIFACT_NAMES
            }
            model = source / "model_ep2.pt"
            original_stat = model.stat()
            model.write_bytes(b"X" * original_stat.st_size)
            os.utime(model, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
            self.assertEqual(collector.stats(model), expected_stats["model_ep2.pt"])
            destination = root / "snapshot"
            with self.assertRaisesRegex(collector.CollectionMismatch, "producer bytes changed"):
                collector.copy_generation(
                    source,
                    destination,
                    expected_hashes=expected_hashes,
                    expected_stats=expected_stats,
                )
            self.assertFalse(destination.exists())


if __name__ == "__main__":
    unittest.main()
