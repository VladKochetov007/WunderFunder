from __future__ import annotations

import base64
import json
import tempfile
import unittest
from pathlib import Path

import terminal_gate as gate

STEP_IDS = (
    ("stage1_wide", "fit_stage1", "wide"),
    ("stage1_deep", "fit_stage1", "deep"),
    ("continuation_wide", "continue_fixed_six_epoch_fit", "wide"),
    ("continuation_deep", "continue_fixed_six_epoch_fit", "deep"),
    ("export_wide", "export_fixed_epoch5", "wide"),
    ("export_deep", "export_fixed_epoch5", "deep"),
    ("grade", "paired_official_grade", "wide_vs_deep"),
)


def plan_fixture() -> dict[str, object]:
    """Build only the seven step identities used by the sequence gate."""
    return {
        "steps": [
            {"step_id": step_id, "kind": kind, "arm": arm, "attempt": f"attempt-{index}"}
            for index, (step_id, kind, arm) in enumerate(STEP_IDS)
        ]
    }


def resource_fixture(index: int, operation: str = "collector") -> dict[str, object]:
    """Build a tiny cgroup tree with all active plan children accounted."""
    plan = plan_fixture()
    step = plan["steps"][index]
    current = {
        "MemoryCurrent": "3000000000",
        "CPUQuotaPerSecUSec": "2s",
        "MemoryHigh": "18000000000",
        "MemoryMax": "20000000000",
        "MemorySwapMax": "0",
    }
    worker_memory = 4_000_000_000 if index == 6 else 8_000_000_000
    worker = f"wnn-bandit-{step['attempt']}.service"
    return {
        "parent_properties": current,
        "parent_cgroup_procs": "",
        "top_units": [
            {
                "unit": gate.CONTROLLER,
                "cpu_cores": 0.1,
                "memory_max": 512_000_000,
                "memory_current": 100_000_000,
                "swap_max": 0,
                "populated": True,
            },
            {
                "unit": worker,
                "cpu_cores": 1.0,
                "memory_max": worker_memory,
                "memory_current": 1_400_000_000,
                "swap_max": 0,
                "populated": True,
            },
        ],
        "populated_paths": [gate.CONTROLLER, f"{gate.CONTROLLER}/child", worker],
    }


class SequenceGateTests(unittest.TestCase):
    """Exercise fit, both exports, grade, and finished sequence states."""

    def test_wide_terminal_accepts_each_later_sequence_state(self) -> None:
        plan = plan_fixture()
        for index in (3, 4, 5, 6):
            sequence = {
                "plan_sha256": gate.PLAN_SHA256,
                "status": "running",
                "index": index,
                "attempt": plan["steps"][index]["attempt"],
            }
            self.assertEqual(gate.validate_pointer(plan, sequence, "wide"), plan["steps"][index])
        sequence = {
            "plan_sha256": gate.PLAN_SHA256,
            "status": "finished",
            "index": 6,
            "attempt": plan["steps"][6]["attempt"],
        }
        self.assertIsNone(gate.validate_pointer(plan, sequence, "wide"))

    def test_deep_terminal_accepts_exports_grade_and_finished(self) -> None:
        plan = plan_fixture()
        for index in (4, 5, 6):
            sequence = {
                "plan_sha256": gate.PLAN_SHA256,
                "status": "running",
                "index": index,
                "attempt": plan["steps"][index]["attempt"],
            }
            self.assertEqual(gate.validate_pointer(plan, sequence, "deep"), plan["steps"][index])
        sequence = {
            "plan_sha256": gate.PLAN_SHA256,
            "status": "finished",
            "index": 6,
            "attempt": plan["steps"][6]["attempt"],
        }
        self.assertIsNone(gate.validate_pointer(plan, sequence, "deep"))

    def test_refuses_stale_or_mismatched_sequence(self) -> None:
        plan = plan_fixture()
        cases = (
            {
                "plan_sha256": gate.PLAN_SHA256,
                "status": "needs_review",
                "index": 4,
                "attempt": "attempt-4",
            },
            {
                "plan_sha256": gate.PLAN_SHA256,
                "status": "running",
                "index": 2,
                "attempt": "attempt-2",
            },
            {"plan_sha256": gate.PLAN_SHA256, "status": "running", "index": 4, "attempt": "wrong"},
            {"plan_sha256": "0" * 64, "status": "running", "index": 4, "attempt": "attempt-4"},
        )
        for sequence in cases:
            with self.subTest(sequence=sequence), self.assertRaises(gate.GateError):
                gate.validate_pointer(plan, sequence, "deep")

    def test_native_attempt_must_finish_with_exit_zero(self) -> None:
        successful = {"id": "native", "status": "finished", "exit_status": 0}
        gate.validate_successful_exit(successful, "native")
        for record in (
            {"id": "other", "status": "finished", "exit_status": 0},
            {"id": "native", "status": "running", "exit_status": None},
            {"id": "native", "status": "finished", "exit_status": 1},
        ):
            with self.subTest(record=record), self.assertRaises(gate.GateError):
                gate.validate_successful_exit(record, "native")


class ResourceGateTests(unittest.TestCase):
    """Reject unknown descendants and enforce operation-specific headroom."""

    def test_fit_export_grade_and_finished_envelopes(self) -> None:
        for index in (3, 4, 5, 6):
            result = gate.validate_resource_snapshot(
                resource_fixture(index), plan_fixture()["steps"][index], "collector"
            )
            self.assertLess(result["committed_memory_bytes"], 17_000_000_000)
            self.assertLessEqual(result["committed_cpu_cores"], 2.0)
            self.assertIn(f"wnn-bandit-attempt-{index}.service", result["known_top_units"])
        finished = {
            "parent_properties": {
                "MemoryCurrent": "2000000000",
                "CPUQuotaPerSecUSec": "2s",
                "MemoryHigh": "18000000000",
                "MemoryMax": "20000000000",
                "MemorySwapMax": "0",
            },
            "parent_cgroup_procs": "",
            "top_units": [],
            "populated_paths": [],
        }
        result = gate.validate_resource_snapshot(finished, None, "verifier")
        self.assertEqual(result["known_top_units"], [])
        controller_only = {
            **finished,
            "top_units": [
                {
                    "unit": gate.CONTROLLER,
                    "cpu_cores": 0.1,
                    "memory_max": 512_000_000,
                    "memory_current": 25_000_000,
                    "swap_max": 0,
                    "populated": True,
                }
            ],
            "populated_paths": [gate.CONTROLLER, f"{gate.CONTROLLER}/controller"],
        }
        result = gate.validate_resource_snapshot(controller_only, None, "verifier")
        self.assertEqual(result["known_top_units"], [gate.CONTROLLER])
        verifier = gate.validate_resource_snapshot(
            resource_fixture(5, "verifier"), plan_fixture()["steps"][5], "verifier"
        )
        self.assertEqual(verifier["support_memory_bytes"], 1_500_000_000)

    def test_refuses_unknown_child_nested_root_and_overcommit(self) -> None:
        valid = resource_fixture(4)
        valid["top_units"].append(
            {
                "unit": "foreign.service",
                "cpu_cores": 0.1,
                "memory_max": 100_000_000,
                "memory_current": 10_000_000,
                "swap_max": 0,
                "populated": True,
            }
        )
        with self.assertRaises(gate.GateError):
            gate.validate_resource_snapshot(valid, plan_fixture()["steps"][4], "collector")
        unknown_nested = resource_fixture(4)
        unknown_nested["populated_paths"].append("foreign.service/worker")
        with self.assertRaises(gate.GateError):
            gate.validate_resource_snapshot(unknown_nested, plan_fixture()["steps"][4], "collector")
        high = resource_fixture(4)
        high["parent_properties"]["MemoryCurrent"] = "16000000000"
        with self.assertRaises(gate.GateError):
            gate.validate_resource_snapshot(high, plan_fixture()["steps"][4], "verifier")
        direct = resource_fixture(4)
        direct["parent_cgroup_procs"] = "123"
        with self.assertRaises(gate.GateError):
            gate.validate_resource_snapshot(direct, plan_fixture()["steps"][4], "collector")


class CaptureSafetyTests(unittest.TestCase):
    """Require successful native exit, intact raw bytes, and create-only writes."""

    def test_raw_bytes_are_hash_checked_before_capture(self) -> None:
        raw = json.dumps({"status": "finished", "exit_status": 0}).encode()
        files = {
            "attempt.json": {
                "bytes": len(raw),
                "sha256": gate.sha256_bytes(raw),
                "base64": base64.b64encode(raw).decode(),
            }
        }
        self.assertEqual(gate.decode_raw_files(files)["attempt.json"], raw)
        files["attempt.json"]["sha256"] = "0" * 64
        with self.assertRaises(gate.GateError):
            gate.decode_raw_files(files)

    def test_create_only_capture_preserves_existing_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "capture.json"
            path.write_bytes(b"original")
            with self.assertRaises(FileExistsError):
                gate.write_exclusive(path, b"replacement")
            self.assertEqual(path.read_bytes(), b"original")


if __name__ == "__main__":
    unittest.main()
