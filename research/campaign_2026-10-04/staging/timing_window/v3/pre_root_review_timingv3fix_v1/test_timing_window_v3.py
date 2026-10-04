"""Stdlib-only checks for v3 timing identity, absolute paths, and quiet-window order."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[6]
CAMPAIGN = ROOT / "research/campaign_2026-10-04"
STAGING = CAMPAIGN / "staging/timing_window"
V3 = STAGING / "v3"
E462_SOURCE = CAMPAIGN / "staging/e462/time_public_q8.py"
E450_SOURCE = STAGING / "time_seed4_i16.py"
CONTROLLER_PATH = V3 / "run_timing_window_v3.py"
SPEC_PATHS = {
    "e450_seed4_i16_timing_v3": CAMPAIGN / "specs/e450_seed4_i16_timing_v3.draft.json",
    "e462_public_q8_timing_v3": CAMPAIGN / "specs/e462_public_q8_timing_v3.draft.json",
}
SPEC = importlib.util.spec_from_file_location("timing_window_v3_controller", CONTROLLER_PATH)
assert SPEC is not None and SPEC.loader is not None
CONTROLLER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CONTROLLER)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def arg(argv: list[str], flag: str) -> str:
    return argv[argv.index(flag) + 1]


class TimingWindowV3Tests(unittest.TestCase):
    def test_nested_controller_root_and_queue_gate(self) -> None:
        self.assertEqual(CONTROLLER.ROOT, ROOT)
        self.assertEqual(CONTROLLER.PREDECESSOR_SERVICE, "wnn-e426-clock-queue-v3.service")
        self.assertEqual(CONTROLLER.E426_QUEUE, CAMPAIGN / "checks/e426_clock_queue_v3.json")
        self.assertEqual(
            CONTROLLER.TIMINGS,
            (
                ("e450_seed4_i16_timing_v3", "e450-seed4-i16-timing-v3"),
                ("e462_public_q8_timing_v3", "e462-public-q8-timing-v3"),
            ),
        )
        self.assertIn("After=wnn-e426-clock-queue-v3.service", (V3 / "systemd/wnn-timing-window-v3.service").read_text())
        self.assertEqual(CONTROLLER.MAX_LOAD1, 0.3)
        self.assertEqual(CONTROLLER.WORKER_MAX_LOAD1, 1.0)
        self.assertEqual(CONTROLLER.QUIET_WAIT_SECONDS, 900)

    def test_frozen_loops_and_absolute_cli_operands(self) -> None:
        self.assertEqual(sha256(E450_SOURCE), CONTROLLER.E450_TIMING_SOURCE_SHA256)
        self.assertEqual(sha256(E462_SOURCE), CONTROLLER.E462_TIMING_SOURCE_SHA256)
        old_artifacts: set[str] = set()
        for old_name in ("e450_seed4_i16_timing_v2", "e462_public_q8_timing_v2"):
            old = json.loads((CAMPAIGN / f"specs/{old_name}.sealed.json").read_text())
            old_artifacts.update(old["artifacts"])

        expected_worker = {
            "e450_seed4_i16_timing_v3": E450_SOURCE,
            "e462_public_q8_timing_v3": E462_SOURCE,
        }
        path_flags = (
            "--control", "--seed4", "--readiness", "--parity-attempt", "--data", "--output",
            "--q8r", "--q8x", "--build-dir", "--parity-report", "--score-report",
        )
        for name, path in SPEC_PATHS.items():
            with self.subTest(spec=name):
                value = json.loads(path.read_text())
                staged_copy = json.loads((V3 / "specs" / path.name).read_text())
                self.assertEqual(value, staged_copy)
                argv = value["argv"]
                self.assertEqual(value["name"], name)
                self.assertEqual(value["kind"], "cpu")
                self.assertIn(str(pathlib_absolute(expected_worker[name])), argv)
                self.assertIn(str(expected_worker[name].relative_to(ROOT)), [item["path"] for item in value["code"]])
                self.assertEqual(arg(argv, "--cpu"), "15")
                self.assertEqual(arg(argv, "--max-load1"), "1.0")
                repeat = "--repeats" if name.startswith("e450") else "--paired-repeats"
                self.assertEqual(arg(argv, repeat), "5")
                self.assertIn(str(pathlib_absolute(STAGING / "run_pinned_cpu.py")), argv)
                self.assertTrue(all(Path(arg(argv, flag)).is_absolute() for flag in path_flags if flag in argv))
                output = Path(arg(argv, "--output"))
                self.assertTrue(output.is_absolute())
                self.assertIn("staging/timing_window/v3/results/", output.as_posix())
                self.assertFalse(old_artifacts.intersection(value["artifacts"]))
                self.assertEqual(
                    set(value["artifacts"]),
                    {f"{output.relative_to(ROOT)}/trials.jsonl", f"{output.relative_to(ROOT)}/timing_report.json"},
                )
                self.assertEqual(
                    value["limits"],
                    {"cpus": 1, "ram_bytes": 2 * 1024**3, "gpu_memory_bytes": 0,
                     "gpu_stop_bytes": 0, "timeout_seconds": 1800},
                )
                CONTROLLER.validate_timing_spec(value, name)

    def test_relative_cli_path_is_rejected_before_native_worker(self) -> None:
        value = json.loads(SPEC_PATHS["e450_seed4_i16_timing_v3"].read_text())
        argv = list(value["argv"])
        readiness_index = argv.index("--readiness") + 1
        argv[readiness_index] = "research/campaign_2026-10-04/checks/e450_seed4_native_parity_v1/readiness.json"
        broken = {**value, "argv": argv}
        with self.assertRaisesRegex(ValueError, "absolute and inside repository"):
            CONTROLLER.validate_timing_spec(broken, value["name"])

    def test_absolute_operands_resolve_to_existing_fixed_inputs(self) -> None:
        for value_path in SPEC_PATHS.values():
            value = json.loads(value_path.read_text())
            argv = value["argv"]
            for flag in ("--control", "--data", "--output"):
                path = Path(arg(argv, flag))
                self.assertTrue(path.is_absolute())
                self.assertTrue(path.resolve().is_relative_to(ROOT))
            self.assertTrue(Path(arg(argv, "--control")).is_file())
            self.assertTrue(Path(arg(argv, "--data")).is_file())
        seed4 = json.loads(SPEC_PATHS["e450_seed4_i16_timing_v3"].read_text())["argv"]
        for flag in ("--seed4", "--readiness", "--parity-attempt"):
            self.assertTrue(Path(arg(seed4, flag)).is_absolute())
        q8 = json.loads(SPEC_PATHS["e462_public_q8_timing_v3"].read_text())["argv"]
        for flag in ("--build-dir", "--parity-report", "--score-report", "--q8r", "--q8x"):
            self.assertTrue(Path(arg(q8, flag)).is_absolute())


def pathlib_absolute(path: Path) -> Path:
    return (ROOT / path).resolve() if not path.is_absolute() else path.resolve()


if __name__ == "__main__":
    unittest.main()
