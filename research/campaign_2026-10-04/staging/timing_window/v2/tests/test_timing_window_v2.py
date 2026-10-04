"""Pure-standard-library checks for the new timing-window identities."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[6]
CAMPAIGN = ROOT / "research/campaign_2026-10-04"
STAGING = CAMPAIGN / "staging/timing_window"
V2 = STAGING / "v2"
E462_SOURCE = CAMPAIGN / "staging/e462/time_public_q8.py"
E450_SOURCE = STAGING / "time_seed4_i16.py"
CONTROLLER_PATH = V2 / "run_timing_window_v2.py"
SPEC_PATHS = {
    "e450_seed4_i16_timing_v2": CAMPAIGN / "specs/e450_seed4_i16_timing_v2.draft.json",
    "e462_public_q8_timing_v2": CAMPAIGN / "specs/e462_public_q8_timing_v2.draft.json",
}
SPEC = importlib.util.spec_from_file_location("timing_window_v2_controller", CONTROLLER_PATH)
assert SPEC is not None and SPEC.loader is not None
CONTROLLER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CONTROLLER)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def arg(argv: list[str], flag: str) -> str:
    return argv[argv.index(flag) + 1]


class TimingWindowV2Tests(unittest.TestCase):
    def test_nested_controller_resolves_workspace_root(self) -> None:
        self.assertEqual(CONTROLLER.ROOT, ROOT)
        self.assertEqual(
            CONTROLLER.expected_worker_unit("e450-seed4-i16-timing-v2"),
            "wnn-c04-e450-seed4-i16-timing-v2.service",
        )

    def test_startup_and_worker_load_gates_are_distinct(self) -> None:
        self.assertEqual(CONTROLLER.MAX_LOAD1, 0.3)
        self.assertEqual(CONTROLLER.WORKER_MAX_LOAD1, 1.0)
        self.assertEqual(CONTROLLER.QUIET_WAIT_SECONDS, 900)
        self.assertEqual(
            CONTROLLER.TIMINGS,
            (
                ("e450_seed4_i16_timing_v2", "e450-seed4-i16-timing-v2"),
                ("e462_public_q8_timing_v2", "e462-public-q8-timing-v2"),
            ),
        )

    def test_specs_keep_original_sources_and_new_unique_outputs(self) -> None:
        old_artifacts = set()
        for name in ("e450_seed4_i16_timing_v1", "e462_public_q8_timing_v1"):
            old = read_json(CAMPAIGN / f"specs/{name}.sealed.json")
            old_artifacts.update(old["artifacts"])

        expected_sources = {
            "e450_seed4_i16_timing_v2": E450_SOURCE,
            "e462_public_q8_timing_v2": E462_SOURCE,
        }
        for name, path in SPEC_PATHS.items():
            with self.subTest(spec=name):
                value = read_json(path)
                self.assertEqual(value["name"], name)
                self.assertEqual(value["kind"], "cpu")
                self.assertTrue(
                    all(isinstance(item, str) for item in value["code"] + value["inputs"])
                )
                self.assertIn(str(expected_sources[name].relative_to(ROOT)), value["argv"])
                self.assertIn(str(expected_sources[name].relative_to(ROOT)), value["code"])
                self.assertEqual(
                    value["limits"],
                    {
                        "cpus": 1,
                        "gpu_memory_bytes": 0,
                        "gpu_stop_bytes": 0,
                        "ram_bytes": 2 * 1024**3,
                        "timeout_seconds": 1800,
                    },
                )
                self.assertEqual(arg(value["argv"], "--cpu"), "15")
                self.assertEqual(arg(value["argv"], "--max-load1"), "1.0")
                repeat_flag = "--repeats" if name.startswith("e450") else "--paired-repeats"
                self.assertEqual(arg(value["argv"], repeat_flag), "5")
                output = arg(value["argv"], "--output")
                self.assertIn("staging/timing_window/v2/results/", output)
                self.assertFalse(old_artifacts.intersection(value["artifacts"]))
                self.assertEqual(
                    set(value["artifacts"]),
                    {
                        f"{output}/trials.jsonl",
                        f"{output}/timing_report.json",
                    },
                )
                CONTROLLER.validate_timing_spec(value, name)

    def test_seed4_worker_first_and_q8_loop_hashes_unchanged(self) -> None:
        seed4 = read_json(SPEC_PATHS["e450_seed4_i16_timing_v2"])
        q8 = read_json(SPEC_PATHS["e462_public_q8_timing_v2"])
        self.assertEqual(sha256(E450_SOURCE), CONTROLLER.E450_TIMING_SOURCE_SHA256)
        self.assertEqual(sha256(E462_SOURCE), CONTROLLER.E462_TIMING_SOURCE_SHA256)
        self.assertEqual(arg(seed4["argv"], "--expected-e462-timing-sha256"), sha256(E462_SOURCE))
        self.assertEqual(
            arg(seed4["argv"], "--output"),
            "research/campaign_2026-10-04/staging/timing_window/v2/results/e450_seed4_i16_v2",
        )
        self.assertEqual(
            arg(q8["argv"], "--output"),
            "research/campaign_2026-10-04/staging/timing_window/v2/results/e462_q8_v2",
        )

    def test_v2_unit_is_bounded_and_old_v1_unit_remains_separate(self) -> None:
        unit = (V2 / "systemd/wnn-timing-window-v2.service").read_text()
        self.assertIn("Type=oneshot", unit)
        self.assertIn("CPUQuota=25%", unit)
        self.assertIn("MemoryMax=512M", unit)
        self.assertIn("MemorySwapMax=0", unit)
        self.assertIn("TimeoutStartSec=6h", unit)
        self.assertIn("run_timing_window_v2.py", unit)
        self.assertTrue((STAGING / "systemd/wnn-timing-window-v1.service").is_file())
        self.assertTrue((STAGING / "timing_window_v1.json").is_file())


if __name__ == "__main__":
    unittest.main()
