from __future__ import annotations

import importlib.util
import sys
import time
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[1] / "run_loaded_host_timing_v4.py"
SPEC = importlib.util.spec_from_file_location("loaded_host_timing_v4_test", SOURCE)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load timing-v4 module")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class FakeTiming:
    def __init__(self) -> None:
        self.time = time

    def _trial(self, _label: str, _package: Path, _streams: list[Any], _cycles: Any) -> dict[str, Any]:
        start = self.time.perf_counter()
        time.sleep(0.001)
        elapsed = self.time.perf_counter() - start
        return {"rows": MODULE.EXPECTED_ROWS, "elapsed_seconds": elapsed, "cycles": 12_000}


class LoadedHostTimingTests(unittest.TestCase):
    def test_host_load_is_recorded_but_does_not_reject_a_trial(self) -> None:
        high_load = {
            "loadavg": [99.0, 80.0, 70.0],
            "proc_stat": {"cpu": [100, 0, 100, 100, 0], "cpu15": [10, 0, 10, 10, 0]},
            "pressure_cpu": None,
            "pressure_memory": None,
            "vmstat": {},
        }
        hashes = {
            "package_sha256": {"seed3_i16": "a" * 64},
            "native_library_sha256": {"seed3_i16": "b" * 64},
        }
        timing = FakeTiming()
        with patch.object(MODULE, "read_cpu_snapshot", return_value=high_load):
            result = MODULE.run_one_trial(
                timing,
                None,
                "seed3_i16",
                Path("seed3.zip"),
                [],
                "synthetic-pair-1",
                ["seed3_i16", "seed4_i16"],
                hashes,
            )
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["host_before"]["loadavg"][0], 99.0)
        self.assertEqual(result["hardware_thread_cycles_per_row"], 0.2)
        self.assertEqual(len(result["clock_boundary_samples"]), 2)

    def test_host_counter_delta_reports_cpu_and_vm_pressure_without_a_gate(self) -> None:
        before = {
            "proc_stat": {"cpu": [100, 0, 100, 100, 0], "cpu15": [10, 0, 10, 10, 0]},
            "vmstat": {"pgfault": 20, "pgmajfault": 2},
        }
        after = {
            "proc_stat": {"cpu": [110, 0, 110, 120, 0], "cpu15": [12, 0, 13, 15, 0]},
            "vmstat": {"pgfault": 29, "pgmajfault": 3},
        }
        deltas = MODULE.snapshot_deltas(before, after)
        self.assertAlmostEqual(deltas["host_cpu_busy_fraction"], 20 / 40)
        self.assertAlmostEqual(deltas["cpu15_busy_fraction"], 5 / 10)
        self.assertEqual(deltas["vmstat_counter_delta"], {"pgfault": 9, "pgmajfault": 1})

    def test_five_pair_log_interval_is_exact_for_constant_ratios(self) -> None:
        result = MODULE.geometric_mean_ci([1.2] * MODULE.PAIRS_PER_CONTRAST)
        self.assertAlmostEqual(result["geometric_mean"], 1.2)
        self.assertAlmostEqual(result["ci95_low"], 1.2)
        self.assertAlmostEqual(result["ci95_high"], 1.2)

    def test_protocol_uses_three_fixed_contrasts_and_alternates_pair_order(self) -> None:
        self.assertEqual(len(MODULE.PHASES), 3)
        for _, control, variant in MODULE.PHASES:
            order0 = [control, variant]
            order1 = [variant, control]
            self.assertEqual(set(order0), set(order1))
            self.assertEqual(order0, list(reversed(order1)))
        source = SOURCE.read_text()
        self.assertNotIn("MAX_LOAD", source)
        self.assertNotIn("loadavg[0] >", source)


if __name__ == "__main__":
    unittest.main()
