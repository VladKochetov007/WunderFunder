from __future__ import annotations

import subprocess
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from wnn.training.sequence_monitor_resources import (
    MAX_NVIDIA_OUTPUT_CHARS,
    MIB_BYTES,
    NVIDIA_QUERY,
    NVIDIA_TIMEOUT_SECONDS,
    HostResourceSampler,
    _parse_rss_bytes,
    _query_gpu,
)


class HostResourceSamplerTests(unittest.TestCase):
    """Check resource metrics with mocked procfs, disk, clocks, and GPU sensors."""

    def test_process_cpu_uses_elapsed_clock_and_omits_first_sample(self) -> None:
        sampler = HostResourceSampler(Path("unused-output"))
        with (
            patch(
                "wnn.training.sequence_monitor_resources.time.process_time", side_effect=[1.0, 1.75]
            ),
            patch(
                "wnn.training.sequence_monitor_resources.time.monotonic", side_effect=[10.0, 10.5]
            ),
            patch("wnn.training.sequence_monitor_resources._read_proc_file", return_value=None),
            patch("wnn.training.sequence_monitor_resources.shutil.disk_usage") as disk_usage,
        ):
            disk_usage.return_value = SimpleNamespace(total=200, used=150, free=50)
            first = sampler.sample()
            second = sampler.sample()
        self.assertNotIn("cpu/process_utilization_pct", first)
        self.assertEqual(second["cpu/process_utilization_pct"], 150.0)
        self.assertEqual(second["disk/free_bytes"], 50)
        self.assertEqual(second["disk/used_pct"], 75.0)
        disk_usage.assert_called_with(Path("unused-output"))

    def test_proc_memory_metrics_and_rss_fallback(self) -> None:
        proc_payloads = {
            "/proc/self/status": b"Name:\tpython\nVmRSS:\t1234 kB\n",
            "/proc/self/statm": b"10 5 0 0 0 0 0\n",
            "/proc/meminfo": b"MemTotal: 4096 kB\nMemAvailable: 1024 kB\n",
        }
        sampler = HostResourceSampler(Path("unused-output"))
        with (
            patch(
                "wnn.training.sequence_monitor_resources._read_proc_file",
                side_effect=lambda path: proc_payloads[path],
            ),
            patch("wnn.training.sequence_monitor_resources.shutil.disk_usage", side_effect=OSError),
        ):
            metrics = sampler.sample()
        self.assertEqual(metrics["cpu/rss_bytes"], 1_234 * 1_024)
        self.assertEqual(metrics["cpu/memory_available_bytes"], 1_024 * 1_024)
        self.assertEqual(metrics["cpu/memory_total_bytes"], 4_096 * 1_024)
        self.assertNotIn("disk/free_bytes", metrics)
        with patch(
            "wnn.training.sequence_monitor_resources.os.sysconf", return_value=4_096, create=True
        ):
            self.assertEqual(_parse_rss_bytes(None, b"10 6 0 0 0 0 0"), 6 * 4_096)

    def test_missing_proc_entries_and_disk_errors_are_optional(self) -> None:
        sampler = HostResourceSampler(Path("unavailable-output"))
        with (
            patch("wnn.training.sequence_monitor_resources._read_proc_file", return_value=None),
            patch("wnn.training.sequence_monitor_resources.shutil.disk_usage", side_effect=OSError),
        ):
            metrics = sampler.sample()
        self.assertEqual(metrics, {})

    def test_uuid_gpu_query_parses_values_and_uses_fixed_arguments(self) -> None:
        identifier = "GPU-12345678-abcd-4321-9876-1234567890ab"
        expected = [
            "nvidia-smi",
            "--id",
            identifier,
            f"--query-gpu={NVIDIA_QUERY}",
            "--format=csv,noheader,nounits",
        ]
        completed = subprocess.CompletedProcess(expected, 0, "75, 512, 8192, 140.5, 65\n", "")
        with patch(
            "wnn.training.sequence_monitor_resources.subprocess.run", return_value=completed
        ) as run:
            metrics = _query_gpu(identifier)
        run.assert_called_once_with(
            expected,
            capture_output=True,
            text=True,
            timeout=NVIDIA_TIMEOUT_SECONDS,
            check=True,
        )
        self.assertEqual(metrics["gpu/utilization_pct"], 75.0)
        self.assertEqual(metrics["gpu/device_memory_used_bytes"], 512 * MIB_BYTES)
        self.assertEqual(metrics["gpu/device_memory_total_bytes"], 8_192 * MIB_BYTES)
        self.assertEqual(metrics["gpu/power_w"], 140.5)
        self.assertEqual(metrics["gpu/temperature_c"], 65.0)

    def test_invalid_logical_ids_and_unavailable_sensors_are_omitted(self) -> None:
        with patch("wnn.training.sequence_monitor_resources.subprocess.run") as run:
            self.assertEqual(_query_gpu("0"), {})
            self.assertEqual(_query_gpu("GPU-12345678"), {})
            self.assertEqual(_query_gpu("GPU-12345678/../../other"), {})
            run.assert_not_called()
            run.side_effect = FileNotFoundError("nvidia-smi unavailable")
            self.assertEqual(_query_gpu("MIG-GPU-12345678-abcd-4321-9876-1234567890ab/1/0"), {})

    def test_timeout_and_n_a_values_do_not_become_zero_readings(self) -> None:
        identifier = "GPU-12345678-abcd-4321-9876-1234567890ab"
        with patch(
            "wnn.training.sequence_monitor_resources.subprocess.run",
            side_effect=subprocess.TimeoutExpired("nvidia-smi", NVIDIA_TIMEOUT_SECONDS),
        ):
            self.assertEqual(_query_gpu(identifier), {})
        oversized = subprocess.CompletedProcess([], 0, "x" * (MAX_NVIDIA_OUTPUT_CHARS + 1), "")
        with patch(
            "wnn.training.sequence_monitor_resources.subprocess.run", return_value=oversized
        ):
            self.assertEqual(_query_gpu(identifier), {})
        completed = subprocess.CompletedProcess([], 0, "N/A, N/A, 8192, N/A, 65\n", "")
        with patch(
            "wnn.training.sequence_monitor_resources.subprocess.run", return_value=completed
        ):
            metrics = _query_gpu(identifier)
        self.assertNotIn("gpu/utilization_pct", metrics)
        self.assertNotIn("gpu/device_memory_used_bytes", metrics)
        self.assertEqual(metrics["gpu/device_memory_total_bytes"], 8_192 * MIB_BYTES)
        self.assertNotIn("gpu/power_w", metrics)
        self.assertEqual(metrics["gpu/temperature_c"], 65.0)


if __name__ == "__main__":
    unittest.main()
