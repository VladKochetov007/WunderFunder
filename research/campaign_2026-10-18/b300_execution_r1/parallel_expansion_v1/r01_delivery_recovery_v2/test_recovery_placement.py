"""CPU placement provenance accepts a bound launch and rejects substitutions."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from r01_recovery_placement import verify_placement


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class PlacementTests(unittest.TestCase):
    """Exercise changed placement with fixed bytes and independently varied evidence."""

    def test_bound_successor_and_rejected_pid_argv_and_control(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            control = root / "runtime/r01-density4-v1"
            control.mkdir(parents=True)
            source = control / "controller.py"
            source.write_text("# fixture controller\n", encoding="utf-8")
            files = {source.name: _sha(source)}
            manifest = control / "CONTROL_MANIFEST.json"
            manifest.write_text(
                json.dumps(
                    {
                        "schema": "tabdata-r01-density4-control-bundle.v1",
                        "files": files,
                        "sha256": hashlib.sha256(
                            json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
                        ).hexdigest(),
                    }
                ),
                encoding="utf-8",
            )
            events = root / "state/r01-density4-v1/events.jsonl"
            events.parent.mkdir(parents=True)
            job = {
                "id": "wide_f0_s1004",
                "arm": "wide",
                "fold": 0,
                "seed": 1004,
                "gpu_uuid": "GPU-fixed",
                "cpu_ids_initial": [8, 10],
                "argv": ["python", "fit"],
            }
            resource = {"pid": 580678, "visible_devices": "GPU-fixed", "cpu_affinity": [56, 58]}
            event = {
                "event": "fit_started",
                "job_id": job["id"],
                "arm": "wide",
                "fold": 0,
                "seed": 1004,
                "pid": 580678,
                "pgid": 580678,
                "pid_start_ticks": 230132388,
                "cpu_ids": [56, 58],
                "gpu_uuid": "GPU-fixed",
                "argv_sha256": hashlib.sha256(
                    json.dumps(job["argv"], separators=(",", ":")).encode()
                ).hexdigest(),
            }
            events.write_text(json.dumps(event) + "\n", encoding="utf-8")
            with patch("r01_recovery_placement.DENSITY_MANIFEST_SHA256", _sha(manifest)):
                result = verify_placement(root, job, resource)
                self.assertEqual(result["authority"], "sealed_density4_launch_event")
                for key, value in (("pid", 1), ("argv_sha256", "0" * 64), ("seed", 1005)):
                    with self.subTest(key=key):
                        changed = dict(event)
                        changed[key] = value
                        events.write_text(json.dumps(changed) + "\n", encoding="utf-8")
                        with self.assertRaises(ValueError):
                            verify_placement(root, job, resource)
                events.write_text(json.dumps(event) + "\n", encoding="utf-8")
                source.write_text("# changed\n", encoding="utf-8")
                with self.assertRaises(ValueError):
                    verify_placement(root, job, resource)

    def test_initial_cpu_does_not_authorize_gpu_move(self) -> None:
        job = {"gpu_uuid": "GPU-fixed", "cpu_ids_initial": [8, 10]}
        resource = {"visible_devices": "GPU-fixed", "cpu_affinity": [8, 10]}
        self.assertEqual(
            verify_placement(Path("."), job, resource)["authority"],
            "frozen_panel_initial_placement",
        )
        with self.assertRaises(ValueError):
            verify_placement(Path("."), job, dict(resource, visible_devices="GPU-other"))


if __name__ == "__main__":
    unittest.main()
