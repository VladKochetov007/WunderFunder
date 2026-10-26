"""Receiver-only tests for recipe admission and validation-stop boundaries."""

from __future__ import annotations

import copy
import json
import tempfile
import time
import unittest
from pathlib import Path

from curves import selection
from evaluate import _check_recipe
from execute import validation_failure


class OperatorContractTests(unittest.TestCase):
    """Check materially different recipe and failure states without fitting."""

    def test_configs_and_rejections(self) -> None:
        root = Path("/data/epanov_s/tabdata/runs/compact-factor-sweep-20261026")
        for wave in ["regularization_factors", "learning_rate"]:
            for path in (root / wave / "configs").glob("*.json"):
                cfg = json.loads(path.read_text())
                _check_recipe(path.stem, cfg, wave)
                wrong = copy.deepcopy(cfg)
                wrong["lr"] = 0.12345
                with self.assertRaises(ValueError):
                    _check_recipe(path.stem, wrong, wave)
                with self.assertRaises(ValueError):
                    _check_recipe(path.stem, cfg, "unknown")

    def test_fresh_heartbeat_does_not_hide_admission_failure(self) -> None:
        with tempfile.TemporaryDirectory(dir="/data/epanov_s/tabdata/tmp") as temp:
            root = Path(temp)
            folder = root / "observer-b128"
            folder.mkdir()
            for job_status, epoch_status in [
                ("invalid_epoch_index", "complete"),
                ("running", "admission_failed"),
                ("running", "failed_preserved"),
            ]:
                (folder / "MONITOR.json").write_text(
                    json.dumps(
                        {
                            "status": "running",
                            "jobs": {
                                "test": {
                                    "status": job_status,
                                    "epochs": {"1": {"status": epoch_status}},
                                }
                            },
                        }
                    )
                )
                self.assertIsNotNone(validation_failure(root, time.time()))
            (folder / "MONITOR.json").write_text(
                json.dumps(
                    {
                        "status": "running",
                        "jobs": {
                            "test": {
                                "status": "running",
                                "epochs": {"1": {"status": "complete"}, "2": {"status": "running"}},
                            }
                        },
                    }
                )
            )
            self.assertIsNone(validation_failure(root, time.time()))

    def test_common_epoch_not_per_fold_cherry_pick(self) -> None:
        records = []
        for fold in [0, 2]:
            for epoch in range(1, 6):
                wp = 0.9 if (fold, epoch) in {(0, 1), (2, 2)} else 0.1
                records.append(
                    {"arm": "control", "fold": fold, "epoch": epoch, "validation_wp": wp}
                )
        result = selection(records)["selection"]["control"]
        self.assertEqual(result["epoch"], 1)
        self.assertAlmostEqual(result["mean_wp"], 0.5)


if __name__ == "__main__":
    unittest.main()
