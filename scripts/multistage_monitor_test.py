from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from scripts.multistage_monitor import (
    MAX_METADATA_BYTES,
    _load_fit_run,
    _promote_run_view,
)
from wnn.dashboard.reader import MAX_SELECTED_RUNS, DashboardReader
from wnn.dashboard.schema import sanitize_config, sanitize_json
from wnn.dashboard.storage import MAX_LISTED_RUNS


class MultistageMonitorIdentityTests(unittest.TestCase):
    """Cover stage-role binding and canonical metadata identity deduplication."""

    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.config = {"model": {"architecture": "GRU"}, "kd_weight": 0.0}
        self.config_bytes = json.dumps(self.config, sort_keys=True).encode()
        self.source_sha = "a" * 64
        self.auxiliary_sha = "b" * 64
        self.primary_sha = "c" * 64
        self.metadata = {
            "source": {"sha256": self.source_sha},
            "config": self.config,
            "provenance": {
                "data": {
                    "evaluation_role": "control",
                    "split_sha256": self.auxiliary_sha,
                    "primary_split_sha256": self.primary_sha,
                    "teacher": None,
                    "initializer": None,
                }
            },
        }
        self.metadata_bytes = json.dumps(self.metadata, sort_keys=True).encode()

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _campaign_run(self, name: str, stage: int) -> Path:
        trial_dir = self.root / "campaign" / f"stage{stage}" / name
        run_dir = trial_dir / "fit_output"
        run_dir.mkdir(parents=True)
        (run_dir / "run_metadata.json").write_bytes(self.metadata_bytes)
        config_path = trial_dir / "config.json"
        config_path.write_bytes(self.config_bytes)
        request = {
            "schema": "wnn.multistage-trial-plan.v1",
            "stage": stage,
            "family": "GRU",
            "config_path": str(config_path),
            "config_sha256": hashlib.sha256(self.config_bytes).hexdigest(),
            "source_sha256": self.source_sha,
            "auxiliary_split_sha256": self.auxiliary_sha,
            "primary_split_sha256": self.primary_sha,
            "objective_scope": "inner_train_proxy",
            "primary_valid_used": False,
        }
        (trial_dir / "trial_request.json").write_text(json.dumps(request), encoding="utf-8")
        return run_dir

    def test_label_only_stage_comes_from_bound_trial_request(self) -> None:
        run_dir = self._campaign_run("label-only", stage=3)
        run = _load_fit_run(run_dir)
        self.assertEqual(run.stage, 3)
        self.assertEqual(run.stage_evidence, "trial_request")

    def test_unbound_label_only_fit_requires_registered_stage(self) -> None:
        run_dir = self.root / "unregistered"
        run_dir.mkdir()
        (run_dir / "run_metadata.json").write_bytes(self.metadata_bytes)
        with self.assertRaisesRegex(ValueError, "needs an owned trial_request.stage"):
            _load_fit_run(run_dir)
        registered = _load_fit_run(run_dir, registered_stage=1)
        self.assertEqual(registered.stage, 1)
        self.assertEqual(registered.stage_evidence, "registered_role")

    def test_hardlinked_aliases_share_canonical_run_identity(self) -> None:
        first = self._campaign_run("anchor", stage=1)
        second = self._campaign_run("preexisting", stage=1)
        second_metadata = second / "run_metadata.json"
        second_metadata.unlink()
        second_metadata.hardlink_to(first / "run_metadata.json")
        left = _load_fit_run(first)
        right = _load_fit_run(second)
        self.assertEqual(left.metadata_sha256, right.metadata_sha256)
        self.assertEqual(left.run_id, right.run_id)
        self.assertNotEqual(left.source_dir, right.source_dir)

    def test_later_bound_role_promotes_one_existing_mirror(self) -> None:
        stage1_dir = self._campaign_run("stage1-role", stage=1)
        stage3_dir = self._campaign_run("stage3-role", stage=3)
        old = _load_fit_run(stage1_dir)
        new = _load_fit_run(stage3_dir)
        new = replace(new, source_dir=stage3_dir)
        dashboard_root = self.root / "dashboard"
        old_mirror = dashboard_root / old.run_id
        old_mirror.mkdir(parents=True)
        (old_mirror / "run_metadata.json").write_bytes(old.metadata_bytes)
        with patch("scripts.multistage_monitor._fsync_directory"):
            _promote_run_view(old, new, dashboard_root)
        self.assertFalse(old_mirror.exists())
        self.assertEqual(
            (dashboard_root / new.run_id / "run_metadata.json").read_bytes(),
            old.metadata_bytes,
        )

    def test_metadata_cap_and_dashboard_visibility_bounds(self) -> None:
        self.assertEqual(MAX_METADATA_BYTES, 16 * 1_048_576)
        self.assertEqual(MAX_LISTED_RUNS, 48)
        self.assertEqual(MAX_SELECTED_RUNS, 8)
        root = self.root / "dashboard"
        root.mkdir()
        for index in range(MAX_LISTED_RUNS + 1):
            run_dir = root / f"run-{index:02d}"
            run_dir.mkdir()
            (run_dir / "run_metadata.json").write_text(
                json.dumps({"source": "fixture", "config": {}, "provenance": {}}),
                encoding="utf-8",
            )
        listing = DashboardReader(root).list_runs()
        self.assertEqual(len(listing["runs"]), MAX_LISTED_RUNS)
        self.assertTrue(listing["truncated"])

    def test_display_truncation_is_explicit_and_config_is_complete(self) -> None:
        source = {"members": list(range(305)), "note": "x" * 4_100}
        sanitized = sanitize_json(source)
        self.assertEqual(
            sanitized["members"][-1],
            "[display truncated: 5 of 305 items omitted]",
        )
        self.assertIn("display truncated: 100 characters omitted", sanitized["note"])
        config = {"long_value": "x" * 4_100, "all_values": list(range(305))}
        complete = sanitize_config(config)
        self.assertEqual(complete, config)


if __name__ == "__main__":
    unittest.main()
