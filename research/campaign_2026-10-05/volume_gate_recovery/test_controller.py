from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

import controller


class ControllerTests(unittest.TestCase):
    def test_fixed_admission_is_denied_at_30gb_even_with_zero_current_usage(self):
        with self.assertRaisesRegex(RuntimeError, "RAM admission denied"):
            controller.check_resource_admission(current_bytes=0)
        self.assertEqual(controller.WORKER_RAM_BYTES + controller.SUPPORT_RESERVE_BYTES, 30_064_771_072)

    def test_timebox_stops_new_admission_and_keeps_full_future_inventory(self):
        self.assertTrue(controller.may_admit_new_stage(datetime(2026, 10, 5, 6, 59, tzinfo=timezone.utc)))
        self.assertFalse(controller.may_admit_new_stage(datetime(2026, 10, 5, 7, 0, tzinfo=timezone.utc)))
        self.assertEqual([s["stage"] for s in controller.stage_inventory()], ["smoke", "fit", "export", "score"])
        gpu = [s for s in controller.stage_inventory() if s.get("device", "gpu") == "gpu"]
        self.assertEqual(sum(3 * s["seconds_each"] for s in gpu), 18_900)

    def test_declared_input_cache_advice_is_exact_and_hash_checked(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            file = root / "declared.bin"
            file.write_bytes(b"small fixture")
            entry = {"path": "declared.bin", "size": file.stat().st_size,
                     "sha256": controller.sha256_file(file)}
            advised = controller.advise_exact_declared_inputs(root, [entry])
            self.assertEqual([x["path"] for x in advised], ["declared.bin"])
            with self.assertRaisesRegex(ValueError, "identity changed"):
                controller.advise_exact_declared_inputs(root, [{**entry, "sha256": "0" * 64}])

    def test_attempt_artifact_missing_hash_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "out").write_bytes(b"x")
            with self.assertRaisesRegex(ValueError, "size/hash"):
                controller.verify_artifacts(root, {"spec": {"artifacts": ["out"], "code_sha256": "a" * 64},
                    "spec_sha256": "b" * 64, "artifacts": [{"path": "out", "size": 1}]})

    def test_fresh_output_guard_rejects_existing_target(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            target = root / "out"
            target.write_text("old")
            spec = root / "spec.json"
            spec.write_text(json.dumps({"artifacts": ["out"]}))
            with self.assertRaisesRegex(FileExistsError, "already exists"):
                controller.require_fresh_outputs(root, spec)

    def test_e464_identity_and_attempt_states_reject_wrong_partial_failed_or_bad_memory(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            campaign = root / "research/campaign_2026-10-04"
            (campaign / "checks").mkdir(parents=True)
            (campaign / "runner/attempts").mkdir(parents=True)
            (root / Path(controller.EXPECTED_REPORT).parent).mkdir(parents=True)
            report = {"metric_id": "global_wp_clip2_v1", "n_sequences": 368,
                      "screen": {"folds": ["holdout"], "comparisons": [{} for _ in range(16)]}}
            report_path = root / controller.EXPECTED_REPORT
            report_path.write_text(json.dumps(report))
            sequence = {"status": "complete", "completed": list(controller.EXPECTED_COMPLETED),
                        "paired_score_path": controller.EXPECTED_REPORT, "paired_comparisons": 16,
                        "bootstrap_replicates": 2000, "paired_score_sha256": controller.sha256_file(report_path)}
            sequence_path = campaign / "checks/e464_sequence_v1.json"
            sequence_path.write_text(json.dumps(sequence))
            for name in controller.EXPECTED_ATTEMPTS:
                (campaign / "runner/attempts" / f"{name}.json").write_text(json.dumps({
                    "id": name, "status": "finished", "exit_code": 0, "teardown_verified": True,
                    "failure": None, "kind": "cpu" if name == controller.EXPECTED_SCORE else "gpu",
                    "final_memory_events": {"oom": 0, "oom_kill": 0, "oom_group_kill": 0},
                    "spec_sha256": "b" * 64, "spec": {"name": name, "code_sha256": "a" * 64,
                        "artifacts": [f"proof/{name}.txt"]},
                    "artifacts": []}))
                ap = root / f"proof/{name}.txt"
                ap.parent.mkdir(exist_ok=True)
                ap.write_text(name)
                record_path = campaign / "runner/attempts" / f"{name}.json"
                record = json.loads(record_path.read_text())
                record["artifacts"] = [{"path": f"proof/{name}.txt", "size": ap.stat().st_size,
                                        "sha256": controller.sha256_file(ap)}]
                record_path.write_text(json.dumps(record))
            self.assertEqual(len(controller.verify_e464(root)["attempts"]), 17)
            seq_bad = {**sequence, "completed": sequence["completed"][:-1]}
            sequence_path.write_text(json.dumps(seq_bad))
            with self.assertRaisesRegex(ValueError, "exact completed 17"):
                controller.verify_e464(root)
            sequence_path.write_text(json.dumps(sequence))
            bad = campaign / "runner/attempts" / f"{controller.EXPECTED_FITS[0]}.json"
            record = json.loads(bad.read_text())
            record["exit_code"] = 15
            bad.write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError, "successful terminal"):
                controller.verify_e464(root)
            record["exit_code"] = 0
            record["final_memory_events"]["oom_kill"] = 1
            bad.write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError, "zero OOM"):
                controller.verify_e464(root)


if __name__ == "__main__":
    unittest.main(verbosity=2)
