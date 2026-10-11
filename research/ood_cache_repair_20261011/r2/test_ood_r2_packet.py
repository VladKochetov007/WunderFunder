"""Bounded integrity and launch-contract checks for the frozen OOD R2 packet."""

from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

PACKET = Path(__file__).resolve().parent
RUNNER = PACKET / "ood_student_pipeline_shared_r2.py"
BUILDER = PACKET / "build_distill_cache_r2.py"
MANIFEST = PACKET / "OOD_R2_BUNDLE_MANIFEST.json"
JOBS = PACKET / "OOD_R2_RETRY_JOBS.json"


def digest(path: Path) -> str:
    """Hash a small packet input in bounded memory."""
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            hasher.update(block)
    return hasher.hexdigest()


class OodR2PacketTests(unittest.TestCase):
    """Verify frozen source lineage, guarded imports and explicit job argv."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = json.loads(MANIFEST.read_text())
        cls.jobs = json.loads(JOBS.read_text())

    def test_bundle_files_and_58_file_source_manifest_are_hash_bound(self) -> None:
        self.assertEqual(self.manifest["frozen_source_manifest_sha256"], digest(PACKET / "frozen_source_manifest_r1.json"))
        sources = json.loads((PACKET / "frozen_source_manifest_r1.json").read_text())["sources"]
        self.assertEqual(len(sources), 58)
        expected = {
            "ood_student_pipeline_shared_r2.py": RUNNER,
            "build_distill_cache_r2.py": BUILDER,
            "frozen_source_manifest_r1.json": PACKET / "frozen_source_manifest_r1.json",
        }
        self.assertEqual(set(self.manifest["bundle_files"]), set(expected))
        for name, path in expected.items():
            self.assertEqual(self.manifest["bundle_files"][name], digest(path), name)

    def test_runner_and_builder_verify_sources_before_project_imports(self) -> None:
        runner = RUNNER.read_text()
        builder = BUILDER.read_text()
        self.assertLess(runner.index("BUNDLE = _preimport_bootstrap"), runner.index("import numpy as np"))
        self.assertLess(builder.index("_verify_before_project_imports()"), builder.index("import numpy as np"))
        self.assertIn("/root/wnn/lr377_source_r1", runner)
        self.assertIn('PROJECT_ROOT / "lr377_source_r1"', builder)
        self.assertIn("combine_teacher_chunk", builder)
        self.assertNotIn("distill_pipeline", runner)
        self.assertNotIn("dp.predict", runner)
        self.assertIn("predict_run.py", runner)
        self.assertIn("train_rnn.py", runner)
        self.assertIn(".partial", runner)

    def test_four_job_vectors_use_frozen_python_timeout_and_separate_environment(self) -> None:
        self.assertEqual(len(self.jobs["jobs"]), 4)
        expected_arms = {"splice_r2", "spmix_r2", "swn_r2", "swn25_r2"}
        self.assertEqual({item["arm"] for item in self.jobs["jobs"]}, expected_arms)
        manifest_sha = digest(MANIFEST)
        for item in self.jobs["jobs"]:
            argv = item["argv"]
            self.assertEqual(argv[:5], ["timeout", "--signal=TERM", "--kill-after=10s", "14400s", "/opt/conda/bin/python"])
            self.assertEqual(argv[5], "/root/wnn/research/ood_cache_repair_20261011/r2/ood_student_pipeline_shared_r2.py")
            self.assertIn(item["arm"], argv)
            self.assertEqual(argv[argv.index("--bundle-manifest-sha256") + 1], manifest_sha)
            self.assertEqual(argv[argv.index("--other-live-reservations-bytes") + 1], "6000000000")
            self.assertFalse(any(value.startswith("WNN_PROJECT_ROOT=") for value in argv))
            self.assertEqual(item["env"]["WNN_PROJECT_ROOT"], "/root/wnn")
            self.assertEqual(item["env"]["WNN_FROZEN_SOURCE_ROOT"], "/root/wnn/lr377_source_r1")
            self.assertEqual(item["env"]["PYTHONPATH"], "/root/wnn/lr377_source_r1/src")

    def test_disk_reservation_declares_all_fixed_terms(self) -> None:
        reserve = self.manifest["disk_reservation"]
        self.assertEqual(reserve["compact_cache_payload_bytes"], 1_911_840_000)
        self.assertEqual(reserve["prediction_rows"], 1873)
        self.assertEqual(reserve["prediction_bytes_each"], 299_680_000)
        self.assertEqual(reserve["checkpoint_state_reserve_each_bytes"], 67_108_864)
        self.assertEqual(reserve["other_live_reservations_bytes"], 6_000_000_000)
        self.assertEqual(reserve["minimum_free_bytes"], 20 * (1 << 30))


if __name__ == "__main__":
    unittest.main()
