"""Offline contract checks for the bounded export-only packet."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

PACKET_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKET_ROOT))

import build_export_packet as builder  # noqa: E402


class ExportPacketContractTests(unittest.TestCase):
    """Prevent source drift, training, scoring, and provider calls in delivery repair."""

    def test_parent_resource_worker_changes_only_identity_and_wall_cap(self) -> None:
        basis = PACKET_ROOT / "basis/parent_profile_worker.py"
        original = basis.read_bytes()
        patched = builder.patch_worker_identity(original)
        reverted = (
            patched.decode()
            .replace(
                f'PACKET_ID = "{builder.PACKET_ID}"',
                'PACKET_ID = "E529_H232_GRU384_B8_S3_AMP4096_PROACTIVE_CACHE_RECOVERY_20261008_V7A"',
            )
            .replace("WORKER_WALL_CAP_S = 4_500", "WORKER_WALL_CAP_S = 16_000")
        )
        self.assertEqual(reverted.encode(), original)

    def test_export_worker_has_no_training_or_grading_entrypoint(self) -> None:
        source = (PACKET_ROOT / "export_worker.py").read_text()
        self.assertNotIn("train_rnn.main", source)
        self.assertNotIn("profile_kernel.main", source)
        self.assertNotIn("global_wp_clip2_v1", source)
        self.assertNotIn("evaluate_bandit_continuation", source)
        self.assertEqual(builder.EXPORT_CAP_SECONDS, 1_800)
        self.assertIn('"--epoch",\n        "3"', source)
        self.assertIn('"--inference-precision",\n        "fp32"', source)

    def test_npy_payload_bytes_exclude_header(self) -> None:
        import numpy as np
        from export_worker import EXPECTED_PREDICTION_NBYTES, npy_payload_layout

        self.assertEqual(1_873 * 20_000 * 2 * 4, EXPECTED_PREDICTION_NBYTES)

        path = PACKET_ROOT / "tests" / "npy_header_regression.npy"
        array = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(3, 7, 2))
        array[:] = 0
        array.flush()
        expected_payload = 3 * 7 * 2 * np.dtype(np.float32).itemsize
        layout = npy_payload_layout(path, np)
        self.assertEqual(array.nbytes, expected_payload)
        self.assertEqual(layout["payload_bytes"], expected_payload)
        self.assertEqual(layout["file_bytes"], path.stat().st_size)
        self.assertGreater(layout["file_bytes"], layout["payload_bytes"])
        path.unlink()

    def test_built_packet_is_hash_consistent_and_provider_push_disabled(self) -> None:
        packet = PACKET_ROOT / "packet_r4_metadata_slugfix"
        if not (packet / "kernel-metadata.json").is_file():
            self.skipTest("packet build has not been materialized")
        payload_zip, payload_hashes = builder._zip_payload(packet)
        builder.verify_packet(packet, payload_zip, payload_hashes)

    def test_kernel_identity_is_new_and_private(self) -> None:
        self.assertLess(len(builder.KERNEL_TITLE), 50)
        self.assertIn("20261008", builder.KERNEL_ID)
        self.assertTrue(builder.KERNEL_ID.startswith("vladkochetov07/"))
        builder.validate_kernel_identity(builder.KERNEL_ID, builder.KERNEL_TITLE)

    def test_kernel_title_slug_must_match_the_id(self) -> None:
        with self.assertRaisesRegex(ValueError, "does not match kernel ID"):
            builder.validate_kernel_identity(
                "vladkochetov07/h232-gru384-v7b-ep3-export-20261008-r2",
                "H232 GRU384 V7B EP3 Export R2",
            )
        builder.validate_kernel_identity(
            "vladkochetov07/h232-gru384-v7b-ep3-export-20261008-r3",
            "H232 GRU384 V7B EP3 Export 20261008 R3",
        )


if __name__ == "__main__":
    unittest.main()
