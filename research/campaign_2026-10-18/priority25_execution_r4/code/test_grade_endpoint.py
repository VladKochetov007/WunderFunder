"""Small CPU fixtures for export identity and official metric integration."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from grade_endpoint import (
    membership_sha256,
    validate_export_manifest,
    validate_source_rows,
)

from wnn.data import SEQUENCE_LENGTH
from wnn.gmetric import METRIC_ID, GlobalAccumulator, reference_global_wp


class GradeEndpointTests(unittest.TestCase):
    def _manifest(self) -> dict[str, object]:
        groups = [7, 3, 11]
        return {
            "run": "candidate",
            "split": "valid",
            "epoch": 5,
            "split_manifest_sha256": "a" * 64,
            "campaign_split_manifest_sha256": "a" * 64,
            "campaign_export_scope": "evaluation",
            "feature_only": True,
            "inference_precision": "fp32",
            "row_groups": groups,
            "evaluation_membership_sha256": membership_sha256(groups),
        }

    def test_ordered_row_groups_are_part_of_export_identity(self) -> None:
        manifest = self._manifest()
        with self.assertRaisesRegex(ValueError, "row_groups"):
            validate_export_manifest(
                manifest,
                run="candidate",
                epoch=5,
                split_sha256="a" * 64,
                row_groups=[3, 7, 11],
            )

    def test_campaign_manifest_hash_must_match(self) -> None:
        manifest = self._manifest()
        manifest["campaign_split_manifest_sha256"] = "b" * 64
        with self.assertRaisesRegex(ValueError, "campaign_split_manifest_sha256"):
            validate_export_manifest(
                manifest,
                run="candidate",
                epoch=5,
                split_sha256="a" * 64,
                row_groups=[7, 3, 11],
            )

    def test_source_row_order_labels_and_official_mask_are_bound(self) -> None:
        steps = np.arange(SEQUENCE_LENGTH, dtype=np.int64)
        need = steps >= 99
        scored = (steps % 3) != 0
        labels = np.column_stack((steps / 1000, -steps / 2000)).astype(np.float32)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "datasets").mkdir()
            table = pa.table(
                {
                    "seq_ix": np.full(SEQUENCE_LENGTH, 19, dtype=np.int64),
                    "step_in_seq": steps,
                    "need_prediction": need,
                    "is_scored": scored,
                    "t0": labels[:, 0],
                    "t1": labels[:, 1],
                }
            )
            pq.write_table(
                table, root / "datasets" / "valid.parquet", row_group_size=SEQUENCE_LENGTH
            )
            masks = scored[None, :]
            targets = labels[None, :, :]
            official = validate_source_rows(root, [0], {0: 0}, np.array([19]), masks, targets)
            np.testing.assert_array_equal(official[0], need & scored)
            with self.assertRaisesRegex(ValueError, "step order"):
                reordered = pa.concat_tables([table.slice(1), table.slice(0, 1)])
                pq.write_table(
                    reordered, root / "datasets" / "valid.parquet", row_group_size=SEQUENCE_LENGTH
                )
                validate_source_rows(root, [0], {0: 0}, np.array([19]), masks, targets)

    def test_chunked_official_metric_matches_independent_reference(self) -> None:
        rng = np.random.default_rng(314159)
        targets = rng.normal(size=(5, 317, 2)).astype(np.float32)
        predictions = (0.45 * targets + rng.normal(size=targets.shape)).astype(np.float32)
        mask = rng.random(size=targets.shape[:2]) > 0.28
        mask[:, :99] = False
        targets[~mask] = np.nan
        predictions[~mask] = np.nan
        reference = reference_global_wp(targets, predictions, mask)
        accumulator = GlobalAccumulator()
        accumulator.add(targets[:2], predictions[:2], mask[:2])
        accumulator.add(targets[2:], predictions[2:], mask[2:])
        result = accumulator.result()
        self.assertEqual(result["metric_id"], METRIC_ID)
        self.assertAlmostEqual(float(result["wp"]), reference, places=12)


if __name__ == "__main__":
    unittest.main()
