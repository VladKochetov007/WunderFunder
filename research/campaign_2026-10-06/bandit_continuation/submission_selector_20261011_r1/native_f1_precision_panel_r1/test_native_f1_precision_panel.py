"""Focused tests for the fixed panel and selective transport contracts."""

from __future__ import annotations

import io
import json
import unittest
from pathlib import Path

import numpy as np
from build_selective_fp16_transport import (
    CONVERTED_TENSORS,
    convert_npz,
    validate_archive_paths,
)
from freeze_candidate_catalog import fixed_members_for_candidate
from freeze_native_f1_panel import validate_selection

PACKET_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[5]


def make_npz(names: set[str], values: np.ndarray | None = None) -> bytes:
    arrays = {
        name: np.array(
            values if name == "gru.weight_ih_l0" and values is not None else [[0.25]],
            dtype=np.float32,
        )
        for name in names
    }
    output = io.BytesIO()
    np.savez(output, **arrays)
    return output.getvalue()


class NativeF1PrecisionPanelTests(unittest.TestCase):
    """Check fixed membership and the exact selective storage conversion."""

    def test_root_selected_panel_is_41_groups_from_14_whole_components(self) -> None:
        panel_path = PACKET_DIR / "references" / "NATIVE_PRECISION_COMPONENT_PANEL_R1.json"
        split_path = PROJECT_ROOT / "research/campaign_2026-09-29/artifacts/splits/fold1.json"
        panel = json.loads(panel_path.read_text())
        split = json.loads(split_path.read_text())
        mapping = validate_selection(panel, split)
        self.assertEqual(len(mapping), 41)
        self.assertEqual(len({row["component"] for row in mapping}), 14)

    def test_selective_conversion_changes_only_declared_matrices(self) -> None:
        names = CONVERTED_TENSORS | {"inp.weight", "head.2.weight", "head.2.bias"}
        source = make_npz(names, np.array([[0.33333334]], dtype=np.float32))
        transported, receipt = convert_npz(source, "weights_0.npz")
        with (
            np.load(io.BytesIO(source), allow_pickle=False) as original,
            np.load(io.BytesIO(transported), allow_pickle=False) as stored,
        ):
            for name in names:
                if name in CONVERTED_TENSORS:
                    self.assertEqual(stored[name].dtype, np.dtype("float16"))
                    np.testing.assert_array_equal(
                        stored[name].astype(np.float32),
                        original[name].astype(np.float16).astype(np.float32),
                    )
                else:
                    self.assertEqual(stored[name].dtype, np.dtype("float32"))
                    np.testing.assert_array_equal(stored[name], original[name])
        self.assertEqual(
            {row["name"] for row in receipt["tensors"] if row["converted"]}, CONVERTED_TENSORS
        )

    def test_conversion_rejects_missing_selected_matrix(self) -> None:
        payload = make_npz(CONVERTED_TENSORS - {"gru.weight_hh_l1"})
        with self.assertRaisesRegex(ValueError, "lacks required dense matrices"):
            convert_npz(payload, "weights_0.npz")

    def test_conversion_rejects_half_storage_overflow(self) -> None:
        values = np.array([[100_000.0]], dtype=np.float32)
        payload = make_npz(CONVERTED_TENSORS, values)
        with (
            np.errstate(over="ignore"),
            self.assertRaisesRegex(ValueError, "overflows in binary16"),
        ):
            convert_npz(payload, "weights_0.npz")

    def test_archive_paths_reject_traversal_and_duplicates(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsafe package path"):
            validate_archive_paths(["../weights.npz"])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_archive_paths(["weights.npz", "weights.npz"])

    def test_transport_child_inherits_exact_fixed_members_from_build_source(self) -> None:
        source = {
            "candidate_id": "teacher-trio-fp32",
            "members": [
                {"run": "teacher_s3@ep2", "coefficient": 1 / 3},
                {"run": "teacher_s4@ep2", "coefficient": 1 / 3},
                {"run": "teacher_s5@ep2", "coefficient": 1 / 3},
            ],
        }
        child = {
            "candidate_id": "teacher-trio-fp16-transport",
            "source_build_candidate": "teacher-trio-fp32",
        }
        fixed = fixed_members_for_candidate(child, {source["candidate_id"]: source})
        self.assertEqual(
            fixed,
            [
                {"run_reference": "teacher_s3@ep2", "coefficient": 1 / 3},
                {"run_reference": "teacher_s4@ep2", "coefficient": 1 / 3},
                {"run_reference": "teacher_s5@ep2", "coefficient": 1 / 3},
            ],
        )

    def test_transport_child_rejects_missing_source_membership(self) -> None:
        child = {
            "candidate_id": "teacher-trio-fp16-transport",
            "source_build_candidate": "absent",
        }
        with self.assertRaisesRegex(ValueError, "no fixed members"):
            fixed_members_for_candidate(child, {})


if __name__ == "__main__":
    unittest.main()
