"""Regression coverage using the actual V7B ep3 provenance and frozen fold 0."""

from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path

PACKET_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = Path(__file__).resolve().parents[6]
FIXTURE = PACKET_ROOT / "tests/fixtures/v7b_ep3_membership.json"
FOLD = REPOSITORY_ROOT / "research/campaign_2026-09-29/artifacts/splits/fold0.json"
SPLIT_SHA256 = "09263ad4fe2b18e2b2f3b79f4562678dce801b067b868134ddc79279f9316a3d"
sys.path.insert(0, str(PACKET_ROOT))

from membership_audit import MembershipError, audit_fit_membership  # noqa: E402


def load_snapshot() -> tuple[dict[str, object], dict[str, object]]:
    """Load compact actual membership provenance and the tracked fold manifest."""
    return json.loads(FIXTURE.read_text()), json.loads(FOLD.read_text())


class MembershipAuditTests(unittest.TestCase):
    """Check source-qualified group and feature-component identity invariants."""

    def test_actual_v7b_snapshot_passes_and_reports_integer_aliases(self) -> None:
        provenance, manifest = load_snapshot()
        result = audit_fit_membership(provenance, manifest, expected_split_sha256=SPLIT_SHA256)
        self.assertEqual(result["fit_counts"], {"train": 10_535, "valid": 1_405})
        self.assertEqual(result["evaluation_count"], 468)
        self.assertEqual(result["cross_source_integer_alias_count"], 463)
        self.assertTrue(result["cross_source_integer_aliases_are_not_overlap"])
        self.assertEqual(result["fit_evaluation_component_overlap_count"], 0)
        self.assertEqual(
            result["source_qualified_example"],
            {
                "train_row_group": 4,
                "train_component": 4,
                "valid_row_group": 4,
                "valid_component": 2_799,
            },
        )

    def test_same_valid_source_group_overlap_is_rejected(self) -> None:
        provenance, manifest = load_snapshot()
        mutated = copy.deepcopy(provenance)
        mutated["evaluation_row_groups"].append(mutated["fit_row_groups"]["valid"][0])
        with self.assertRaisesRegex(MembershipError, "same-source valid"):
            audit_fit_membership(mutated, manifest, expected_split_sha256=SPLIT_SHA256)

    def test_component_overlap_is_rejected_even_with_disjoint_group_ids(self) -> None:
        provenance, manifest = load_snapshot()
        mutated = copy.deepcopy(manifest)
        fit_group = mutated["fit"]["valid"][0]
        eval_group = mutated["evaluation"]["valid"][0]
        records = {node["row_group"]: node for node in mutated["sequences"]["valid"]}
        records[fit_group]["component"] = records[eval_group]["component"]
        with self.assertRaisesRegex(MembershipError, "component overlap"):
            audit_fit_membership(provenance, mutated, expected_split_sha256=SPLIT_SHA256)

    def test_changed_fit_member_is_rejected(self) -> None:
        provenance, manifest = load_snapshot()
        mutated = copy.deepcopy(provenance)
        mutated["fit_row_groups"]["valid"][0] = mutated["fit_row_groups"]["valid"][1]
        with self.assertRaisesRegex(MembershipError, "duplicate"):
            audit_fit_membership(mutated, manifest, expected_split_sha256=SPLIT_SHA256)

    def test_fit_membership_hash_tampering_is_rejected(self) -> None:
        provenance, manifest = load_snapshot()
        mutated = copy.deepcopy(provenance)
        mutated["fit_membership_identity"]["train"]["ordered_int64_le_sha256"] = "0" * 64
        with self.assertRaisesRegex(MembershipError, "digest differs"):
            audit_fit_membership(mutated, manifest, expected_split_sha256=SPLIT_SHA256)


if __name__ == "__main__":
    unittest.main()
