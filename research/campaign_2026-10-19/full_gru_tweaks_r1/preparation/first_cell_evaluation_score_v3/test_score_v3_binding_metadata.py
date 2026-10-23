"""Metadata-only proof of the scorer's V3 binding and evidence-root update."""

from __future__ import annotations

import ast
import hashlib
import json
import unittest
from pathlib import Path

PACKET = Path(__file__).resolve().parent
PREPARATION = PACKET.parent
SCORE_V2 = PREPARATION / "first_cell_evaluation_score_v2"
EXPORT_V2 = PREPARATION / "first_cell_evaluation_export_v2"
EXPORT_V3 = PREPARATION / "first_cell_evaluation_export_v3"
BINDING_BYTES = 15_798
BINDING_SHA256 = "372ec16016517baa510cc1899f140cd0978a35b83f96a4a5fa1ff4bda8b39730"
OLD_EVIDENCE_BASENAME = "fg19_first_cell_f0_s1001_evaluation_v1"
NEW_EVIDENCE_BASENAME = "fg19_first_cell_f0_s1001_evaluation_v2"


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"))


def _normalize_core(tree: ast.Module) -> ast.Module:
    assignments = {
        "EXPECTED_BINDING_BYTES": BINDING_BYTES,
        "EXPECTED_BINDING_SHA256": BINDING_SHA256,
    }
    seen: set[str] = set()
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            name = node.targets[0].id
            if name in assignments:
                seen.add(name)
                node.value = ast.Constant(
                    value="<binding-pin>" if name.endswith("SHA256") else "<binding-bytes>"
                )
    if seen != set(assignments):
        raise AssertionError("expected scorer binding assignments were not found")
    return tree


def _normalize_contract(tree: ast.Module) -> ast.Module:
    found_binding = 0
    found_root = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if node.value == "EXPORT_BINDING_V3.json" or node.value == "EXPORT_BINDING_V1.json":
                node.value = "EXPORT_BINDING_<VERSION>.json"
                found_binding += 1
            if node.value == NEW_EVIDENCE_BASENAME:
                node.value = OLD_EVIDENCE_BASENAME
                found_root += 1
            elif node.value == OLD_EVIDENCE_BASENAME:
                found_root += 1
    if found_binding != 2 or found_root != 1:
        raise AssertionError(
            f"unexpected contract delta counts: binding={found_binding}, root={found_root}"
        )
    return tree


class ScoreV3MetadataFixture(unittest.TestCase):
    def test_binding_is_v2_with_only_four_bound_root_paths_changed(self) -> None:
        old_path = EXPORT_V2 / "EXPORT_BINDING_V1.json"
        new_path = EXPORT_V3 / "EXPORT_BINDING_V3.json"
        old_raw, new_raw = old_path.read_bytes(), new_path.read_bytes()
        self.assertEqual(
            (len(new_raw), hashlib.sha256(new_raw).hexdigest()), (BINDING_BYTES, BINDING_SHA256)
        )
        old, new = json.loads(old_raw), json.loads(new_raw)
        old["evidence_root"] = new["evidence_root"]
        old["output_dir"] = new["output_dir"]
        old["arms"][4]["exact_stop"]["absolute_path"] = new["arms"][4]["exact_stop"][
            "absolute_path"
        ]
        old["arms"][4]["terminal_acceptance"]["absolute_path"] = new["arms"][4][
            "terminal_acceptance"
        ]["absolute_path"]
        self.assertEqual(old, new)
        self.assertEqual(new["evidence_root"].rsplit("/", 1)[-1], NEW_EVIDENCE_BASENAME)
        self.assertEqual(new["output_dir"], new["evidence_root"] + "/exports")

    def test_scorer_delta_is_only_binding_pin_and_fixed_filename_root(self) -> None:
        old_core, new_core = (
            SCORE_V2 / "first_cell_score_core.py",
            PACKET / "first_cell_score_core.py",
        )
        self.assertEqual(
            ast.dump(_normalize_core(_tree(old_core)), include_attributes=False),
            ast.dump(_normalize_core(_tree(new_core)), include_attributes=False),
        )
        old_contract = _normalize_contract(_tree(SCORE_V2 / "first_cell_score_contract.py"))
        new_contract = _normalize_contract(_tree(PACKET / "first_cell_score_contract.py"))
        self.assertEqual(
            ast.dump(old_contract, include_attributes=False),
            ast.dump(new_contract, include_attributes=False),
        )
        for name in ("first_cell_score_predictions.py", "score_first_cell.py"):
            self.assertEqual((SCORE_V2 / name).read_bytes(), (PACKET / name).read_bytes())
        source = (PACKET / "first_cell_score_core.py").read_text()
        self.assertIn(f"EXPECTED_BINDING_BYTES = {BINDING_BYTES}", source)
        self.assertIn(f'EXPECTED_BINDING_SHA256 = "{BINDING_SHA256}"', source)

    def test_manifest_pins_exact_exporter_and_new_consumer(self) -> None:
        manifest = json.loads((PACKET / "dependency_manifest.json").read_bytes())
        self.assertEqual(manifest["schema"], "fg19.first-cell-score-dependency-manifest.v3")
        producer = manifest["producer_contract"]
        self.assertEqual(
            producer["path"], "../first_cell_evaluation_export_v3/export_first_cell_v3.py"
        )
        exporter_raw = (EXPORT_V3 / "export_first_cell_v3.py").read_bytes()
        self.assertEqual(
            (producer["bytes"], producer["sha256"]),
            (len(exporter_raw), hashlib.sha256(exporter_raw).hexdigest()),
        )
        self.assertEqual(producer["binding"]["runtime_filename"], "EXPORT_BINDING_V3.json")
        self.assertEqual(producer["binding"]["sha256"], BINDING_SHA256)
        source_raw = (EXPORT_V3 / "EXPORT_SOURCE_MANIFEST.json").read_bytes()
        source_pin = producer["source_manifest"]
        self.assertEqual(
            (source_pin["bytes"], source_pin["sha256"]),
            (len(source_raw), hashlib.sha256(source_raw).hexdigest()),
        )
        self.assertEqual(producer["source_review"]["status"], "PENDING_SOURCE_ONLY_REVIEW")
        for name in (
            "first_cell_score_core.py",
            "first_cell_score_contract.py",
            "first_cell_score_predictions.py",
            "score_first_cell.py",
            "test_score_v3_binding_metadata.py",
        ):
            pin = next(row for row in manifest["score_consumer_files"] if row["path"] == name)
            raw = (PACKET / name).read_bytes()
            self.assertEqual(
                (pin["bytes"], pin["sha256"]), (len(raw), hashlib.sha256(raw).hexdigest())
            )


if __name__ == "__main__":
    unittest.main()
