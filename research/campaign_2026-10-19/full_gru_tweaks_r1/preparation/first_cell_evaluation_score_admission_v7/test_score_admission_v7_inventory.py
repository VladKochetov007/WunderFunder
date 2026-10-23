"""Tiny metadata fixtures for V7's fail-closed source inventory gate."""

from __future__ import annotations

import ast
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import ModuleType

PACKET = Path(__file__).resolve().parent
PREPARATION = PACKET.parent
BODY_V6 = (
    PREPARATION / "first_cell_evaluation_score_admission_v6" / "SCORE_ADMISSION_BODY_TEMPLATE_V6.py"
)
BODY_V7 = PACKET / "SCORE_ADMISSION_BODY_TEMPLATE_V7.py"
PLAN = PACKET / "SCORE_ADMISSION_BINDING_PLAN_V7.json"
V6_GAP = PACKET / "V6_SOURCE_INVENTORY_GAP_NOTE_V1.json"


def _load_body() -> ModuleType:
    """Load the stdlib-only unrendered template without calling its entrypoint."""
    spec = importlib.util.spec_from_file_location("score_admission_v7_body", BODY_V7)
    if spec is None or spec.loader is None:
        raise AssertionError("V7 body template could not be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def require(condition: bool, message: str) -> None:
        if not condition:
            raise RuntimeError(message)

    module.require = require
    return module


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _without_inventory_functions(tree: ast.Module) -> ast.Module:
    allowed_changes = {"inventory_exact_source_files", "check_source_tree"}
    tree.body = [
        node
        for node in tree.body
        if not (isinstance(node, ast.FunctionDef) and node.name in allowed_changes)
    ]
    return tree


class ScoreAdmissionV7Inventory(unittest.TestCase):
    def test_exact_small_tree_matches_expected_manifest_paths(self) -> None:
        body = _load_body()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            (root / "wnn" / "metrics").mkdir(parents=True)
            (root / "wnn" / "__init__.py").write_bytes(b"# package\n")
            (root / "wnn" / "metrics" / "score.py").write_bytes(b"score = 1\n")
            expected = {"wnn/__init__.py", "wnn/metrics/score.py"}
            observed = body.inventory_exact_source_files(root, expected)
            self.assertEqual(observed, {"file_count": 2, "directory_count": 2, "entry_count": 4})

    def test_each_unlisted_code_or_runtime_file_fails_before_hashing(self) -> None:
        body = _load_body()
        for filename in ("extra.py", "extra.so", "extra.pyc"):
            with self.subTest(filename=filename), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary).resolve()
                (root / "wnn").mkdir()
                (root / "wnn" / "main.py").write_bytes(b"main = 1\n")
                (root / "wnn" / filename).write_bytes(b"unlisted\n")
                with self.assertRaisesRegex(RuntimeError, "unlisted file"):
                    body.inventory_exact_source_files(root, {"wnn/main.py"})

    def test_symlink_file_and_directory_entries_fail_closed(self) -> None:
        body = _load_body()
        for kind in ("file", "directory"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary).resolve()
                (root / "wnn").mkdir()
                target = root / "wnn" / "main.py"
                target.write_bytes(b"main = 1\n")
                link = root / "wnn" / ("linked.py" if kind == "file" else "linked_dir")
                link.symlink_to(
                    target if kind == "file" else root / "wnn",
                    target_is_directory=kind == "directory",
                )
                with self.assertRaisesRegex(RuntimeError, "symbolic link"):
                    body.inventory_exact_source_files(root, {"wnn/main.py"})

    def test_source_inventory_call_precedes_every_source_file_hash(self) -> None:
        tree = _tree(BODY_V7)
        checker = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "check_source_tree"
        )
        inventory_positions = [
            index
            for index, node in enumerate(ast.walk(checker))
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "inventory_exact_source_files"
        ]
        hash_positions = [
            index
            for index, node in enumerate(ast.walk(checker))
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "file_hash"
        ]
        self.assertEqual(len(inventory_positions), 1)
        self.assertEqual(len(hash_positions), 1)
        inventory_call = next(
            node
            for node in ast.walk(checker)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "inventory_exact_source_files"
        )
        hash_call = next(
            node
            for node in ast.walk(checker)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "file_hash"
        )
        self.assertLess(inventory_call.lineno, hash_call.lineno)

    def test_v7_body_matches_v6_outside_the_inventory_closure(self) -> None:
        self.assertEqual(
            ast.dump(_without_inventory_functions(_tree(BODY_V6)), include_attributes=False),
            ast.dump(_without_inventory_functions(_tree(BODY_V7)), include_attributes=False),
        )

    def test_plan_keeps_future_export_receipts_unbound_and_v6_gap_pinned(self) -> None:
        plan = json.loads(PLAN.read_bytes())
        self.assertEqual(plan["schema"], "fg19.first-cell-score-admission-binding-plan.v7")
        for name in ("acceptance", "exact_stop"):
            self.assertIsNone(plan["export_authority"][name]["path"])
            self.assertIsNone(plan["export_authority"][name]["bytes"])
            self.assertIsNone(plan["export_authority"][name]["sha256"])
        gap = json.loads(V6_GAP.read_bytes())
        self.assertEqual(gap["schema"], "fg19.first-cell-score-admission-v6-inventory-gap.v1")
        self.assertEqual(
            gap["v6_body"]["path"],
            "first_cell_evaluation_score_admission_v6/SCORE_ADMISSION_BODY_TEMPLATE_V6.py",
        )


if __name__ == "__main__":
    unittest.main()
