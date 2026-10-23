"""Focused local checks for bounded score-capture metadata readers."""

from __future__ import annotations

import ast
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType

PACKET = Path(__file__).parent
PREPARATION = PACKET.parent
TEMPLATE_V3 = (
    PREPARATION / "first_cell_evaluation_score_capture_v3" / "SCORE_RESULT_CAPTURE_TEMPLATE_V3.py"
)
TEMPLATE_V4 = PACKET / "SCORE_RESULT_CAPTURE_TEMPLATE_V4.py"
CHANGED_FUNCTIONS = {"_show", "_cgroup_populated", "_cgroup_text", "_gpu_identity"}
NEW_FUNCTIONS = {"_run_bounded_command", "_bounded_metadata_read"}
NEW_CONSTANTS = {
    "MAX_CGROUP_DIRECTORIES",
    "MAX_CGROUP_ENTRIES_PER_DIRECTORY",
    "MAX_CGROUP_PROCS_FILE_BYTES",
    "MAX_CGROUP_PROCS_AGGREGATE_BYTES",
    "MAX_CGROUP_METADATA_FILE_BYTES",
    "MAX_PROC_CGROUP_BYTES",
    "MAX_COMMAND_CHUNK_BYTES",
}


def _load_template() -> ModuleType:
    """Load only the standalone standard-library capture template."""
    spec = importlib.util.spec_from_file_location("score_result_capture_v4_fixture", TEMPLATE_V4)
    if spec is None or spec.loader is None:
        raise AssertionError("V4 result-capture template could not be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _normalized_tree(path: Path) -> ast.Module:
    """Remove only the reviewed bounded-reader changes from a template AST."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    body: list[ast.stmt] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            node.names = [alias for alias in node.names if alias.name not in {"selectors", "time"}]
            if not node.names:
                continue
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id in NEW_CONSTANTS for target in node.targets
        ):
            continue
        if isinstance(node, ast.FunctionDef) and node.name in NEW_FUNCTIONS:
            continue
        if isinstance(node, ast.FunctionDef) and node.name in CHANGED_FUNCTIONS:
            body.append(ast.Expr(value=ast.Constant(value=f"<bounded:{node.name}>")))
            continue
        body.append(node)
    tree.body = body
    return tree


class ScoreResultCaptureV4BoundedMetadata(unittest.TestCase):
    def test_only_bounded_observer_leaves_and_their_imports_changed(self) -> None:
        self.assertEqual(
            ast.dump(_normalized_tree(TEMPLATE_V3), include_attributes=False),
            ast.dump(_normalized_tree(TEMPLATE_V4), include_attributes=False),
        )

    def test_subprocess_reader_accepts_bounded_output_and_rejects_overflow(self) -> None:
        module = _load_template()
        command = [
            sys.executable,
            "-B",
            "-c",
            "import sys;sys.stdout.write('ok');sys.stderr.write('err')",
        ]
        self.assertEqual(module._run_bounded_command(command, 3, 2, 3), (0, "ok", "err"))
        for stream, limit in (("stdout", 8), ("stderr", 9)):
            code = f"import sys;sys.{stream}.write('x'*{limit + 100})"
            with (
                self.subTest(stream=stream),
                self.assertRaisesRegex(RuntimeError, "time or output limit"),
            ):
                module._run_bounded_command([sys.executable, "-B", "-c", code], 3, limit, limit)

    def test_subprocess_timeout_kills_the_owned_child(self) -> None:
        module = _load_template()
        command = [sys.executable, "-B", "-c", "import time;time.sleep(5)"]
        with self.assertRaisesRegex(RuntimeError, "time or output limit"):
            module._run_bounded_command(command, 0.1, 64, 64)

    def test_capture_has_no_posthoc_or_unbounded_reader_path(self) -> None:
        tree = ast.parse(TEMPLATE_V4.read_text(encoding="utf-8"))
        prohibited_calls = {"rglob", "read_text", "read_bytes"}
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                self.assertNotIn(node.func.attr, prohibited_calls)
                self.assertFalse(
                    node.func.attr == "run"
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "subprocess"
                )
            if isinstance(node, ast.Name):
                self.assertNotEqual(node.id, "capture_output")

    def test_cgroup_inventory_is_tree_bounded_and_preserves_empty_entries(self) -> None:
        module = _load_template()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "scope"
            child = root / "worker"
            child.mkdir(parents=True)
            (root / "cgroup.procs").write_text("123\n", encoding="ascii")
            (child / "cgroup.procs").write_text("", encoding="ascii")
            observed = module._cgroup_populated(root)
            self.assertEqual(observed[str(root / "cgroup.procs")], "123")
            self.assertEqual(observed[str(child / "cgroup.procs")], "")

    def test_metadata_file_reader_enforces_byte_cap_and_rejects_symlinks(self) -> None:
        module = _load_template()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            small = root / "small"
            small.write_bytes(b"abc")
            self.assertEqual(module._bounded_metadata_read(small, 3), b"abc")
            with self.assertRaisesRegex(ValueError, "exceeded 2 bytes"):
                module._bounded_metadata_read(small, 2)
            link = root / "link"
            link.symlink_to(small)
            with self.assertRaisesRegex(ValueError, "not a regular file"):
                module._bounded_metadata_read(link, 4)

    def test_cgroup_directory_and_aggregate_limits_fail_closed(self) -> None:
        module = _load_template()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "entry_cap"
            root.mkdir()
            (root / "cgroup.procs").write_text("", encoding="ascii")
            for index in range(module.MAX_CGROUP_ENTRIES_PER_DIRECTORY):
                (root / f"entry_{index:03d}").touch()
            with self.assertRaisesRegex(ValueError, "entry cap"):
                module._cgroup_populated(root)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "aggregate_cap"
            root.mkdir()
            (root / "cgroup.procs").write_bytes(b"1" * module.MAX_CGROUP_PROCS_FILE_BYTES)
            for index in range(8):
                child = root / f"child_{index:02d}"
                child.mkdir()
                (child / "cgroup.procs").write_bytes(b"2" * module.MAX_CGROUP_PROCS_FILE_BYTES)
            with self.assertRaisesRegex(ValueError, "aggregate byte cap"):
                module._cgroup_populated(root)

    def test_empty_or_root_unit_cgroup_is_not_a_host_root_fallback(self) -> None:
        tree = ast.parse(TEMPLATE_V4.read_text(encoding="utf-8"))
        main = next(
            node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main"
        )
        skips_empty_or_root = any(
            isinstance(node, ast.Compare)
            and isinstance(node.left, ast.Name)
            and node.left.id == "unit_control_group"
            and any(
                isinstance(comparator, ast.Tuple)
                and [value.value for value in comparator.elts if isinstance(value, ast.Constant)]
                == ["", "/"]
                for comparator in node.comparators
            )
            for node in ast.walk(main)
        )
        self.assertTrue(skips_empty_or_root)


if __name__ == "__main__":
    unittest.main()
