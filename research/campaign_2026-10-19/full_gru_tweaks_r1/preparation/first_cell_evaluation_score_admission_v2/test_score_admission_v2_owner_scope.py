"""Focused policy fixture for V2 same-UID GPU owner metadata reads."""

from __future__ import annotations

import ast
import importlib.util
import unittest
from pathlib import Path
from types import ModuleType

PACKET = Path(__file__).resolve().parent
V1_BODY = (
    PACKET.parent
    / "first_cell_evaluation_score_admission_v1"
    / "SCORE_ADMISSION_BODY_TEMPLATE_V1.py"
)
V2_BODY = PACKET / "SCORE_ADMISSION_BODY_TEMPLATE_V2.py"


def load_body() -> ModuleType:
    """Import the pure body module without invoking its main entrypoint."""
    spec = importlib.util.spec_from_file_location("score_admission_body_v2_fixture", V2_BODY)
    if spec is None or spec.loader is None:
        raise RuntimeError("V2 body module could not be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class OwnerScopeFixture(unittest.TestCase):
    def test_same_uid_only_proc_detail_reads_and_narrow_body_delta(self) -> None:
        body = load_body()
        foreign_calls: list[tuple[Path, int]] = []

        def foreign_reader(path: Path, limit: int) -> bytes:
            foreign_calls.append((path, limit))
            raise AssertionError("foreign process detail must not be opened")

        foreign = body.read_gpu_owner_details(Path("/proc/731"), 501, 502, foreign_reader)
        self.assertEqual(foreign, (None, None))
        self.assertEqual(foreign_calls, [])

        same_uid_calls: list[tuple[Path, int]] = []
        fixture_files = {
            Path("/fixture/732/cgroup"): b"0::/user.slice\n",
            Path("/fixture/732/cmdline"): b"python\0score_worker.py\0",
        }

        def same_uid_reader(path: Path, limit: int) -> bytes:
            same_uid_calls.append((path, limit))
            return fixture_files[path]

        same_uid = body.read_gpu_owner_details(Path("/fixture/732"), 502, 502, same_uid_reader)
        self.assertEqual(same_uid, ("0::/user.slice", "python\0score_worker.py\0"))
        self.assertEqual(
            same_uid_calls,
            [
                (Path("/fixture/732/cgroup"), 32_768),
                (Path("/fixture/732/cmdline"), 8_192),
            ],
        )

        v1_tree = ast.parse(V1_BODY.read_text(encoding="utf-8"))
        v2_tree = ast.parse(V2_BODY.read_text(encoding="utf-8"))
        excluded = {"check_selected_gpu", "read_gpu_owner_details"}

        def unaffected_nodes(tree: ast.Module) -> list[str]:
            return [
                ast.dump(node, include_attributes=False)
                for node in tree.body
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                or node.name not in excluded
            ]

        self.assertEqual(unaffected_nodes(v1_tree), unaffected_nodes(v2_tree))
        gpu_check = next(
            node
            for node in v2_tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "check_selected_gpu"
        )
        calls = [
            node
            for node in ast.walk(gpu_check)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "read_gpu_owner_details"
        ]
        self.assertEqual(len(calls), 1)
        self.assertEqual(
            [ast.dump(argument, include_attributes=False) for argument in calls[0].args],
            [
                ast.dump(ast.Name(id="proc", ctx=ast.Load()), include_attributes=False),
                ast.dump(
                    ast.Attribute(
                        value=ast.Name(id="before", ctx=ast.Load()),
                        attr="st_uid",
                        ctx=ast.Load(),
                    ),
                    include_attributes=False,
                ),
                ast.dump(
                    ast.Call(
                        func=ast.Attribute(
                            value=ast.Name(id="os", ctx=ast.Load()),
                            attr="getuid",
                            ctx=ast.Load(),
                        ),
                        args=[],
                        keywords=[],
                    ),
                    include_attributes=False,
                ),
                ast.dump(
                    ast.Name(id="_bounded_proc_read", ctx=ast.Load()), include_attributes=False
                ),
            ],
        )


if __name__ == "__main__":
    unittest.main()
