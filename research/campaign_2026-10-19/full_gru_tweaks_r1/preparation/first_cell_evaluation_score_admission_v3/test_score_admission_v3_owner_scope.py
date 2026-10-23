"""Focused policy fixture for V3 cgroup and command-line owner checks."""

from __future__ import annotations

import ast
import importlib.util
import unittest
from pathlib import Path
from types import ModuleType

PACKET = Path(__file__).resolve().parent
V2_BODY = (
    PACKET.parent
    / "first_cell_evaluation_score_admission_v2"
    / "SCORE_ADMISSION_BODY_TEMPLATE_V2.py"
)
V3_BODY = PACKET / "SCORE_ADMISSION_BODY_TEMPLATE_V3.py"


def load_body() -> ModuleType:
    """Import the pure body module without invoking its main entrypoint."""
    spec = importlib.util.spec_from_file_location("score_admission_body_v3_fixture", V3_BODY)
    if spec is None or spec.loader is None:
        raise RuntimeError("V3 body module could not be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class OwnerScopeFixture(unittest.TestCase):
    def test_cgroup_all_uids_cmdline_same_uid_and_narrow_body_delta(self) -> None:
        body = load_body()
        foreign_calls: list[tuple[Path, int]] = []
        foreign_files = {Path("/fixture/731/cgroup"): b"0::/user.slice\n"}

        def foreign_reader(path: Path, limit: int) -> bytes:
            foreign_calls.append((path, limit))
            return foreign_files[path]

        foreign = body.read_gpu_owner_details(Path("/fixture/731"), 501, 502, foreign_reader)
        self.assertEqual(foreign, ("0::/user.slice", None))
        self.assertEqual(foreign_calls, [(Path("/fixture/731/cgroup"), 32_768)])

        same_uid_calls: list[tuple[Path, int]] = []
        same_uid_files = {
            Path("/fixture/732/cgroup"): b"0::/user.slice\n",
            Path("/fixture/732/cmdline"): b"python\0score_worker.py\0",
        }

        def same_uid_reader(path: Path, limit: int) -> bytes:
            same_uid_calls.append((path, limit))
            return same_uid_files[path]

        same_uid = body.read_gpu_owner_details(Path("/fixture/732"), 502, 502, same_uid_reader)
        self.assertEqual(same_uid, ("0::/user.slice", "python\0score_worker.py\0"))
        self.assertEqual(
            same_uid_calls,
            [
                (Path("/fixture/732/cgroup"), 32_768),
                (Path("/fixture/732/cmdline"), 8_192),
            ],
        )

        v2_tree = ast.parse(V2_BODY.read_text(encoding="utf-8"))
        v3_tree = ast.parse(V3_BODY.read_text(encoding="utf-8"))
        excluded = {"check_selected_gpu", "read_gpu_owner_details"}

        def unaffected_nodes(tree: ast.Module) -> list[str]:
            return [
                ast.dump(node, include_attributes=False)
                for node in tree.body
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                or node.name not in excluded
            ]

        self.assertEqual(unaffected_nodes(v2_tree), unaffected_nodes(v3_tree))
        gpu_check = next(
            node
            for node in v3_tree.body
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
        campaign_assignment = next(
            node
            for node in ast.walk(gpu_check)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "campaign_owned"
                for target in node.targets
            )
        )
        campaign_predicate = compile(ast.Expression(campaign_assignment.value), "<fixture>", "eval")
        fixture_root = Path("/fixture/project")
        self.assertTrue(
            eval(
                campaign_predicate,
                {
                    "__builtins__": {"str": str},
                    "PROJECT_BASE": fixture_root,
                    "cgroup": "0::/system.slice/wnn-campaign.slice/worker.service",
                    "command": None,
                },
            )
        )
        self.assertTrue(
            eval(
                campaign_predicate,
                {
                    "__builtins__": {"str": str},
                    "PROJECT_BASE": fixture_root,
                    "cgroup": "0::/user.slice/foreign.service",
                    "command": f"python {fixture_root}/worker.py",
                },
            )
        )
        self.assertFalse(
            eval(
                campaign_predicate,
                {
                    "__builtins__": {"str": str},
                    "PROJECT_BASE": fixture_root,
                    "cgroup": "0::/user.slice/unrelated.service",
                    "command": None,
                },
            )
        )


if __name__ == "__main__":
    unittest.main()
