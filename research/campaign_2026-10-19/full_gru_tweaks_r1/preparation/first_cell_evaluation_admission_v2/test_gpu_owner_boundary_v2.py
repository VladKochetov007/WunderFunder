"""Focused source-fragment fixtures for V2 GPU process ownership inspection."""

from __future__ import annotations

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace

DIRECTORY = Path(__file__).resolve().parent
BODY_TEMPLATE = DIRECTORY / "ADMISSION_BODY_TEMPLATE_V2.py"


def gpu_gate_nodes() -> list[ast.stmt]:
    """Select the exact ownership assignment, same-UID branch, and fail-closed gate."""
    tree = ast.parse(BODY_TEMPLATE.read_text(encoding="utf-8"), filename=str(BODY_TEMPLATE))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "gpu_owners"
    )

    def find_gate(statements: list[ast.stmt]) -> list[ast.stmt] | None:
        for index, node in enumerate(statements):
            is_initial_assignment = (
                isinstance(node, ast.Assign)
                and any(
                    isinstance(target, ast.Name) and target.id == "campaign_owned"
                    for target in node.targets
                )
                and any(
                    isinstance(value, ast.Constant) and value.value == "wnn-campaign.slice"
                    for value in ast.walk(node.value)
                )
            )
            if is_initial_assignment:
                return statements[index : index + 3]
            for field in ("body", "orelse", "finalbody"):
                child = getattr(node, field, None)
                if isinstance(child, list):
                    found = find_gate(child)
                    if found is not None:
                        return found
        return None

    selected = find_gate(function.body)
    if selected is None or len(selected) != 3:
        raise AssertionError("GPU ownership source statements changed")
    assignment, branch, decision = selected
    if not isinstance(branch, ast.If) or not isinstance(decision, ast.Expr):
        raise AssertionError("GPU ownership branch/gate source layout changed")
    if not isinstance(decision.value, ast.Call) or not isinstance(decision.value.func, ast.Name):
        raise AssertionError("GPU ownership fail-closed call changed")
    if decision.value.func.id != "require":
        raise AssertionError("GPU ownership does not use the expected require gate")
    return [assignment, branch, decision]


def run_gpu_gate(
    *, current_uid: int, process_uid: int, cgroup: str, command: bytes
) -> tuple[bool, list[Path]]:
    """Execute only the three selected statements with a spy for cmdline access."""
    reads: list[Path] = []
    process_path = Path("/proc/1234")

    def bounded_proc_read(path: Path, maximum: int) -> bytes:
        if path.name != "cmdline" or maximum != 8_192:
            raise AssertionError("unexpected proc read in ownership fragment")
        reads.append(path)
        return command

    def require(condition: bool, message: str) -> None:
        if not condition:
            raise RuntimeError(message)

    namespace = {
        "after": SimpleNamespace(st_uid=process_uid),
        "cgroup": cgroup,
        "os": SimpleNamespace(getuid=lambda: current_uid),
        "proc": process_path,
        "_bounded_proc_read": bounded_proc_read,
        "PROJECT_BASE": Path("/home/grinchenkoeg/computer-vision-homework-20261018"),
        "require": require,
    }
    module = ast.Module(body=gpu_gate_nodes(), type_ignores=[])
    code = compile(ast.fix_missing_locations(module), str(BODY_TEMPLATE), "exec")
    try:
        exec(code, namespace)
    except RuntimeError:
        return True, reads
    return False, reads


class GPUOwnerBoundaryTests(unittest.TestCase):
    def test_foreign_uid_never_reads_cmdline_but_campaign_cgroup_still_rejects(self) -> None:
        rejected, reads = run_gpu_gate(
            current_uid=1061,
            process_uid=2002,
            cgroup="/user.slice/user-2002.slice/session-1.scope",
            command=b"/home/grinchenkoeg/computer-vision-homework-20261018/secret-argv",
        )
        self.assertFalse(rejected)
        self.assertEqual(reads, [])

        rejected, reads = run_gpu_gate(
            current_uid=1061,
            process_uid=2002,
            cgroup="/user.slice/user-2002.slice/wnn-campaign.slice/worker.service",
            command=b"foreign command must remain unread",
        )
        self.assertTrue(rejected)
        self.assertEqual(reads, [])

    def test_current_uid_project_command_is_read_and_rejected(self) -> None:
        rejected, reads = run_gpu_gate(
            current_uid=1061,
            process_uid=1061,
            cgroup="/user.slice/user-1061.slice/session-2.scope",
            command=b"/home/grinchenkoeg/computer-vision-homework-20261018/export.py\0--fixed",
        )
        self.assertTrue(rejected)
        self.assertEqual(reads, [Path("/proc/1234/cmdline")])

    def test_current_uid_unrelated_command_is_preserved(self) -> None:
        rejected, reads = run_gpu_gate(
            current_uid=1061,
            process_uid=1061,
            cgroup="/user.slice/user-1061.slice/session-2.scope",
            command=b"/usr/bin/other-job\0--preserve",
        )
        self.assertFalse(rejected)
        self.assertEqual(reads, [Path("/proc/1234/cmdline")])


if __name__ == "__main__":
    unittest.main()
