"""Focused static fixtures for the selected-GPU gate added to NLL verifier V2."""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

FOLDER = Path(__file__).resolve().parent
BODY = FOLDER / "REMOTE_SUBMISSION_V2.py"


class SelectedGpuGateAstTests(unittest.TestCase):
    """Inspect only the new gate's source; never import or execute the remote body."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.source = BODY.read_text(encoding="utf-8")
        cls.tree = ast.parse(cls.source)
        cls.functions = {
            node.name: node for node in cls.tree.body if isinstance(node, ast.FunctionDef)
        }
        cls.gate = cls.functions["check_selected_gpu_ownership"]
        cls.owner = cls.functions["_gpu_process_owner"]
        cls.query = cls.functions["_bounded_gpu_query"]
        cls.main = cls.functions["main"]

    def test_selected_gpu_uuid_and_queries_are_pinned(self) -> None:
        constants = {
            node.targets[0].id: ast.literal_eval(node.value)
            for node in self.tree.body
            if isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "SELECTED_GPU_UUID"
        }
        self.assertEqual(constants["SELECTED_GPU_UUID"], "GPU-ec74220d-78d7-cc47-69fc-0de1fb580944")
        self.assertIn("--query-gpu=uuid,name,memory.free,memory.total", self.source)
        self.assertIn("--query-compute-apps=pid,gpu_uuid,used_memory", self.source)

    def test_gate_is_before_create_and_its_observation_is_recorded(self) -> None:
        statements = self.main.body
        gate_index = next(
            index
            for index, node in enumerate(statements)
            if isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "check_selected_gpu_ownership"
        )
        create_index = next(
            index
            for index, node in enumerate(statements)
            if isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Attribute)
            and isinstance(node.value.func.value, ast.Name)
            and node.value.func.value.id == "D"
            and node.value.func.attr == "mkdir"
        )
        self.assertLess(gate_index, create_index)
        self.assertIn('"selected_gpu_ownership": gpu_ownership', self.source)

    def test_foreign_processes_are_recorded_as_preserved(self) -> None:
        owner_source = ast.get_source_segment(self.source, self.owner)
        self.assertIsNotNone(owner_source)
        self.assertIn('"preserved": True', owner_source)
        self.assertIn("if before.st_uid == os.getuid():", owner_source)
        self.assertIn("selected-GPU PID identity became unavailable", owner_source)
        self.assertNotIn("exited_during_observation", owner_source)
        self.assertIn('"wnn-campaign.slice" in cgroup', owner_source)
        self.assertIn("command_has_project_path", owner_source)
        self.assertNotIn("kill(", owner_source)
        self.assertNotIn("terminate(", owner_source)

    def test_process_inventory_and_proc_reads_have_explicit_bounds(self) -> None:
        self.assertIn("len(rows) <= 256", self.source)
        self.assertIn('_bounded_proc_read(proc / "cgroup", 32768)', self.source)
        self.assertIn('_bounded_proc_read(proc / "cmdline", 8192)', self.source)
        self.assertIn('_bounded_proc_read(proc / "stat", 8192)', self.source)
        self.assertIn("identity_before == identity_after", self.source)
        query_source = ast.get_source_segment(self.source, self.query)
        self.assertIsNotNone(query_source)
        self.assertIn("65536", query_source)
        self.assertIn("time.monotonic() + 15", query_source)

    def test_cpu_gate_has_no_vram_reservation_threshold(self) -> None:
        gate_source = ast.get_source_segment(self.source, self.gate)
        self.assertIsNotNone(gate_source)
        self.assertIn('"minimum_free_memory_required_bytes": 0', gate_source)
        self.assertNotIn("MIN_GPU_FREE", gate_source)
        self.assertNotIn("memory_free_bytes >", gate_source)


if __name__ == "__main__":
    unittest.main()
