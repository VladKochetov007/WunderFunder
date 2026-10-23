from __future__ import annotations

import ast
import hashlib
import json
import os
import stat
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
V10_DIR = ROOT / "preparation/first_cell_evaluation_score_admission_v10"
V11_DIR = Path(__file__).resolve().parent
BODY = V11_DIR / "REMOTE_SUBMISSION_V11.py"
V10_BINDING = json.loads((V10_DIR / "SCORE_ADMISSION_SOURCE_BINDING_V10.json").read_text())
V11_BINDING = json.loads((V11_DIR / "SCORE_ADMISSION_SOURCE_BINDING_V11.json").read_text())


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


class CountingOS:
    O_RDONLY = os.O_RDONLY
    O_CLOEXEC = getattr(os, "O_CLOEXEC", 0)
    O_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)

    def __init__(self, read_override=None):
        self.read_requests: list[int] = []
        self.open_count = 0
        self.read_override = read_override

    def open(self, path, flags):
        self.open_count += 1
        return os.open(path, flags)

    @staticmethod
    def fstat(descriptor):
        return os.fstat(descriptor)

    def read(self, descriptor, size):
        self.read_requests.append(size)
        if self.read_override is not None:
            return self.read_override(descriptor, size)
        return os.read(descriptor, size)

    @staticmethod
    def close(descriptor):
        os.close(descriptor)


class SourceHashCapTests(unittest.TestCase):
    def invoke(self, path: Path, cap: int, fake_os: CountingOS):
        namespace = {
            "Path": Path,
            "dict": dict,
            "int": int,
            "str": str,
            "stat": stat,
            "os": fake_os,
            "hashlib": hashlib,
            "require": require,
        }
        function_node = next(
            node
            for node in ast.parse(BODY.read_text()).body
            if isinstance(node, ast.FunctionDef) and node.name == "file_hash"
        )
        module = ast.fix_missing_locations(ast.Module(body=[function_node], type_ignores=[]))
        exec(compile(module, str(BODY), "exec"), namespace)
        return namespace["file_hash"](path, cap)

    def test_zero_and_exact_expected_file_size_are_hashable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            empty = root / "empty.py"
            exact = root / "exact.py"
            empty.write_bytes(b"")
            exact.write_bytes(b"abc")
            empty_hash, empty_info = self.invoke(empty, 0, CountingOS())
            exact_hash, exact_info = self.invoke(exact, 3, CountingOS())
        self.assertEqual(empty_hash, hashlib.sha256(b"").hexdigest())
        self.assertEqual(empty_info["bytes"], 0)
        self.assertEqual(exact_hash, hashlib.sha256(b"abc").hexdigest())
        self.assertEqual(exact_info["bytes"], 3)

    def test_oversize_live_file_is_rejected_before_open(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "larger.py"
            path.write_bytes(b"abc")
            fake_os = CountingOS()
            with self.assertRaisesRegex(ValueError, "exceeds cap"):
                self.invoke(path, 2, fake_os)
        self.assertEqual(fake_os.open_count, 0)

    def test_growth_is_detected_with_only_one_sentinel_byte_read(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "growing.py"
            path.write_bytes(b"ab")
            fake_os = CountingOS(read_override=lambda _descriptor, _size: b"abc")
            with self.assertRaisesRegex(ValueError, "exceeded cap"):
                self.invoke(path, 2, fake_os)
        self.assertEqual(fake_os.read_requests, [3])

    def test_tree_sums_manifest_and_passes_remaining_cap_before_hash(self) -> None:
        tree = ast.parse(BODY.read_text())
        function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "check_source_tree"
        )
        source = ast.unparse(function)
        self.assertLess(source.index("expected_total_bytes"), source.index("file_hash("))
        self.assertIn("expected_total_bytes <= MAX_SOURCE_AGGREGATE_BYTES", source)
        self.assertIn("live_size == expected_bytes", source)
        self.assertIn("min(expected_bytes, remaining_source_bytes)", source)
        self.assertEqual(V10_BINDING["scorer_payloads"], V11_BINDING["scorer_payloads"])
        self.assertEqual(len(V11_BINDING["scorer_payloads"]), 6)


if __name__ == "__main__":
    unittest.main()
