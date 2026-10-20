from __future__ import annotations

import ast
import hashlib
import json
import sys
import unittest
from pathlib import Path

PREPARATION = Path(__file__).resolve().parents[1]
SOURCE_ROOT = Path("/Users/nb/Documents/wunder/WunderFunder-strong-gru-tweaks-20261019")
SOURCE_RELATIVE = "src/wnn/resources.py"
SOURCE_SHA256 = "95fdc839ff1561622ac06c465cfc9a01370130d99be1e832b3a33c79bcb37930"
MANIFEST = PREPARATION / "teacher_cache_cpu_repair_v2/SOURCE_MANIFEST.json"
sys.path.insert(0, str(PREPARATION))

from parent_fit_terminal_check_v3 import (  # noqa: E402
    CUDA_POLICY_SOURCE_SHA,
    _check_cuda_cudnn_deterministic,
)


def _attribute_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _attribute_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else None
    return None


class CudnnDeterminismTests(unittest.TestCase):
    """Exercise the source-bound runtime bool invariant without GPU/data reads."""

    def test_exact_true_is_required(self) -> None:
        _check_cuda_cudnn_deterministic(True)
        for value in (False, 1, 0, None, "true", "False"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                _check_cuda_cudnn_deterministic(value)

    def test_resource_policy_manifest_and_source_are_pinned(self) -> None:
        self.assertEqual(CUDA_POLICY_SOURCE_SHA, SOURCE_SHA256)
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        rows = [row for row in manifest["files"] if row["path"] == SOURCE_RELATIVE]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["sha256"], SOURCE_SHA256)
        source = SOURCE_ROOT / SOURCE_RELATIVE
        self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), SOURCE_SHA256)

        tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
        function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "configure_torch"
        )
        cuda_guard = next(
            node
            for node in ast.walk(function)
            if isinstance(node, ast.If)
            and ast.dump(node.test, include_attributes=False)
            == ast.dump(
                ast.parse("torch.cuda.is_available()").body[0].value, include_attributes=False
            )
        )
        assignments = {
            _attribute_name(node.targets[0]): node.value
            for node in ast.walk(cuda_guard)
            if isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and _attribute_name(node.targets[0])
            in {
                "torch.backends.cudnn.deterministic",
                "torch.backends.cudnn.benchmark",
            }
        }
        self.assertEqual(
            set(assignments),
            {
                "torch.backends.cudnn.deterministic",
                "torch.backends.cudnn.benchmark",
            },
        )
        self.assertIs(assignments["torch.backends.cudnn.deterministic"].value, True)
        self.assertIs(assignments["torch.backends.cudnn.benchmark"].value, False)


if __name__ == "__main__":
    unittest.main()
