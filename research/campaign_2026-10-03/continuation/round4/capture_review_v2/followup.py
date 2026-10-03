"""Small independent AST and typed RNG edge checks without rerunning the smoke."""

from __future__ import annotations

import ast
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

OWN = Path(__file__).resolve().parent
ROOT = OWN.parents[4]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(OWN.parent / "live_capture_v2"))
import baselines
import live_capture as capture

LOGGER = logging.getLogger(__name__)


class EraseObservations(ast.NodeTransformer):
    def visit_If(self, node):
        if ast.unparse(node.test) == "_capture.enabled":
            return None
        return self.generic_visit(node)

    def visit_IfExp(self, node):
        if ast.unparse(node.test) == "_capture.enabled":
            return self.visit(node.orelse)
        return self.generic_visit(node)

    def visit_Expr(self, node):
        if isinstance(node.value, ast.Call) and ast.unparse(node.value.func).startswith("_baseline."):
            if ast.unparse(node.value.func) == "_baseline.record_clip":
                return ast.Expr(value=self.visit(node.value.args[0]))
            return None
        return self.generic_visit(node)

    def visit_Call(self, node):
        if ast.unparse(node.func) == "_baseline.component":
            return self.visit(node.args[1])
        return self.generic_visit(node)


def rejected(a, b) -> bool:
    try:
        capture.compare(a, b)
    except AssertionError:
        return True
    return False


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    original = ast.dump(ast.Module(body=capture.chunk_nodes(), type_ignores=[]), include_attributes=False)
    trees = {
        "online": ast.Module(body=capture.instrument_body(capture.chunk_nodes()), type_ignores=[]),
        "original_baseline": ast.Module(body=baselines.independent_original_nodes(False), type_ignores=[]),
    }
    equality = {name: ast.dump(EraseObservations().visit(tree), include_attributes=False) == original for name, tree in trees.items()}
    result = {"AST_erased_equals_source": equality,
              "signed_zero_without_tolerance_rejected": rejected(torch.tensor(0.0), torch.tensor(-0.0)),
              "numpy_RNG_value_change_rejected": rejected(np.array([1], dtype=np.uint32), np.array([2], dtype=np.uint32)),
              "numpy_RNG_dtype_change_rejected": rejected(np.array([1], dtype=np.uint32), np.array([1], dtype=np.uint64))}
    (OWN / "followup_results.json").write_text(json.dumps(result, indent=2) + "\n")
    LOGGER.info("FOLLOWUP %s", json.dumps(result))


if __name__ == "__main__":
    main()
