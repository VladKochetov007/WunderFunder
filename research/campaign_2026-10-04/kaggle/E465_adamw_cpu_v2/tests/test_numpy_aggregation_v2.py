from __future__ import annotations

import ast
import unittest
from pathlib import Path

import numpy as np


def find_repo_root() -> Path:
    for path in Path(__file__).resolve().parents:
        if (path / "scripts/train_rnn.py").is_file():
            return path
    raise RuntimeError("repository root not found")


class AdamWAggregationTests(unittest.TestCase):
    def test_version_two_is_only_the_numpy_indexing_fix(self) -> None:
        root = find_repo_root()
        scripts = root / "research/campaign_2026-10-04/staging/jax_tpu_e465/scripts"
        old = (scripts / "check_e465_adamw_fp32_parity.py").read_text()
        revised = (scripts / "check_e465_adamw_fp32_parity_v2.py").read_text()
        self.assertEqual(
            old.replace(
                "error_by_value[indices]",
                "error_by_value[list(indices)]",
                1,
            ),
            revised,
        )

    def test_actual_aggregation_expression_selects_expected_pair_groups(self) -> None:
        root = find_repo_root()
        source = (
            root
            / "research/campaign_2026-10-04/staging/jax_tpu_e465/scripts/check_e465_adamw_fp32_parity_v2.py"
        ).read_text()
        module = ast.parse(source)
        main = next(
            node
            for node in module.body
            if isinstance(node, ast.FunctionDef) and node.name == "main"
        )
        assignment = next(
            node
            for node in main.body
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "maxima" for target in node.targets)
        )
        expression = ast.Expression(body=assignment.value)
        ast.fix_missing_locations(expression)
        namespace = {
            "np": np,
            "float": float,
            "error_by_value": np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        }
        maxima = eval(compile(expression, "<adamw-aggregation>", "eval"), namespace)
        self.assertEqual(
            maxima,
            {"parameter": 2.0, "first_moment": 4.0, "second_moment": 6.0},
        )


if __name__ == "__main__":
    unittest.main()
