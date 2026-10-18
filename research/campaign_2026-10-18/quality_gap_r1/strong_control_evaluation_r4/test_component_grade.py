"""Synthetic checks for the centered component-bootstrap adapter."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
from grade_control import _group_statistics

from wnn.evaluation import paired_component_bootstrap
from wnn.gmetric import global_wp


class ComponentGradeTests(unittest.TestCase):
    """Check that group moments preserve the canonical pooled paired difference."""

    def test_grade_source_paths_resolve_from_repository_root(self) -> None:
        repo_root = Path(__file__).resolve().parents[4]
        self.assertTrue((repo_root / "src/wnn/gmetric.py").is_file())
        self.assertTrue((repo_root / "src/wnn/evaluation.py").is_file())

    def test_centered_component_bootstrap_reproduces_pooled_delta(self) -> None:
        targets = np.array(
            [
                [-0.8, -0.4],
                [0.1, 0.5],
                [0.7, 1.1],
                [-1.1, -0.6],
                [0.4, 0.2],
                [1.2, 0.9],
                [-0.2, -1.0],
                [0.6, 0.3],
                [1.0, 1.4],
                [-1.3, -0.9],
                [0.2, 0.7],
                [0.9, 1.2],
            ],
            dtype=np.float32,
        )
        candidate = targets * np.float32(0.82) + np.float32(0.07)
        reference = targets * np.float32(0.76) - np.float32(0.03)
        row_group = np.repeat(np.arange(4, dtype=np.int64), 3)
        component = np.repeat(np.array([10, 10, 20, 20], dtype=np.int64), 3)
        candidate_stats, reference_stats, group_components = _group_statistics(
            targets, candidate, reference, row_group, component
        )
        result = paired_component_bootstrap(
            candidate_stats,
            reference_stats,
            group_components,
            n_boot=200,
            seed=271828,
            centered=True,
        )
        expected = global_wp(targets, candidate, np.ones(len(targets), dtype=bool))["wp"]
        expected -= global_wp(targets, reference, np.ones(len(targets), dtype=bool))["wp"]
        self.assertAlmostEqual(result["mean_diff"], expected, delta=5e-10)
        self.assertEqual(result["n_components"], 2)


if __name__ == "__main__":
    unittest.main()
