"""Stdlib negative cases for E464 queue spec, domain, budget and artifact gates."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[6]
CAMPAIGN = ROOT / "research/campaign_2026-10-04"
sys.path.insert(0, str(CAMPAIGN / "checks"))


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


queue = load_module("e464_queue_controller_under_test", CAMPAIGN / "checks/run_e464_sequence.py")
builder = load_module(
    "e464_queue_spec_builder_under_test",
    CAMPAIGN / "staging/e464/scripts/build_queue_specs.py",
)


def canonical_spec() -> dict[str, object]:
    return {
        "version": 1,
        "name": "e464_predict_fixture_fp32_v1",
        "kind": "gpu",
        "argv": ["{python}", "predict.py"],
        "code": ["src/model.py"],
        "inputs": ["artifacts/future_model.pt"],
        "artifacts": ["cache/future.npy"],
        "limits": {"cpus": 1, "ram_bytes": 2**30, "gpu_memory_bytes": 100,
                   "gpu_stop_bytes": 200, "timeout_seconds": 60},
        "environment": {},
    }


class E464QueueControllerTests(unittest.TestCase):
    def test_rich_design_is_reduced_to_path_only_runtime_schema(self) -> None:
        rich = canonical_spec() | {
            "architecture_scope": {"freeze": ["inp", "gru"]},
            "code": [{"path": "src/model.py", "sha256": "a" * 64}],
            "inputs": ["artifacts/future_model.pt", "artifacts/future_model.pt"],
        }
        reduced = builder.canonicalize_design_spec(rich)
        self.assertEqual(set(reduced), queue.EXECUTION_SPEC_FIELDS)
        self.assertEqual(reduced["code"], ["src/model.py"])
        self.assertEqual(reduced["inputs"], ["artifacts/future_model.pt"])
        queue.validate_path_only_spec(reduced)

    def test_runtime_schema_rejects_rich_design_fields(self) -> None:
        rich = canonical_spec() | {"architecture_scope": {"freeze": ["inp"]}}
        with self.assertRaisesRegex(ValueError, "noncanonical"):
            queue.validate_path_only_spec(rich)

    def test_runtime_schema_rejects_content_descriptors(self) -> None:
        spec = canonical_spec()
        spec["code"] = [{"path": "src/model.py", "sha256": "a" * 64}]
        with self.assertRaisesRegex(ValueError, "only path strings"):
            queue.validate_path_only_spec(spec)

    def test_runtime_schema_rejects_path_traversal(self) -> None:
        spec = canonical_spec()
        spec["inputs"] = ["../outside.pt"]
        with self.assertRaisesRegex(ValueError, "root-relative"):
            queue.validate_path_only_spec(spec)

    def test_missing_predecessor_unit_handle_fails_closed(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "missing or unavailable"):
            queue.require_loaded_service(
                {"LoadState": "not-found", "ActiveState": "inactive", "ExecMainStatus": "0"},
                "wnn-e425-sequence-v1.service",
            )

    def test_prediction_domain_requires_exact_order_and_integer_types(self) -> None:
        self.assertTrue(queue.prediction_domain_is_exact(list(range(1873))))
        self.assertFalse(queue.prediction_domain_is_exact(list(range(1, 1874))))
        self.assertFalse(queue.prediction_domain_is_exact(list(range(1872)) + [1870]))
        self.assertFalse(queue.prediction_domain_is_exact(list(range(1872)) + [True]))
        self.assertFalse(queue.prediction_domain_is_exact(tuple(range(1873))))

    def test_budget_gate_includes_full_next_timeout_at_boundary(self) -> None:
        self.assertTrue(queue.reserve_fits(10_200, 600, 10_800))
        self.assertFalse(queue.reserve_fits(10_201, 600, 10_800))
        self.assertFalse(queue.reserve_fits(10_200, 601, 10_800))

    def test_budget_gate_rejects_invalid_numeric_types_and_negative_charge(self) -> None:
        self.assertFalse(queue.reserve_fits(True, 60, 10_800))
        self.assertFalse(queue.reserve_fits(-1, 60, 10_800))
        self.assertFalse(queue.reserve_fits(1.5, 60.0, 10_800))

    def test_artifact_coverage_fails_when_a_declared_output_is_missing(self) -> None:
        spec = {"artifacts": ["cache/a.npy", "cache/a.json"]}
        record = {"artifacts": [{"path": "cache/a.npy", "size": 3, "sha256": "a" * 64}]}
        with self.assertRaisesRegex(ValueError, "exactly cover"):
            queue.verify_recorded_artifacts(record, spec)

    def test_artifact_hash_must_match_current_file_bytes(self) -> None:
        old_root = queue.ROOT
        with tempfile.TemporaryDirectory() as temporary:
            queue.ROOT = Path(temporary)
            target = queue.ROOT / "cache/a.npy"
            target.parent.mkdir(parents=True)
            target.write_bytes(b"actual")
            record = {"artifacts": [{"path": "cache/a.npy", "size": 6,
                                     "sha256": "0" * 64}]}
            try:
                with self.assertRaisesRegex(ValueError, "hash changed"):
                    queue.verify_recorded_artifacts(record, {"artifacts": ["cache/a.npy"]})
            finally:
                queue.ROOT = old_root

    def test_sealed_identity_rejects_rich_draft_before_hash_comparison(self) -> None:
        rich = canonical_spec() | {"review_note": "not runner schema"}
        with tempfile.TemporaryDirectory() as temporary:
            draft_path = Path(temporary) / "rich.draft.json"
            draft_path.write_text(json.dumps(rich))
            with self.assertRaisesRegex(ValueError, "noncanonical"):
                queue.verify_sealed_matches_draft(draft_path, {})


if __name__ == "__main__":
    unittest.main()
