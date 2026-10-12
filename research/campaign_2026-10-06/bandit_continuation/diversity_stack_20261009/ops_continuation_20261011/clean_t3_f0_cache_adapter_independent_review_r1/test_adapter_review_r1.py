"""Independent low-cost checks against the clean T3 adapter's real functions."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

ADAPTER_DIR = (
    Path(__file__).resolve().parents[1]
    / "clean_t3_f0_matched_student_r5_cache_r1"
    / "cache_adapter_r1"
)
ADAPTER_FILE = ADAPTER_DIR / "build_clean_t3_f0_homogeneous_cache_r1.py"
_spec = importlib.util.spec_from_file_location("reviewed_cache_adapter", ADAPTER_FILE)
if _spec is None or _spec.loader is None:
    raise RuntimeError("could not load the reviewed adapter")
adapter = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(adapter)


class AdapterReviewTests(unittest.TestCase):
    """Exercise production arithmetic and output-collision guards only."""

    def test_target_operation_matches_frozen_float32_reduce(self) -> None:
        generator = np.random.default_rng(20261012)
        arrays = [
            generator.normal(size=(2, 20_000, 2)).astype(np.float32)
            for _ in range(3)
        ]
        stacked = np.stack(arrays, axis=0)
        reference = np.add.reduce(stacked, axis=0, dtype=np.float32)
        reference /= np.float32(3.0)
        reference *= np.float32(0.8125)
        observed = adapter.homogeneous_float32_chunk(*arrays)
        self.assertEqual(observed.dtype, np.float32)
        np.testing.assert_array_equal(observed, reference)

    def test_existing_output_and_partial_are_rejected_without_mutation(self) -> None:
        for existing_suffix in ("", ".partial"):
            with self.subTest(existing_suffix=existing_suffix):
                with tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary)
                    output = root / "cache"
                    collision = root / f"cache{existing_suffix}"
                    collision.mkdir()
                    sentinel = collision / "sentinel"
                    sentinel.write_text("preserve")
                    with self.assertRaises(FileExistsError):
                        adapter._write_output(output, {}, [], [], 8)
                    self.assertEqual(sentinel.read_text(), "preserve")
                    self.assertEqual(
                        sorted(path.name for path in root.iterdir()),
                        [f"cache{existing_suffix}"],
                    )

    def test_actual_sidecars_match_terminal_handoffs_and_contract(self) -> None:
        main = Path("/home/vlad/development/wunder/wnn_connectome_starterpack")
        ops = (
            main
            / "research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009"
            / "ops_continuation_20261011"
        )
        packet = ops / "clean_t3_f0_matched_student_r5_cache_r1"
        contract = json.loads(
            (packet / "cache_adapter_r1/CACHE_ADAPTER_CONTRACT_R1.json").read_text()
        )
        s3 = json.loads(
            (ops / "clean_t3_local_s3_terminal_r1/LOCAL_TERMINAL_HANDOFF_R1.json").read_text()
        )
        s4_s5 = json.loads(
            (
                ops
                / "clean_t3_local_s4_s5_terminal_r1/LOCAL_EXPORT_TERMINAL_HANDOFF_R1.json"
            ).read_text()
        )
        receipt_by_seed = {
            3: {
                "sidecar_sha256": s3["output_validation"]["sidecar_sha256"],
                "prediction_sha256": s3["output_validation"]["npy_sha256"],
            },
            **{
                item["seed"]: {
                    "sidecar_sha256": item["sidecar_sha256"],
                    "prediction_sha256": item["prediction_sha256"],
                }
                for item in s4_s5["outputs"]
            },
        }
        for teacher in contract["teachers"]:
            with self.subTest(seed=teacher["seed"]):
                binding = teacher["fit_valid_full_valid_prediction"]
                sidecar_path = main / binding["sidecar_path"]
                sidecar_bytes = sidecar_path.read_bytes()
                sidecar = json.loads(sidecar_bytes)
                terminal = receipt_by_seed[teacher["seed"]]
                self.assertEqual(
                    hashlib.sha256(sidecar_bytes).hexdigest(), binding["sidecar_sha256"]
                )
                self.assertEqual(binding["sidecar_sha256"], terminal["sidecar_sha256"])
                self.assertEqual(sidecar["prediction_sha256"], binding["prediction_sha256"])
                self.assertEqual(binding["prediction_sha256"], terminal["prediction_sha256"])
                self.assertEqual(sidecar["run"], teacher["run"])
                self.assertEqual(sidecar["epoch"], 2)
                self.assertEqual(sidecar["inference_precision"], "fp32")
                self.assertEqual(sidecar["campaign_fit_scope"], "full-valid-feature-only")
                self.assertEqual(
                    sidecar["checkpoint_sha256"], teacher["checkpoint_sha256"]
                )
                self.assertEqual(
                    sidecar["split_manifest_sha256"],
                    contract["lineage"]["split_manifest_sha256"],
                )
                self.assertEqual(sidecar["row_groups"], list(range(1873)))
                prediction = main / binding["path"]
                self.assertEqual(prediction.stat().st_size, binding["expected_file_bytes"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
