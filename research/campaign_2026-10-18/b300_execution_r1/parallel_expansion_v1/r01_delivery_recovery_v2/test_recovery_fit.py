"""CPU checks for original R01 fit actor completion evidence."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from r01_recovery_fit import _verify_completed_training_phase


def _write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _resource(*, status: str = "actor_failed", error_type: str = "ValueError") -> dict:
    return {
        "schema": "tabdata-b300-process-envelope.v1",
        "status": status,
        "script": "train_cell.py",
        "visible_devices": "GPU-example",
        "gpu": "NVIDIA B300",
        "allocator_budget_bytes": 6 * 1024**3,
        "torch_threads": 2,
        "cpu_affinity": [1, 3],
        "cuda_peak_allocated_bytes": 128,
        "cuda_peak_reserved_bytes": 256,
        "elapsed_seconds": 1.0,
        "error_type": error_type,
    }


class CompletedPhaseTests(unittest.TestCase):
    """Accept only the known post-fit receipt failure or a normal producer return."""

    def test_accepts_only_identified_postfit_receipt_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            _write(run_dir / "train_cell_resource_receipt.json", _resource())
            job = {"gpu_uuid": "GPU-example", "cpu_ids_initial": [1, 3]}
            result = _verify_completed_training_phase(run_dir, job)
            self.assertEqual(result["status"], "actor_failed")
            self.assertIsNone(result["original_producer_receipt_sha256"])

    def test_rejects_other_actor_failure_and_train_failure_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            resource_path = run_dir / "train_cell_resource_receipt.json"
            _write(resource_path, _resource(error_type="RuntimeError"))
            job = {"gpu_uuid": "GPU-example", "cpu_ids_initial": [1, 3]}
            with self.assertRaises(ValueError):
                _verify_completed_training_phase(run_dir, job)
            _write(resource_path, _resource())
            _write(
                run_dir / "producer_receipt.json",
                {"status": "trainer_failed", "error_type": "RuntimeError"},
            )
            with self.assertRaises(ValueError):
                _verify_completed_training_phase(run_dir, job)

    def test_normal_actor_return_requires_completed_producer_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            _write(
                run_dir / "train_cell_resource_receipt.json",
                _resource(status="actor_returned", error_type=""),
            )
            job = {"gpu_uuid": "GPU-example", "cpu_ids_initial": [1, 3]}
            with self.assertRaises(ValueError):
                _verify_completed_training_phase(run_dir, job)
            _write(
                run_dir / "producer_receipt.json",
                {"status": "completed_fixed_epoch5"},
            )
            result = _verify_completed_training_phase(run_dir, job)
            self.assertEqual(result["status"], "actor_returned")
            self.assertIsNotNone(result["original_producer_receipt_sha256"])


if __name__ == "__main__":
    unittest.main()
