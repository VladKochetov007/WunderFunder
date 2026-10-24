"""Stdlib-only validation tests for the batch-amended cohort controller."""

from __future__ import annotations

import copy
import importlib.util
import sys
import unittest
from pathlib import Path
from typing import Any

OPERATOR_PATH = Path(__file__).with_name("run_teacher_cohort_batch.py")
SPEC = importlib.util.spec_from_file_location("run_teacher_cohort_batch", OPERATOR_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("could not load the batch-amended cohort operator")
OPERATOR = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = OPERATOR
SPEC.loader.exec_module(OPERATOR)


def make_amendment() -> dict[str, Any]:
    """Return one complete amendment contract for local validation tests."""
    return {
        "schema": OPERATOR.BATCH_AMENDMENT_SCHEMA,
        "authority_id": OPERATOR.BATCH_AUTHORITY_ID,
        "operator_sha256": OPERATOR.sha256_file(OPERATOR_PATH.resolve()),
        "original_global_batch": 16,
        "per_job_batch": {
            "t1280_f0": 8,
            "t1792_f0": 16,
            "t1280_f2": 96,
            "t1792_f2": 128,
        },
        "new_attempt_from_seed": True,
    }


class BatchAmendmentTests(unittest.TestCase):
    """Reject batch plans that are incomplete or depart from the frozen recipe."""

    def test_batch_96_and_128_require_the_exact_max_throughput_authority(self) -> None:
        self.assertEqual(
            OPERATOR.BATCH_AUTHORITY_ID,
            "user-2026-10-24-max-throughput-batch-on-8-b300",
        )
        batches = OPERATOR.validate_batch_amendment({"batch_amendment": make_amendment()})
        self.assertEqual((batches["t1280_f2"], batches["t1792_f2"]), (96, 128))

    def test_exact_authorized_map_is_returned(self) -> None:
        batches = OPERATOR.validate_batch_amendment({"batch_amendment": make_amendment()})
        self.assertEqual(
            batches,
            {
                "t1280_f0": 8,
                "t1792_f0": 16,
                "t1280_f2": 96,
                "t1792_f2": 128,
            },
        )

    def test_undeclared_or_extra_cells_are_rejected(self) -> None:
        for change in ("missing", "extra"):
            with self.subTest(change=change):
                amendment = make_amendment()
                if change == "missing":
                    amendment["per_job_batch"].pop("t1792_f2")
                else:
                    amendment["per_job_batch"]["t9999_f0"] = 16
                with self.assertRaisesRegex(ValueError, "exactly the four"):
                    OPERATOR.validate_batch_amendment({"batch_amendment": amendment})

    def test_undeclared_and_mismatched_config_batches_are_rejected(self) -> None:
        batches = OPERATOR.validate_batch_amendment({"batch_amendment": make_amendment()})
        with self.assertRaisesRegex(ValueError, "no per-job batch"):
            OPERATOR.validate_job_batch("t1792_f2", 64, {})
        with self.assertRaisesRegex(ValueError, "differs from batch_amendment"):
            OPERATOR.validate_job_batch("t1792_f2", 32, batches)

    def test_only_declared_batch_sizes_are_accepted(self) -> None:
        for invalid in (4, 12, 256, True):
            with self.subTest(batch=invalid):
                amendment = make_amendment()
                amendment["per_job_batch"]["t1280_f0"] = invalid
                with self.assertRaisesRegex(ValueError, "8, 16, 32, 64, 96 or 128"):
                    OPERATOR.validate_batch_amendment({"batch_amendment": amendment})

    def test_authority_baseline_and_fresh_seed_attempt_are_required(self) -> None:
        for field, invalid in (
            ("authority_id", "unspecified"),
            ("original_global_batch", 32),
            ("new_attempt_from_seed", False),
        ):
            with self.subTest(field=field):
                amendment = make_amendment()
                amendment[field] = invalid
                with self.assertRaises(ValueError):
                    OPERATOR.validate_batch_amendment({"batch_amendment": amendment})

    def test_frozen_recipe_allows_declared_batch_only(self) -> None:
        repository = next(
            parent
            for parent in Path(__file__).resolve().parents
            if (parent / "pyproject.toml").is_file()
        )
        base_path = (
            repository
            / "research/experiment_drafts/b300-transformer-distillation-runtime"
            / "large_teacher_r3_muon30/configs/t1280_f0.json"
        )
        base = OPERATOR.read_json(base_path)
        amended = copy.deepcopy(base)
        amended["batch_size"] = 16
        amended["required_world_size"] = 2
        amended["precision_mode"] = "bf16_bf16x9"
        amended["model"]["checkpoint_blocks"] = False
        amended["model"]["attention_backend"] = "efficient"
        amended["model"]["compile_blocks"] = True
        OPERATOR.validate_frozen_recipe(repository, "t1280_f0", amended, "bf16_bf16x9", 16)

        amended["lr"] = 0.0002
        with self.assertRaisesRegex(ValueError, "unapproved deviation"):
            OPERATOR.validate_frozen_recipe(repository, "t1280_f0", amended, "bf16_bf16x9", 16)


if __name__ == "__main__":
    unittest.main()
