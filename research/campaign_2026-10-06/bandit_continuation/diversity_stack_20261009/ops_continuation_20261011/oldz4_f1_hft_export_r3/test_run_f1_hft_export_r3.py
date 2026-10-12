"""Focused tests for the R3 export-only consumer."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

PACKET = Path(__file__).resolve().parent
MAIN = Path("/home/vlad/development/wunder/wnn_connectome_starterpack")
R2 = MAIN / (
    "research/campaign_2026-10-06/bandit_continuation/"
    "diversity_stack_20261009/ops_continuation_20261011/oldz4_f1_core_hft_r2"
)
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
SPEC = importlib.util.spec_from_file_location(
    "oldz4_hft_export_r3_tested", PACKET / "run_f1_hft_export_r3.py"
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ExportConsumerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.contract = json.loads((PACKET / "HFT_EXPORT_CONTRACT_R3.json").read_text())

    def test_actual_producer_preflight_binds_cp0_without_export(self) -> None:
        receipt_path = PACKET / "CP0_EXPORT_PREFLIGHT_R3.json"
        self.assertTrue(receipt_path.is_file())
        receipt = json.loads(receipt_path.read_text())
        self.assertEqual(
            receipt["contract_sha256"], MODULE.sha256_file(PACKET / "HFT_EXPORT_CONTRACT_R3.json")
        )
        self.assertEqual(receipt["runner_sha256"], MODULE.sha256_file(Path(MODULE.__file__)))
        self.assertEqual(receipt["status"], "PASS_PRODUCER_PREFLIGHT")
        self.assertFalse(receipt["export_performed"])
        self.assertFalse(receipt["public_or_fresh_holdout_claim"])
        self.assertTrue(receipt["predictor_cli_probe"]["reached_configure_torch"])
        self.assertFalse(receipt["predictor_cli_probe"]["device_started"])
        self.assertTrue(receipt["predictor_cli_probe"]["output_paths_unchanged_and_absent"])
        producer = receipt["producer"]
        self.assertEqual(producer["checkpoint_epoch"], 0)
        self.assertEqual(
            producer["checkpoint_sha256"], self.contract["artifacts"]["checkpoint_sha256"]
        )
        self.assertEqual(producer["fit_counters"]["optimizer_attempts"], 880)
        self.assertEqual(producer["fit_counters"]["optimizer_completed_updates"], 877)
        self.assertEqual(producer["fit_counters"]["amp_skipped_updates"], 3)
        self.assertFalse(producer["strict_profile_gate_passed"])
        self.assertFalse(producer["profile_valid_for_throughput"])
        self.assertEqual(receipt["source_and_membership"]["counts"]["valid_fit"], 1416)
        self.assertEqual(receipt["source_and_membership"]["counts"]["valid_evaluation"], 457)
        self.assertEqual(receipt["source_and_membership"]["outer_evaluation_fit_overlap"], 0)

    def test_r2_actual_contract_locates_fixed_epoch_and_trainer_argv(self) -> None:
        fit = json.loads((R2 / "HFT_FIT_CONTRACT_R2.json").read_text())
        export = json.loads((R2 / "HFT_EXPORT_CONTRACT_R2.json").read_text())
        self.assertEqual(fit["expected_checkpoint_epoch"], 0)
        self.assertIsNone(fit.get("checkpoint_epoch"))
        self.assertIn("trainer_argv", fit)
        self.assertNotIn("trainer_argv", export)
        self.assertEqual(export["predictor_argv"], self.contract["prediction"]["argv"])
        self.assertEqual(self.contract["fit_epoch"], 0)

    def test_collision_guard_covers_partial_and_predictor_deleted_sequence_sums(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            partial = MODULE.expected_output_paths(root, self.contract)["partial"]
            partial.parent.mkdir(parents=True)
            partial.write_bytes(b"partial")
            with self.assertRaisesRegex(ValueError, "output collision"):
                MODULE.assert_outputs_absent(root, self.contract)
            partial.unlink()

            sequence_sums = MODULE.expected_output_paths(root, self.contract)["sequence_sums"]
            sequence_sums.write_bytes(b"stale")
            with self.assertRaisesRegex(ValueError, "output collision"):
                MODULE.assert_outputs_absent(root, self.contract)
            sequence_sums.unlink()
            result_receipt = MODULE.expected_output_paths(root, self.contract)["result_receipt"]
            result_receipt.parent.mkdir(parents=True, exist_ok=True)
            result_receipt.write_bytes(b"existing")
            with self.assertRaisesRegex(ValueError, "output collision"):
                MODULE.assert_outputs_absent(root, self.contract)
            result_receipt.unlink()
            self.assertEqual(
                MODULE.assert_outputs_absent(root, self.contract),
                {
                    "prediction": False,
                    "partial": False,
                    "sidecar": False,
                    "sequence_sums": False,
                    "result_receipt": False,
                },
            )

    def test_terminal_gate_requires_matching_digest_and_schema(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "terminal.json"
            path.write_text('{"schema":"wrong"}\n', encoding="utf-8")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            with self.assertRaisesRegex(ValueError, "SHA differs"):
                MODULE._check_gate(path, "0" * 64, {"schema": "expected"})
            with self.assertRaisesRegex(ValueError, "binding mismatch"):
                MODULE._check_gate(path, digest, {"schema": "expected"})

    def test_export_requires_frozen_result_receipt_before_root_gate_reads(self) -> None:
        with (
            patch.object(
                sys,
                "argv",
                [
                    "run_f1_hft_export_r3.py",
                    "--stage",
                    "export",
                    "--terminal-gate-sha256",
                    "0" * 64,
                    "--export-gate-sha256",
                    "0" * 64,
                ],
            ),
            self.assertRaisesRegex(ValueError, "fixed result receipt path"),
        ):
            MODULE.main()


if __name__ == "__main__":
    unittest.main()
