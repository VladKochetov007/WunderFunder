"""Focused checks for the conditional dictionary verifier preparation packet."""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

import prepare_child_dictionary_terminal_verifier_v1 as prepare
import render_child_dictionary_terminal_verifier_v1 as render


class ConditionalPreparationTests(unittest.TestCase):
    """Exercise only small local receipt and preparation metadata."""

    def test_original_dictionary_receipts_join_to_actual_root_acceptance(self) -> None:
        result = prepare._check_original_receipts()

        self.assertEqual(result["status"], "PASS_ORIGINAL_RECEIPTS")
        self.assertEqual(
            result["hashes"],
            {
                "ADMISSION.json": prepare.ADMISSION_SHA256,
                "LAUNCH.json": prepare.LAUNCH_SHA256,
                "LAUNCH_ACCEPTANCE.json": prepare.LAUNCH_ACCEPTANCE_SHA256,
            },
        )
        self.assertFalse(result["launch_body_sha256_field_assumed"])
        self.assertFalse(result["native_terminal_inferred"])

    def test_five_frozen_helpers_are_verified_at_the_outer_boundary(self) -> None:
        result = prepare._check_code_inputs()
        dependencies = prepare._check_dependency_manifest()

        self.assertEqual(set(result["frozen_helpers"]), set(prepare.HELPER_INPUTS))
        self.assertIn("child_dictionary_terminal_check_v1.py", result["files_to_stage"])
        self.assertEqual(len(result["files_to_stage"]), 6)
        self.assertIn("before launching Python", result["outer_boundary"])
        self.assertTrue(result["stage_only_after_native_exit0_and_final_source_review"])
        self.assertEqual(
            dependencies["status"],
            "CONDITIONAL_NATIVE_PROOF_PENDING",
        )

    def test_final_root_preparation_and_source_review_bind_exact_six_code_files(self) -> None:
        raw = render._read_regular(render.ROOT_PREPARATION, render.MAX_METADATA_BYTES)
        root, code, source_review, proof_pins = render._check_root_preparation(raw)

        self.assertEqual(
            root["status"], "PASS_LOCAL_PREPARATION_ONLY_NOT_STAGED_ADMITTED_OR_EXECUTED"
        )
        self.assertEqual(len(code), 6)
        self.assertEqual(source_review["status"], "PASS_SOURCE_ONLY")
        self.assertEqual(len(proof_pins), 24)
        self.assertFalse(
            any(
                Path(str(item["path"])).suffix.lower()
                in {".pt", ".pth", ".npy", ".npz", ".parquet"}
                for item in proof_pins
            )
        )

    def test_template_is_conditional_and_renderer_fails_closed_without_native_proof(self) -> None:
        template = render._read_regular(render.TEMPLATE, 100_000).decode("utf-8")
        self.assertIn("PAYLOAD = {}", template)
        self.assertIn("FINAL_REVIEW_BINDING = {}", template)
        self.assertIn("ROOT_PREPARATION_AND_SOURCE_REVIEW_AND_NATIVE_EXIT0_BOUND", template)
        self.assertFalse(render.BODY.exists())

        original_read = render._read_regular

        def fail_before_native_proof(path: Path, cap: int) -> bytes:
            if path == render.TERMINAL:
                raise FileNotFoundError(path)
            return original_read(path, cap)

        with (
            patch.object(render, "_read_regular", side_effect=fail_before_native_proof),
            self.assertRaises(FileNotFoundError),
        ):
            render.render_body()
        self.assertFalse(render.BODY.exists())

    def test_local_preparation_has_no_payload_reads(self) -> None:
        observed_paths: list[Path] = []
        original_read = prepare._read_regular

        def read_spy(path: Path, cap: int) -> bytes:
            observed_paths.append(path)
            return original_read(path, cap)

        absent_terminal_path = prepare.OUTPUT / "NO_NATIVE_EVIDENCE_FIXTURE.json"
        with (
            patch.object(prepare, "_read_regular", side_effect=read_spy),
            patch.object(prepare, "NATIVE_TERMINAL_PATH", absent_terminal_path),
        ):
            report = prepare.build_local_report()

        suffixes = {path.suffix for path in observed_paths}
        self.assertTrue(suffixes <= {".json", ".py"})
        self.assertFalse(
            any(
                path.name.endswith((".pt", ".npy", ".npz", ".parquet"))
                or path.name in {"history.json", "unit.log"}
                for path in observed_paths
            )
        )
        self.assertEqual(report["status"], "CONDITIONAL_NOT_RENDERED")
        self.assertFalse(report["source_tree_content_hashed"])
        self.assertFalse(report["remote_or_b_action"])
        self.assertFalse(report["body_rendered"])
        self.assertFalse(report["conditional_body_template"]["production_body_emitted"])
        self.assertTrue(report["conditional_renderer"]["requires_actual_typed_and_raw_exit0_proof"])
        self.assertEqual(
            report["ema_v3_fresh_gate_template"],
            {
                "bytes": prepare.EMA_V3_TEMPLATE_BYTES,
                "sha256": prepare.EMA_V3_TEMPLATE_SHA256,
            },
        )
        self.assertEqual(report["native_terminal"]["status"], "WAITING_FOR_NATIVE_EXIT0_EVIDENCE")


if __name__ == "__main__":
    unittest.main()
