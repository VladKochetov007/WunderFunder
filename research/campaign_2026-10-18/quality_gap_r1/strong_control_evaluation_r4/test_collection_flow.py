"""Tiny end-to-end check for preserving and consuming export capture bindings."""

from __future__ import annotations

import importlib
import json
import sys
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import collect_control
import collection_support
import numpy as np
from collection_flow_fixture import build_tiny_fixture
from source_alignment import sha256_file


class CollectionFlowTests(unittest.TestCase):
    """Exercise the real collector with bounded synthetic source and export metadata."""

    def test_collect_rehashes_capture_bindings_and_writes_diagnostic_ready_inputs(self) -> None:
        acceptance_path = Path(__file__).resolve().parent / "fixtures/COLLECT_ACCEPTANCE_V1.json"
        acceptance = json.loads(acceptance_path.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as directory:
            case = build_tiny_fixture(Path(directory))
            extraction_finished = [False]
            post_extraction_hashes: set[Path] = set()
            original_hash_file = collect_control.sha256_file
            validated_export_output: list[
                tuple[np.ndarray, dict[str, object], dict[str, object], dict[str, object]]
            ] = []

            def diagnostic_result(
                _: Path,
            ) -> tuple[dict[str, object], dict[str, object], Path, dict[str, object], Path]:
                return (
                    case.diagnostic,
                    case.bundle_spec,
                    case.bundle_path,
                    case.q8_receipt,
                    case.q8_receipt_path,
                )

            def track_post_extraction_hash(path: Path, max_bytes: int | None = None) -> str:
                if extraction_finished[0]:
                    post_extraction_hashes.add(Path(path).resolve())
                return original_hash_file(path, max_bytes)

            def extracted_rows(
                _parquet: object,
                _evaluation_groups: list[int],
                _identities: dict[int, dict[str, int]],
                _prediction: np.ndarray,
                _row_map: dict[int, int],
            ) -> tuple[dict[str, np.ndarray], list[dict[str, object]]]:
                extraction_finished[0] = True
                return case.extracted, case.group_audit

            def validate_and_capture_export(
                prediction_path: Path,
                sidecar_path: Path,
                contract_path: Path,
                expected_contract_sha256: str,
                receipt_path: Path,
                expected_sidecar_sha256: str,
                expected_receipt_sha256: str,
            ) -> tuple[np.ndarray, dict[str, object], dict[str, object], dict[str, object]]:
                result = collection_support.validate_export(
                    prediction_path,
                    sidecar_path,
                    contract_path,
                    expected_contract_sha256,
                    receipt_path,
                    expected_sidecar_sha256,
                    expected_receipt_sha256,
                )
                validated_export_output.append(result)
                return result

            with ExitStack() as stack:
                stack.enter_context(
                    patch.object(
                        collect_control,
                        "resolve_create_only_output",
                        side_effect=lambda path: path.absolute(),
                    )
                )
                for module, name, value in (
                    (collection_support, "ROW_GROUPS", 4),
                    (collection_support, "ROWS_PER_GROUP", 3),
                    (collect_control, "ROW_GROUPS", 4),
                    (collect_control, "ROWS_PER_GROUP", 3),
                    (collect_control, "SELECTED_ROWS", 4),
                    (collect_control, "EVALUATION_GROUPS", 2),
                    (collect_control, "EVALUATION_COMPONENTS", 2),
                    (collection_support, "SELECTED_ROWS", 4),
                    (collection_support, "BUNDLE_SHA256", case.bundle_spec["sha256"]),
                    (collection_support, "DIAGNOSTIC_INPUTS_SHA256", "c" * 64),
                ):
                    stack.enter_context(patch.object(module, name, value))
                stack.enter_context(
                    patch.object(
                        collection_support,
                        "validate_contract",
                        return_value=({}, case.args.expected_export_contract_sha256),
                    )
                )
                stack.enter_context(
                    patch.object(collection_support, "validate_export_output_paths")
                )
                stack.enter_context(
                    patch.object(
                        collection_support,
                        "validate_receipt",
                        return_value=(case.export_receipt, case.capture_bindings),
                    )
                )
                stack.enter_context(
                    patch.object(
                        collect_control,
                        "validate_export",
                        side_effect=validate_and_capture_export,
                    )
                )
                stack.enter_context(
                    patch.object(
                        collection_support,
                        "validate_h390_sidecar",
                        return_value={1: 0, 2: 1},
                    )
                )
                stack.enter_context(
                    patch.object(
                        collect_control,
                        "validate_split_membership",
                        return_value=case.split_result,
                    )
                )
                stack.enter_context(
                    patch.object(
                        collect_control,
                        "validate_diagnostic_manifest",
                        side_effect=diagnostic_result,
                    )
                )
                stack.enter_context(
                    patch.object(
                        collect_control,
                        "load_bundle_arrays",
                        return_value=case.expected_bundle,
                    )
                )
                stack.enter_context(
                    patch.object(
                        collect_control,
                        "validate_parquet_source",
                        return_value=object(),
                    )
                )
                stack.enter_context(
                    patch.object(
                        collect_control,
                        "collect_evaluation_rows",
                        side_effect=extracted_rows,
                    )
                )
                stack.enter_context(
                    patch.object(
                        collect_control,
                        "sha256_file",
                        side_effect=track_post_extraction_hash,
                    )
                )
                result = collect_control.collect(case.args)

                self.assertEqual(result["status"], "passed")
                self.assertEqual(result["selected_rows"], 4)
                self.assertEqual(len(validated_export_output), 1)
                returned_bindings = validated_export_output[0][3]
                for key, value in case.capture_bindings.items():
                    self.assertEqual(returned_bindings[key], value, key)

                output_path = case.args.output
                self.assertEqual(
                    sorted(path.name for path in output_path.iterdir()),
                    acceptance["expected_output_files"],
                )
                expected_rehashed = {
                    case.args.export_contract.resolve(),
                    case.args.sidecar.resolve(),
                    case.args.export_receipt.resolve(),
                    case.raw_guard_path.resolve(),
                    case.captured_sidecar_path.resolve(),
                    case.preinference_path.resolve(),
                }
                self.assertTrue(expected_rehashed.issubset(post_extraction_hashes))

                alignment_path = output_path / collect_control.ALIGNMENT_FILE
                derived_path = output_path / collect_control.DERIVED_INPUTS_FILE
                prediction_out = output_path / collect_control.ALIGNED_FILE
                alignment = json.loads(alignment_path.read_text(encoding="utf-8"))
                derived = json.loads(derived_path.read_text(encoding="utf-8"))
                raw_evidence = alignment["export"]["captured_raw_evidence"]
                self.assertEqual(sorted(raw_evidence), acceptance["captured_raw_evidence_fields"])
                self.assertEqual(raw_evidence["guard_attempt_path"], str(case.raw_guard_path))
                self.assertEqual(
                    raw_evidence["guard_attempt_sha256"],
                    case.capture_bindings["raw_guard_attempt_sha256"],
                )
                self.assertEqual(
                    raw_evidence["pre_inference_runtime_path"],
                    str(case.preinference_path),
                )
                self.assertEqual(
                    raw_evidence["pre_inference_runtime_sha256"],
                    case.capture_bindings["raw_preinference_sha256"],
                )
                self.assertEqual(
                    raw_evidence["sidecar_sha256"],
                    case.capture_bindings["captured_sidecar_sha256"],
                )
                self.assertEqual(alignment["export"]["attempt_id"], "tiny-export-attempt")
                self.assertEqual(
                    alignment["export"]["captured_ordered_row_groups_sha256"],
                    case.capture_bindings["ordered_row_groups_sha256"],
                )
                model = derived["models"]["strong_control"]
                self.assertEqual(model["kind"], acceptance["strong_control_model"]["kind"])
                self.assertEqual(model["sha256"], sha256_file(prediction_out))
                self.assertEqual(model["paired_bundle_rows"], 4)
                self.assertEqual(
                    model["alignment_receipt"]["fields"],
                    acceptance["strong_control_model"]["alignment_receipt_fields"],
                )
                self.assertEqual(model["alignment_receipt"]["sha256"], sha256_file(alignment_path))
                self.assertTrue(np.array_equal(np.load(prediction_out), case.control))

                # Validate the diagnostic's new model bindings; frozen baseline receipts stay intact.
                project_root = Path(__file__).resolve().parents[4]
                diagnostics_dir = (
                    project_root / "research/campaign_2026-10-18/quality_gap_r1/diagnostics"
                )
                sys.path.insert(0, str(project_root / "src"))
                sys.path.insert(0, str(diagnostics_dir))
                input_bindings = importlib.import_module("input_bindings")
                stack.enter_context(
                    patch.object(input_bindings, "_validate_receipts", return_value={})
                )
                _, _, accepted_predictions, _ = input_bindings._load_inputs(derived_path)
                accepted = accepted_predictions["strong_control"]
                self.assertTrue(np.array_equal(accepted, case.control))
                self.assertEqual(
                    list(accepted.shape), acceptance["strong_control_model"]["prediction_shape"]
                )
                self.assertEqual(
                    str(accepted.dtype), acceptance["strong_control_model"]["prediction_dtype"]
                )


if __name__ == "__main__":
    unittest.main()
