"""Producer-shaped checks for the sealed H390 full-valid export contract."""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from export_contract import (
    EXPECTED_B_BASE,
    EXPECTED_B_HOME,
    EXPECTED_PROJECT_ROOT,
    EXPECTED_RESOURCE_ENVELOPE,
    EXPECTED_UTILITY_IDENTITIES,
    EXPECTED_UTILITY_NAMES,
    EXPORT_PREP_REL,
    OUTPUT_NAME,
    _expected_predictor_command,
    validate_contract,
    validate_export_output_paths,
)
from export_receipt import (
    _check_successful_attempt,
    _expected_worker_command,
    _validate_preinference_paths,
)
from h390_sidecar import validate_h390_sidecar
from pins import (
    CANONICAL_SPLIT_SHA256,
    CHECKPOINT_SHA256,
    CONTROL_RUN,
    OLD_SPLIT_SHA256,
    OVERLAY_MANIFEST_SHA256,
    PREDICTOR_SHA256,
    ROW_GROUPS,
    SOURCE_MANIFEST_SHA256,
)


class ExportContractTests(unittest.TestCase):
    """Exercise the current exporter schema without loading artifacts or arrays."""

    def _contract(self, contract_path: Path) -> dict[str, object]:
        project = EXPECTED_PROJECT_ROOT
        execution_dir = project / EXPORT_PREP_REL / "export_execution_r1"
        campaign_root = EXPECTED_B_BASE / "source-a266b036"
        utility_code = {
            name: {
                "bytes": EXPECTED_UTILITY_IDENTITIES[name][0],
                "sha256": EXPECTED_UTILITY_IDENTITIES[name][1],
            }
            for name in sorted(EXPECTED_UTILITY_NAMES)
        }
        prediction = project / "cache" / f"pred_{OUTPUT_NAME}_valid.npy"
        return {
            "schema": "quality_gap.f0_strong_control_export_contract.r1",
            "status": "READY_FOR_ROOT_GUARDED_GPU_EXPORT",
            "attempt_id": "fixture-attempt-01",
            "owner": "root",
            "contract_path": str(contract_path.resolve()),
            "project_root": str(project),
            "execution_dir": str(execution_dir),
            "resource_envelope": EXPECTED_RESOURCE_ENVELOPE,
            "gpu_uuid": "GPU-00000000-0000-0000-0000-000000000001",
            "pins": {
                "predictor_sha256": PREDICTOR_SHA256,
                "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
                "old_split_sha256": OLD_SPLIT_SHA256,
                "canonical_split_sha256": CANONICAL_SPLIT_SHA256,
                "overlay_manifest_sha256": OVERLAY_MANIFEST_SHA256,
            },
            "source_snapshot": {
                "path": str(project / "frozen_source_snapshot"),
                "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
                "predictor_sha256": PREDICTOR_SHA256,
                "file_count": 42,
            },
            "artifacts": {"control_checkpoint": {"sha256": CHECKPOINT_SHA256}},
            "execution": {
                "python": str(EXPECTED_B_HOME / ".venv/bin/python"),
                "limited": str(campaign_root / "scripts/limited"),
                "guard": str(campaign_root / "scripts/bandit_resources.py"),
                "guard_state": str(EXPECTED_B_BASE / "runtime"),
                "guard_unit_template": "wnn-bandit-{attempt}.service",
                "physical_tmp": str(EXPECTED_B_BASE / "research_tmp"),
                "pythonpath": (
                    f"{project / EXPORT_PREP_REL / 'compatibility_overlay_r2/src'}:"
                    f"{project / 'frozen_source_snapshot/src'}"
                ),
                "predictor": str(project / "frozen_source_snapshot/scripts/predict_run.py"),
                "command": _expected_predictor_command(project),
            },
            "runtime": {
                "python_executable": str(EXPECTED_B_HOME / ".venv/bin/python"),
                "bandit_resources_sha256": "3732ffeef79e635c8867d83e618f8d51b2fa8beffd7badf11f60412a864210f7",
                "limited_sha256": "d9bcfc3b671af3237b607e18087fde43fc7451b6482891e4abadd8580bfb8661",
                "export_utility_code": utility_code,
            },
            "expected_output": {
                "split": "valid",
                "epoch": 5,
                "row_groups": ROW_GROUPS,
                "shape": [ROW_GROUPS, 20_000, 2],
                "dtype": "<f4",
                "bytes": 299_680_128,
                "grading_run": False,
                "training_run": False,
            },
            "outputs_create_only": {
                "prediction": str(prediction),
                "sidecar": str(prediction.with_suffix(".json")),
                "partial": str(prediction.with_name(prediction.name + ".partial")),
                "family_prediction": str(project / "cache" / f"allout_{OUTPUT_NAME}_valid.npy"),
                "stale_seqsum": str(project / "cache" / f"seqsums_{OUTPUT_NAME}.npz"),
            },
        }

    def _write_contract(self, path: Path, value: dict[str, object]) -> str:
        path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def test_actual_exporter_contract_shape_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "contract.json"
            contract = self._contract(path)
            digest = self._write_contract(path, contract)
            validated, actual = validate_contract(path, digest)
            self.assertEqual(validated["execution"], contract["execution"])
            self.assertEqual(actual, digest)

    def test_wrong_source_root_command_or_output_inventory_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "contract.json"
            contract = self._contract(path)
            contract["source_snapshot"] = dict(contract["source_snapshot"])
            contract["source_snapshot"]["path"] = "/wrong/snapshot"
            digest = self._write_contract(path, contract)
            with self.assertRaisesRegex(ValueError, "snapshot"):
                validate_contract(path, digest)

            contract = self._contract(path)
            contract["execution"] = dict(contract["execution"])
            contract["execution"]["command"] = list(contract["execution"]["command"]) + [
                "--training"
            ]
            digest = self._write_contract(path, contract)
            with self.assertRaisesRegex(ValueError, "command"):
                validate_contract(path, digest)

            contract = self._contract(path)
            contract["outputs_create_only"] = dict(contract["outputs_create_only"])
            del contract["outputs_create_only"]["family_prediction"]
            digest = self._write_contract(path, contract)
            with self.assertRaisesRegex(ValueError, "create-only"):
                validate_contract(path, digest)

            contract = self._contract(path)
            contract["runtime"] = dict(contract["runtime"])
            contract["runtime"]["export_utility_code"] = dict(
                contract["runtime"]["export_utility_code"]
            )
            contract["runtime"]["export_utility_code"]["capture.py"] = {
                "bytes": 1,
                "sha256": "0" * 64,
            }
            digest = self._write_contract(path, contract)
            with self.assertRaisesRegex(ValueError, "reviewed R3 source"):
                validate_contract(path, digest)

    def test_prediction_and_sidecar_must_match_contract_output_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            prediction = root / "expected.npy"
            sidecar = root / "expected.json"
            prediction.write_bytes(b"npy fixture")
            sidecar.write_text("{}", encoding="utf-8")
            contract = {
                "outputs_create_only": {
                    "prediction": str(prediction),
                    "sidecar": str(sidecar),
                }
            }
            validate_export_output_paths(contract, prediction, sidecar)

            alternate_prediction = root / "alternate.npy"
            alternate_prediction.write_bytes(b"same-shaped fixture")
            with self.assertRaisesRegex(ValueError, "exact contract output"):
                validate_export_output_paths(contract, alternate_prediction, sidecar)

            prediction_alias = root / "alias.npy"
            prediction_alias.symlink_to(prediction)
            with self.assertRaisesRegex(ValueError, "exact contract output"):
                validate_export_output_paths(contract, prediction_alias, sidecar)

    def test_preinference_capture_uses_a_distinct_archival_path(self) -> None:
        record_dir = Path("/execution/records/attempt-01")
        captured = {"path": str(record_dir / "CAPTURED_WORKER_PRE_INFERENCE.json")}
        original = {"path": str(record_dir / "WORKER_PRE_INFERENCE.json")}
        _validate_preinference_paths(captured, original, record_dir)

        with self.assertRaisesRegex(ValueError, "paths differ"):
            _validate_preinference_paths(original, original, record_dir)

    def test_worker_command_matches_exporter_builder(self) -> None:
        exporter_path = (
            Path(__file__).resolve().parents[1] / "export_preparation/export_execution_r1"
        )
        previous_bytecode_setting = sys.dont_write_bytecode
        sys.dont_write_bytecode = True
        sys.path.insert(0, str(exporter_path))
        try:
            from launch import worker_command  # type: ignore[import-not-found]

            contract = self._contract(Path("/tmp/export-contract.json"))
            contract["execution_dir"] = str(exporter_path)
            contract["runtime"] = dict(contract["runtime"])
            contract["attempt_id"] = "fixture-attempt-01"
            contract["root_admission"] = {"gpu_uuid": "GPU-00000000-0000-0000-0000-000000000001"}
            contract_path = Path("/tmp/export-contract.json")
            contract_sha = "a" * 64
            expected = _expected_worker_command(contract, contract_path, contract_sha)
            actual = worker_command(contract_path, contract_sha, contract)
            self.assertEqual(actual, expected)
        finally:
            sys.path.remove(str(exporter_path))
            sys.dont_write_bytecode = previous_bytecode_setting

    def test_captured_guard_schema_binds_worker_exit_and_utility_sources(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            contract_path = Path(directory) / "contract.json"
            contract_path.write_text("{}", encoding="utf-8")
            contract = self._contract(contract_path)
            contract["gpu_uuid"] = "GPU-00000000-0000-0000-0000-000000000001"
            contract_sha = hashlib.sha256(contract_path.read_bytes()).hexdigest()
            event_deltas = {"oom": 0, "oom_kill": 0, "oom_group_kill": 0}
            runtime = contract["runtime"]
            utilities = runtime["export_utility_code"]
            sources = {"frozen_source_snapshot/scripts/predict_run.py": PREDICTOR_SHA256}
            execution_dir = Path(contract["execution_dir"])
            project = Path(contract["project_root"])
            for name, identity in utilities.items():
                sources[str((execution_dir / name).relative_to(project))] = identity["sha256"]
            sources[str(EXPECTED_B_BASE / "source-a266b036/scripts/bandit_resources.py")] = runtime[
                "bandit_resources_sha256"
            ]
            sources[str(EXPECTED_B_BASE / "source-a266b036/scripts/limited")] = runtime[
                "limited_sha256"
            ]
            source_manifest_sha256 = hashlib.sha256(
                json.dumps(sources, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            raw_attempt = {
                "id": contract["attempt_id"],
                "status": "finished",
                "phase": "terminal",
                "exit_status": 0,
                "failure": None,
                "kind": "gpu",
                "unit": f"wnn-bandit-{contract['attempt_id']}.service",
                "gpu_device": contract["gpu_uuid"],
                "reservation": {"cpus": 1, "ram_bytes": 8_000_000_000},
                "contract": str(contract_path.resolve()),
                "contract_sha256": contract_sha,
                "command": _expected_worker_command(contract, contract_path, contract_sha),
                "profile": EXPECTED_RESOURCE_ENVELOPE["profile"],
                "gpu_memory_safety_reserve_bytes": 1_000_000_000,
                "elapsed_seconds": 12.5,
                "source_sha256": sources,
                "source_manifest_sha256": source_manifest_sha256,
                "final_memory_event_deltas": event_deltas,
                "final_parent_memory_event_deltas": event_deltas,
                "final_parent_hierarchical_memory_event_deltas": event_deltas,
            }
            raw_bytes = json.dumps(raw_attempt, sort_keys=True).encode()
            guard = {
                "status": "finished",
                "exit_status": 0,
                "unit": raw_attempt["unit"],
                "reservation": raw_attempt["reservation"],
                "elapsed_seconds": raw_attempt["elapsed_seconds"],
                "bytes": len(raw_bytes),
                "sha256": hashlib.sha256(raw_bytes).hexdigest(),
                "final_memory_event_deltas": event_deltas,
                "final_parent_memory_event_deltas": event_deltas,
                "final_parent_hierarchical_memory_event_deltas": event_deltas,
            }
            _check_successful_attempt(
                raw_attempt, raw_bytes, guard, contract, contract_path, contract_sha
            )

            changed_sources = dict(raw_attempt)
            changed_sources["source_sha256"] = dict(sources)
            changed_sources["source_sha256"][
                str(EXPECTED_B_BASE / "source-a266b036/scripts/limited")
            ] = "0" * 64
            changed_sources["source_manifest_sha256"] = hashlib.sha256(
                json.dumps(
                    changed_sources["source_sha256"], sort_keys=True, separators=(",", ":")
                ).encode()
            ).hexdigest()
            changed_bytes = json.dumps(changed_sources, sort_keys=True).encode()
            changed_guard = dict(guard)
            changed_guard["bytes"] = len(changed_bytes)
            changed_guard["sha256"] = hashlib.sha256(changed_bytes).hexdigest()
            with self.assertRaisesRegex(ValueError, "guard or limited source"):
                _check_successful_attempt(
                    changed_sources,
                    changed_bytes,
                    changed_guard,
                    contract,
                    contract_path,
                    contract_sha,
                )

            changed_manifest = dict(raw_attempt)
            changed_manifest["source_manifest_sha256"] = "0" * 64
            changed_bytes = json.dumps(changed_manifest, sort_keys=True).encode()
            changed_guard = dict(guard)
            changed_guard["bytes"] = len(changed_bytes)
            changed_guard["sha256"] = hashlib.sha256(changed_bytes).hexdigest()
            with self.assertRaisesRegex(ValueError, "source manifest digest"):
                _check_successful_attempt(
                    changed_manifest,
                    changed_bytes,
                    changed_guard,
                    contract,
                    contract_path,
                    contract_sha,
                )

            raw_attempt["command"] = [*raw_attempt["command"], "--training"]
            raw_bytes = json.dumps(raw_attempt, sort_keys=True).encode()
            guard["bytes"] = len(raw_bytes)
            guard["sha256"] = hashlib.sha256(raw_bytes).hexdigest()
            with self.assertRaisesRegex(ValueError, "successful exit-0"):
                _check_successful_attempt(
                    raw_attempt, raw_bytes, guard, contract, contract_path, contract_sha
                )

    def test_h390_sidecar_shape_and_explicit_order_are_accepted(self) -> None:
        groups = list(reversed(range(ROW_GROUPS)))
        ordered_digest = hashlib.sha256(np.asarray(groups, dtype="<i8").tobytes()).hexdigest()
        sidecar = {
            "run": CONTROL_RUN,
            "split": "valid",
            "epoch": 5,
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "prediction_sha256": "a" * 64,
            "split_manifest_sha256": OLD_SPLIT_SHA256,
            "quote_cache": None,
            "gate_mode": None,
            "inference_precision": "fp32",
            "quantization_preset": None,
            "recurrent_rank": None,
            "sentinel_preprocessing": None,
            "numeric_preprocessing": None,
            "row_groups": groups,
            "evaluation_membership_sha256": None,
            "fit_subset_manifest_sha256": None,
            "campaign_split_manifest_sha256": OLD_SPLIT_SHA256,
            "campaign_fit_scope": "full-valid-feature-only",
            "campaign_evaluation_membership_sha256": None,
        }
        receipt = {
            "ordered_rows": {
                "count": ROW_GROUPS,
                "ordered_int64_le_sha256": ordered_digest,
                "first": groups[0],
                "last": groups[-1],
            },
            "sidecar": {
                "campaign_fit_scope": "full-valid-feature-only",
                "schema_fields": sorted(sidecar),
            },
        }
        row_map = validate_h390_sidecar(sidecar, receipt)
        self.assertEqual(row_map[ROW_GROUPS - 1], 0)
        self.assertEqual(row_map[0], ROW_GROUPS - 1)

        sidecar["campaign_export_scope"] = "evaluation-only"
        with self.assertRaisesRegex(ValueError, "field inventory"):
            validate_h390_sidecar(sidecar, receipt)


if __name__ == "__main__":
    unittest.main()
