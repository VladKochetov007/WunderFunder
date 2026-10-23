"""Synthetic strict-metadata tests for ROOT score-result acceptance."""

from __future__ import annotations

import base64
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path, PurePosixPath
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import score_result_acceptance as acceptance  # noqa: E402

PREPARATION = HERE.parent
CAMPAIGN = PREPARATION.parent
RECORD = CAMPAIGN / "records/first_cell_evaluation_v1/export_v5"
STATUS = RECORD / "status_20261021T1107Z"
BINDING_RAW = (PREPARATION / "first_cell_evaluation_export_v4/EXPORT_BINDING_V4.json").read_bytes()
EXPORT_ACCEPTANCE_RAW = (RECORD / "EXPORT_ACCEPTANCE.json").read_bytes()
EXPORT_STOP_RAW = (RECORD / "exact_stop/EXACT_STOP.json").read_bytes()
EXPORT_MANIFEST_RAW = (STATUS / "EXPORT_MANIFEST_V1.json").read_bytes()


def _encode(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n"
    ).encode()


def _pin(raw: bytes) -> dict[str, object]:
    return {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def _base64_record(raw: bytes) -> dict[str, object]:
    return {**_pin(raw), "base64": base64.b64encode(raw).decode("ascii")}


def _fixture_inputs() -> dict[str, bytes]:
    binding = json.loads(BINDING_RAW)
    manifest = json.loads(EXPORT_MANIFEST_RAW)
    membership = manifest["membership_sha256"]
    output_dir = binding["output_dir"]
    fixed_models = acceptance._expected_fixed_models(binding)
    exported_file_pins = acceptance._expected_file_pins(binding, manifest)
    source_ids = binding["registered_data"]["source_identities"]
    observations: dict[str, dict[str, object]] = {}
    for relative, identity in source_ids.items():
        parent_path = str(PurePosixPath(identity["path"]).parent)
        parent_resolved = str(PurePosixPath(identity["resolved"]).parent)
        parent_group = "cache" if relative.startswith("cache/") else "datasets"
        parent = {
            "path": parent_path,
            "resolved": parent_resolved,
            "device": 64512,
            "inode": 880_000_000 + (1 if parent_group == "cache" else 2),
        }
        observations[relative] = {
            key: identity[key]
            for key in ("path", "resolved", "bytes", "device", "inode", "islink", "mtime_ns")
        } | {"parent": parent}
    registered_data = {
        "admission": binding["registered_data"]["admission"],
        "portable_manifest": binding["registered_data"]["portable_manifest"],
        "registered_source_identities": source_ids,
        "observed_before_split": observations,
        "observed_after_split": observations,
        "observed_before_scoring": observations,
        "observed_after_scoring": observations,
    }
    pooled = {"control": 0.1, "ema": 0.2, "sparse": 0.15, "nll": 0.05, "allthree": 0.25}
    score_report = {
        "schema": acceptance.RESULT_SCHEMA,
        "status": acceptance.RESULT_STATUS,
        "created_utc": "2026-10-21T12:00:00+00:00",
        "scope": acceptance.SCORE_SCOPE,
        "source_commit": acceptance.SOURCE_COMMIT,
        "source_manifest_sha256": acceptance.SOURCE_MANIFEST_SHA256,
        "source_set_sha256": acceptance.SOURCE_SET_SHA256,
        "split_manifest_sha256": acceptance.SPLIT_SHA256,
        "ordered_membership_sha256": membership,
        "fixed_models": fixed_models,
        "export_manifest": {
            "path": str(PurePosixPath(output_dir) / "EXPORT_MANIFEST_V1.json"),
            **_pin(EXPORT_MANIFEST_RAW),
        },
        "exported_file_pins": exported_file_pins,
        "registered_data": registered_data,
        "scoring": {
            "metric_id": acceptance.METRIC_ID,
            "ordered_group_count": acceptance.N_GROUPS,
            "distinct_component_count": 120,
            "official_mask": "is_scored AND (t >= 99)",
            "arms": {
                name: {
                    "t0": score,
                    "t1": score,
                    "wp": score,
                    "selected_rows": 1_000,
                    "metric_id": acceptance.METRIC_ID,
                }
                for name, score in pooled.items()
            },
            "paired_contrasts_vs_control": {
                name: {
                    "candidate_minus_control": {
                        "candidate_wp": pooled[name],
                        "reference_wp": pooled["control"],
                        "mean_diff": pooled[name] - pooled["control"],
                        "bootstrap_sd": 0.04,
                        "ci95": [-0.08, 0.12],
                        "p_le_0": 0.5,
                        "n_sequences": acceptance.N_GROUPS,
                        "n_components": 120,
                        "n_boot": acceptance.BOOTSTRAP_REPLICATES,
                        "seed": acceptance.BOOTSTRAP_SEED,
                        "metric_id": acceptance.METRIC_ID,
                        "centered_moments": True,
                    }
                }
                for name in acceptance.CONTRAST_ARMS
            },
            "uncertainty_scope": "paired feature-only successor-component bootstrap conditional on these five fixed epoch-5 models; not fit-seed or public-transfer uncertainty",
            "bootstrap": {
                "replicates": acceptance.BOOTSTRAP_REPLICATES,
                "seed": acceptance.BOOTSTRAP_SEED,
                "centered_moments": True,
            },
        },
        "limitations": acceptance.LIMITATIONS,
    }
    result_raw = _encode(score_report)
    score_admission = {
        "schema": "fg19.first-cell-score-admission.v1",
        "status": "FRESH_GATES_PASSED_SUBMISSION_PENDING",
        "unit": acceptance.SCORE_UNIT,
        "export_unit": acceptance.EXPORT_UNIT,
        "export_invocation_id": acceptance.EXPORT_INVOCATION_ID,
        "export_acceptance": {
            **_pin(EXPORT_ACCEPTANCE_RAW),
            "staged_path": "accepted_export/EXPORT_ACCEPTANCE.json",
        },
        "export_exact_stop": {
            **_pin(EXPORT_STOP_RAW),
            "staged_path": "accepted_export/EXPORT_EXACT_STOP.json",
        },
        "staged_files": {
            name: {"bytes": size, "sha256": digest}
            for name, (size, digest) in acceptance.EXPECTED_SCORE_CODE_PINS.items()
        },
        "result_path": acceptance.RESULT_PATH,
        "result_create_only_and_absent_before_mutation": True,
    }
    admission_raw = _encode(score_admission)
    invocation = "0123456789abcdef0123456789abcdef"
    score_launch = {
        "schema": "fg19.first-cell-score-launch.v1",
        "unit": acceptance.SCORE_UNIT,
        "invocation_id": invocation,
        "export_unit": acceptance.EXPORT_UNIT,
        "export_invocation_id": acceptance.EXPORT_INVOCATION_ID,
        "admission_sha256": hashlib.sha256(admission_raw).hexdigest(),
        "submission": {"outcome": "submitted", "returncode": 0},
        "systemctl_after_submission": {
            "outcome": "captured",
            "properties": {"Id": acceptance.SCORE_UNIT, "InvocationID": invocation},
        },
    }
    launch_raw = _encode(score_launch)
    native_unit = {
        "Id": acceptance.SCORE_UNIT,
        "InvocationID": invocation,
        "ActiveState": "active",
        "SubState": "exited",
        "MainPID": "0",
        "ExecMainCode": "1",
        "ExecMainStatus": "0",
        "Result": "success",
    }
    empty_sha = hashlib.sha256(b"").hexdigest()
    capture = {
        "schema": "fg19.first-cell-score-bounded-status.v1",
        "captured_utc": "2026-10-21T12:01:00+00:00",
        "unit_name": acceptance.SCORE_UNIT,
        "invocation_id": invocation,
        "terminal_success": True,
        "unit": native_unit,
        "parent": {},
        "parent_events": {},
        "own_cgroup_files": {},
        "parent_populated_files": {},
        "selected_gpu": {"selected_gpu_uuid": "GPU-test", "processes": []},
        "files": {
            "ADMISSION.json": _base64_record(admission_raw),
            "LAUNCH.json": _base64_record(launch_raw),
            "unit.log": {
                "bytes": 0,
                "captured_bytes": 0,
                "tail_offset_bytes": 0,
                "tail_sha256": empty_sha,
                "tail_base64": "",
            },
            "RESULT.json": {"read": True, **_base64_record(result_raw)},
        },
        "result_capture": {
            "read": True,
            "bytes": len(result_raw),
            "sha256": hashlib.sha256(result_raw).hexdigest(),
        },
        "code_pins": {
            name: {"bytes": size, "sha256": digest}
            for name, (size, digest) in acceptance.EXPECTED_SCORE_CODE_PINS.items()
        },
        "scope": "bounded strict-success result capture fixture",
    }
    return {
        "result_raw": result_raw,
        "capture_raw": _encode(capture),
        "score_admission_raw": admission_raw,
        "score_launch_raw": launch_raw,
        "export_acceptance_raw": EXPORT_ACCEPTANCE_RAW,
        "export_exact_stop_raw": EXPORT_STOP_RAW,
        "export_binding_raw": BINDING_RAW,
        "export_manifest_raw": EXPORT_MANIFEST_RAW,
    }


def _build(inputs: dict[str, bytes]) -> dict[str, Any]:
    return acceptance.build_root_acceptance(**inputs)


class ScoreResultAcceptanceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.inputs = _fixture_inputs()

    def _replace_report(self, change: Any) -> None:
        report = json.loads(self.inputs["result_raw"])
        change(report)
        result_raw = _encode(report)
        capture = json.loads(self.inputs["capture_raw"])
        capture["files"]["RESULT.json"] = {"read": True, **_base64_record(result_raw)}
        capture["result_capture"] = {
            "read": True,
            "bytes": len(result_raw),
            "sha256": hashlib.sha256(result_raw).hexdigest(),
        }
        self.inputs["result_raw"] = result_raw
        self.inputs["capture_raw"] = _encode(capture)

    def test_valid_exact_metadata_joins_create_compact_acceptance(self) -> None:
        receipt = _build(self.inputs)
        self.assertEqual(receipt["schema"], acceptance.ACCEPTANCE_SCHEMA)
        self.assertEqual(receipt["status"], acceptance.ACCEPTANCE_STATUS)
        self.assertEqual(receipt["result"]["bytes"], len(self.inputs["result_raw"]))
        self.assertEqual(
            receipt["result"]["sha256"], hashlib.sha256(self.inputs["result_raw"]).hexdigest()
        )
        self.assertEqual(receipt["scalar_tables"]["ordered_group_count"], acceptance.N_GROUPS)
        self.assertEqual(
            set(receipt["scalar_tables"]["paired_contrasts_vs_control"]),
            set(acceptance.CONTRAST_ARMS),
        )

    def test_writer_is_create_only_and_preserves_receipt_bytes(self) -> None:
        receipt = _build(self.inputs)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ACCEPTANCE.json"
            written = acceptance.write_root_acceptance_exclusive(path, receipt)
            self.assertEqual(path.read_bytes(), written)
            with self.assertRaises(FileExistsError):
                acceptance.write_root_acceptance_exclusive(path, receipt)
            self.assertEqual(path.read_bytes(), written)

    def test_native_invocation_must_join_receipts(self) -> None:
        capture = json.loads(self.inputs["capture_raw"])
        capture["unit"]["InvocationID"] = "f" * 32
        self.inputs["capture_raw"] = _encode(capture)
        with self.assertRaisesRegex(ValueError, "terminal tuple"):
            _build(self.inputs)

    def test_embedded_result_bytes_must_join_exact_result(self) -> None:
        capture = json.loads(self.inputs["capture_raw"])
        capture["files"]["RESULT.json"]["base64"] = base64.b64encode(b"{}").decode()
        self.inputs["capture_raw"] = _encode(capture)
        with self.assertRaisesRegex(
            ValueError, "byte count or SHA-256 differs|exact native capture"
        ):
            _build(self.inputs)

    def test_bool_is_not_accepted_as_a_numeric_score(self) -> None:
        self._replace_report(
            lambda report: report["scoring"]["arms"]["ema"].__setitem__("t0", True)
        )
        with self.assertRaisesRegex(ValueError, "finite JSON number"):
            _build(self.inputs)

    def test_missing_contrast_is_rejected(self) -> None:
        self._replace_report(
            lambda report: report["scoring"]["paired_contrasts_vs_control"].pop("nll")
        )
        with self.assertRaisesRegex(ValueError, "key set differs"):
            _build(self.inputs)

    def test_duplicate_json_keys_and_nonfinite_numbers_are_rejected(self) -> None:
        inputs = dict(self.inputs)
        inputs["result_raw"] = b'{"schema":"a","schema":"b"}'
        with self.assertRaisesRegex(ValueError, "duplicate JSON object key"):
            _build(inputs)
        inputs["result_raw"] = b'{"score":NaN}'
        with self.assertRaisesRegex(ValueError, "nonstandard JSON number"):
            _build(inputs)


if __name__ == "__main__":
    unittest.main()
