"""CPU-only fixtures for additive full-primary reporting admission and labels."""

from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

from multistage_github_common import (
    canonical_json,
    make_manifest,
    make_snapshot,
    sha256_bytes,
    verify_snapshot,
)
from multistage_github_primary import PRIMARY, SOURCE, _primary_run, validate_primary_snapshot


def _write(path: Path, value: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = canonical_json(value) + b"\n"
    path.write_bytes(raw)
    return sha256_bytes(raw)


def _fixture(root: Path) -> dict[str, object]:
    prepared, runtime = root / "prepared", root / "runtime"
    fit_id = "primary-refit-01-s1t000001-gru"
    attempt = f"stage4-primary-refit-{fit_id}"
    output = runtime / "runs" / fit_id
    profile = runtime / "profiles" / f"{fit_id}-profile"
    config = {"model": {"architecture": "gru", "hint_dim": 0}, "kd_weight": 0, "hint_weight": 0}
    config_sha = sha256_bytes(canonical_json(config))
    data = {
        "split_sha256": PRIMARY,
        "primary_split_sha256": None,
        "evaluation_role": "valid",
        "evaluation_source": "valid",
        "fit_members": [["train", i] for i in range(10325)],
        "evaluation_members": [["valid", i] for i in range(1873)],
        "teacher": None,
        "initializer": None,
        "normalization": {"source_fingerprints": {"datasets/train.parquet": {"sha256": "a" * 64}}},
    }
    identity = {
        "config_sha256": config_sha,
        "source_sha256": SOURCE,
        "source_files": {},
        "data": data,
        "data_sha256": sha256_bytes(canonical_json(data)),
        "runtime": {"world_size": 1},
        "world_size": 1,
        "model_parameters": {"total": 1},
    }
    metadata = {
        "config": config,
        "source": {"sha256": SOURCE, "files": {}},
        "provenance": {
            key: identity[key]
            for key in ("data", "data_sha256", "runtime", "world_size", "model_parameters")
        },
    }
    request = {
        "schema": "wnn.multistage-primary-refit-request.v1",
        "fit_id": fit_id,
        "training_source": {"sha256": SOURCE},
        "config": {"sha256": "c" * 64, "normalized_identity_sha256": config_sha},
        "primary_split": {
            "sha256": PRIMARY,
            "fit_train_sequences": 10325,
            "fit_valid_sequences": 0,
            "evaluation_sequences": 1873,
        },
        "profile": {"output": str(profile)},
        "fit": {
            "output": str(output),
            "resume": False,
            "initializer": None,
            "teacher_or_cache": None,
            "transferred_auxiliary_weights": False,
            "argv_after_approval_and_reservation": ["python", "train_sequence.py"],
        },
    }
    request_path = prepared / "requests" / f"{fit_id}.json"
    request_sha = _write(request_path, request)
    selected = {
        "fit_id": fit_id,
        "request_path": str(request_path.resolve()),
        "request_sha256": request_sha,
    }
    selection_sha = "d" * 64
    approval = {
        "schema": "wnn.primary-refit-root-native-approval.v1",
        "status": "approved",
        "approved": True,
        "approver_id": "root",
        "native_fit_authorized": True,
        "scientific_candidate_promotion": False,
        "primary_valid_used_for_selection": False,
        "source_sha256": SOURCE,
        "primary_split_sha256": PRIMARY,
        "fit_id": fit_id,
        "request_sha256": request_sha,
        "config_sha256": "c" * 64,
        "native_profile_sha256": "e" * 64,
        "engineering_review_sha256": "f" * 64,
    }
    approval_path = runtime / "receipts" / "root-approval.json"
    approval_sha = _write(approval_path, approval)
    _write(
        profile / "approved_profile_receipt.json",
        {
            "status": "profile_complete",
            "approved": True,
            "identity": identity,
            "root_native_approval": {"path": str(approval_path), "sha256": approval_sha},
        },
    )
    fit = {
        "stage": 4,
        "fit_attempt_id": attempt,
        "source_sha256": SOURCE,
        "primary_split_sha256": PRIMARY,
        "selection_receipt_sha256": selection_sha,
        "config_sha256": "c" * 64,
        "command": request["fit"]["argv_after_approval_and_reservation"],
        "output_dir": str(output.resolve()),
        "root_native_approval_sha256": approval_sha,
        "profile_sha256": "e" * 64,
        "engineering_review_sha256": "f" * 64,
    }
    _write(runtime / "requests" / f"{attempt}.json", fit)
    _write(
        runtime / "receipts" / f"{attempt}.owner.json",
        {
            "fit_id": fit_id,
            "stage": 4,
            "budget_attempt_id": attempt,
            "approval_sha256": approval_sha,
            "profile_sha256": "e" * 64,
        },
    )
    _write(output / "run_metadata.json", metadata)
    event = {
        "schema": "wnn.monitor.v1",
        "kind": "evaluation",
        "step": 1,
        "epoch": 1,
        "wall_time": "2026-10-31T04:00:00Z",
        "metrics": {"valid/wp": 0.3, "valid/score": 0.3, "train_eval/wp": 0.4},
        "context": {"valid": {"scope": "primary_valid", "role": "valid", "member_count": 1873}},
    }
    for name in ("monitor_events.jsonl", "monitor_evaluations.jsonl"):
        _write(output / name, event)
    return _primary_run(prepared, runtime, selection_sha, selected)


class PrimaryReportingTests(unittest.TestCase):
    def test_receipt_bound_native_valid_roundtrip_preserves_labels_and_context(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run = _fixture(Path(temporary))
        payload, compressed, _ = make_snapshot([run])
        restored, _ = verify_snapshot(
            compressed, make_manifest(payload, compressed, "2026-10-31T04:00:00Z")
        )
        result = restored["runs"][0]
        self.assertEqual(result["stage"], 4)
        self.assertEqual(result["phase"], "primary_refit")
        self.assertIsNone(result["metadata"]["provenance"]["data"]["primary_split_sha256"])
        event = result["evaluation_journal"]["events"][0]
        self.assertEqual(
            event["metrics"], {"valid/wp": 0.3, "valid/score": 0.3, "train_eval/wp": 0.4}
        )
        self.assertEqual(event["context"]["valid"]["role"], "valid")

    def test_wrong_split_members_and_ancestry_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            original = _fixture(Path(temporary))
        for key, value in (
            ("split_sha256", "b" * 64),
            ("evaluation_role", "control"),
            ("teacher", {"checkpoint": "auxiliary"}),
            ("initializer", {"checkpoint": "auxiliary"}),
            ("fit_members", [["train", i] for i in range(8260)]),
        ):
            run = copy.deepcopy(original)
            run["metadata"]["provenance"]["data"][key] = value
            run["canonical_metadata_sha256"] = sha256_bytes(canonical_json(run["metadata"]))
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_primary_snapshot(run)

    def test_control_metrics_and_unapproved_receipts_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run = _fixture(Path(temporary))
        wrong = copy.deepcopy(run)
        wrong["evaluation_journal"]["events"][0]["metrics"]["control/wp"] = 0.9
        with self.assertRaises(ValueError):
            validate_primary_snapshot(wrong)
        run["primary_refit_binding"]["root_approval"]["approved"] = False
        with self.assertRaises(ValueError):
            validate_primary_snapshot(run)


if __name__ == "__main__":
    unittest.main()
