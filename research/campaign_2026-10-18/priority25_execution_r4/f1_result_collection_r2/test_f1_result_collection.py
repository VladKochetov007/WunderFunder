"""Focused standard-library tests for F1 receipt and snapshot boundaries."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import struct
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from collect_f1_results import collect
from result_checks import (
    BASE_SHA,
    FOLD_SHA,
    RECOVERY_SHA,
    V2_SUPPORT_SHA,
    V4_SUPPORT_SHA,
    _sealed_phase,
    validate_attempt_record,
    validate_grade,
    validate_helper_profiles,
    validate_plans,
    validate_predictor_source_binding,
    validate_sidecar,
    validate_terminal_seal,
)
from result_io import CollectionError, copy_stable, hash_stable, inspect_npy, membership_sha256

PACKET = Path(__file__).resolve().parents[1]
PROJECT = PACKET.parents[2]


def _sha(letter: str) -> str:
    return letter * 64


def _npy_bytes(shape: tuple[int, ...], values: tuple[float, ...]) -> bytes:
    header = repr({"descr": "<f4", "fortran_order": False, "shape": shape}).encode("latin1")
    padded_size = (16 - ((10 + len(header) + 1) % 16)) % 16
    header = header + b" " * padded_size + b"\n"
    return (
        b"\x93NUMPY\x01\x00"
        + struct.pack("<H", len(header))
        + header
        + struct.pack(f"<{len(values)}f", *values)
    )


def _score(
    run: str, checkpoint: str, provenance: str, prediction: str, sidecar: str, wp: float
) -> dict[str, object]:
    return {
        "run": run,
        "epoch": 5,
        "checkpoint_sha256": checkpoint,
        "provenance_sha256": provenance,
        "prediction_sha256": prediction,
        "sidecar_sha256": sidecar,
        "score": {
            "metric_id": "global_wp_clip2_v1",
            "wp": wp,
            "t0": wp - 0.01,
            "t1": wp + 0.01,
            "selected_rows": 123456,
        },
    }


class F1ResultCollectionTests(unittest.TestCase):
    def _predictor_fixture(self, root: Path) -> tuple[Path, Path, list[str], dict[str, str]]:
        project = root / "project"
        predictor = project / "source-a266b036" / "scripts" / "predict_run.py"
        predictor.parent.mkdir(parents=True)
        predictor.write_bytes((PROJECT / "scripts" / "predict_run.py").read_bytes())
        command = ["/usr/bin/env", str(predictor), "rr18_r01_wide_f1_s1001_e6"]
        source_map = {
            "scripts/predict_run.py": hashlib.sha256(predictor.read_bytes()).hexdigest(),
            "src/wnn/model.py": _sha("a"),
            "/home/grinchenkoeg/computer-vision-homework-20261018/splits/fold0.json": _sha("b"),
        }
        return project, predictor, command, source_map

    def test_predictor_source_map_uses_exact_relative_key_root_and_pin(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            project, predictor, command, source_map = self._predictor_fixture(Path(temporary))
            path, expected_sha = validate_predictor_source_binding(project, command, source_map)
            self.assertEqual(path, predictor)
            self.assertEqual(
                expected_sha, "750b9e77aa55673f61a4dd84d809d91dd315bf86eb36c36b065bc67a7d0f4b65"
            )

    def test_predictor_source_binding_rejects_wrong_command_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            project, _, command, source_map = self._predictor_fixture(Path(temporary))
            command[1] = str(project / "source-other" / "scripts" / "predict_run.py")
            with self.assertRaisesRegex(CollectionError, "pinned source root"):
                validate_predictor_source_binding(project, command, source_map)

    def test_predictor_source_binding_rejects_wrong_relative_key(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            project, _, command, source_map = self._predictor_fixture(Path(temporary))
            source_map["source-a266b036/scripts/predict_run.py"] = source_map.pop(
                "scripts/predict_run.py"
            )
            with self.assertRaisesRegex(CollectionError, "canonical predictor key"):
                validate_predictor_source_binding(project, command, source_map)

    def test_predictor_source_binding_rejects_missing_key(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            project, _, command, source_map = self._predictor_fixture(Path(temporary))
            del source_map["scripts/predict_run.py"]
            with self.assertRaisesRegex(CollectionError, "canonical predictor key"):
                validate_predictor_source_binding(project, command, source_map)

    def test_predictor_source_binding_rejects_wrong_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            project, _, command, source_map = self._predictor_fixture(Path(temporary))
            source_map["scripts/predict_run.py"] = _sha("f")
            with self.assertRaisesRegex(CollectionError, "frozen source"):
                validate_predictor_source_binding(project, command, source_map)

    def test_predictor_source_binding_rejects_changed_source_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            project, predictor, command, source_map = self._predictor_fixture(Path(temporary))
            predictor.write_bytes(b"changed after export")
            with self.assertRaisesRegex(CollectionError, "file hash"):
                validate_predictor_source_binding(project, command, source_map)

    def test_generic_attempt_validator_keeps_absolute_external_grader_keys(self) -> None:
        source_map = {"/project/grader-6f9fd05e/code/grade_endpoint.py": _sha("a")}
        record = {
            "id": "grade-attempt",
            "status": "finished",
            "exit_status": 0,
            "command": ["/usr/bin/python", "/project/grader-6f9fd05e/code/grade_endpoint.py"],
            "source_sha256": source_map,
            "source_manifest_sha256": hashlib.sha256(
                json.dumps(source_map, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
        }
        with tempfile.TemporaryDirectory() as temporary:
            record_path = Path(temporary) / "attempt.json"
            record_path.write_text(json.dumps(record), encoding="utf-8")
            validated, _ = validate_attempt_record(record_path, "grade-attempt", record["command"])
            self.assertEqual(validated["source_sha256"], source_map)

    def test_fold_ordered_sidecar_is_independent_of_training_order_hashes(self) -> None:
        groups = [9, 2, 14]
        sidecar = {
            "run": "wide-f1",
            "split": "valid",
            "epoch": 5,
            "split_manifest_sha256": FOLD_SHA,
            "campaign_split_manifest_sha256": FOLD_SHA,
            "campaign_export_scope": "evaluation",
            "feature_only": True,
            "inference_precision": "fp32",
            "row_groups": groups,
            "evaluation_membership_sha256": membership_sha256(groups),
            "checkpoint_sha256": _sha("a"),
            "prediction_sha256": _sha("b"),
            "fit_membership_identity": {"train": {"ordered_int64_le_sha256": _sha("c")}},
            "training_evaluation_rng12345_order_sha256": _sha("d"),
        }
        validate_sidecar(sidecar, "wide-f1", groups, FOLD_SHA, _sha("a"), _sha("b"))
        sidecar["row_groups"] = list(reversed(groups))
        with self.assertRaisesRegex(CollectionError, "row_groups"):
            validate_sidecar(sidecar, "wide-f1", groups, FOLD_SHA, _sha("a"), _sha("b"))

    def test_grade_accepts_observed_row_count_and_score_without_f0_constants(self) -> None:
        groups = [9, 2, 14]
        arms = {
            "wide": {
                "run": "wide-f1",
                "checkpoint_sha256": _sha("a"),
                "provenance_sha256": _sha("b"),
            },
            "deep": {
                "run": "deep-f1",
                "checkpoint_sha256": _sha("c"),
                "provenance_sha256": _sha("d"),
            },
        }
        predictions = {"wide": _sha("e"), "deep": _sha("f")}
        sidecars = {"wide": _sha("1"), "deep": _sha("2")}
        grade = {
            "schema": "priority25.endpoint_grade.v1",
            "fold": 1,
            "n_folds": 4,
            "split_manifest_sha256": FOLD_SHA,
            "evaluation_row_groups": groups,
            "evaluation_components": 2,
            "metric_id": "global_wp_clip2_v1",
            "target_cache_sha256": _sha("3"),
            "endpoint": _score(
                "wide-f1", _sha("a"), _sha("b"), predictions["wide"], sidecars["wide"], 0.671
            ),
            "reference": _score(
                "deep-f1", _sha("c"), _sha("d"), predictions["deep"], sidecars["deep"], 0.669
            ),
            "paired_component_bootstrap": {
                "resamples": 10000,
                "n_boot": 10000,
                "seed": 271828,
                "components": 2,
                "n_components": 2,
                "n_sequences": 3,
                "centered_moments": True,
                "metric_id": "global_wp_clip2_v1",
                "method": "paired whole-component bootstrap using centered moments",
                "mean_diff": 0.002,
                "bootstrap_sd": 0.01,
                "p_le_0": 0.42,
                "candidate_wp": 0.671,
                "reference_wp": 0.669,
                "ci95": [-0.018, 0.022],
            },
        }
        validate_grade(grade, groups, 2, FOLD_SHA, arms, predictions, sidecars, 10000)
        grade["reference"]["score"]["selected_rows"] = 123457
        with self.assertRaisesRegex(CollectionError, "selected-row"):
            validate_grade(grade, groups, 2, FOLD_SHA, arms, predictions, sidecars, 10000)

    def test_streamed_npy_header_payload_and_finiteness(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "tiny.npy"
            path.write_bytes(_npy_bytes((2, 1, 2), (0.25, -0.5, 1.0, 0.0)))
            header, digest, _ = inspect_npy(path, [2, 1, 2])
            self.assertEqual(header["bytes"], path.stat().st_size)
            self.assertEqual(len(digest), 64)
            path.write_bytes(_npy_bytes((2, 1, 2), (0.25, math.nan, 1.0, 0.0)))
            with self.assertRaisesRegex(CollectionError, "nonfinite"):
                inspect_npy(path, [2, 1, 2])

    def test_copy_is_create_only_and_detects_changed_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, archived = root / "source.json", root / "archive" / "source.json"
            source.write_bytes(b"same bytes")
            digest, source_identity = hash_stable(source, 1024)
            copied = copy_stable(source, archived, digest, source_identity)
            self.assertEqual(copied["sha256"], digest)
            with self.assertRaisesRegex(CollectionError, "overwrite"):
                copy_stable(source, archived, digest, source_identity)
            changed = root / "changed.json"
            changed.write_bytes(b"before")
            changed_digest, changed_identity = hash_stable(changed, 1024)
            changed.write_bytes(b"after!")
            with self.assertRaisesRegex(CollectionError, "changed after preflight"):
                copy_stable(
                    changed, root / "archive" / "changed.json", changed_digest, changed_identity
                )

    def test_exact_frozen_plan_recovery_vector_and_schedule(self) -> None:
        base_path = PACKET / "f1_beleriand_fallback" / "F1_BELERIAND_PLAN.json"
        recovery_path = PACKET / "f1_recovery_r1" / "F1_RECOVERY_PLAN.json"
        base_bytes, recovery_bytes = base_path.read_bytes(), recovery_path.read_bytes()
        self.assertEqual(hashlib.sha256(base_bytes).hexdigest(), BASE_SHA)
        self.assertEqual(hashlib.sha256(recovery_bytes).hexdigest(), RECOVERY_SHA)
        base, recovery = json.loads(base_bytes), json.loads(recovery_bytes)
        fold = {
            "fold": 1,
            "n_folds": 4,
            "evaluation": {"valid": [9, 2, 14]},
            "fit": {"train": [101, 108], "valid": [202]},
            "sequences": {
                "valid": [
                    {"row_group": 9, "component": 1},
                    {"row_group": 2, "component": 1},
                    {"row_group": 14, "component": 2},
                ]
            },
        }
        base_steps, recovery_steps, groups, components, expected = validate_plans(
            base, recovery, fold, BASE_SHA, RECOVERY_SHA, FOLD_SHA
        )
        self.assertEqual(groups, [9, 2, 14])
        self.assertEqual(components, 2)
        self.assertEqual(expected["terminal_updates"], 89580)
        self.assertEqual(
            recovery_steps["continuation_wide"]["attempt"],
            "computer-vision-homework-r01-wide-f1s1001-ep25-v2",
        )
        self.assertEqual(
            base_steps["continuation_deep"]["attempt"],
            "computer-vision-homework-r01-deep-f1s1001-ep25-v1",
        )
        altered_recovery = copy.deepcopy(recovery)
        for step in altered_recovery["steps"]:
            if step["step_id"] == "continuation_wide":
                step["attempt"] = "computer-vision-homework-r01-wide-f1s1001-ep25-v1"
        with self.assertRaisesRegex(CollectionError, "recovery wide attempt ID"):
            validate_plans(base, altered_recovery, fold, BASE_SHA, RECOVERY_SHA, FOLD_SHA)

    def test_invalid_fold_fails_before_snapshot_or_receipt_creation(self) -> None:
        base_plan = PACKET / "f1_beleriand_fallback/F1_BELERIAND_PLAN.json"
        recovery_plan = PACKET / "f1_recovery_r1/F1_RECOVERY_PLAN.json"
        with tempfile.TemporaryDirectory(dir=PROJECT) as temporary:
            root = Path(temporary)
            fold = root / "bad-fold.json"
            fold.write_text(json.dumps({"fold": 0, "n_folds": 4}), encoding="utf-8")
            output, receipt = root / "snapshot", root / "result.json"
            paths = {
                "project_root": PROJECT,
                "base_plan": base_plan,
                "recovery_plan": recovery_plan,
                "fold_manifest": fold,
                "terminal_seal": root / "seal.json",
                "snapshot_dir": output,
                "receipt": receipt,
            }
            for arm in ("wide", "deep"):
                for phase in ("stage1", "terminal"):
                    paths[f"{arm}_{phase}_collection"] = root / f"{arm}-{phase}-collection.json"
                    paths[f"{arm}_{phase}_verification"] = root / f"{arm}-{phase}-verification.json"
            with self.assertRaisesRegex(CollectionError, "base, recovery or fold manifest"):
                collect(SimpleNamespace(**paths))
            self.assertFalse(output.exists())
            self.assertFalse(receipt.exists())

    def test_support_profiles_match_actual_v2_and_v4_manifests(self) -> None:
        v2_manifest = json.loads(
            (
                PACKET / "f1_beleriand_fallback/checkpoints_v2/PREPARATION_MANIFEST_V2_R2.json"
            ).read_text()
        )
        v4_manifest = json.loads(
            (PACKET / "f1_recovery_r1/checkpoints_v4/PREPARATION_MANIFEST_V4.json").read_text()
        )
        v2_files = v2_manifest["helper_files_sha256"]
        v4_files = v4_manifest["support_files_sha256"]
        train_sha, _ = hash_stable(PROJECT / "src/wnn/training_state.py", 1024 * 1024)

        def profile(files: dict[str, str]) -> dict[str, object]:
            return {
                "collector_sha256": files["collect_f1_checkpoint.py"],
                "collector_dependencies": {
                    "f1_checkpoint_common.py": files["f1_checkpoint_common.py"]
                },
                "verification_helper_sha256": files["verify_f1_checkpoint.py"],
                "verification_dependencies": {
                    name: files[name]
                    for name in (
                        "f1_checkpoint_common.py",
                        "f1_checkpoint_evidence.py",
                        "f1_checkpoint_lineage.py",
                        "f1_checkpoint_tensors.py",
                        "collect_f1_checkpoint.py",
                    )
                }
                | {"wnn/training_state.py": train_sha},
            }

        def phase(
            profile_value: dict[str, object], attempt: str
        ) -> tuple[dict[str, object], dict[str, object]]:
            collection = {
                "collector_sha256": profile_value["collector_sha256"],
                "collector_dependencies": profile_value["collector_dependencies"],
            }
            verification = {
                "verification_helper_sha256": profile_value["verification_helper_sha256"],
                "verification_dependencies": profile_value["verification_dependencies"],
                "attempt": {"id": attempt},
            }
            return collection, verification

        stage_wide = phase(profile(v2_files), "wide-stage1")
        terminal_wide = phase(profile(v4_files), "wide-terminal")
        stage_deep = phase(profile(v2_files), "deep-stage1")
        terminal_deep = phase(profile(v2_files), "deep-terminal")
        meta = {
            "wide": {
                "stage1_collection": stage_wide[0],
                "stage1_verification": stage_wide[1],
                "collection": terminal_wide[0],
                "verification": terminal_wide[1],
            },
            "deep": {
                "stage1_collection": stage_deep[0],
                "stage1_verification": stage_deep[1],
                "collection": terminal_deep[0],
                "verification": terminal_deep[1],
            },
        }
        paths, hashes, manifests = validate_helper_profiles(
            PROJECT, meta, "a266b036b02e8e012c6eb31072291dad618a169d"
        )
        self.assertEqual(manifests, {"v2": V2_SUPPORT_SHA, "wide_terminal_v4": V4_SUPPORT_SHA})
        self.assertIn("support/v2/verify_f1_checkpoint.py", paths)
        self.assertIn("support/v4/verify_f1_checkpoint.py", paths)
        self.assertEqual(len(hashes), len(paths))
        meta["deep"]["verification"]["verification_dependencies"]["f1_checkpoint_common.py"] = _sha(
            "f"
        )
        with self.assertRaisesRegex(CollectionError, "helper profile"):
            validate_helper_profiles(PROJECT, meta, "a266b036b02e8e012c6eb31072291dad618a169d")

    def test_arrival_seal_binds_dynamic_deep_receipts_and_support(self) -> None:
        def checkpoint_phase(tag: str) -> tuple[dict[str, object], dict[str, object], str, str]:
            collection = {
                "schema": f"collection-{tag}",
                "collector_sha256": _sha("a"),
                "collector_dependencies": {"base.py": _sha("b")},
            }
            verification = {
                "schema": f"verification-{tag}",
                "verification_helper_sha256": _sha("c"),
                "verification_dependencies": {"base.py": _sha("d")},
                "attempt": {"id": f"attempt-{tag}"},
            }
            return collection, verification, _sha("e"), _sha("f")

        support = {
            "collect_f1_results.py": _sha("1"),
            "result_checks.py": _sha("2"),
            "result_io.py": _sha("3"),
        }
        manifests = {"v2": V2_SUPPORT_SHA, "wide_terminal_v4": V4_SUPPORT_SHA}
        terminal_meta: dict[str, dict[str, object]] = {}
        arms = {}
        seal_arms = {}
        reviews = {}
        for arm in ("wide", "deep"):
            stage_c, stage_v, stage_c_sha, stage_v_sha = checkpoint_phase(f"{arm}-stage1")
            term_c, term_v, term_c_sha, term_v_sha = checkpoint_phase(f"{arm}-terminal")
            hashes = {"stage1_collection": stage_c_sha, "stage1_verification": stage_v_sha}
            meta = {
                "stage1_collection": stage_c,
                "stage1_verification": stage_v,
                "collection": term_c,
                "verification": term_v,
                "hashes": hashes,
            }
            terminal_meta[arm] = meta
            arms[arm] = (term_c, term_v, term_c_sha, term_v_sha, {})
            phase_bindings = {
                "stage1": _sealed_phase(stage_c, stage_v, stage_c_sha, stage_v_sha),
                "terminal": _sealed_phase(term_c, term_v, term_c_sha, term_v_sha),
            }
            seal_arms[arm] = {"receipts": phase_bindings}
            review = {
                "schema": "priority25.f1_terminal_result_review.v1",
                "status": "passed",
                "arm": arm,
                "base_plan_sha256": BASE_SHA,
                "recovery_plan_sha256": RECOVERY_SHA,
                "fold1_manifest_sha256": FOLD_SHA,
                "receipts": phase_bindings,
                "collector_support_sha256": support,
                "checkpoint_support_manifests_sha256": manifests,
            }
            reviews[arm] = (review, _sha("9"))
        seal = {
            "schema": "priority25.f1_terminal_evidence_seal.v2",
            "status": "sealed",
            "review_status": "passed",
            "base_plan_sha256": BASE_SHA,
            "recovery_plan_sha256": RECOVERY_SHA,
            "fold1_manifest_sha256": FOLD_SHA,
            "collector_support_sha256": support,
            "checkpoint_support_manifests_sha256": manifests,
            "arms": seal_arms,
            "review_receipts": {arm: {"sha256": _sha("9"), "status": "passed"} for arm in reviews},
        }
        validate_terminal_seal(
            seal,
            _sha("8"),
            BASE_SHA,
            RECOVERY_SHA,
            FOLD_SHA,
            arms,
            terminal_meta,
            reviews,
            support,
            manifests,
        )
        seal["arms"]["deep"]["receipts"]["terminal"]["verification_dependencies"] = {
            "other.py": _sha("0")
        }
        with self.assertRaisesRegex(CollectionError, "receipt/helper"):
            validate_terminal_seal(
                seal,
                _sha("8"),
                BASE_SHA,
                RECOVERY_SHA,
                FOLD_SHA,
                arms,
                terminal_meta,
                reviews,
                support,
                manifests,
            )


if __name__ == "__main__":
    unittest.main()
