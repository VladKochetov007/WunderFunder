from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import f3_result_checks as checks
from result_io import CollectionError, membership_sha256


def _write_json(path: Path, value: object) -> tuple[int, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(value, sort_keys=True) + "\n").encode()
    path.write_bytes(raw)
    return len(raw), hashlib.sha256(raw).hexdigest()


def _attempt(
    project: Path, step: dict[str, object], contract: Path
) -> tuple[dict[str, object], str]:
    source = {str(contract): checks.PLAN_SHA, "src/wnn/training_state.py": "a" * 64}
    source_manifest = hashlib.sha256(
        json.dumps(source, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    record = {
        "id": step["attempt"],
        "command": step["expected_command"],
        "unit": f"wnn-bandit-{step['attempt']}.service",
        "status": "finished",
        "exit_status": 0,
        "contract": str(contract),
        "contract_sha256": checks.PLAN_SHA,
        "source_sha256": source,
        "source_manifest_sha256": source_manifest,
    }
    path = project / "runtime/attempts" / f"{step['attempt']}.json"
    _, digest = _write_json(path, record)
    return record, digest


class F3ResultChecksTests(unittest.TestCase):
    def test_plan_step_set_rejects_non_v1_attempt(self) -> None:
        plan_path = (
            Path(__file__).resolve().parents[1]
            / "f3_beleriand_preparation_r1/F3_BELERIAND_PLAN.json"
        )
        plan = json.loads(plan_path.read_text())
        self.assertEqual(len(checks.step_map(plan)), 7)
        plan["steps"][0]["attempt"] = "computer-vision-homework-r01-wide-f3s1001-ep01-v2"
        with self.assertRaisesRegex(CollectionError, "only original F3 v1"):
            checks.step_map(plan)

    def test_attempt_binds_command_contract_and_source_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            project = Path(temp).resolve()
            contract = project / "plan.json"
            contract.write_text("{}")
            step = {
                "attempt": "f3-fit-v1",
                "expected_command": ["python", "train_rnn.py", "--seed", "1001"],
                "step_id": "stage1_wide",
            }
            _, digest = _attempt(project, step, contract)
            path = project / "runtime/attempts/f3-fit-v1.json"
            record, actual, source = checks.validate_attempt(
                path, step, project, contract, checks.PLAN_SHA
            )
            self.assertEqual(actual, digest)
            self.assertEqual(source, record["source_sha256"])
            record["contract_sha256"] = "0" * 64
            _write_json(path, record)
            with self.assertRaisesRegex(CollectionError, "exact F3 plan contract"):
                checks.validate_attempt(path, step, project, contract, checks.PLAN_SHA)

    def test_evaluation_sidecar_uses_f3_order_and_epoch5(self) -> None:
        groups = [8, 2, 9]
        sidecar = {
            "run": "rr18_r01_wide_f3_s1001_e6",
            "split": "valid",
            "epoch": 5,
            "split_manifest_sha256": checks.FOLD_SHA,
            "campaign_split_manifest_sha256": checks.FOLD_SHA,
            "campaign_export_scope": "evaluation",
            "feature_only": True,
            "inference_precision": "fp32",
            "row_groups": groups,
            "evaluation_membership_sha256": membership_sha256(groups),
            "checkpoint_sha256": "b" * 64,
            "prediction_sha256": "c" * 64,
        }
        checks.validate_sidecar(sidecar, sidecar["run"], groups, "b" * 64, "c" * 64)
        sidecar["campaign_export_scope"] = "full-valid-feature-only"
        with self.assertRaisesRegex(CollectionError, "sidecar fields differ"):
            checks.validate_sidecar(sidecar, sidecar["run"], groups, "b" * 64, "c" * 64)

    def test_grade_binds_both_fixed_endpoints_and_components(self) -> None:
        groups = [10, 20]
        arms = {
            "wide": {"run": "wide", "checkpoint_sha256": "a" * 64, "provenance_sha256": "b" * 64},
            "deep": {"run": "deep", "checkpoint_sha256": "c" * 64, "provenance_sha256": "d" * 64},
        }
        scores = {
            "endpoint": {
                "run": "wide",
                "epoch": 5,
                "checkpoint_sha256": "a" * 64,
                "provenance_sha256": "b" * 64,
                "prediction_sha256": "e" * 64,
                "sidecar_sha256": "f" * 64,
                "score": {
                    "metric_id": "global_wp_clip2_v1",
                    "wp": 0.6,
                    "t0": 0.5,
                    "t1": 0.7,
                    "selected_rows": 2,
                },
            },
            "reference": {
                "run": "deep",
                "epoch": 5,
                "checkpoint_sha256": "c" * 64,
                "provenance_sha256": "d" * 64,
                "prediction_sha256": "1" * 64,
                "sidecar_sha256": "2" * 64,
                "score": {
                    "metric_id": "global_wp_clip2_v1",
                    "wp": 0.55,
                    "t0": 0.45,
                    "t1": 0.65,
                    "selected_rows": 2,
                },
            },
        }
        paired = {
            "resamples": 10000,
            "n_boot": 10000,
            "seed": 271828,
            "components": 1,
            "n_components": 1,
            "n_sequences": 2,
            "centered_moments": True,
            "metric_id": "global_wp_clip2_v1",
            "method": "paired whole-component bootstrap using centered moments",
            "mean_diff": 0.05,
            "bootstrap_sd": 0.1,
            "p_le_0": 0.2,
            "candidate_wp": 0.6,
            "reference_wp": 0.55,
            "ci95": [-0.1, 0.2],
        }
        grade = {
            "schema": "priority25.endpoint_grade.v1",
            "fold": 3,
            "n_folds": 4,
            "split_manifest_sha256": checks.FOLD_SHA,
            "evaluation_row_groups": groups,
            "evaluation_components": 1,
            "metric_id": "global_wp_clip2_v1",
            **scores,
            "paired_component_bootstrap": paired,
        }
        checks.validate_grade(
            grade,
            groups,
            1,
            {"wide": arms["wide"], "deep": arms["deep"]},
            {"wide": "e" * 64, "deep": "1" * 64},
            {"wide": "f" * 64, "deep": "2" * 64},
            10000,
        )
        grade["fold"] = 1
        with self.assertRaisesRegex(CollectionError, "fold/membership/metric"):
            checks.validate_grade(
                grade,
                groups,
                1,
                arms,
                {"wide": "e" * 64, "deep": "1" * 64},
                {"wide": "f" * 64, "deep": "2" * 64},
                10000,
            )

    def test_terminal_seal_requires_actual_receipt_and_reviewer_bindings(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            project = Path(temp).resolve()
            collector_hashes = {
                name: str(i) * 64 for i, name in enumerate(checks.COLLECTOR_FILES, 1)
            }
            arms: dict[str, dict[str, object]] = {}
            entries = {}
            for arm in ("wide", "deep"):
                arms[arm] = {}
                phase_receipts = {}
                paths = {}
                for phase in ("stage1", "terminal"):
                    collection_path = project / f"receipts/{arm}_{phase}_c.json"
                    verification_path = project / f"receipts/{arm}_{phase}_v.json"
                    meta = {
                        "collection_path": collection_path,
                        "collection_bytes": 31,
                        "collection_sha": "a" * 64,
                        "verification_path": verification_path,
                        "verification_bytes": 33,
                        "verification_sha": "b" * 64,
                        "snapshot_path": project / f"snapshots/{arm}/{phase}",
                    }
                    arms[arm][phase] = meta
                    paths[phase] = str(meta["snapshot_path"])
                    phase_receipts[phase] = {
                        "collection": {
                            "path": str(collection_path),
                            "bytes": 31,
                            "sha256": "a" * 64,
                        },
                        "verification": {
                            "path": str(verification_path),
                            "bytes": 33,
                            "sha256": "b" * 64,
                        },
                    }
                review = {
                    "schema": "priority25.f3_terminal_result_review.v1",
                    "status": "passed",
                    "arm": arm,
                    "plan_sha256": checks.PLAN_SHA,
                    "fold3_manifest_sha256": checks.FOLD_SHA,
                    "checkpoint_support_manifest_sha256": checks.CHECKPOINT_MANIFEST_SHA,
                    "collector_support_sha256": collector_hashes,
                    "receipts": phase_receipts,
                    "snapshot_paths": paths,
                }
                review_path = project / f"reviews/{arm}.json"
                byte_count, review_sha = _write_json(review_path, review)
                entries[arm] = {
                    "receipts": phase_receipts,
                    "snapshot_paths": paths,
                    "review_receipt": {
                        "path": str(review_path),
                        "bytes": byte_count,
                        "sha256": review_sha,
                        "status": "passed",
                    },
                }
            seal = {
                "schema": "priority25.f3_terminal_evidence_seal.v1",
                "status": "sealed",
                "review_status": "passed",
                "plan_sha256": checks.PLAN_SHA,
                "fold3_manifest_sha256": checks.FOLD_SHA,
                "checkpoint_support_manifest_sha256": checks.CHECKPOINT_MANIFEST_SHA,
                "collector_support_sha256": collector_hashes,
                "arms": entries,
            }
            seal_path = project / "seal.json"
            _, seal_sha = _write_json(seal_path, seal)
            review_bindings = checks.validate_seal(
                project, seal_path, seal_sha, arms, collector_hashes
            )
            self.assertEqual(review_bindings["wide"][0], project / "reviews/wide.json")
            self.assertEqual(
                review_bindings["wide"][1], entries["wide"]["review_receipt"]["sha256"]
            )
            seal["arms"]["deep"]["review_receipt"]["sha256"] = "0" * 64
            _, bad_sha = _write_json(seal_path, seal)
            with self.assertRaisesRegex(CollectionError, "review receipt file binding"):
                checks.validate_seal(project, seal_path, bad_sha, arms, collector_hashes)

    def test_actual_f3_stage1_receipt_shapes_and_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            project = Path(temp).resolve()
            snapshot = project / "checkpoint_snapshots/rr18_r01_wide_f3_s1001/stage1_epoch1"
            snapshot.mkdir(parents=True)
            names = {
                "model_ep0.pt",
                "model_ep1.pt",
                "provenance.json",
                "training_state.pt",
                "history.json",
            }
            artifacts: dict[str, dict[str, object]] = {}
            for name in names:
                data = f"tiny {name}".encode()
                path = snapshot / name
                path.write_bytes(data)
                artifacts[name] = {
                    "path": str(path),
                    "bytes": len(data),
                    "sha256": hashlib.sha256(data).hexdigest(),
                }
            command = ["python", "train_rnn.py", "--seed", "1001"]
            step = {
                "step_id": "stage1_wide",
                "run_name": "rr18_r01_wide_f3_s1001",
                "attempt": "computer-vision-homework-r01-wide-f3s1001-ep01-v1",
                "archive_directory": str(snapshot),
                "expected_command": command,
            }
            source_map = {"plan": checks.PLAN_SHA}
            source_manifest = hashlib.sha256(
                json.dumps(source_map, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            attempt = {
                "id": step["attempt"],
                "status": "finished",
                "exit_status": 0,
                "contract_sha256": checks.PLAN_SHA,
                "source_manifest_sha256": source_manifest,
                "source_sha256": source_map,
            }
            raw_attempt = {**attempt, "command": command}
            attempt_bytes, attempt_sha = _write_json(
                project / "runtime/attempts/f3-attempt.json", raw_attempt
            )
            attempt_copy = snapshot / "attempt.json"
            copied_bytes, copied_sha = _write_json(attempt_copy, raw_attempt)
            assert (attempt_bytes, attempt_sha) == (copied_bytes, copied_sha)
            support = {
                "collect_f3_checkpoint.py": "a" * 64,
                "f3_checkpoint_common.py": "b" * 64,
                "verify_f3_checkpoint.py": "c" * 64,
                "f3_checkpoint_evidence.py": "d" * 64,
                "f3_checkpoint_lineage.py": "e" * 64,
                "f3_checkpoint_tensors.py": "f" * 64,
                "f3_checkpoint_membership.py": "1" * 64,
                "f3_checkpoint_semantics.py": "2" * 64,
            }
            collection = {
                "schema": "priority25.f3_checkpoint_collection.v1",
                "arm": "wide",
                "phase": "stage1",
                "run": step["run_name"],
                "attempt": step["attempt"],
                "plan_sha256": checks.PLAN_SHA,
                "snapshot_directory": str(snapshot),
                "source_directory": str(project / "data/artifacts/runs" / step["run_name"]),
                "source_stats_before": {"same": 1},
                "source_stats_after": {"same": 1},
                "source_snapshot_hash_match": True,
                "producer_model_hash_matches": True,
                "attempt_record_sha256": attempt_sha,
                "collector_sha256": support["collect_f3_checkpoint.py"],
                "collector_dependencies": {
                    "f3_checkpoint_common.py": support["f3_checkpoint_common.py"]
                },
                "artifacts": artifacts,
                "attempt_evidence": {
                    "path": str(attempt_copy),
                    "bytes": copied_bytes,
                    "sha256": copied_sha,
                    **attempt,
                },
            }
            counts = {
                "optimizer_attempts": 29_760,
                "optimizer_completed_updates": 29_760,
                "amp_skipped_updates": 0,
                "ema_updates": 0,
                "completed_batches": 2_976,
            }
            fold_membership = {
                "manifest_sha256": checks.FOLD_SHA,
                "fit_train_groups": 10_539,
                "fit_valid_groups": 1_368,
                "evaluation_groups": 505,
                "evaluation_components": 145,
                "fit_membership_identity": {
                    "train": {"count": 10_607, "ordered_int64_le_sha256": "4" * 64},
                    "valid": {"count": 1_368, "ordered_int64_le_sha256": "5" * 64},
                    "valid_repeat": 1,
                    "dataset_length": 11_907,
                },
                "ordered_evaluation_membership_sha256": "6" * 64,
            }
            state_source = project / "source-a266b036/src/wnn/training_state.py"
            state_source.parent.mkdir(parents=True)
            state_source.write_bytes(b"tiny loader")
            dependencies = {
                key: value for key, value in support.items() if key != "verify_f3_checkpoint.py"
            }
            dependencies["wnn/training_state.py"] = hashlib.sha256(
                state_source.read_bytes()
            ).hexdigest()
            verification = {
                "schema": "priority25.f3_checkpoint_verification.v1",
                "status": "passed",
                "arm": "wide",
                "phase": "stage1",
                "run": step["run_name"],
                "next_epoch": 2,
                "plan_sha256": checks.PLAN_SHA,
                "fold3_manifest_sha256": checks.FOLD_SHA,
                "source_revision": checks.SOURCE_REVISION,
                "source_archive_manifest_sha256": checks.ARCHIVE_SHA,
                "portable_registry_sha256": checks.REGISTRY_SHA,
                "collection_receipt_sha256": "3" * 64,
                "attempt": {
                    "id": step["attempt"],
                    "record_sha256": attempt_sha,
                    "command": command,
                    "status": "finished",
                    "exit_status": 0,
                },
                "completed_update_counts": counts,
                "planned_schedule_updates": 89_280,
                "planned_schedule_batches": 8_928,
                "scheduler_last_epoch": 29_760,
                "scheduler_step_count": 29_761,
                "history_epochs_verified": 2,
                "fold3_membership": fold_membership,
                "verification_dependencies": dependencies,
                "verification_helper_sha256": support["verify_f3_checkpoint.py"],
                "artifacts": artifacts,
            }
            collection_path = project / "receipts/collection.json"
            verification_path = project / "receipts/verification.json"
            _, collection_sha = _write_json(collection_path, collection)
            verification["collection_receipt_sha256"] = collection_sha
            _, verification_sha = _write_json(verification_path, verification)
            out = checks.validate_checkpoint_phase(
                project,
                {},
                {"stage1_wide": step},
                "wide",
                "stage1",
                collection_path,
                verification_path,
                raw_attempt,
                attempt_sha,
                support,
                fold_membership,
            )
            self.assertEqual(out[2:], (collection_sha, verification_sha, artifacts))
            verification["completed_update_counts"]["optimizer_completed_updates"] = 29_759
            _write_json(verification_path, verification)
            with self.assertRaisesRegex(CollectionError, "counters/scheduler differ"):
                checks.validate_checkpoint_phase(
                    project,
                    {},
                    {"stage1_wide": step},
                    "wide",
                    "stage1",
                    collection_path,
                    verification_path,
                    raw_attempt,
                    attempt_sha,
                    support,
                    fold_membership,
                )


if __name__ == "__main__":
    unittest.main()
