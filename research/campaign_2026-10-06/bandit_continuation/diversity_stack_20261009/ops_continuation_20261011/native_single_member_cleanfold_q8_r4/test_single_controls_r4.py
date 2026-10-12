"""Bounded standard-library tests for the R4 fail-closed gates."""

from __future__ import annotations

import importlib.util
import struct
import tempfile
import unittest
from pathlib import Path
from typing import Any

RUNNER_PATH = Path(__file__).with_name("run_single_controls_r4.py")
SPEC = importlib.util.spec_from_file_location("single_controls_r4_under_test", RUNNER_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load R4 test subject")
R4 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(R4)


def _admission_contract() -> dict[str, Any]:
    """Return the minimum binding shape expected by the exact admission gate."""
    return {
        "r3_binding": {
            "contract_sha256": "a" * 64,
            "input_preflight_sha256": "b" * 64,
            "valid_dataset_identity": {
                "bytes": 100,
                "footer_bytes": 12,
                "footer_sha256_including_8_byte_trailer": "c" * 64,
            },
        },
        "models": {
            fold: {
                "evaluation_group_list_sha256": f"{fold}" * 64,
                "models": {
                    role: {"checkpoint_sha256": "d" * 64, "provenance_sha256": "e" * 64}
                    for role in R4.ROLES
                },
            }
            for fold in R4.FOLDS
        },
        "weights": {"archive_sha256": "f" * 64, "receipt_sha256": "1" * 64},
        "runtime_assets": {"native_elf": {"sha256": "2" * 64}},
        "output": {"root_repo_path": "research/out"},
        "root_admission": {"schema": "test.admission.r3", "decision": "PASS_FOR_TEST"},
        "resource_budget": {
            "cpu_cores": 1,
            "ram_max_bytes": 1_500_000_000,
            "wall_seconds": 1800,
            "gpu": False,
            "new_spend_usd": 0,
            "thread_count": 1,
            "provider_actions": False,
        },
    }


class R4GateTests(unittest.TestCase):
    def test_changed_parquet_footer_fails_against_prior_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "valid.parquet"
            footer = b"footer-r4-v1"

            def write_parquet_footer(value: bytes) -> None:
                content = b"PAR1" + b"BODY" + value + struct.pack("<I", len(value)) + b"PAR1"
                path.write_bytes(content)

            write_parquet_footer(footer)
            with path.open("rb") as stream:
                original = R4.file_footer_identity(stream, path, len(footer))
            r3 = {
                "valid_dataset": {
                    key: original[key]
                    for key in ("bytes", "footer_bytes", "footer_sha256_including_8_byte_trailer")
                }
            }
            preflight = {"valid_dataset_identity": {key: original[key] for key in ("inode", "mtime_ns")}}
            R4.validate_live_identity(original, r3, preflight)

            write_parquet_footer(b"footer-r4-v2")
            with path.open("rb") as stream:
                changed = R4.file_footer_identity(stream, path, len(footer))
            with self.assertRaises(R4.GateError):
                R4.validate_live_identity(changed, r3, preflight)

    def test_missing_and_wrong_root_admission_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            missing = Path(temp) / "ROOT_ADMISSION_R3.json"
            with self.assertRaisesRegex(R4.GateError, "missing"):
                R4.read_root_admission(missing)

        contract = _admission_contract()
        args = (contract, "3" * 64, "4" * 64, "5" * 64, "6" * 64)
        expected = R4.admission_expected_bindings(*args)
        admission = {
            "schema": "test.admission.r3",
            "status": "admitted",
            "authorized": True,
            "decision": "PASS_FOR_TEST",
            "bindings": expected,
            "budget": contract["resource_budget"],
        }
        R4.verify_admission_object(admission, *args)
        wrong = dict(admission, status="pending")
        with self.assertRaisesRegex(R4.GateError, "unauthorized"):
            R4.verify_admission_object(wrong, *args)
        wrong = dict(admission, bindings={**expected, "unbound": True})
        with self.assertRaisesRegex(R4.GateError, "binding set/value"):
            R4.verify_admission_object(wrong, *args)

    def test_repository_path_binding_rejects_absolute_traversal_and_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as temp, tempfile.TemporaryDirectory() as outside:
            root = Path(temp)
            file = root / "inside" / "receipt.json"
            file.parent.mkdir()
            file.write_text("{}", encoding="utf-8")
            self.assertEqual(R4.resolve_repo_path(root, "inside/receipt.json"), file.resolve())
            for unsafe in (str(file), "../escape", "inside\\receipt.json"):
                with self.subTest(path=unsafe), self.assertRaises(R4.GateError):
                    R4.resolve_repo_path(root, unsafe, must_exist=False)
            link = root / "linked.json"
            link.symlink_to(Path(outside) / "external.json")
            with self.assertRaises(R4.GateError):
                R4.resolve_repo_path(root, "linked.json", must_exist=False)

    def test_result_accounting_requires_exact_groups_members_and_outputs(self) -> None:
        plan = {}
        for fold in R4.FOLDS:
            for role in R4.ROLES:
                key = f"fold{fold}_{role}"
                plan[key] = {
                    "filename": f"predictions_fold{fold}_{role}.npy",
                    "shape": [2, 20_000, 2],
                    "dtype": "float32",
                    "bytes": 320_000,
                    "fold": fold,
                    "role": role,
                    "evaluation_group_list_sha256": f"{fold}" * 64,
                    "selected_rows": 101,
                }
        inventory = {
            item["filename"]: {"bytes": item["bytes"], "dtype": item["dtype"], "shape": item["shape"], "sha256": "a" * 64, "all_finite": True}
            for item in plan.values()
        }
        folds = {fold: {} for fold in R4.FOLDS}
        for fold in R4.FOLDS:
            for role in R4.ROLES:
                spec = plan[f"fold{fold}_{role}"]
                folds[fold][role] = {
                    "evaluation_groups": 2,
                    "evaluation_group_list_sha256": spec["evaluation_group_list_sha256"],
                    "callback_rows": 40_000,
                    "member_count": 1,
                    "precision": "q8vnni",
                    "metric": {"selected_rows": 101},
                    "prediction_file": inventory[spec["filename"]],
                }
        result = {
            "folds_evaluated": list(R4.FOLDS),
            "roles_evaluated_independently": list(R4.ROLES),
            "member_count_per_prediction": 1,
            "precision": "q8vnni",
            "blend_search": False,
            "parameter_averaging": False,
            "training_performed": False,
            "fresh_holdout_claim": False,
            "folds": folds,
            "prediction_inventory": inventory,
        }
        R4.validate_result_accounting(result, plan)
        folds["1"]["hft"]["evaluation_groups"] = 1
        with self.assertRaisesRegex(R4.GateError, "group count"):
            R4.validate_result_accounting(result, plan)
        folds["1"]["hft"]["evaluation_groups"] = 2
        result["prediction_inventory"].pop("predictions_fold0_student.npy")
        with self.assertRaisesRegex(R4.GateError, "inventory"):
            R4.validate_result_accounting(result, plan)

    def test_prediction_plan_accounts_for_npy_header_bytes(self) -> None:
        r3 = {"folds": {}, "expected_output": {"prediction_files": {}}}
        for fold, groups in (("0", 468), ("1", 457)):
            r3["folds"][fold] = {
                "evaluation_groups": groups,
                "evaluation_group_list_sha256": fold * 64,
                "fixed_half_output_reference": {"selected_rows": 101},
            }
            for role in R4.ROLES:
                r3["expected_output"]["prediction_files"][f"fold{fold}_{role}"] = {
                    "shape": [groups, 20_000, 2],
                    "dtype": "float32",
                    "bytes": groups * 20_000 * 2 * 4 + 128,
                }
        plan = R4.prediction_plan(r3)
        self.assertEqual(plan["fold0_student"]["payload_bytes"], 468 * 20_000 * 2 * 4)
        self.assertEqual(plan["fold0_student"]["npy_header_bytes"], 128)
        self.assertEqual(plan["fold1_hft"]["bytes"], 457 * 20_000 * 2 * 4 + 128)
        r3["expected_output"]["prediction_files"]["fold0_student"]["bytes"] -= 1
        with self.assertRaisesRegex(R4.GateError, "NPY header/file byte accounting"):
            R4.prediction_plan(r3)

    def test_wrapper_requires_actual_cpu_memory_and_no_gpu_caps(self) -> None:
        contract = _admission_contract()
        env = {
            "WNN_SLICE": "wnn-campaign.slice",
            "WNN_CPU_QUOTA": "100%",
            "WNN_MEM_MAX": "1500000000",
            "WNN_RUNTIME_MAX": "1800s",
            "CUDA_VISIBLE_DEVICES": "",
            **{key: "1" for key in R4.THREAD_ENV},
        }
        cgroup = {"path": "/user.slice/wnn-campaign.slice/support.scope", "cpu_quota": 100_000, "cpu_period": 100_000, "memory_max_bytes": 1_500_000_000, "memory_current_bytes": 100_000_000, "memory_swap_max_bytes": 0}
        R4.verify_wrapper_limits(contract["resource_budget"], env, cgroup)
        with self.assertRaisesRegex(R4.GateError, "GPU"):
            R4.verify_wrapper_limits(contract["resource_budget"], {key: value for key, value in env.items() if key != "CUDA_VISIBLE_DEVICES"}, cgroup)
        with self.assertRaisesRegex(R4.GateError, "CPU"):
            R4.verify_wrapper_limits(contract["resource_budget"], env, {**cgroup, "cpu_quota": 200_000})
        with self.assertRaisesRegex(R4.GateError, "campaign cgroup"):
            R4.verify_wrapper_limits(contract["resource_budget"], env, {**cgroup, "path": "/user.slice/other.slice/job.scope"})


if __name__ == "__main__":
    unittest.main()
