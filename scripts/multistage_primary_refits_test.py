"""CPU-only falsifiers for configured-final reporting and fixed four-member ensembles."""

from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from multistage_primary_refits_inputs import (
    HARD_DEADLINE,
    bind_final_model,
    configured_final,
    owned_group_active,
    validate_weights,
)
from multistage_primary_refits_outputs import export_report
from report_multistage_primary_ensemble_worker import (
    PRIMARY_ROWS,
    PRIMARY_SHA256,
    _canonical_sha256,
)
from report_multistage_primary_refits import watchdog_command
from report_multistage_primary_refits_worker import SOURCE4, common_mask, validate_endpoint


def _endpoint() -> tuple[dict[str, object], dict[str, object], object]:
    primary = SimpleNamespace(
        train_groups=tuple(range(10325)), evaluation_groups=tuple(range(1873))
    )
    config = {"epochs": 4, "precision_mode": "fp32", "kd_weight": 0, "hint_weight": 0}
    data = {
        "split_sha256": PRIMARY_SHA256,
        "primary_split_sha256": None,
        "fit_members": [["train", group] for group in primary.train_groups],
        "evaluation_members": [["valid", group] for group in primary.evaluation_groups],
        "evaluation_role": "valid",
        "evaluation_source": "valid",
        "evaluation_mask": "is_scored",
        "teacher": None,
        "initializer": None,
        "normalization": {"source_fingerprints": {"datasets/train.parquet": {}}},
    }
    identity = {"source_sha256": SOURCE4, "config_sha256": _canonical_sha256(config), "data": data}
    member = {
        "config": config,
        "identity": identity,
        "artifacts": {"approved_profile": {"sha256": "a" * 64}},
    }
    payload = {
        "checkpoint_schema": "wnn.sequence-fit.v1",
        "completed_epochs": 4,
        "epoch": 3,
        "config": config,
        "data": data,
        "provenance": {
            "identity": identity,
            "fixed_checkpoint_policy": "configured-final-epoch",
            "profile_receipt": {"sha256": "a" * 64},
        },
    }
    return payload, member, primary


class FullPrimaryRefitTests(unittest.TestCase):
    def test_independent_watchdog_wraps_gpu_and_cpu_commands_with_cleanup_margin(self) -> None:
        for mode in ("member", "blend"):
            worker = ["python", "worker.py", "--mode", mode]
            wrapped, wall = watchdog_command(worker, HARD_DEADLINE, HARD_DEADLINE - 20)
            self.assertEqual(
                wrapped[:4], ["/usr/bin/timeout", "--signal=TERM", "--kill-after=5s", "10s"]
            )
            self.assertEqual(wrapped[4:], worker)
            self.assertEqual(wall, 20)
            self.assertLessEqual((HARD_DEADLINE - 20) + 10 + 5, HARD_DEADLINE - 5)
        with self.assertRaises(RuntimeError):
            watchdog_command(["python"], HARD_DEADLINE, HARD_DEADLINE - 10)

    def test_native_terminal_model_checksum_must_match_final_bytes(self) -> None:
        bind_final_model({"model_sha256": "a" * 64}, {"sha256": "a" * 64})
        for terminal in ({}, {"model_sha256": "b" * 64}):
            with self.assertRaises(ValueError):
                bind_final_model(terminal, {"sha256": "a" * 64})

    def test_zombies_are_gone_but_live_owned_group_members_block_scoring(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            proc = Path(temporary)

            def write_stat(pid: int, state: str) -> None:
                directory = proc / str(pid)
                directory.mkdir(exist_ok=True)
                fields = ["0"] * 20
                fields[0], fields[2], fields[19] = state, "101", "7"
                (directory / "stat").write_text(
                    f"{pid} (owned worker) " + " ".join(fields), encoding="ascii"
                )

            identity = {"pid": 101, "pgid": 101, "birth": "7"}
            write_stat(101, "Z")
            self.assertFalse(owned_group_active(identity, proc))
            write_stat(102, "S")
            self.assertTrue(owned_group_active(identity, proc))
            write_stat(102, "X")
            self.assertFalse(owned_group_active(identity, proc))

    def test_terminal_gate_rejects_censored_and_early_endpoints(self) -> None:
        child = {"status": "completed", "exit_code": 0, "timed_out": False}
        terminal = {
            "status": "completed",
            "completion_kind": "max_epochs",
            "completed_epochs": 4,
            "requested_epochs": 4,
        }
        self.assertTrue(configured_final(child, terminal, 4))
        for changed in (
            {"status": "timed_out"},
            {"exit_code": 1},
            {"timed_out": True},
            {"exit_code": False},
        ):
            self.assertFalse(configured_final(child | changed, terminal, 4))
        for changed in (
            {"status": "early_stopped"},
            {"completion_kind": "declared_early_stop"},
            {"completed_epochs": 3},
        ):
            self.assertFalse(configured_final(child, terminal | changed, 4))

    def test_frozen_weights_are_bound_before_fit_and_never_reduced_to_partial_subset(self) -> None:
        selection = {
            "selected": [
                {"fit_id": f"fit-{i}", "source_result_path": f"result-{i}"} for i in range(4)
            ]
        }
        stage4 = {
            "ensemble": {
                "members": [
                    {"trial_result_path": f"result-{i}", "weight": float(i + 1)} for i in range(4)
                ]
            }
        }
        frozen = {
            "schema": "wnn.primary-refit-frozen-ensembles.v1",
            "selection_receipt_sha256": "s",
            "stage4_result_sha256": "t",
            "frozen_before_new_refit_fitting": True,
            "primary_valid_used_for_weights": False,
            "fit_ids": [f"fit-{i}" for i in range(4)],
            "frozen_epoch": 1.0,
            "equal_weight": [0.25] * 4,
            "stage4_direct_renormalized": [0.1, 0.2, 0.3, 0.4],
        }
        self.assertEqual(
            validate_weights(frozen, selection, "s", stage4, "t", 2.0)["equal_weight"], [0.25] * 4
        )
        for changed in (
            {"frozen_epoch": 2.0},
            {"primary_valid_used_for_weights": True},
            {"fit_ids": ["fit-0", "fit-1", "fit-2"]},
            {"equal_weight": [1 / 3] * 3},
            {"stage4_direct_renormalized": [0.4, 0.3, 0.2, 0.1]},
        ):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                validate_weights(frozen | changed, selection, "s", stage4, "t", 2.0)

    def test_loaded_payload_requires_exact_full_primary_source_and_ancestry(self) -> None:
        payload, member, primary = _endpoint()
        validate_endpoint(payload, member, primary)
        for key, value in (("completed_epochs", 3), ("checkpoint_schema", "wnn.sequence-epoch.v1")):
            wrong = copy.deepcopy(payload)
            wrong[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_endpoint(wrong, member, primary)
        for key, value in (
            ("teacher", {"path": "auxiliary"}),
            ("initializer", {"path": "auxiliary"}),
            ("fit_members", [["train", i] for i in range(8260)]),
            ("evaluation_source", "train"),
            ("evaluation_mask", "all_after_warmup"),
            ("split_sha256", "b" * 64),
            ("primary_split_sha256", "b" * 64),
            ("normalization", {"source_fingerprints": {"datasets/valid.parquet": {}}}),
        ):
            wrong_payload, wrong_member, primary = _endpoint()
            wrong_payload["data"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_endpoint(wrong_payload, wrong_member, primary)
        payload, member, primary = _endpoint()
        member["identity"]["source_sha256"] = "b" * 64
        with self.assertRaises(ValueError):
            validate_endpoint(payload, member, primary)

    def test_masks_and_order_must_match_all_four_singletons(self) -> None:
        mask = {
            "membership_sha256": "a" * 64,
            "mask_sha256": "b" * 64,
            "selected_row_index_sha256": "c" * 64,
            "selected_rows": PRIMARY_ROWS,
            "mask": "is_scored AND row_index>=99",
        }
        receipts = [{"status": "complete", "predictions": dict(mask)} for _ in range(4)]
        self.assertEqual(common_mask(receipts)["mask_sha256"], "b" * 64)
        for key, value in (
            ("mask_sha256", "d" * 64),
            ("selected_row_index_sha256", "d" * 64),
            ("selected_rows", 4),
        ):
            wrong = copy.deepcopy(receipts)
            wrong[-1]["predictions"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                common_mask(wrong)
        with self.assertRaises(ValueError):
            common_mask(receipts[:3])

    def test_partial_outcomes_are_exported_without_fixed_ensembles_or_fabricated_scores(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            members = [
                {
                    "index": i,
                    "fit_id": f"fit-{i}",
                    "family": "gru",
                    "configured_final_complete": False,
                    "child_status": "timed_out",
                    "native_terminal": {"completed_epochs": i},
                    "config": {"epochs": 4},
                    "receipt_path": str(output / f"missing-{i}.json"),
                }
                for i in range(4)
            ]
            request = {
                "output_dir": str(output),
                "members": members,
                "primary_split": {"sha256": PRIMARY_SHA256},
                "source6": {},
                "selection": {},
                "frozen_weights": {},
                "weights": {"equal_weight": [0.25] * 4},
                "ensemble_policy": "all four complete required",
            }
            result = export_report(
                request, output / "score_request.json", "fixed", {}, {"status": "skipped"}
            )
            self.assertEqual(result["status"], "partial_or_singletons_only")
            self.assertEqual(len(result["comparison_rows"]), 5)
            self.assertTrue(all(row["wp"] is None for row in result["comparison_rows"][:4]))
            self.assertTrue((output / "primary_refit_comparison.png").is_file())
            self.assertTrue((output / "primary_refit_comparison.svg").is_file())
            self.assertFalse((output / "equal_weight_scored_predictions.npy").exists())


if __name__ == "__main__":
    unittest.main()
