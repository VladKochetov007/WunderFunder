from __future__ import annotations

import unittest

from scripts.w18_l4_quality_20261104.contract import ContractError, validate_panel


def _recipe(run_name: str, seed: int) -> list[str]:
    return [
        "--name",
        run_name,
        "--seed",
        str(seed),
        "--train-source",
        "train",
        "--train-seqs",
        "10325",
        "--eval-source",
        "dev",
        "--full-valid",
        "--split-manifest",
        "/data/project/chronological_primary.json",
        "--precision",
        "fp32",
        "--checkpoint-policy",
        "fixed",
        "--checkpoint-epoch",
        "1",
        "--lr",
        ".0001",
        "--epochs",
        "2",
        "--train-eval-seqs",
        "32",
        "--save-training-state",
    ]


def _panel() -> dict[str, object]:
    digest = "a" * 64
    cells: list[dict[str, object]] = []
    configurations = [("control", "control", None)] + [
        (f"arm{index}", "mechanism", f"H{index}") for index in range(1, 4)
    ]
    for arm, role, hypothesis in configurations:
        for seed in (1001, 1002):
            run_name = f"{arm}_s{seed}"
            index = len(cells) + 1
            args = _recipe(run_name, seed)
            if arm == "arm1":
                args.extend(["--research-features", "causal_innovations"])
            elif arm == "arm2":
                args[args.index("--lr") + 1] = ".0002"
            elif arm == "arm3":
                args[args.index("--lr") + 1] = ".0003"
            cells.append(
                {
                    "cell_id": f"{arm}_{seed}",
                    "configuration_id": arm,
                    "role": role,
                    "hypothesis_id": hypothesis,
                    "seed": seed,
                    "gpu_uuid": f"GPU-00000000-0000-0000-0000-{index:012d}",
                    "cpu_threads": 8,
                    "run_name": run_name,
                    "train_args": args,
                    "fit_timeout_seconds": 3600,
                }
            )
    return {
        "schema": "wnn.w18-l4-cohort.v1",
        "status": "recipes_frozen",
        "attempt_id": "w18-l4-r1",
        "paths": {
            "source_root": "/data/source",
            "project_root": "/data/project",
            "python": "/data/.venv/bin/python",
            "split_manifest": "/data/project/chronological_primary.json",
            "runtime_root": "/data/runtime",
            "control_root": "/data/control",
            "disk_mount": "/data",
        },
        "identity": {
            "source_binding_sha256": digest,
            "data_binding_sha256": digest,
            "split_sha256": digest,
            "fit_members_sha256": digest,
            "fit_sequence_count": 10325,
            "valid_members_sha256": digest,
            "valid_sequence_count": 1873,
            "valid_scored_row_count": 3_528_273,
            "baseline_ancestry_sha256": digest,
        },
        "budget": {
            "max_scientific_fits": 8,
            "max_profiles": 16,
            "max_wave_seconds": 43_200,
            "collection_reserve_seconds": 2700,
            "max_output_bytes": 8_000_000_000,
            "minimum_free_bytes": 3_000_000_000,
            "aggregate_cpu_threads": 64,
            "aggregate_memory_bytes": 400 * (1 << 30),
        },
        "profile_policy": {"warmup_batches": 1, "timed_batches": 3, "timeout_seconds": 900},
        "admission_receipt": "/data/runtime/admission.json",
        "reference_predictions": [
            {
                "run_name": "clean_reference",
                "weight": 1.0,
                "prediction_path": "/data/cache/pred_reference.npy",
                "sidecar_path": "/data/cache/pred_reference.json",
                "checkpoint_sha256": digest,
            }
        ],
        "fixed_blend_weight": 0.1,
        "cells": cells,
    }


class W18L4CohortContractTests(unittest.TestCase):
    def test_frozen_eight_cell_panel_is_accepted(self) -> None:
        validate_panel(_panel())

    def test_full_valid_is_mandatory(self) -> None:
        panel = _panel()
        cells = panel["cells"]
        cells[0]["train_args"].remove("--full-valid")
        with self.assertRaisesRegex(ContractError, "full-valid"):
            validate_panel(panel)

    def test_gpu_is_unique_per_fit(self) -> None:
        panel = _panel()
        panel["cells"][1]["gpu_uuid"] = panel["cells"][0]["gpu_uuid"]
        with self.assertRaisesRegex(ContractError, "unique physical GPU"):
            validate_panel(panel)

    def test_matched_seeds_may_change_only_seed_and_run_name(self) -> None:
        panel = _panel()
        second_seed_args = panel["cells"][1]["train_args"]
        second_seed_args[second_seed_args.index("--lr") + 1] = ".0002"
        with self.assertRaises(ContractError):
            validate_panel(panel)

    def test_mechanism_recipes_must_differ_from_control(self) -> None:
        panel = _panel()
        for candidate_index, control_index in ((4, 0), (5, 1)):
            candidate = panel["cells"][candidate_index]
            matched_control = panel["cells"][control_index]["train_args"]
            candidate["train_args"] = list(matched_control)
            candidate["train_args"][1] = candidate["run_name"]
        with self.assertRaisesRegex(ContractError, "recipes must be distinct"):
            validate_panel(panel)

    def test_hypothesis_ids_are_unique_per_mechanism(self) -> None:
        panel = _panel()
        panel["cells"][6]["hypothesis_id"] = panel["cells"][4]["hypothesis_id"]
        panel["cells"][7]["hypothesis_id"] = panel["cells"][4]["hypothesis_id"]
        with self.assertRaisesRegex(ContractError, "unique hypothesis ids"):
            validate_panel(panel)

    def test_fp32_does_not_admit_quantized_training(self) -> None:
        panel = _panel()
        panel["cells"][0]["train_args"].extend(["--qat", "int8"])
        panel["cells"][1]["train_args"].extend(["--qat", "int8"])
        with self.assertRaisesRegex(ContractError, "quantized training is forbidden"):
            validate_panel(panel)

    def test_cpu_affinity_must_be_disjoint(self) -> None:
        panel = _panel()
        for index, cell in enumerate(panel["cells"]):
            cell["cpu_affinity"] = list(range(index * 8, index * 8 + 8))
        panel["cells"][1]["cpu_affinity"] = panel["cells"][0]["cpu_affinity"]
        with self.assertRaisesRegex(ContractError, "must not overlap"):
            validate_panel(panel)


if __name__ == "__main__":
    unittest.main()
