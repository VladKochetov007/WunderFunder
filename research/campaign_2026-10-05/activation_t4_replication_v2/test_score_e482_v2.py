"""Synthetic, fit-data-free contract tests for the E482 independent grader."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "research/campaign_2026-10-05/recurrent_activation_curvature"))

import score_e482_v2 as grader  # noqa: E402
import score_recurrent_curvature_v2 as e481grade  # noqa: E402


def resource_fixture() -> dict:
    """Return a successful synthetic two-T4 supervisor receipt."""
    host = {
        "capacity_bytes": 32 * 1024**3,
        "available_bytes": 29 * 1024**3,
        "affinity": [0, 1, 2, 3],
    }
    devices = [
        {
            "index": index,
            "uuid": f"GPU-{index}",
            "name": "Tesla T4",
            "total_bytes": 16 * 1024**3,
            "used_bytes": 0,
        }
        for index in (0, 1)
    ]
    events = {"oom": 0, "oom_kill": 0, "oom_group_kill": 0}
    return {
        "resource_start": {"host": host, "devices": devices, "memory_events": events},
        "resource_end": {"host": host, "devices": devices, "memory_events": events},
        "exit_codes": {optimizer: 0 for optimizer in grader.OPTIMIZERS},
        "peak_process_tree_rss_bytes_by_optimizer": {
            optimizer: 1024**3 for optimizer in grader.OPTIMIZERS
        },
        "thresholds": {
            "ram_worker_bytes": grader.RAM_LIMIT_BYTES,
            "gpu_physical_stop_bytes": grader.GPU_STOP_BYTES,
            "host_stop_fraction": grader.HOST_STOP_FRACTION,
        },
    }


def probe_fixture(optimizer: str, uuid: str, sources: dict[str, str]) -> dict:
    """Return a structurally valid probe receipt with synthetic-only values."""
    precision = {
        "device_math": "FP32 IEEE",
        "autocast": False,
        "torch.backends.fp32_precision": "ieee",
        "torch.backends.cuda.matmul.fp32_precision": "ieee",
        "torch.backends.cudnn.fp32_precision": "ieee",
        "torch.backends.cudnn.rnn.fp32_precision": "ieee",
    }
    return {
        "status": "probe_passed",
        "phase": "probe",
        "optimizer": optimizer,
        "fit_only": True,
        "test_rows_loaded": False,
        "source_sha256": sources,
        "dataset_sha256": grader.DATASET_SHA256,
        "cohort_sha256": grader.COHORT_SHA256,
        "assigned_device": {"uuid": uuid, "name": "Tesla T4"},
        "device_properties": {"uuid": uuid},
        "native_parity": {
            "optimizer": optimizer,
            "output_state_input_gradient_parameter_gradient_and_clipped_step": "passed",
            "precision": precision,
        },
        "peak_allocated_bytes": 100,
        "peak_reserved_bytes": 200,
        "arms": [
            {
                "seed": grader.SEED,
                "optimizer": optimizer,
                "shape": shape,
                "batch": 4,
                "chunk_rows": grader.CHUNK_ROWS,
                "updates": 1,
                "changed_parameter_tensors": 1,
                "parameters": grader.PARAMETERS,
                "loss": 1.0,
                "telemetry_step0": {
                    "update_index_zero_based": 0,
                    "preclip_gradient_rms": {family: 0.1 for family in ("inp", "gru", "head")},
                    "parameter_update_rms": {family: 0.01 for family in ("inp", "gru", "head")},
                },
            }
            for shape in grader.SHAPES
        ],
    }


class E482GraderTests(unittest.TestCase):
    """Exercise provenance gates without loading model data or provider outputs."""

    def test_resource_receipt_requires_no_oom_and_distinct_t4s(self) -> None:
        resources = resource_fixture()
        valid = grader.validate_supervision(resources, "synthetic probe", initial=True)
        self.assertEqual(set(valid["uuids"]), {"GPU-0", "GPU-1"})
        self.assertLess(valid["start_host_used_fraction"], grader.HOST_STOP_FRACTION)
        failed = resource_fixture()
        failed["resource_end"]["memory_events"] = {
            "oom": 1,
            "oom_kill": 0,
            "oom_group_kill": 0,
        }
        with self.assertRaisesRegex(ValueError, "OOM counter"):
            grader.validate_supervision(failed, "synthetic OOM")
        collided = resource_fixture()
        collided["resource_start"]["devices"][1]["uuid"] = "GPU-0"
        with self.assertRaisesRegex(ValueError, "invalid/used T4"):
            grader.validate_supervision(collided, "synthetic GPU collision")

    def test_probe_receipt_requires_all_three_cells_and_ieee_parity(self) -> None:
        sources = {"worker": "a" * 64}
        receipt = probe_fixture("adamw", "GPU-0", sources)
        cells = grader.validate_probe_receipt(
            receipt,
            "adamw",
            sources,
            {"uuid": "GPU-0", "name": "Tesla T4", "index": 0},
        )
        self.assertEqual(len(cells), 3)
        receipt["arms"].pop()
        with self.assertRaisesRegex(ValueError, "omits registered shapes"):
            grader.validate_probe_receipt(
                receipt,
                "adamw",
                sources,
                {"uuid": "GPU-0", "name": "Tesla T4", "index": 0},
            )

    def test_synthetic_fit_triplet_checks_order_telemetry_and_checkpoint_state(self) -> None:
        nodes = grader.canonical_fit_nodes()
        order = np.arange(24)
        np.random.default_rng(grader.SEED).shuffle(order)
        row_order = [nodes[int(index)]["row_group"] for index in order]
        sources = grader.expected_sources()
        optimizer = "adamw"
        fit_manifest = {
            "cohort": nodes,
            "normalization": {"fit_mean": [0.0] * 42, "fit_scale": [1.0] * 42},
        }
        with tempfile.TemporaryDirectory(dir=HERE) as temporary:
            run = Path(temporary)
            arms = []
            for shape in grader.SHAPES:
                model = e481grade.CurvatureModel(shape)
                checkpoint_name = f"seed{grader.SEED}_{optimizer}_{shape}.pt"
                checkpoint = run / "fits" / optimizer / checkpoint_name
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), checkpoint)
                samples = []
                for step in grader.TELEMETRY_STEPS:
                    batch = step // 79
                    samples.append(
                        {
                            "update_index_zero_based": step,
                            "fit_row_groups": row_order[batch * 4 : (batch + 1) * 4],
                            "chunk_start": (step % 79) * grader.CHUNK_ROWS,
                            "loss": 1.0,
                            "state_abs_max": 0.9,
                            "preclip_gradient_rms": {
                                family: 0.1 for family in ("inp", "gru", "head")
                            },
                            "parameter_update_rms": {
                                family: 0.01 for family in ("inp", "gru", "head")
                            },
                        }
                    )
                arms.append(
                    {
                        "seed": grader.SEED,
                        "optimizer": optimizer,
                        "shape": shape,
                        "updates": grader.UPDATES,
                        "parameters": grader.PARAMETERS,
                        "fit_row_groups": row_order,
                        "telemetry": samples,
                        "fit_seconds": 10.0,
                        "checkpoint": checkpoint_name,
                        "checkpoint_sha256": grader.file_sha256(checkpoint),
                    }
                )
            receipt = {
                "status": "three_fit_arms_complete_test_unread",
                "phase": "fit",
                "fit_only": True,
                "test_rows_loaded": False,
                "planned_fit_count": 3,
                "batch": 4,
                "chunk_rows": grader.CHUNK_ROWS,
                "updates_per_arm": grader.UPDATES,
                "optimizer": optimizer,
                "seed": grader.SEED,
                "dataset_sha256": grader.DATASET_SHA256,
                "cohort_sha256": grader.COHORT_SHA256,
                "source_sha256": sources,
                "assigned_device": {"uuid": "GPU-0", "name": "Tesla T4"},
                "fit_only_cohort": fit_manifest,
                "parent_source_sha256": {"e470": grader.E470_SHA256, "e474": grader.E474_SHA256},
                "architecture": "E481 frozen; 42 features, two-layer H64 GRU, 61,058 parameters",
                "arms": arms,
                "precision": {"device_math": "FP32 IEEE", "autocast": False},
            }
            cells = grader.validate_fit_receipt(
                receipt,
                optimizer,
                sources,
                {"uuid": "GPU-0", "name": "Tesla T4", "index": 0},
                run,
                nodes,
            )
            self.assertEqual(len(cells), 3)
            malformed_path = run / "fits" / optimizer / arms[0]["checkpoint"]
            malformed_state = torch.load(malformed_path, map_location="cpu", weights_only=True)
            malformed_state["head.2.weight"] = malformed_state["head.2.weight"].transpose(0, 1).contiguous()
            torch.save(malformed_state, malformed_path)
            malformed_arm = dict(arms[0])
            malformed_arm["checkpoint_sha256"] = grader.file_sha256(malformed_path)
            with self.assertRaisesRegex(ValueError, "tensor shapes"):
                grader.validate_checkpoint(malformed_path, malformed_arm, optimizer, "baseline")
            receipt["arms"][0]["fit_row_groups"] = list(reversed(row_order))
            with self.assertRaisesRegex(ValueError, "traversal"):
                grader.validate_fit_receipt(
                    receipt,
                    optimizer,
                    sources,
                    {"uuid": "GPU-0", "name": "Tesla T4", "index": 0},
                    run,
                    nodes,
                )

    def test_provider_proof_binds_status_timestamps_and_download_hashes(self) -> None:
        packet = {
            "build_receipt": "1" * 64,
            "bundle_manifest": "2" * 64,
            "bundle_archive": "3" * 64,
            "kernel_source": grader.file_sha256(
                HERE / "provider_metadata_v1/wnn-e482-t4-sixfit-20261005-v2-runner.py"
            ),
            "kernel_metadata": grader.file_sha256(
                grader.PACKET / "bundle/kernel/kernel-metadata.json"
            ),
            "worker": "6" * 64,
        }
        with tempfile.TemporaryDirectory(dir=HERE) as temporary:
            root = Path(temporary)
            relative_outputs = [entry["path"] for entry in json.loads(
                (HERE / "provider-proof.json").read_text()
            )["outputs"]]
            original_root = grader.ROOT
            source = original_root / "research/campaign_2026-10-05/activation_t4_replication_v2/provider_metadata_v1/wnn-e482-t4-sixfit-20261005-v2-runner.py"
            metadata_source = original_root / "research/campaign_2026-10-05/activation_t4_replication_v2/provider_metadata_v1/kernel-metadata.json"
            run = root / "research/campaign_2026-10-05/activation_t4_replication_v2/remote_v1"
            run.mkdir(parents=True)
            observed = datetime(2026, 10, 5, 6, 0, tzinfo=UTC)
            for relative in relative_outputs:
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                if relative.endswith("provider_complete_status_v1.json"):
                    payload = {
                        "kernel_ref": grader.KERNEL_REF,
                        "status": "COMPLETE",
                        "exit_code": 0,
                        "observed_utc": observed.isoformat(),
                        "stdout": "synthetic status observation",
                    }
                    path.write_text(json.dumps(payload))
                elif relative.endswith("kernel-metadata.json"):
                    path.write_bytes(metadata_source.read_bytes())
                elif relative.endswith("wnn-e482-t4-sixfit-20261005-v2-runner.py"):
                    path.write_bytes(source.read_bytes())
                else:
                    path.write_text("synthetic provider artifact")
            proof = {
                "kernel_ref": grader.KERNEL_REF,
                "status": "COMPLETE",
                "private": True,
                "bundle_dataset": grader.DATASET_SLUG,
                "input_dataset": grader.INPUT_DATASET,
                "submitted_utc": "2026-10-05T05:00:00Z",
                "observed_terminal_utc": observed.isoformat(),
                "status_observation_file": "research/campaign_2026-10-05/activation_t4_replication_v2/provider_complete_status_v1.json",
                "provider_metadata_file": "research/campaign_2026-10-05/activation_t4_replication_v2/provider_metadata_v1/kernel-metadata.json",
                "download_root": "research/campaign_2026-10-05/activation_t4_replication_v2/remote_v1",
                "outputs": [],
            }
            proof_path = root / "research/campaign_2026-10-05/activation_t4_replication_v2/provider-proof.json"
            proof_path.parent.mkdir(parents=True, exist_ok=True)
            with patch.object(grader, "ROOT", root):
                proof["outputs"] = [
                    {
                        "path": relative,
                        "size": (root / relative).stat().st_size,
                        "sha256": grader.file_sha256(root / relative),
                    }
                    for relative in relative_outputs
                ]
                proof_path.write_text(json.dumps(proof))
                result = grader.validate_provider_proof(proof_path, run, packet)
                self.assertEqual(result["kernel_status"], "COMPLETE")
                metadata_path = root / proof["provider_metadata_file"]
                normalized = json.loads(metadata_path.read_text())
                normalized["keywords"] = ["unregistered"]
                metadata_path.write_text(json.dumps(normalized))
                for item in proof["outputs"]:
                    if item["path"] == proof["provider_metadata_file"]:
                        item["size"] = metadata_path.stat().st_size
                        item["sha256"] = grader.file_sha256(metadata_path)
                proof_path.write_text(json.dumps(proof))
                with self.assertRaisesRegex(ValueError, "unregistered source changes"):
                    grader.validate_provider_proof(proof_path, run, packet)
                proof["status"] = "ERROR"
                proof_path.write_text(json.dumps(proof))
                with self.assertRaisesRegex(ValueError, "successful private E482"):
                    grader.validate_provider_proof(proof_path, run, packet)

    def test_registered_e482_contrast_inventory_is_ten(self) -> None:
        scores = {
            f"{optimizer}__{shape}": float(i)
            for i, (optimizer, shape) in enumerate(
                (optimizer, shape) for optimizer in grader.OPTIMIZERS for shape in grader.SHAPES
            )
        }
        draws = {
            key: np.linspace(value - 0.1, value + 0.1, grader.BOOTSTRAP_COUNT)
            for key, value in scores.items()
        }
        contrasts = e481grade.registered_contrasts(scores, draws)
        self.assertEqual(len(contrasts), 10)
        self.assertEqual(
            sum(item["family"] == "shape_within_optimizer" for item in contrasts.values()), 6
        )
        self.assertEqual(
            sum(item["family"] == "optimizer_within_shape" for item in contrasts.values()), 3
        )
        self.assertEqual(
            sum(item["family"] == "shape_by_optimizer_interaction" for item in contrasts.values()),
            1,
        )


if __name__ == "__main__":
    unittest.main()
