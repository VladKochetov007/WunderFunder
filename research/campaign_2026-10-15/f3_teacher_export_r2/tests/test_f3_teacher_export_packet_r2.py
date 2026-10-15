from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import collect_terminal_r2 as collector
import prepare_export_packet_r2 as builder


class FakeHelper:
    @staticmethod
    def relative_result_path(name: str) -> PurePosixPath:
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or not name.startswith("results/"):
            raise ValueError("unsafe results path")
        return path


class CollectionInventoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.helper = FakeHelper()
        self.files = [
            SimpleNamespace(file_name=name, url=f"https://storage.invalid/{index}")
            for index, name in enumerate(sorted(collector.expected_remote_paths()))
        ]

    def test_exact_complete_inventory_is_bounded(self) -> None:
        with patch.object(collector, "declared_length", return_value=1024):
            rows, missing = collector.inventory(self.files, 2048, self.helper, complete=True)
        self.assertEqual(len(rows), 14)
        self.assertEqual(missing, [])
        self.assertEqual({row["path"] for row in rows}, collector.expected_remote_paths())

    def test_terminal_partial_inventory_is_preserved_as_missing(self) -> None:
        partial = self.files[:-1]
        with patch.object(collector, "declared_length", return_value=256):
            rows, missing = collector.inventory(partial, 512, self.helper, complete=True)
        self.assertEqual(len(rows), 13)
        self.assertEqual(len(missing), 1)

    def test_unexpected_path_fails_before_size_requests(self) -> None:
        unexpected = [SimpleNamespace(file_name="results/unexpected.bin", url="https://invalid")]
        with (
            patch.object(collector, "declared_length", side_effect=AssertionError),
            self.assertRaisesRegex(ValueError, "undeclared"),
        ):
            collector.inventory(unexpected, 64, self.helper, complete=False)

    def test_inventory_cap_is_enforced_before_download(self) -> None:
        with (
            patch.object(collector, "declared_length", return_value=collector.MAX_BYTES),
            self.assertRaisesRegex(ValueError, "2.5 GB"),
        ):
            collector.inventory([self.files[0]], 1, self.helper, complete=False)


class KernelDeadlineTests(unittest.TestCase):
    def test_outer_cap_covers_worker_and_declared_reserve(self) -> None:
        self.assertEqual(builder.WORKER_SECONDS, 18_000)
        self.assertEqual(builder.KERNEL_PROCESS_WALL_SECONDS_MAX, 21_600)
        self.assertEqual(
            builder.KERNEL_PROCESS_WALL_SECONDS_MAX - builder.WORKER_SECONDS,
            3_600,
        )

    def test_generated_bootstrap_is_lowered_once_and_idempotently(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory)
            kernel = bundle / "kernel"
            kernel.mkdir()
            source = kernel / "kernel.py"
            source.write_text("KERNEL_PROCESS_WALL_SECONDS_MAX = 43_200\n")

            builder.enforce_kernel_wall_cap(bundle)
            self.assertEqual(source.read_text(), "KERNEL_PROCESS_WALL_SECONDS_MAX = 21_600\n")
            builder.enforce_kernel_wall_cap(bundle)
            self.assertEqual(source.read_text(), "KERNEL_PROCESS_WALL_SECONDS_MAX = 21_600\n")

    def test_release_receipts_reject_the_r1_outer_deadline(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            packet = root / collector.PACKET
            packet.mkdir(parents=True)
            binding = {
                "bundle": {
                    "archive_sha256": "a" * 64,
                    "kernel_metadata_sha256": "b" * 64,
                    "kernel_process_wall_seconds_max": 21_600,
                }
            }
            admission = {
                "status": "admitted",
                "kernel_id": collector.KERNEL_ID,
                "bundle_archive_sha256": "a" * 64,
                "kernel_process_wall_seconds_max": 21_600,
                "worker_timeout_seconds": 18_000,
            }
            push = {
                "kernel_id": collector.KERNEL_ID,
                "kernel_version": 1,
                "bundle_archive_sha256": "a" * 64,
                "kernel_metadata_sha256": "b" * 64,
                "private": True,
            }
            (packet / "SOURCE_BINDING_R2.json").write_text(json.dumps(binding))
            (packet / "ROOT_ADMISSION_R2.json").write_text(json.dumps(admission))
            (packet / "KERNEL_PUSH_R2.json").write_text(json.dumps(push))

            receipt = collector.verify_release_receipts(root)
            self.assertEqual(receipt["kernel_process_wall_seconds_max"], 21_600)
            admission["kernel_process_wall_seconds_max"] = 43_200
            (packet / "ROOT_ADMISSION_R2.json").write_text(json.dumps(admission))
            with self.assertRaisesRegex(ValueError, "do not bind"):
                collector.verify_release_receipts(root)


class CollectedPayloadTests(unittest.TestCase):
    def _make_payload(self, root: Path, *, nonfinite: bool) -> tuple[Path, list[dict[str, object]]]:
        import hashlib

        import numpy as np

        stages = [{**stage, "rows": 1} for stage in collector.PREDICTION_STAGES]
        sequence_length = 2
        downloaded: list[dict[str, object]] = []
        manifest: list[dict[str, object]] = []
        pipeline_stages: list[dict[str, object]] = []
        total_bytes = 0

        def save_bytes(path: Path, raw: bytes) -> tuple[int, str]:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
            return len(raw), hashlib.sha256(raw).hexdigest()

        def record_artifact(relative: str, path: Path) -> None:
            size = path.stat().st_size
            digest = collector.sha256_file(path)
            manifest.append({"path": relative, "size": size, "sha256": digest})
            remote = f"results/{collector.JOB_NAME}/artifacts/{relative}"
            downloaded.append({"path": remote, "size_bytes": size, "sha256": digest})

        for index, stage in enumerate(stages):
            array_relative = stage["array"]
            sidecar_relative = str(Path(array_relative).with_suffix(".json"))
            remote_array = f"results/{collector.JOB_NAME}/artifacts/{array_relative}"
            array_path = root / remote_array
            values = np.full((1, sequence_length, 2), index + 0.5, dtype=np.float32)
            if nonfinite and index == 0:
                values[0, 0, 0] = np.nan
            array_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(array_path, values, allow_pickle=False)
            array_bytes = array_path.stat().st_size
            array_hash = collector.sha256_file(array_path)
            sidecar = {
                "run": collector.OWNER_RUN,
                "split": stage["split"],
                "epoch": 2,
                "checkpoint_sha256": collector.CHECKPOINT_SHA256,
                "split_manifest_sha256": collector.SPLIT_SHA256,
                "inference_precision": "fp32",
                "campaign_fit_scope": stage["scope"],
                "row_groups": [index + 1],
                "prediction_sha256": array_hash,
            }
            sidecar_path = root / f"results/{collector.JOB_NAME}/artifacts/{sidecar_relative}"
            sidecar_path.parent.mkdir(parents=True, exist_ok=True)
            sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")
            record_artifact(array_relative, array_path)
            record_artifact(sidecar_relative, sidecar_path)
            finite_values = int(np.isfinite(values).size)
            pipeline_stages.append(
                {
                    "name": stage["name"],
                    "status": "complete",
                    "array_path": array_relative,
                    "sidecar_path": sidecar_relative,
                    "array_sha256": array_hash,
                    "array_bytes": array_bytes,
                    "sidecar_bytes": sidecar_path.stat().st_size,
                    "shape": [1, sequence_length, 2],
                    "dtype": "float32",
                    "finite_values": finite_values,
                    "row_group_count": 1,
                }
            )
            total_bytes += array_bytes

        pipeline = {
            "schema": "wnn.f3-teacher-cp2-export-pipeline.r2",
            "status": "complete",
            "run": collector.OWNER_RUN,
            "checkpoint_epoch": 2,
            "checkpoint_sha256": collector.CHECKPOINT_SHA256,
            "provenance_sha256": collector.PROVENANCE_SHA256,
            "split_manifest_sha256": collector.SPLIT_SHA256,
            "source_wrapper_sha256": collector.WRAPPER_SHA256,
            "source_predictor_sha256": collector.PREDICTOR_SHA256,
            "stages": pipeline_stages,
            "prediction_payload_bytes": total_bytes,
        }
        pipeline_relative = "results/f3_teacher_export_pipeline_r2.json"
        pipeline_path = root / f"results/{collector.JOB_NAME}/artifacts/{pipeline_relative}"
        _, _ = save_bytes(pipeline_path, json.dumps(pipeline).encode("utf-8"))
        record_artifact(pipeline_relative, pipeline_path)

        result_path = root / "results" / collector.JOB_NAME / "result.json"
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(
            json.dumps(
                {
                    "name": collector.JOB_NAME,
                    "status": "complete",
                    "returncode": 0,
                    "artifacts": {"files": manifest, "missing": []},
                }
            ),
            encoding="utf-8",
        )
        timeout_path = result_path.with_name("timeout_resolution.json")
        timeout_path.write_text(
            json.dumps(
                {
                    "status": "resolved_before_worker_launch",
                    "resolved_worker_timeout_seconds": 18_000,
                    "kernel_deadline_monotonic": 1000.0,
                    "finalization_reserve_seconds": 300,
                }
            ),
            encoding="utf-8",
        )
        return root, downloaded

    def _validate_fixture(self, *, nonfinite: bool) -> dict[str, object]:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stages = [{**stage, "rows": 1} for stage in collector.PREDICTION_STAGES]
            with (
                patch.object(collector, "PREDICTION_STAGES", stages),
                patch.object(collector, "SEQUENCE_LENGTH", 2),
            ):
                output, downloaded = self._make_payload(root, nonfinite=nonfinite)
                return collector.validate_collected_payload(output, downloaded)

    def test_complete_downloaded_triplet_passes_hash_geometry_and_finiteness(self) -> None:
        receipt = self._validate_fixture(nonfinite=False)
        self.assertEqual(receipt["status"], "passed")
        self.assertEqual(len(receipt["stages"]), 3)

    def test_complete_downloaded_triplet_rejects_nonfinite_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-finite prediction values"):
            self._validate_fixture(nonfinite=True)


if __name__ == "__main__":
    unittest.main()
