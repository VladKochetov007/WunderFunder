from __future__ import annotations

import importlib.util
import json
import shutil
import tempfile
import unittest
from pathlib import Path

MAIN_ROOT = Path("/home/vlad/development/wunder/wnn_connectome_starterpack")
BUILDER_PATH = Path(__file__).with_name("build_r9_transport.py")


def load_builder():
    spec = importlib.util.spec_from_file_location("build_r9_transport_under_test", BUILDER_PATH)
    if spec is None or spec.loader is None:
        raise AssertionError(f"could not import {BUILDER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


build_r9_transport = load_builder()


def import_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"could not import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class R9TransportTests(unittest.TestCase):
    def test_five_export_packet_pending_gate_and_real_checkpoint_mount(self) -> None:
        with tempfile.TemporaryDirectory(prefix="wnn-r9-test-") as temporary:
            root = Path(temporary)
            packet = root / "packet"
            output = packet / "kernel"
            stage_parent = root / "dataset_upload"
            receipt_path = packet / "R9_BUILD_RECEIPT_R1.json"
            result = build_r9_transport.build(MAIN_ROOT, output, stage_parent, receipt_path)
            self.assertEqual(result["status"], "pending_root_release")
            self.assertEqual(len(result["selected_exports"]), 5)
            self.assertEqual(result["expected_prediction_npy_bytes"], 4_045_600_000)
            metadata = json.loads((output / "kernel-metadata.json").read_text())
            self.assertEqual(metadata["id"], build_r9_transport.R9_ID)
            self.assertEqual(metadata["title"], build_r9_transport.R9_TITLE)
            self.assertEqual(metadata["kernel_sources"], [])
            self.assertIn(build_r9_transport.R9_DATASET_ID, metadata["dataset_sources"])

            wrapper = import_module("r9_wrapper_pending", output / "kernel.py")
            worker_bytes, binding_bytes, pending_bytes = wrapper._decode_embedded()
            with self.assertRaisesRegex(PermissionError, "pending"):
                wrapper._validate_release(binding_bytes, pending_bytes)
            control_dir = root / "working" / "control"
            with self.assertRaisesRegex(PermissionError, "pending"):
                wrapper.run_embedded(control_dir)
            self.assertFalse(control_dir.exists())

            worker_path = root / "worker.py"
            worker_path.write_bytes(worker_bytes)
            worker = import_module("r9_worker_under_test", worker_path)
            bindings = json.loads(binding_bytes)
            source_root = worker.find_source_mount(stage_parent)
            manifest = worker.validate_checkpoint_input_dataset(source_root, bindings)
            self.assertEqual(manifest["dataset_id"], build_r9_transport.R9_DATASET_ID)
            self.assertEqual(len(manifest["files"]), 6)
            for run_name, expected in bindings["checkpoints"].items():
                checkpoint, provenance = worker.canonical_source_artifacts(source_root, run_name)
                worker.sha_member(
                    checkpoint, expected["model_ep2_bytes"], expected["model_ep2_sha256"]
                )
                worker.sha_member(
                    provenance, expected["provenance_bytes"], expected["provenance_sha256"]
                )
                worker.validate_provenance(provenance, run_name, expected)

            admitted = json.loads(pending_bytes)
            admitted.update(status="admitted", authorized=True)
            admitted["binding"] = bindings["root_release_binding"]
            admitted["logical_recipe_target_kernel_id"] = bindings["root_release_binding"][
                "target_kernel_id"
            ]
            admitted["transport_target_kernel_id"] = build_r9_transport.R9_ID
            validated = wrapper._validate_release(
                binding_bytes, build_r9_transport.canonical_json(admitted)
            )
            self.assertEqual(validated["root_release_binding"], bindings["root_release_binding"])
            wrong_target = dict(admitted, transport_target_kernel_id="vladkochetov07/wrong-target")
            with self.assertRaisesRegex(PermissionError, "R9 transport"):
                wrapper._validate_release(
                    binding_bytes, build_r9_transport.canonical_json(wrong_target)
                )

            bad_parent = root / "bad_input"
            shutil.copytree(stage_parent, bad_parent)
            (bad_source,) = list(bad_parent.glob(f"**/{build_r9_transport.R9_DATASET_SLUG}"))
            bad_checkpoint = bad_source / next(
                row["path"] for row in manifest["files"] if row["type"] == "checkpoint"
            )
            with bad_checkpoint.open("ab") as changed:
                changed.write(b"x")
            with self.assertRaisesRegex(ValueError, "changed|manifest"):
                worker.validate_checkpoint_input_dataset(bad_source, bindings)


if __name__ == "__main__":
    unittest.main()
