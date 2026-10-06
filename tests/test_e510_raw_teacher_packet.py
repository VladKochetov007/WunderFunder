from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import unittest
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]
PACKET_DIR = ROOT / "research/campaign_2026-10-06/bandit_continuation/e510_raw_teacher_pair"
WORKER_PATH = PACKET_DIR / "packet_worker.py"
PREPARE_PATH = PACKET_DIR / "prepare_packet.py"


def load_builder() -> ModuleType:
    """Load the pure packet builder helpers without materializing a packet."""
    spec = importlib.util.spec_from_file_location("e510_prepare_packet", PREPARE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load E510 packet builder")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def profile_receipt() -> dict[str, object]:
    """Return a minimal synthetic, no-fit E508 receipt for gate tests."""
    return {
        "packet": "E508_RAW_BASELINE_V1",
        "status": "profile_ready_awaiting_controller_review",
        "fit_admitted": False,
        "fit_stages_started": [],
        "split_sha256": "09263ad4fe2b18e2b2f3b79f4562678dce801b067b868134ddc79279f9316a3d",
        "fit_group_counts": {"train": 10_535, "valid": 1_405},
        "evaluation_valid_groups": 468,
        "target_row_counts": {"train": 10_606, "valid": 1_873},
        "profile": {
            "metrics": {"step_wall": 12.5},
            "peak_reserved_bytes": 1_500_000_000,
            "estimated_pair_fit_and_export_seconds": 3_000.0,
        },
        "hardware": {
            "cuda_available": True,
            "cuda_devices": [{"name": "Tesla T4", "total_memory": 16_000_000_000}],
        },
    }


def function_namespace(
    path: Path, function_name: str, globals_: dict[str, object]
) -> dict[str, object]:
    """Compile one standalone pure worker helper without importing Torch."""
    tree = ast.parse(path.read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == function_name
    )
    module = ast.Module(body=[function], type_ignores=[])
    namespace = dict(globals_)
    exec(compile(module, str(path), "exec"), namespace)
    return namespace


class E510PacketGateTests(unittest.TestCase):
    """Check profile admission, source mapping, and embedded receipt identity."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.builder = load_builder()
        cls.receipt_bytes = (json.dumps(profile_receipt(), sort_keys=True) + "\n").encode()
        cls.receipt_sha = hashlib.sha256(cls.receipt_bytes).hexdigest()

    def test_profile_receipt_requires_matching_review_hash_and_no_fit_status(self) -> None:
        result = self.builder.validate_profile_receipt(
            self.receipt_bytes,
            expected_sha256=self.receipt_sha,
            controller_reviewed_sha256=self.receipt_sha,
        )
        self.assertEqual(result["packet"], "E508_RAW_BASELINE_V1")
        with self.assertRaisesRegex(ValueError, "controller review"):
            self.builder.validate_profile_receipt(
                self.receipt_bytes,
                expected_sha256=self.receipt_sha,
                controller_reviewed_sha256="0" * 64,
            )

        rejected = profile_receipt()
        rejected["fit_stages_started"] = ["fit_seed_3"]
        changed = (json.dumps(rejected, sort_keys=True) + "\n").encode()
        digest = hashlib.sha256(changed).hexdigest()
        with self.assertRaisesRegex(ValueError, "no-fit profile"):
            self.builder.validate_profile_receipt(
                changed, expected_sha256=digest, controller_reviewed_sha256=digest
            )

    def test_profile_timing_allocator_and_group_counts_fail_closed(self) -> None:
        for mutate, message in (
            (
                lambda row: row["profile"].update(estimated_pair_fit_and_export_seconds=6_301),
                "timing",
            ),
            (lambda row: row["profile"].update(peak_reserved_bytes=4_300_000_001), "timing"),
            (lambda row: row.update(fit_group_counts={"train": 10_534, "valid": 1_405}), "fold-0"),
        ):
            row = profile_receipt()
            mutate(row)
            encoded = (json.dumps(row, sort_keys=True) + "\n").encode()
            digest = hashlib.sha256(encoded).hexdigest()
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                self.builder.validate_profile_receipt(
                    encoded, expected_sha256=digest, controller_reviewed_sha256=digest
                )

    def test_packet_builder_refuses_to_materialize_without_real_profile_receipt(self) -> None:
        with self.assertRaisesRegex(ValueError, "sealed until"):
            self.builder.build_packet()
        self.assertFalse((PACKET_DIR / "packet").exists())

    def test_generated_kernel_embeds_exact_profile_and_family_snapshot_hashes(self) -> None:
        source_families = (PACKET_DIR.parent / "SOURCE_FAMILIES.json").read_bytes()
        kernel_bytes = self.builder.kernel_source(
            b"synthetic-archive",
            {"scripts/train_rnn.py": "a" * 64},
            b"print('synthetic worker')\n",
            self.receipt_bytes,
            self.receipt_sha,
            source_families,
        )
        namespace: dict[str, object] = {"__name__": "e510_kernel_test"}
        exec(compile(kernel_bytes, "kernel.py", "exec"), namespace)
        decode_source = namespace["decode_source"]
        recovered = decode_source(namespace["PROFILE_RECEIPT"], namespace["PROFILE_RECEIPT_SHA256"])
        self.assertEqual(recovered, self.receipt_bytes)
        self.assertEqual(namespace["PROFILE_RECEIPT_SHA256"], self.receipt_sha)
        self.assertIn(
            '"WNN_E508_PROFILE_REVIEWED_SHA256": PROFILE_RECEIPT_SHA256', kernel_bytes.decode()
        )

    def test_source_manifest_checkpoint_mapping_uses_exact_logical_path(self) -> None:
        namespace = function_namespace(
            WORKER_PATH,
            "validate_audit_checkpoint_entry",
            {
                "AUDIT_CHECKPOINT_SHA256": "d4a1e50b1892b4d9edd7c255908b68073a760c059ca80223e98a63c7e023371e"
            },
        )
        validate_entry = namespace["validate_audit_checkpoint_entry"]
        accepted = {
            "files": {
                "artifacts/runs/c20261003_combo_s3_f0/model_ep3.pt": {
                    "sha256": "d4a1e50b1892b4d9edd7c255908b68073a760c059ca80223e98a63c7e023371e",
                    "size_bytes": 3_977_183,
                }
            }
        }
        self.assertEqual(validate_entry(accepted)["size_bytes"], 3_977_183)
        for files in (
            {
                "source_bundle/artifacts/runs/c20261003_combo_s3_f0/model_ep3.pt": accepted[
                    "files"
                ]["artifacts/runs/c20261003_combo_s3_f0/model_ep3.pt"]
            },
            {
                "artifacts/runs/c20261003_combo_s3_f0/model_ep3.pt": {
                    **accepted["files"]["artifacts/runs/c20261003_combo_s3_f0/model_ep3.pt"],
                    "sha256": "f" * 64,
                }
            },
        ):
            with self.subTest(files=files), self.assertRaisesRegex(ValueError, "not bound"):
                validate_entry({"files": files})

    def test_worker_checks_profile_hash_before_input_inventory(self) -> None:
        tree = ast.parse(WORKER_PATH.read_text())
        main = next(
            node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main"
        )
        calls: list[str] = []
        for node in ast.walk(main):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                calls.append(node.func.id)
        self.assertIn("validate_external_profile_receipt", calls)
        self.assertIn("input_map", calls)
        main_source = ast.get_source_segment(WORKER_PATH.read_text(), main)
        self.assertLess(
            main_source.index("validate_external_profile_receipt()"),
            main_source.index('mark_phase("input_inventory")'),
        )


if __name__ == "__main__":
    unittest.main()
