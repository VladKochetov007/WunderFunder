from __future__ import annotations

import ast
import base64
import hashlib
import importlib.util
import json
import math
import unittest
import zlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("prepare_packet", HERE / "prepare_packet.py")
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("packet builder cannot be imported")
prepare_packet = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prepare_packet)


class PacketRecipeTests(unittest.TestCase):
    """Protect the sealed command, trainer pin, and upload boundary."""

    def test_training_recipe_is_fixed_and_profile_only_adds_timing(self) -> None:
        fit = prepare_packet.training_args("e508_raw_s3_f0", 3)
        profile = prepare_packet.training_args("e508_raw_profile_s3_f0", 3, profile=True)
        self.assertIn("--raw-only", fit)
        self.assertIn("--no-eval-during-fit", fit)
        self.assertEqual(fit[fit.index("--checkpoint-epoch") + 1], "3")
        self.assertEqual(fit[fit.index("--epochs") + 1], "4")
        self.assertEqual(fit[fit.index("--hidden") + 1], "128")
        self.assertEqual(fit[fit.index("--seed") + 1], "3")
        self.assertEqual(profile[-4:], ["--profile-warmup", "1", "--profile-steps", "3"])
        self.assertEqual(fit.count("--fast"), 1)
        self.assertEqual(fit.count("--distill"), 1)
        self.assertEqual(fit[fit.index("--distill") + 1], "")

    def test_runtime_recipe_matches_the_review_receipt_commands(self) -> None:
        source = (HERE / "packet_worker.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        definitions = {
            node.name: node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name in {"training_argv", "profile_estimate_seconds"}
        }
        namespace: dict[str, object] = {
            "SEEDS": (3, 4),
            "EPOCHS": 4,
            "CHECKPOINT_EPOCH": 3,
            "SPLIT_RELATIVE": Path("research/campaign_2026-09-29/artifacts/splits/fold0.json"),
            "CHUNK": 2_000,
            "FIT_COUNTS": {"train": 10_535, "valid": 1_405},
            "BATCH": 16,
            "TARGET_ORDERS": {"train": 10_606, "valid": 1_873},
            "math": math,
        }
        module = ast.Module(body=list(definitions.values()), type_ignores=[])
        exec(compile(module, "packet_worker_recipe", "exec"), namespace)
        for seed in (3, 4):
            self.assertEqual(
                namespace["training_argv"](f"e508_raw_s{seed}_f0", seed),
                prepare_packet.training_args(f"e508_raw_s{seed}_f0", seed),
            )
        self.assertEqual(
            namespace["training_argv"]("e508_raw_profile_s3_f0", 3, profile=True),
            prepare_packet.training_args("e508_raw_profile_s3_f0", 3, profile=True),
        )
        self.assertEqual(
            namespace["profile_estimate_seconds"](500.0),
            prepare_packet.profile_estimate_seconds(500.0),
        )

    def test_pinned_e463_manifest_binds_checkpoint_at_exact_logical_path(self) -> None:
        worker_tree = ast.parse((HERE / "packet_worker.py").read_text(encoding="utf-8"))
        constants = {
            target.id: ast.literal_eval(node.value)
            for node in worker_tree.body
            if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Name)
            and target.id
            in {
                "AUDIT_CHECKPOINT_RELATIVE",
                "AUDIT_CHECKPOINT_SHA256",
                "AUDIT_CHECKPOINT_BYTES",
            }
        }
        helper = next(
            node
            for node in worker_tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "audit_checkpoint_manifest_entry"
        )
        namespace: dict[str, object] = dict(constants)
        exec(
            compile(ast.Module(body=[helper], type_ignores=[]), "packet_manifest_audit", "exec"),
            namespace,
        )

        pinned_manifest = (
            HERE.parents[3]
            / "research/campaign_2026-10-04/kaggle/jax_tpu_realdata_preflight_v1"
            / "upload_dataset/source-manifest.json"
        )
        manifest_bytes = pinned_manifest.read_bytes()
        self.assertEqual(
            hashlib.sha256(manifest_bytes).hexdigest(),
            "851898475dfa701f2089ec76cfe9592538aa2e7f9b70ebcb68c0a614e332d6fb",
        )
        manifest = json.loads(manifest_bytes)
        relative = constants["AUDIT_CHECKPOINT_RELATIVE"]
        self.assertEqual(relative, "artifacts/runs/c20261003_combo_s3_f0/model_ep3.pt")
        entry = manifest["files"][relative]
        self.assertEqual(namespace["audit_checkpoint_manifest_entry"](manifest), entry)
        self.assertEqual(entry["sha256"], constants["AUDIT_CHECKPOINT_SHA256"])
        self.assertEqual(entry["size_bytes"], constants["AUDIT_CHECKPOINT_BYTES"])

        wrong_key_manifest = json.loads(manifest_bytes)
        wrong_key_manifest["files"]["source_bundle/" + relative] = wrong_key_manifest[
            "files"
        ].pop(relative)
        with self.assertRaisesRegex(ValueError, "not bound by its source manifest"):
            namespace["audit_checkpoint_manifest_entry"](wrong_key_manifest)

    def test_profile_cost_gate_rejects_nan_and_bounds_pair(self) -> None:
        self.assertAlmostEqual(prepare_packet.profile_estimate_seconds(500.0), 4469.75)
        with self.assertRaises(ValueError):
            prepare_packet.profile_estimate_seconds(math.nan)
        with self.assertRaises(ValueError):
            prepare_packet.profile_estimate_seconds(0.0)

    def test_pinned_archive_contains_exact_trainer_and_valid_split_checker(self) -> None:
        archive, identities = prepare_packet.source_archive()
        self.assertGreater(len(archive), 100_000)
        self.assertEqual(identities["scripts/train_rnn.py"], prepare_packet.TRAINER_SHA256)
        self.assertIn("src/wnn/splits.py", identities)
        self.assertIn("src/wnn/portability.py", identities)

    def test_generated_kernel_is_valid_python_and_keeps_upload_private(self) -> None:
        archive, identities = prepare_packet.source_archive()
        worker = (HERE / "packet_worker.py").read_bytes()
        source = prepare_packet.kernel_source(archive, identities, worker).decode("utf-8")
        ast.parse(source)
        builder = (HERE / "prepare_packet.py").read_text(encoding="utf-8")
        self.assertIn('"is_private": True', builder)
        self.assertIn('"enable_internet": False', builder)
        self.assertIn("start_new_session=True", source)
        self.assertIn("GLOBAL_TIMEOUT_SECONDS = 3600", source)
        self.assertIn("os.set_blocking(output_fd, False)", source)
        self.assertNotIn("process.stdout.readline()", source)
        self.assertIn("child_cleanup_error", source)
        self.assertIn("stop_worker(process)", source)

    def test_prediction_export_projects_features_without_labels_or_masks(self) -> None:
        source = (HERE / "packet_worker.py").read_text(encoding="utf-8")
        self.assertIn("targets=False, mask=False", source)
        self.assertIn('"label_columns_read": False', source)
        self.assertIn('"mask_columns_read": False', source)
        self.assertIn("PROFILE_ONLY = True", source)
        self.assertIn("profile_ready_awaiting_controller_review", source)

    def test_materialized_packet_hashes_and_embedded_worker_match_receipt(self) -> None:
        packet = HERE / "packet_v2"
        receipt = json.loads((packet / "review_packet.json").read_text(encoding="utf-8"))
        kernel = (packet / "kernel.py").read_bytes()
        metadata = (packet / "kernel-metadata.json").read_bytes()
        self.assertEqual(hashlib.sha256(kernel).hexdigest(), receipt["packet_sha256"]["kernel.py"])
        self.assertEqual(receipt["packet_id"], "E508_RAW_BASELINE_V2")
        self.assertEqual(
            receipt["kernel_id"],
            "vladkochetov07/wnn-raw-teacher-pair-e508-20261007-v2",
        )
        self.assertEqual(
            hashlib.sha256(metadata).hexdigest(),
            receipt["packet_sha256"]["kernel-metadata.json"],
        )
        source = kernel.decode("utf-8")
        tree = ast.parse(source)
        assignments = {
            target.id: ast.literal_eval(node.value)
            for node in tree.body
            if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Name)
            and target.id in {"WORKER_CODE", "WORKER_SHA256", "GLOBAL_TIMEOUT_SECONDS"}
        }
        worker = zlib.decompress(base64.b85decode(assignments["WORKER_CODE"].encode("ascii")))
        self.assertEqual(hashlib.sha256(worker).hexdigest(), assignments["WORKER_SHA256"])
        self.assertEqual(
            hashlib.sha256(worker).hexdigest(),
            hashlib.sha256((HERE / "packet_worker.py").read_bytes()).hexdigest(),
        )
        self.assertEqual(assignments["WORKER_SHA256"], receipt["packet_sha256"]["packet_worker.py"])
        self.assertEqual(assignments["GLOBAL_TIMEOUT_SECONDS"], 3_600)
        worker_text = worker.decode("utf-8")
        self.assertIn("PROFILE_ONLY = True", worker_text)
        self.assertIn("profile_ready_awaiting_controller_review", worker_text)
        self.assertIn("os.set_blocking(output_fd, False)", source)
        self.assertIn('"is_private": true', metadata.decode("utf-8"))
        self.assertTrue(
            receipt["commands"]["kaggle_push_command_prepared_not_run"].endswith("-t 4200")
        )
        self.assertEqual(receipt["profile_gate"]["profile_provider_watch_seconds"], 4_200)


if __name__ == "__main__":
    unittest.main()
