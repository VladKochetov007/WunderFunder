from __future__ import annotations

import ast
import base64
import hashlib
import importlib.util
import json
import math
import os
import tempfile
import unittest
import zlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("prepare_packet", HERE / "prepare_packet.py")
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("packet builder cannot be imported")
prepare_packet = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prepare_packet)
SPEC_V5 = importlib.util.spec_from_file_location("prepare_packet_v5", HERE / "prepare_packet_v5.py")
if SPEC_V5 is None or SPEC_V5.loader is None:
    raise RuntimeError("v5 packet builder cannot be imported")
prepare_packet_v5 = importlib.util.module_from_spec(SPEC_V5)
SPEC_V5.loader.exec_module(prepare_packet_v5)


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

    def test_v5_uses_the_trainer_supported_cuda_selector(self) -> None:
        fit = prepare_packet_v5.training_args("e508_raw_s3_f0", 3)
        profile = prepare_packet_v5.training_args("e508_raw_profile_s3_f0", 3, profile=True)
        self.assertEqual(fit[fit.index("--device") + 1], "cuda")
        self.assertEqual(profile[profile.index("--device") + 1], "cuda")
        self.assertNotIn("cuda:0", fit)

        source = (HERE / "packet_worker_v5.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        training_argv = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "training_argv"
        )
        namespace: dict[str, object] = {
            "SEEDS": (3, 4),
            "EPOCHS": 4,
            "CHECKPOINT_EPOCH": 3,
            "SPLIT_RELATIVE": Path("research/campaign_2026-09-29/artifacts/splits/fold0.json"),
            "CHUNK": 2_000,
            "BATCH": 16,
            "math": math,
        }
        exec(
            compile(ast.Module(body=[training_argv], type_ignores=[]), "v5_training_argv", "exec"),
            namespace,
        )
        self.assertEqual(namespace["training_argv"]("e508_raw_s3_f0", 3), fit)

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

    def test_corrupt_group_guard_uses_train_table_namespace_only(self) -> None:
        worker_tree = ast.parse((HERE / "packet_worker.py").read_text(encoding="utf-8"))
        constants = {
            target.id: ast.literal_eval(node.value)
            for node in worker_tree.body
            if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Name)
            and target.id in {"CORRUPT_TRAIN_GROUP", "FIT_COUNTS", "EVAL_VALID_COUNT"}
        }
        helper = next(
            node
            for node in worker_tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "exact_group_sets"
        )
        namespace: dict[str, object] = {**constants, "set": set, "map": map, "int": int}
        exec(
            compile(ast.Module(body=[helper], type_ignores=[]), "exact_group_sets", "exec"),
            namespace,
        )

        train = set(range(10_536)) - {528}
        valid = set(range(1_405))  # Valid-table row group 528 is a separate group.
        evaluation = set(range(20_000, 20_468))
        groups = type(
            "Groups",
            (),
            {
                "train_groups": train,
                "valid_groups": valid,
                "evaluation_groups": evaluation,
            },
        )()
        result = namespace["exact_group_sets"](groups)
        self.assertNotIn(528, result["train"])
        self.assertIn(528, result["valid"])

        invalid_train = set(train)
        invalid_train.remove(10_535)
        invalid_train.add(528)
        groups.train_groups = invalid_train
        with self.assertRaisesRegex(
            ValueError, "corrupt train row group entered train fit membership"
        ):
            namespace["exact_group_sets"](groups)

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
            if isinstance(node, ast.FunctionDef) and node.name == "audit_checkpoint_manifest_entry"
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
        wrong_key_manifest["files"]["source_bundle/" + relative] = wrong_key_manifest["files"].pop(
            relative
        )
        with self.assertRaisesRegex(ValueError, "not bound by its source manifest"):
            namespace["audit_checkpoint_manifest_entry"](wrong_key_manifest)

    def test_parquet_footer_identity_hashes_metadata_and_eight_byte_trailer(self) -> None:
        worker_tree = ast.parse((HERE / "packet_worker.py").read_text(encoding="utf-8"))
        helper = next(
            node
            for node in worker_tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "parquet_footer_identity"
        )
        namespace: dict[str, object] = {"hashlib": hashlib, "os": os, "Path": Path}
        exec(
            compile(ast.Module(body=[helper], type_ignores=[]), "parquet_footer_identity", "exec"),
            namespace,
        )
        prefix = b"parquet-data-prefix"
        footer = b"serialized row-group and schema metadata"
        trailer = len(footer).to_bytes(4, "little") + b"PAR1"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.parquet"
            path.write_bytes(prefix + footer + trailer)
            actual = namespace["parquet_footer_identity"](path)
            self.assertEqual(actual["footer_bytes"], len(footer))
            self.assertEqual(actual["footer_range_bytes_including_trailer"], len(footer) + 8)
            self.assertEqual(actual["footer_range_start_offset"], len(prefix))
            self.assertEqual(actual["trailer_magic"], "PAR1")
            self.assertEqual(actual["footer_sha256"], hashlib.sha256(footer + trailer).hexdigest())
            path.write_bytes(prefix + footer + len(footer).to_bytes(4, "little") + b"BAD!")
            with self.assertRaisesRegex(ValueError, "invalid parquet trailer"):
                namespace["parquet_footer_identity"](path)

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

    def test_v4_generated_kernel_binds_current_split_guard_worker(self) -> None:
        archive, identities = prepare_packet.source_archive()
        worker = (HERE / "packet_worker.py").read_bytes()
        source = prepare_packet.kernel_source(archive, identities, worker).decode("utf-8")
        tree = ast.parse(source)
        assignments = {
            target.id: ast.literal_eval(node.value)
            for node in tree.body
            if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Name)
            and target.id in {"PACKET_ID", "WORKER_CODE", "WORKER_SHA256"}
        }
        embedded = zlib.decompress(base64.b85decode(assignments["WORKER_CODE"].encode("ascii")))
        self.assertEqual(assignments["PACKET_ID"], "E508_RAW_BASELINE_V4")
        self.assertEqual(hashlib.sha256(embedded).hexdigest(), assignments["WORKER_SHA256"])
        self.assertEqual(embedded, worker)

    def test_prediction_export_projects_features_without_labels_or_masks(self) -> None:
        source = (HERE / "packet_worker.py").read_text(encoding="utf-8")
        self.assertIn("targets=False, mask=False", source)
        self.assertIn('"label_columns_read": False', source)
        self.assertIn('"mask_columns_read": False', source)
        self.assertIn("PROFILE_ONLY = True", source)
        self.assertIn("profile_ready_awaiting_controller_review", source)

    def test_materialized_packet_hashes_and_embedded_worker_match_receipt(self) -> None:
        packet = HERE / "packet_v3"
        receipt = json.loads((packet / "review_packet.json").read_text(encoding="utf-8"))
        kernel = (packet / "kernel.py").read_bytes()
        metadata = (packet / "kernel-metadata.json").read_bytes()
        self.assertEqual(hashlib.sha256(kernel).hexdigest(), receipt["packet_sha256"]["kernel.py"])
        self.assertEqual(receipt["packet_id"], "E508_RAW_BASELINE_V3")
        self.assertEqual(
            receipt["kernel_id"],
            "vladkochetov07/wnn-raw-teacher-pair-e508-20261007-v3",
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
        self.assertEqual(assignments["WORKER_SHA256"], receipt["packet_sha256"]["packet_worker.py"])
        self.assertEqual(assignments["GLOBAL_TIMEOUT_SECONDS"], 3_600)
        worker_text = worker.decode("utf-8")
        self.assertIn("footer_range_bytes_including_trailer", worker_text)
        self.assertIn("full_file_sha256_matches", worker_text)
        self.assertIn("input_identity.json", worker_text)
        self.assertIn("PROFILE_ONLY = True", worker_text)
        self.assertIn("profile_ready_awaiting_controller_review", worker_text)
        self.assertIn("os.set_blocking(output_fd, False)", source)
        self.assertIn('"is_private": true', metadata.decode("utf-8"))
        self.assertTrue(
            receipt["commands"]["kaggle_push_command_prepared_not_run"].endswith("-t 4200")
        )
        self.assertEqual(receipt["profile_gate"]["profile_provider_watch_seconds"], 4_200)

    def test_materialized_v4_packet_binds_the_split_namespace_repair(self) -> None:
        packet = HERE / "packet_v4"
        receipt = json.loads((packet / "review_packet.json").read_text(encoding="utf-8"))
        kernel = (packet / "kernel.py").read_bytes()
        metadata = (packet / "kernel-metadata.json").read_bytes()
        self.assertEqual(receipt["packet_id"], "E508_RAW_BASELINE_V4")
        self.assertEqual(
            receipt["kernel_id"],
            "vladkochetov07/wnn-raw-teacher-pair-e508-20261007-v4",
        )
        self.assertEqual(hashlib.sha256(kernel).hexdigest(), receipt["packet_sha256"]["kernel.py"])
        self.assertEqual(
            hashlib.sha256(metadata).hexdigest(), receipt["packet_sha256"]["kernel-metadata.json"]
        )
        source = kernel.decode("utf-8")
        tree = ast.parse(source)
        assignments = {
            target.id: ast.literal_eval(node.value)
            for node in tree.body
            if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Name)
            and target.id in {"PACKET_ID", "WORKER_CODE", "WORKER_SHA256"}
        }
        worker = zlib.decompress(base64.b85decode(assignments["WORKER_CODE"].encode("ascii")))
        self.assertEqual(assignments["PACKET_ID"], receipt["packet_id"])
        self.assertEqual(hashlib.sha256(worker).hexdigest(), assignments["WORKER_SHA256"])
        self.assertEqual(assignments["WORKER_SHA256"], receipt["packet_sha256"]["packet_worker.py"])
        self.assertEqual(worker, (HERE / "packet_worker.py").read_bytes())
        self.assertIn("if CORRUPT_TRAIN_GROUP in train:", worker.decode("utf-8"))
        self.assertNotIn("CORRUPT_TRAIN_GROUP in valid", worker.decode("utf-8"))
        self.assertFalse(receipt["profile_gate"]["fit_launch_in_this_packet"])
        self.assertIn('"is_private": true', metadata.decode("utf-8"))
        self.assertIn('"enable_internet": false', metadata.decode("utf-8"))
        self.assertIn("v4", source)

    def test_v5_builder_isolated_from_v4_and_embeds_v5_worker(self) -> None:
        self.assertEqual(prepare_packet_v5.PACKET_ID, "E508_RAW_BASELINE_V5")
        self.assertEqual(
            prepare_packet_v5.KERNEL_ID, "vladkochetov07/wnn-raw-teacher-pair-e508-20261007-v5"
        )
        self.assertEqual(prepare_packet_v5.PACKET_DIR.name, "packet_v5")
        worker_v4 = (HERE / "packet_worker.py").read_text(encoding="utf-8")
        worker_v5 = (HERE / "packet_worker_v5.py").read_text(encoding="utf-8")
        normalized_v5 = worker_v5.replace(
            'PACKET_ID = "E508_RAW_BASELINE_V5"', 'PACKET_ID = "E508_RAW_BASELINE_V4"', 1
        ).replace('        "cuda",', '        "cuda:0",', 1)
        self.assertEqual(
            ast.dump(ast.parse(normalized_v5), include_attributes=False),
            ast.dump(ast.parse(worker_v4), include_attributes=False),
        )
        self.assertIn(
            'worker = (HERE / "packet_worker_v5.py").read_bytes()',
            (HERE / "prepare_packet_v5.py").read_text(encoding="utf-8"),
        )

    def test_materialized_v5_packet_binds_supported_device_and_profile_only_gate(self) -> None:
        packet = HERE / "packet_v5"
        receipt = json.loads((packet / "review_packet.json").read_text(encoding="utf-8"))
        for name, expected_hash in receipt["packet_sha256"].items():
            if name in {"packet_worker.py", "source_archive.tar"}:
                continue  # These source inputs are embedded in kernel.py.
            self.assertEqual(
                hashlib.sha256((packet / name).read_bytes()).hexdigest(), expected_hash
            )

        kernel = (packet / "kernel.py").read_text(encoding="utf-8")
        tree = ast.parse(kernel)
        assignments = {
            target.id: ast.literal_eval(node.value)
            for node in tree.body
            if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Name)
            and target.id
            in {
                "PACKET_ID",
                "SOURCE_ARCHIVE",
                "SOURCE_ARCHIVE_SHA256",
                "WORKER_CODE",
                "WORKER_SHA256",
            }
        }
        archive = zlib.decompress(base64.b85decode(assignments["SOURCE_ARCHIVE"].encode("ascii")))
        self.assertEqual(
            hashlib.sha256(archive).hexdigest(), receipt["packet_sha256"]["source_archive.tar"]
        )
        self.assertEqual(hashlib.sha256(archive).hexdigest(), assignments["SOURCE_ARCHIVE_SHA256"])
        worker = zlib.decompress(base64.b85decode(assignments["WORKER_CODE"].encode("ascii")))
        self.assertEqual(assignments["PACKET_ID"], "E508_RAW_BASELINE_V5")
        self.assertEqual(hashlib.sha256(worker).hexdigest(), assignments["WORKER_SHA256"])
        self.assertEqual(assignments["WORKER_SHA256"], receipt["packet_sha256"]["packet_worker.py"])
        self.assertEqual(worker, (HERE / "packet_worker_v5.py").read_bytes())
        profile = receipt["commands"]["profile_argv"]
        self.assertEqual(profile[profile.index("--device") + 1], "cuda")
        self.assertFalse(receipt["profile_gate"]["fit_launch_in_this_packet"])

        metadata = json.loads((packet / "kernel-metadata.json").read_text(encoding="utf-8"))
        self.assertTrue(metadata["is_private"])
        self.assertFalse(metadata["enable_internet"])
        self.assertTrue(metadata["enable_gpu"])
        self.assertFalse(metadata["enable_tpu"])
        self.assertIn("PROFILE_ONLY = True", worker.decode("utf-8"))

    def test_v3_raw_identity_binds_api_version_and_full_portable_hashes(self) -> None:
        packet = HERE / "packet_v3"
        receipt = json.loads((packet / "review_packet.json").read_text(encoding="utf-8"))
        portable_path = HERE.parents[3] / "research/campaign_2026-10-04/portable_inputs.json"
        portable = json.loads(portable_path.read_text(encoding="utf-8"))
        raw = receipt["input_map"]["raw"]
        self.assertEqual(raw["dataset"], "vladkochetov07/wunder-funder")
        self.assertEqual(raw["dataset_id"], 12352811)
        self.assertEqual(raw["dataset_version_at_preregistration"], 1)
        self.assertEqual(raw["dataset_version_observed_at_utc"], "2026-10-07T00:11:43+00:00")
        for logical_path in ("datasets/train.parquet", "datasets/valid.parquet"):
            registry = portable["sources"][logical_path]
            self.assertEqual(
                raw["expected_files"][logical_path],
                {
                    **registry["original_identity"],
                    "sha256": registry["sha256"],
                },
            )


if __name__ == "__main__":
    unittest.main()
