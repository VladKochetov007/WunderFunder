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

    def test_corrupt_train_group_exclusion_is_scoped_to_train_file(self) -> None:
        worker_tree = ast.parse((HERE / "packet_worker.py").read_text(encoding="utf-8"))
        helper = next(
            node
            for node in worker_tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "exact_group_sets"
        )
        namespace: dict[str, object] = {
            "FIT_COUNTS": {"train": 10_535, "valid": 1_405},
            "EVAL_VALID_COUNT": 468,
            "CORRUPT_TRAIN_GROUP": 528,
        }
        exec(
            compile(ast.Module(body=[helper], type_ignores=[]), "e508_exact_group_sets", "exec"),
            namespace,
        )
        exact_group_sets = namespace["exact_group_sets"]
        valid_528_split = type(
            "Split",
            (),
            {
                "train_groups": set(range(10_536)) - {528},
                "valid_groups": set(range(1_405)),
                "evaluation_groups": set(range(468)),
            },
        )()
        result = exact_group_sets(valid_528_split)
        self.assertIn(528, result["valid"])
        self.assertNotIn(528, result["train"])

        corrupt_train_split = type(
            "Split",
            (),
            {
                "train_groups": set(range(10_535)),
                "valid_groups": set(range(1_000, 2_405)),
                "evaluation_groups": set(range(468)),
            },
        )()
        with self.assertRaisesRegex(ValueError, "corrupt train row group"):
            exact_group_sets(corrupt_train_split)

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
