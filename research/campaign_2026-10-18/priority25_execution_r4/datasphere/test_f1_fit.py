"""Offline recipe and artifact tests; no trainer, GPU or provider is invoked."""

from __future__ import annotations

import hashlib
import json
import signal
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import f1_fit_bundle
from f1_fit_bundle import verify_bundle_archive, verify_hash_manifest, write_bundle
from f1_fit_entry import _collect_and_bundle, _request_stop
from f1_fit_outputs import (
    collect_artifacts,
    validate_completed_fit,
    validate_prediction_sidecar,
)
from f1_fit_recipe import ARMS, FOLD1_SHA256, prediction_args, training_args, validate_fit_spec
from f1_fit_staging import check_packet_binding

TEMPLATE = Path(__file__).with_name("f1_fit_spec.template.json")


def sealed_fit_spec() -> dict[str, object]:
    """Return a detached synthetic execution spec with measured deadlines."""
    spec = json.loads(TEMPLATE.read_text(encoding="utf-8"))
    spec["state"] = "SEALED_FOR_F1_PAIR"
    spec["runtime"] = {
        "internal_deadline_seconds": 3000,
        "collection_reserve_seconds": 300,
        "external_cancel_deadline_seconds_from_submit": 3600,
    }
    return spec


class FitRecipeTests(unittest.TestCase):
    def test_template_fails_closed_until_deadline_is_measured(self) -> None:
        template = json.loads(TEMPLATE.read_text(encoding="utf-8"))
        with self.assertRaisesRegex(ValueError, "not sealed"):
            validate_fit_spec(template)
        template["state"] = "SEALED_FOR_F1_PAIR"
        with self.assertRaisesRegex(ValueError, "fit, collection and provider deadlines"):
            validate_fit_spec(template)

    def test_fit_spec_accepts_only_fixed_fold_seed_pair_and_bound(self) -> None:
        spec = sealed_fit_spec()
        self.assertEqual(validate_fit_spec(spec), (3000, 300, 3600))
        spec["recipe"]["fold"] = 0
        with self.assertRaisesRegex(ValueError, "authorized recipe"):
            validate_fit_spec(spec)

    def test_fit_spec_requires_collection_reserve_before_external_deadline(self) -> None:
        spec = sealed_fit_spec()
        spec["runtime"]["collection_reserve_seconds"] = None
        with self.assertRaisesRegex(ValueError, "deadlines are unset or invalid"):
            validate_fit_spec(spec)
        spec = sealed_fit_spec()
        spec["runtime"]["external_cancel_deadline_seconds_from_submit"] = 3359
        with self.assertRaisesRegex(ValueError, "explicit collection reserve"):
            validate_fit_spec(spec)

    def test_sigterm_during_collection_is_recorded_without_unbounded_work(self) -> None:
        state = {"phase": "collection", "sigterm_requested": False}
        _request_stop(signal.SIGTERM, None, state)
        self.assertTrue(state["sigterm_requested"])
        with self.assertRaisesRegex(TimeoutError, "collection/finalization deadline"):
            _request_stop(signal.SIGALRM, None, state)
        state["phase"] = "fit"
        with self.assertRaises(InterruptedError):
            _request_stop(signal.SIGTERM, None, state)

    def test_pair_training_commands_differ_only_by_run_and_shape(self) -> None:
        wide = training_args(ARMS[0], Path("splits/fold1.json"))
        deep = training_args(ARMS[1], Path("splits/fold1.json"))
        shape_flags = {"--name", "--hidden", "--layers"}

        def without_shape(values: list[str]) -> list[str]:
            result: list[str] = []
            skip = False
            for value in values:
                if skip:
                    skip = False
                elif value in shape_flags:
                    skip = True
                else:
                    result.append(value)
            return result

        self.assertEqual(without_shape(wide), without_shape(deep))
        self.assertIn("--precision", wide)
        self.assertEqual(wide[wide.index("--precision") + 1], "fp32")
        self.assertEqual(wide[wide.index("--optim") + 1], "adamw")
        self.assertIn("--save-training-state", wide)
        self.assertIn("--no-eval-during-fit", wide)
        for forbidden in ("--init", "--distill", "--distill-cache-root", "--propensity"):
            self.assertNotIn(forbidden, wide)
        exported = prediction_args(ARMS[1], Path("splits/fold1.json"))
        self.assertIn("--campaign-evaluation-only", exported)
        self.assertEqual(exported[exported.index("--epoch") + 1], "5")
        self.assertEqual(exported[exported.index("--inference-precision") + 1], "fp32")

    def test_sdk_manual_environment_uses_inert_nonempty_local_path(self) -> None:
        config = Path(__file__).with_name("f1_fit_config.template.yaml").read_text(encoding="utf-8")
        self.assertIn("local-paths:\n      - runtime_marker.py", config)
        self.assertIn("f1_fit_bundle.py: FIT_BUNDLE", config)
        self.assertTrue(Path(__file__).with_name("runtime_marker.py").is_file())
        self.assertNotIn("local-paths: []", config)

    def test_fit_binds_to_frozen_profile_and_fold1_archive_member(self) -> None:
        profile_path = Path(__file__).with_name("profile-spec.json")
        profile = json.loads(profile_path.read_text(encoding="utf-8"))
        fit = sealed_fit_spec()
        fit["base_profile_spec_sha256"] = (
            "1ca416cac32fb075b4b9d16683e9a07634f3926e625e62576468c99b2b722d56"
        )
        fit["source_commit"] = profile["source_commit"]
        fit["code_archive_sha256"] = profile["code_archive_sha256"]
        fit["supplemental_archive_sha256"] = profile["supplemental_archive_sha256"]
        fit["fold_manifest_sha256"] = profile["supplemental_members"]["splits/fold1.json"]
        check_packet_binding(profile, fit, profile_path)
        fit["fold_manifest_sha256"] = profile["supplemental_members"]["splits/fold0.json"]
        with self.assertRaisesRegex(ValueError, "fold-1 manifest"):
            check_packet_binding(profile, fit, profile_path)


class FitArtifactTests(unittest.TestCase):
    def _write_fixture(self, root: Path, objective: float = 0.5) -> Path:
        arm = ARMS[0]
        run = root / "artifacts/runs" / str(arm["name"])
        run.mkdir(parents=True)
        for name in ("model_ep5.pt", "model.pt", "training_state.pt"):
            (run / name).write_bytes(b"fixture")
        history = [
            {
                "epoch": epoch,
                "evaluation": "withheld",
                "fit_summary": {
                    "objective_mean": objective,
                    "optimizer_updates": {"successful": 12},
                    "learning_rate_by_group": [{"group": 0, "mean": 1e-4}],
                },
            }
            for epoch in range(6)
        ]
        (run / "history.json").write_text(json.dumps(history), encoding="utf-8")
        (run / "provenance.json").write_text(
            json.dumps(
                {
                    "split_manifest_sha256": "d8e9748d770bb3af525943b36b1801e6fd72d78259cc6b823d4661ffd78367d9",
                    "args": {
                        "name": arm["name"],
                        "arch": "gru",
                        "cell": "gru",
                        "enc": "linear",
                        "hidden": arm["hidden"],
                        "layers": arm["layers"],
                        "precision": "fp32",
                        "split_manifest_sha256": FOLD1_SHA256,
                        "init": "",
                        "valid_fold": 1,
                        "n_valid_folds": 4,
                        "full_valid": True,
                        "valid_repeat": 1,
                        "valid_train_fraction": 1.0,
                        "device": "cuda",
                        "epochs": 6,
                        "checkpoint_epoch": 5,
                        "train_source": "train+valid",
                        "train_seqs": 10607,
                        "eval_source": "vfold",
                        "batch": 8,
                        "chunk": 2000,
                        "seed": 1001,
                        "batch_order": "seeded",
                        "loss": "wmse",
                        "loss_norm": "chunk",
                        "dropout": 0.0,
                        "rnn_init": "default",
                        "optim": "adamw",
                        "lr": 0.0001,
                        "beta2": 0.999,
                        "wd": 0.01,
                        "sched": "onecycle",
                        "warmup_frac": 0.05,
                        "head_lr_mult": 1.0,
                        "clip": 1.0,
                        "grad_stats": True,
                        "pin_memory": False,
                        "checkpoint_policy": "fixed",
                        "no_eval_during_fit": True,
                        "save_training_state": True,
                        "resume_state": None,
                        "final_model_adaptation": False,
                        "final_distill_initializer": False,
                        "inherit_init_preprocessing": False,
                        "class_balance": False,
                        "prop_cache": False,
                        "distill_family_weight": 0.0,
                        "distill_boot_h": 0,
                        "fused_adam": True,
                        "sync_free": True,
                        "deterministic": True,
                        "distill": "",
                        "distill_cache_root": "",
                        "propensity": "",
                        "prop_input": "",
                    },
                }
            ),
            encoding="utf-8",
        )
        return run

    def test_completed_fit_requires_ordered_fit_only_summaries(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run = self._write_fixture(Path(temporary))
            evidence = validate_completed_fit(run, ARMS[0])
            self.assertIn("training_state.pt", evidence)

    def test_nonfinite_objective_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run = self._write_fixture(Path(temporary), objective=float("nan"))
            with self.assertRaisesRegex(ValueError, "finite"):
                validate_completed_fit(run, ARMS[0])

    def test_sidecar_binds_exact_checkpoint_and_fold_row_order(self) -> None:
        groups = [14, 20, 21]
        payload = b"prediction-fixture"
        expected_groups_digest = hashlib.sha256(
            b"".join(group.to_bytes(8, byteorder="little", signed=True) for group in groups)
        ).hexdigest()
        with tempfile.TemporaryDirectory() as temporary:
            prediction = Path(temporary) / "prediction.npy"
            prediction.write_bytes(payload)
            metadata = {
                "run": ARMS[0]["name"],
                "split": "valid",
                "epoch": 5,
                "checkpoint_sha256": "a" * 64,
                "split_manifest_sha256": FOLD1_SHA256,
                "campaign_split_manifest_sha256": FOLD1_SHA256,
                "campaign_export_scope": "evaluation",
                "feature_only": True,
                "inference_precision": "fp32",
                "row_groups": groups,
                "evaluation_membership_sha256": expected_groups_digest,
                "prediction_sha256": hashlib.sha256(payload).hexdigest(),
            }
            validate_prediction_sidecar(metadata, ARMS[0], groups, "a" * 64, prediction)
            metadata["row_groups"] = list(reversed(groups))
            with self.assertRaisesRegex(ValueError, "sidecar"):
                validate_prediction_sidecar(metadata, ARMS[0], groups, "a" * 64, prediction)

    def test_collection_bundles_partial_resume_and_export_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "fit-results"
            output.mkdir()
            data = root / "data"
            run = data / "artifacts/runs" / str(ARMS[0]["name"])
            run.mkdir(parents=True)
            (run / "training_state.pt").write_bytes(b"resume")
            cache = data / "cache"
            cache.mkdir()
            (cache / f"pred_{ARMS[0]['name']}_valid.npy.partial").write_bytes(b"partial")
            profile, fit = root / "profile.json", root / "fit.json"
            profile.write_text("{}", encoding="utf-8")
            fit.write_text("{}", encoding="utf-8")

            errors = collect_artifacts(data, output, profile, fit)
            self.assertEqual(errors, [])
            self.assertTrue(
                (output / "artifacts/runs" / str(ARMS[0]["name"]) / "training_state.pt").is_file()
            )
            self.assertTrue(
                (output / "cache" / f"pred_{ARMS[0]['name']}_valid.npy.partial").is_file()
            )
            bundle = root / "fit-results.tar.gz"
            write_bundle(bundle, output)
            self.assertGreater(bundle.stat().st_size, 0)
            manifest = verify_bundle_archive(bundle)
            self.assertTrue(
                any(item["path"].endswith("training_state.pt") for item in manifest["files"])
            )

    def test_downloaded_tree_is_verified_against_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "fit-results"
            output.mkdir()
            (output / "receipt.json").write_text('{"status":"COMPLETE"}\n', encoding="utf-8")
            bundle = root / "fit-results.tar.gz"
            write_bundle(bundle, output)
            with tarfile.open(bundle, "r:gz") as archive:
                archive.extractall(root / "downloaded")
            downloaded = root / "downloaded/fit-results"
            self.assertEqual(verify_hash_manifest(downloaded)["file_count"], 1)
            (downloaded / "receipt.json").write_text("tampered", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hash or size mismatch"):
                verify_hash_manifest(downloaded)

    def test_oversize_provider_bundle_is_removed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "fit-results"
            output.mkdir()
            (output / "payload").write_bytes(b"payload")
            bundle = root / "fit-results.tar.gz"
            original_limit = f1_fit_bundle.MAX_PROVIDER_OUTPUT_BYTES
            try:
                f1_fit_bundle.MAX_PROVIDER_OUTPUT_BYTES = 1
                with self.assertRaisesRegex(ValueError, "768 MiB"):
                    write_bundle(bundle, output)
            finally:
                f1_fit_bundle.MAX_PROVIDER_OUTPUT_BYTES = original_limit
            self.assertFalse(bundle.exists())

    def test_collection_deadline_preserves_partial_receipt_and_skips_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "fit-results"
            output.mkdir()
            receipt = {"status": "INTERRUPTED", "quality_result": None}
            with (
                patch("f1_fit_entry.collect_artifacts", side_effect=TimeoutError("deadline")),
                patch("f1_fit_entry.write_bundle") as bundle,
            ):
                completed = _collect_and_bundle(
                    None,
                    output,
                    root / "profile.json",
                    root / "fit.json",
                    root / "fit-results.tar.gz",
                    receipt,
                    {"phase": "collection", "sigterm_requested": True},
                )
            self.assertFalse(completed)
            self.assertFalse(bundle.called)
            saved = json.loads((output / "fit-pair-receipt.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["collection_status"], "DEADLINE_EXPIRED")
            self.assertEqual(saved["bundle_status"], "NOT_ATTEMPTED_COLLECTION_DEADLINE")
            self.assertTrue(saved["sigterm_requested_during_collection"])

    def test_collection_reraises_deadline_at_copy_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "fit-results"
            output.mkdir()
            profile = root / "profile.json"
            profile.write_text("{}", encoding="utf-8")
            fit = root / "fit.json"
            for error in (TimeoutError("deadline"), InterruptedError("stop")):
                with (
                    self.subTest(exception=type(error).__name__),
                    patch("f1_fit_outputs.shutil.copy2", side_effect=error),
                    self.assertRaises(type(error)),
                ):
                    collect_artifacts(root, output, profile, fit)


if __name__ == "__main__":
    unittest.main()
