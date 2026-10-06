from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from wnn.bandit_campaign_recipe import (
    FAMILY_ORDER,
    RECIPE_VARIANTS,
    verify_derived_distillation_targets,
)
from wnn.distillation import TeacherSource, combine_teacher_chunk
from wnn.distillation_cache import row_groups_sha256, validate_distillation_cache

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "research/campaign_2026-10-06/bandit_continuation"
SOURCE_MANIFEST = json.loads((CAMPAIGN / "SOURCE_FAMILIES.json").read_text())
VALIDATOR_PATH = ROOT / "scripts/validate_bandit_campaign_recipe.py"


def load_validator():
    """Load path validation helpers without invoking the campaign CLI."""
    spec = importlib.util.spec_from_file_location("bandit_recipe_cli", VALIDATOR_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load campaign recipe validator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source_snapshot() -> dict[str, object]:
    """Make synthetic identities for algorithm tests, never profile evidence."""
    rows = copy.deepcopy(SOURCE_MANIFEST["families"])
    for row in rows:
        if row["status"] == "planned_fit":
            row["status"] = "produced_by_e510"
            row["checkpoint_sha256"] = str(row["seed"]) * 64
            row["provenance_sha256"] = "a" * 64
    return {"families": rows}


class DistillationCacheClosureTests(unittest.TestCase):
    """Exercise actual on-disk target checks using small synthetic arrays."""

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.cache_root = self.root / "cache" / "bd26_four_family_f0_v1"
        self.teacher_cache_root = self.root / "cache"
        self.cache_root.mkdir(parents=True)
        self.snapshot = source_snapshot()
        self.target_orders = {"train": [101], "valid": [202]}
        self.fit_groups = {"train": [101], "valid": [202]}
        self.metadata = self._make_files()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _make_files(self) -> dict[str, object]:
        families = self.snapshot["families"]
        sources = [
            TeacherSource(
                family=row["family"],
                run=row["run"],
                coefficient=row["main_coefficient"],
                epoch=row["epoch_zero_based"],
            )
            for row in families
        ]
        coefficient_rows = [
            {
                "family": row["family"],
                "run": row["run"],
                "epoch": row["epoch_zero_based"],
                "coefficient": row["main_coefficient"],
            }
            for row in families
        ]
        source_rows = []
        all_predictions: dict[str, dict[str, np.ndarray]] = {"train": {}, "valid": {}}
        for index, row in enumerate(families, start=1):
            prediction_records: dict[str, dict[str, object]] = {}
            for split_name in ("train", "valid"):
                path = self.teacher_cache_root / f"pred_{row['run']}_{split_name}.npy"
                value = np.full((1, 20_000, 2), index / 10.0, dtype=np.float32)
                np.save(path, value)
                prediction_sha = sha256_file(path)
                row_groups = self.target_orders[split_name]
                sidecar = {
                    "run": row["run"],
                    "split": split_name,
                    "epoch": row["epoch_zero_based"],
                    "checkpoint_sha256": row["checkpoint_sha256"],
                    "split_manifest_sha256": "09263ad4fe2b18e2b2f3b79f4562678dce801b067b868134ddc79279f9316a3d",
                    "prediction_sha256": prediction_sha,
                    "row_groups": row_groups,
                    "feature_only_export": True,
                    "label_columns_read": False,
                    "mask_columns_read": False,
                }
                sidecar_bytes = (json.dumps(sidecar, sort_keys=True) + "\n").encode()
                path.with_suffix(".json").write_bytes(sidecar_bytes)
                prediction_records[split_name] = {
                    "prediction_path": str(path),
                    "prediction_sha256": prediction_sha,
                    "sidecar_sha256": hashlib.sha256(sidecar_bytes).hexdigest(),
                    "checkpoint_sha256": row["checkpoint_sha256"],
                    "epoch": row["epoch_zero_based"],
                    "row_groups_sha256": row_groups_sha256(row_groups),
                }
                all_predictions[split_name][row["run"]] = value
            source_rows.append(
                {
                    "family": row["family"],
                    "run": row["run"],
                    "epoch": row["epoch_zero_based"],
                    "coefficient": row["main_coefficient"],
                    "checkpoint_sha256": row["checkpoint_sha256"],
                    "predictions": prediction_records,
                }
            )

        files: dict[str, dict[str, object]] = {}
        for split_name in ("train", "valid"):
            main, normalized, homogeneous, family_values = combine_teacher_chunk(
                all_predictions[split_name], sources, FAMILY_ORDER, "strong"
            )
            for prefix, values in (
                ("main_source", main),
                ("main_normalized", normalized),
                ("main_homogeneous", homogeneous),
                ("families", family_values),
            ):
                key = f"{prefix}_{split_name}"
                path = self.cache_root / f"{key}.npy"
                np.save(path, values)
                files[key] = {"path": path.name}
        return {
            "sources": source_rows,
            "source_coefficients": coefficient_rows,
            "files": files,
        }

    def verify(self, variant: str, require_families: bool) -> dict[str, object]:
        return verify_derived_distillation_targets(
            self.cache_root,
            self.teacher_cache_root,
            self.metadata,
            self.snapshot,
            self.target_orders,
            self.fit_groups,
            variant=variant,
            require_families=require_families,
            chunk_rows=1,
        )

    def test_each_preregistered_variant_recomputes_bitwise(self) -> None:
        for arm, spec in RECIPE_VARIANTS.items():
            with self.subTest(arm=arm):
                report = self.verify(str(spec["variant"]), float(spec["family_weight"]) > 0)
                self.assertEqual(report["derived_chunks_compared_bitwise"], 2)

    def test_missing_array_and_tampered_array_bytes_fail(self) -> None:
        missing = copy.deepcopy(self.metadata)
        missing["files"]["main_source_train"]["path"] = "missing.npy"
        with self.assertRaises(FileNotFoundError):
            verify_derived_distillation_targets(
                self.cache_root,
                self.teacher_cache_root,
                missing,
                self.snapshot,
                self.target_orders,
                self.fit_groups,
                variant="source",
                require_families=False,
            )

        path = self.cache_root / self.metadata["files"]["main_source_train"]["path"]
        damaged = np.load(path).copy()
        damaged[0, 0, 0] += 0.1
        np.save(path, damaged)
        with self.assertRaisesRegex(ValueError, "derived cache target values"):
            self.verify("source", False)

    def test_prediction_symlink_cannot_escape_registered_cache_root(self) -> None:
        run = self.snapshot["families"][0]["run"]
        source_path = self.teacher_cache_root / f"pred_{run}_train.npy"
        outside_path = self.root / "outside.npy"
        np.save(outside_path, np.zeros((1, 20_000, 2), dtype=np.float32))
        source_path.unlink()
        source_path.symlink_to(outside_path)
        with self.assertRaisesRegex(ValueError, "escapes its registered cache root"):
            self.verify("source", False)

    def test_source_sidecar_and_prediction_hashes_are_bound(self) -> None:
        run = self.snapshot["families"][0]["run"]
        path = self.teacher_cache_root / f"pred_{run}_train.json"
        sidecar = json.loads(path.read_text())
        sidecar["row_groups"] = [102]
        path.write_text(json.dumps(sidecar, sort_keys=True) + "\n")
        with self.assertRaisesRegex(ValueError, "schema/order"):
            self.verify("homogeneous", False)

    def test_source_values_are_checked_for_finiteness_in_bounded_chunks(self) -> None:
        row = self.snapshot["families"][0]
        run = row["run"]
        path = self.teacher_cache_root / f"pred_{run}_train.npy"
        prediction = np.load(path)
        prediction[0, 0, 0] = np.nan
        np.save(path, prediction)
        prediction_sha = sha256_file(path)
        sidecar_path = path.with_suffix(".json")
        sidecar = json.loads(sidecar_path.read_text())
        sidecar["prediction_sha256"] = prediction_sha
        sidecar_bytes = (json.dumps(sidecar, sort_keys=True) + "\n").encode()
        sidecar_path.write_bytes(sidecar_bytes)
        source_record = self.metadata["sources"][0]
        source_record["predictions"]["train"]["prediction_sha256"] = prediction_sha
        source_record["predictions"]["train"]["sidecar_sha256"] = hashlib.sha256(
            sidecar_bytes
        ).hexdigest()
        with self.assertRaisesRegex(ValueError, "source teacher predictions are nonfinite"):
            self.verify("source", False)

    def test_generic_cache_validator_rejects_missing_wrong_shape_and_bad_order(self) -> None:
        # One-group synthetic split keeps the physical validator test compact.
        data_root = self.root / "data"
        data_cache = data_root / "cache"
        data_cache.mkdir(parents=True)
        for name, group in (("train", 101), ("valid", 202)):
            np.savez(data_cache / f"{name}_targets.npz", row_group=np.asarray([group]))
        split = type(
            "SyntheticSplit",
            (),
            {
                "sha256": "09263ad4fe2b18e2b2f3b79f4562678dce801b067b868134ddc79279f9316a3d",
                "fold": 0,
                "train_groups": (101,),
                "valid_groups": (202,),
                "evaluation_groups": (),
            },
        )()
        metadata = {
            "format": "wnn.heterogeneous-distillation-cache",
            "version": 1,
            "fold": 0,
            "split_manifest_sha256": split.sha256,
            "fit_groups": {"train": [101], "valid": [202]},
            "evaluation_valid_groups": [],
            "family_order": list(FAMILY_ORDER),
            "sources": [],
            "source_coefficients": [],
            "coefficient_sum": 0.8125,
            "homogeneous_family": "strong",
            "epoch_convention": {
                "friend_at_ep1": "second additional epoch",
                "student_epochs": 2,
                "student_checkpoint_policy": "fixed",
                "student_checkpoint_epoch": 1,
            },
            "files": {},
        }
        for index in range(8):
            family = FAMILY_ORDER[index // 2]
            run = f"run_{index}"
            metadata["source_coefficients"].append(
                {"family": family, "run": run, "epoch": 3, "coefficient": 0.1015625}
            )
            metadata["sources"].append(
                {
                    "family": family,
                    "run": run,
                    "epoch": 3,
                    "coefficient": 0.1015625,
                    "checkpoint_sha256": "a" * 64,
                }
            )

        metadata_path = self.cache_root / "metadata.json"

        def persist_metadata() -> None:
            metadata_path.write_text(json.dumps(metadata, sort_keys=True) + "\n")

        def write_main(array: np.ndarray, groups: list[int]) -> None:
            path = self.cache_root / "main_homogeneous_train.npy"
            np.save(path, array)
            metadata["files"]["main_homogeneous_train"] = {
                "path": path.name,
                "shape": list(array.shape),
                "dtype": "float32",
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "row_groups": groups,
                "row_groups_sha256": row_groups_sha256(groups),
            }
            persist_metadata()

        valid_path = self.cache_root / "main_homogeneous_valid.npy"
        valid = np.zeros((1, 20_000, 2), dtype=np.float32)
        np.save(valid_path, valid)
        metadata["files"]["main_homogeneous_valid"] = {
            "path": valid_path.name,
            "shape": list(valid.shape),
            "dtype": "float32",
            "bytes": valid_path.stat().st_size,
            "sha256": sha256_file(valid_path),
            "row_groups": [202],
            "row_groups_sha256": row_groups_sha256([202]),
        }
        persist_metadata()
        train = np.zeros((1, 20_000, 2), dtype=np.float32)
        write_main(train, [101])
        report = validate_distillation_cache(
            self.cache_root,
            data_root,
            split,
            "homogeneous",
            require_families=False,
        )
        self.assertEqual(
            set(report["verified_files"]), {"main_homogeneous_train", "main_homogeneous_valid"}
        )

        train_path = self.cache_root / "main_homogeneous_train.npy"
        train_path.unlink()
        with self.assertRaises(FileNotFoundError):
            validate_distillation_cache(
                self.cache_root, data_root, split, "homogeneous", require_families=False
            )
        write_main(train, [101])
        write_main(np.zeros((1, 20_000, 1), dtype=np.float32), [101])
        with self.assertRaisesRegex(ValueError, "shape or dtype"):
            validate_distillation_cache(
                self.cache_root, data_root, split, "homogeneous", require_families=False
            )
        write_main(np.zeros((1, 20_000, 2), dtype=np.float32), [202])
        with self.assertRaisesRegex(ValueError, "row groups"):
            validate_distillation_cache(
                self.cache_root, data_root, split, "homogeneous", require_families=False
            )

        train = np.zeros((1, 20_000, 2), dtype=np.float32)
        train[0, 0, 0] = np.nan
        write_main(train, [101])
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            validate_distillation_cache(
                self.cache_root, data_root, split, "homogeneous", require_families=False
            )

        outside_path = self.root / "outside-target.npy"
        np.save(outside_path, np.zeros((1, 20_000, 2), dtype=np.float32))
        train_path.unlink()
        train_path.symlink_to(outside_path)
        with self.assertRaisesRegex(ValueError, "escapes the registered cache root"):
            validate_distillation_cache(
                self.cache_root, data_root, split, "homogeneous", require_families=False
            )

    def test_cli_registered_roots_reject_traversal_and_symlink_escape(self) -> None:
        validator = load_validator()
        with (
            tempfile.TemporaryDirectory() as root_dir,
            tempfile.TemporaryDirectory() as outside_dir,
        ):
            repository = Path(root_dir)
            outside = Path(outside_dir)
            (repository / "cache").mkdir()
            (repository / "outside").mkdir()
            resolver = validator.resolve_registered_directory
            self.assertEqual(
                resolver(repository / "cache", "cache", "cache", repository_root=repository),
                (repository / "cache").resolve(),
            )
            with self.assertRaisesRegex(ValueError, "registered path"):
                resolver(repository / "outside", "cache", "cache", repository_root=repository)
            (repository / "cache").rmdir()
            (repository / "cache").symlink_to(outside, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "escapes the repository"):
                resolver(repository / "cache", "cache", "cache", repository_root=repository)


def sha256_file(path: Path) -> str:
    """Hash one small synthetic fixture output."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    unittest.main()
