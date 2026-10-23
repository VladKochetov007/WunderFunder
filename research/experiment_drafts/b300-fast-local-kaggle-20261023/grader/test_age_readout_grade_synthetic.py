"""Synthetic-only admission and artifact-contract tests for the age grader."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock

import numpy as np

_argument_parser = argparse.ArgumentParser(add_help=True)
_argument_parser.add_argument("--root", type=Path, default=Path.cwd())
_argument_parser.add_argument("--tmp-root", type=Path, required=True)
_test_args, _unittest_args = _argument_parser.parse_known_args()
ROOT = _test_args.root.resolve(strict=True)
TMP_ROOT = _test_args.tmp_root.resolve()
if not (ROOT / "src/wnn").is_dir():
    raise ValueError("--root must name the Challenge 2 repository checkout")
TMP_ROOT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "research/experiment_drafts/online-relationships-20261022"))

producer = importlib.import_module("wnn.training.age_readout_provenance")


def _load_grader() -> Any:
    spec = importlib.util.spec_from_file_location(
        "age_readout_grade", Path(__file__).with_name("age_readout_grade.py")
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load grader adapter")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


grader = _load_grader()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


class SyntheticCandidate:
    """Construct one complete candidate using the actual actor sidecar builders."""

    def __init__(self, root: Path, *, fortran: bool = False) -> None:
        self.root = root
        self.output = root / "runs/age_readout_f0_s1001_a_r1"
        self.output.mkdir(parents=True)
        self.row_refs = [{"split": "valid", "row_group": 77}]
        self.fold_id = 0
        self.seed = 1001
        self.recipe = {
            "experiment_id": grader.EXPERIMENT_ID,
            "source_cache_recipe": {"path": "cache/source.json"},
            "features": {"mode": "synthetic_fixed_age"},
            "readout": {"architecture": "323-64-2", "objective": "synthetic"},
        }
        self.parent_ancestry = {
            "run": "synthetic_parent_f0",
            "checkpoint_sha256": "parent-checkpoint-sha",
            "provenance_sha256": "parent-provenance-sha",
            "initial_ancestor": "synthetic_initializer",
            "initial_ancestor_checkpoint_sha256": "initializer-sha",
            "mean_identity": "mean-identity",
            "std_identity": "std-identity",
            "historical_precision": "fp16",
            "history_fit_membership": "synthetic-fit-membership",
            "input_width": 211,
            "output_rows": 23,
            "epoch": 2,
        }
        self.fold = {
            "fold": 0,
            "seed": self.seed,
            "parent_cache": "cache/parent_f0",
            "parent": self.parent_ancestry,
            "_expected_eval_refs": self.row_refs,
        }
        self.cell = {
            "run": grader.EXPECTED_CELLS[0],
            "fold": 0,
            "seed": self.seed,
            "arm": "A",
            "output": str(self.output),
        }
        self.eval_membership = grader.row_refs_sha256(self.row_refs)
        parent_prediction_path = root / "cache/parent_f0/evaluation_predictions.npy"
        parent_prediction_path.parent.mkdir(parents=True, exist_ok=True)
        parent_values = np.zeros((1, grader.ROWS, 2), dtype="<f4")
        np.save(parent_prediction_path, parent_values)
        self.parent_prediction_sha = _sha(parent_prediction_path)
        self.parent_manifest_sha = "parent-manifest-sha"
        mean = [0.0] * 211
        std = [1.0] * 211
        parent_meta = {
            "parent": {
                "mean_identity": self.parent_ancestry["mean_identity"],
                "std_identity": self.parent_ancestry["std_identity"],
            },
            "endpoint": {"backend": "cpu", "precision": "strict_fp32"},
        }
        self.parent = {
            "prediction_sha256": self.parent_prediction_sha,
            "sidecar_sha256": self.parent_manifest_sha,
            "meta": parent_meta,
        }
        fit_refs = [{"split": "train", "row_group": 5}]
        self.cache = {}
        for kind, refs in (("fit_sequences", fit_refs), ("evaluation_sequences", self.row_refs)):
            membership = grader.row_refs_sha256(refs)
            metadata = {
                "row_refs": refs,
                "membership_sha256": membership,
                "parent_mean": mean,
                "parent_std": std,
                "shape": [len(refs), grader.ROWS, 117],
                "dtype": "<f4",
            }
            record = {
                "source_path": f"{root}/cache/parent_f0/{kind}.npy",
                "sidecar_path": f"{root}/cache/parent_f0/{kind}.json",
                "sidecar_sha256": f"{kind}-sidecar-sha",
                "sha256": f"{kind}-payload-sha",
            }
            self.cache[(0, kind)] = {"metadata": metadata, "record": record}

        prediction_values = np.zeros(
            (1, grader.ROWS, 2), dtype="<f4", order="F" if fortran else "C"
        )
        prediction_path = self.output / "prediction.npy"
        np.save(prediction_path, prediction_values)
        checkpoint_path = self.output / "model_ep2.pt"
        checkpoint_path.write_bytes(b"synthetic checkpoint payload")
        source_path = root / "src/age_readout.py"
        source_path.parent.mkdir(parents=True, exist_ok=True)
        source_path.write_text("synthetic source\n", encoding="utf-8")
        self.fit_hashes = {"src/age_readout.py": _sha(source_path)}
        self.cache_receipt = {
            "source_cache_recipe_path": str(root / "cache/source.json"),
            "source_cache_recipe_sha256": grader.SOURCE_CACHE_SHA256,
            "fit_cache": {
                "path": self.cache[(0, "fit_sequences")]["record"]["source_path"],
                "sidecar_path": self.cache[(0, "fit_sequences")]["record"]["sidecar_path"],
                "sidecar_sha256": self.cache[(0, "fit_sequences")]["record"]["sidecar_sha256"],
                "payload_sha256": self.cache[(0, "fit_sequences")]["record"]["sha256"],
                "membership_sha256": self.cache[(0, "fit_sequences")]["metadata"][
                    "membership_sha256"
                ],
                "shape": self.cache[(0, "fit_sequences")]["metadata"]["shape"],
                "dtype": "<f4",
            },
            "evaluation_cache": {
                "path": self.cache[(0, "evaluation_sequences")]["record"]["source_path"],
                "sidecar_path": self.cache[(0, "evaluation_sequences")]["record"]["sidecar_path"],
                "sidecar_sha256": self.cache[(0, "evaluation_sequences")]["record"][
                    "sidecar_sha256"
                ],
                "payload_sha256": self.cache[(0, "evaluation_sequences")]["record"]["sha256"],
                "membership_sha256": self.eval_membership,
                "shape": self.cache[(0, "evaluation_sequences")]["metadata"]["shape"],
                "dtype": "<f4",
            },
            "parent_prediction_path": "cache/parent_f0/evaluation_predictions.npy",
            "parent_prediction_sha256": self.parent_prediction_sha,
            "parent_manifest_path": "cache/parent_f0/evaluation_predictions.json",
            "parent_manifest_sha256": self.parent_manifest_sha,
            "parent_normalization": {
                "mean": mean,
                "std": std,
                "mean_identity": self.parent_ancestry["mean_identity"],
                "std_identity": self.parent_ancestry["std_identity"],
            },
            "payload_hash_verified": True,
        }
        identity = {
            "recipe_sha256": grader.RECIPE_SHA256,
            "source_sha256": self.fit_hashes,
        }
        self.checkpoint = {
            "path": str(checkpoint_path),
            "sha256": _sha(checkpoint_path),
        }
        self.prediction = {
            "path": str(prediction_path),
            "sha256": _sha(prediction_path),
            "shape": [1, grader.ROWS, 2],
            "dtype": "<f4",
        }
        runtime = {
            "device": "cpu",
            "precision": "strict_fp32",
            "torch_threads": 1,
            "interop_threads": 1,
            "tf32_matmul": False,
            "tf32_cudnn": False,
            "autocast": False,
            "deterministic_algorithms": True,
            "cuda_memory": {
                "allocator_cap_bytes": None,
                "peak_allocated_bytes": None,
                "peak_reserved_bytes": None,
            },
        }
        self.provenance = producer.build_provenance(
            recipe=self.recipe,
            cell=self.cell,
            fold=self.fold,
            arm="A",
            fit_cache=SimpleNamespace(
                metadata=self.cache[(0, "fit_sequences")]["metadata"], row_refs=fit_refs
            ),
            eval_cache=SimpleNamespace(
                metadata=self.cache[(0, "evaluation_sequences")]["metadata"],
                row_refs=self.row_refs,
            ),
            cache_receipt=self.cache_receipt,
            checkpoint=self.checkpoint,
            prediction=self.prediction,
            identity=identity,
            runtime=runtime,
            fit_records=[],
        )
        provenance_path = self.output / "provenance.json"
        _write_json(provenance_path, self.provenance)
        self.provenance_sha = _sha(provenance_path)
        self.sidecar = producer.build_prediction_sidecar(
            recipe=self.recipe,
            cell=self.cell,
            fold=self.fold,
            arm="A",
            recipe_sha=grader.RECIPE_SHA256,
            cache_receipt=self.cache_receipt,
            eval_cache=SimpleNamespace(
                metadata=self.cache[(0, "evaluation_sequences")]["metadata"],
                row_refs=self.row_refs,
            ),
            prediction=self.prediction,
            checkpoint=self.checkpoint,
            provenance_path=provenance_path,
            provenance_sha=self.provenance_sha,
        )
        _write_json(self.output / "prediction.json", self.sidecar)
        self.manifest_entries = []
        for path in (
            prediction_path,
            self.output / "prediction.json",
            checkpoint_path,
            provenance_path,
            source_path,
        ):
            self.manifest_entries.append(
                {
                    "source_path": str(path),
                    "bundle_path": path.relative_to(root).as_posix(),
                    "sha256": _sha(path),
                    "bytes": path.stat().st_size,
                }
            )
        self.bundle = grader.Bundle(
            root,
            {
                "schema": "wnn.age-readout-grade-bundle.v1",
                "source_root": str(root),
                "files": self.manifest_entries,
            },
        )

    def validate(self) -> dict[str, Any]:
        return grader._validate_candidate(
            self.recipe,
            self.cell,
            self.fold,
            0,
            self.cache,
            self.parent,
            self.bundle,
            self.fit_hashes,
        )


class AgeGradeSyntheticTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(dir=TMP_ROOT, prefix="age-grade-synthetic-")
        self.root = Path(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_source_qualified_overlap_allows_same_numeric_id_across_splits(self) -> None:
        grader._require_source_qualified_disjoint(
            {"train": [7], "valid": [9]},
            [{"split": "valid", "row_group": 7}],
            "unexpected overlap",
        )
        with self.assertRaisesRegex(ValueError, "actual overlap"):
            grader._require_source_qualified_disjoint(
                {"train": [7], "valid": [9]},
                [{"split": "valid", "row_group": 9}],
                "actual overlap",
            )

    def test_actor_produced_complete_candidate_sidecar_and_provenance_pass(self) -> None:
        candidate = SyntheticCandidate(self.root)
        result = candidate.validate()
        self.assertEqual(result["run"], grader.EXPECTED_CELLS[0])
        self.assertEqual(result["prediction"].shape, (1, grader.ROWS, 2))

    def test_missing_candidate_artifact_fails_coverage_gate(self) -> None:
        candidate = SyntheticCandidate(self.root)
        manifest = {
            "schema": "wnn.age-readout-grade-bundle.v1",
            "source_root": str(self.root),
            "files": [
                entry
                for entry in candidate.manifest_entries
                if not entry["source_path"].endswith("provenance.json")
            ],
        }
        bundle = grader.Bundle(self.root, manifest)
        with self.assertRaisesRegex(ValueError, "no mapping"):
            grader._candidate_paths(bundle, self.output_path(candidate))

    def test_missing_panel_candidate_stops_before_target_loader(self) -> None:
        manifest_path = self.root / "bundle_manifest.json"
        admission_path = self.root / "admission.json"
        manifest_path.write_text("{}", encoding="utf-8")
        admission_path.write_text("{}", encoding="utf-8")
        recipe = {
            "cells": [
                {"run": run, "fold": fold_id, "arm": arm, "seed": seed}
                for run, fold_id, seed, arm in zip(
                    grader.EXPECTED_CELLS,
                    (0, 0, 0, 1, 1, 1),
                    (1001, 1001, 1001, 1002, 1002, 1002),
                    ("A", "G", "F", "A", "G", "F"),
                    strict=True,
                )
            ]
        }
        folds = {0: {}, 1: {}}
        fake_bundle = SimpleNamespace(
            verify_all=lambda: {},
            path_for=lambda _path: self.root / "cache-verification.json",
        )
        args = [
            "age_readout_grade.py",
            "--bundle-root",
            str(self.root),
            "--bundle-manifest",
            str(manifest_path),
            "--admission",
            str(admission_path),
            "--admission-sha256",
            "synthetic-admission-pin",
            "--output",
            str(self.root / "grade.json"),
        ]
        with (
            mock.patch.object(sys, "argv", args),
            mock.patch.object(grader, "read_json", return_value={}),
            mock.patch.object(grader, "Bundle", return_value=fake_bundle),
            mock.patch.object(
                grader, "validate_recipe", return_value=(recipe, grader.RECIPE_SHA256, folds)
            ),
            mock.patch.object(
                grader,
                "validate_admission",
                return_value={
                    "cache_verification": {"path": "cache-verification.json"},
                    "fit_source_sha256": {},
                    "shared_readout_device": "cpu",
                },
            ),
            mock.patch.object(grader, "_cache_entries", return_value={}),
            mock.patch.object(grader, "_validate_parent", return_value={}),
            mock.patch.object(
                grader,
                "_validate_candidate",
                side_effect=ValueError("synthetic missing candidate artifact"),
            ),
            mock.patch.object(grader, "_load_targets") as load_targets,
        ):
            with self.assertRaisesRegex(ValueError, "missing candidate"):
                grader.main()
            load_targets.assert_not_called()

    @staticmethod
    def output_path(candidate: SyntheticCandidate) -> Path:
        return Path(candidate.cell["output"])

    def test_wrong_prediction_bytes_fail_before_label_loading(self) -> None:
        candidate = SyntheticCandidate(self.root)
        with (candidate.output / "prediction.npy").open("ab") as stream:
            stream.write(b"tamper")
        with self.assertRaisesRegex(ValueError, "prediction payload hash"):
            candidate.validate()

    def test_fortran_order_header_is_rejected(self) -> None:
        candidate = SyntheticCandidate(self.root, fortran=True)
        with self.assertRaisesRegex(ValueError, "Fortran-order"):
            candidate.validate()

    def test_wrong_source_bytes_fail_before_label_loading(self) -> None:
        candidate = SyntheticCandidate(self.root)
        (self.root / "src/age_readout.py").write_text("tampered source\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "training source changed"):
            candidate.validate()

    def test_missing_mapping_and_symlink_are_rejected(self) -> None:
        target = self.root / "real.txt"
        target.write_text("contents", encoding="utf-8")
        link = self.root / "linked.txt"
        link.symlink_to(target)
        manifest = {
            "schema": "wnn.age-readout-grade-bundle.v1",
            "source_root": str(self.root),
            "files": [
                {
                    "source_path": str(self.root / "linked.txt"),
                    "bundle_path": "linked.txt",
                    "sha256": _sha(target),
                    "bytes": target.stat().st_size,
                }
            ],
        }
        with self.assertRaisesRegex(ValueError, "symlinks are forbidden"):
            grader.Bundle(self.root, manifest)

    def test_content_pin_accepts_moved_mtime_but_rejects_changed_bytes(self) -> None:
        path = self.root / "transported-target.npz"
        payload = b"synthetic target archive; labels are not decoded"
        path.write_bytes(payload)
        original = path.stat().st_mtime_ns
        os.utime(path, ns=(original + 2_000_000_000, original + 2_000_000_000))
        identity = grader._transported_file_identity(
            path, hashlib.sha256(payload).hexdigest(), len(payload)
        )
        self.assertNotEqual(identity["transport_mtime_ns"], original)
        self.assertEqual(identity["sha256"], hashlib.sha256(payload).hexdigest())
        path.write_bytes(payload + b"x")
        with self.assertRaisesRegex(ValueError, "byte count"):
            grader._transported_file_identity(
                path, hashlib.sha256(payload).hexdigest(), len(payload)
            )

    def test_age_score_wrapper_matches_canonical_global_wp(self) -> None:
        import online_relationship_grade_core as helpers

        from wnn.gmetric import global_wp

        rng = np.random.default_rng(8128)
        targets = rng.normal(size=(1, grader.ROWS, 2)).astype("<f4")
        predictions = rng.normal(size=targets.shape).astype("<f4")
        mask = rng.random((1, grader.ROWS)) > 0.15
        scored = mask & (np.arange(grader.ROWS)[None, :] >= 99)
        actual = helpers.official_scores(targets, mask, predictions)["overall"]
        expected = global_wp(targets, predictions, scored)
        for name in ("wp", "t0", "t1"):
            self.assertAlmostEqual(actual[name], expected[name], places=12)


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0], *_unittest_args], verbosity=2)
