"""Admission-gated grader for the six-cell age readout screen."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

RECIPE_RELATIVE = Path("research/experiment_drafts/b300-fast-local-kaggle-20261023/recipe.json")
RECIPE_SHA256 = "0a67fe12c520e56bcf14a3a1efecb5030f3aa7ce932101814c9dd1c675c930fd"
TARGET_SHA256 = "dbb7a1a48dd0af9fa524a80c2694008a56cb91d5dae80e8d0d4984c0c5622955"
SOURCE_CACHE_SHA256 = "edc873c813de77f55859d1a13180d907004e5f18fcc537e64eda7af1308e2508"
PARENT_MILESTONE_RELATIVE = Path(
    "research/experiment_drafts/online-relationships-20261022/PARENT_CACHE_MILESTONE_R1.json"
)
PARENT_MILESTONE_SHA256 = "0dd9f9cec9bf54a4c2abdaeb835ad13f099e201a4b578714838014e39eb4d11e"
CACHE_VERIFY_ADMISSION_RELATIVE = Path(
    "research/experiment_drafts/b300-fast-local-kaggle-20261023/CACHE_VERIFY_ADMISSION_R1.json"
)
CACHE_VERIFY_ADMISSION_SHA256 = "052424d4d631bce24741ea0b2f75218618e78ca04b9be812ed3a616a42ccfff6"
CACHE_VERIFY_SPEC_RELATIVE = Path(
    "research/experiment_drafts/b300-fast-local-kaggle-20261023/CACHE_VERIFY_SPEC_R1.json"
)
CACHE_VERIFY_SPEC_SHA256 = "ebf1f1fa163f46bbae4e690bdf4507b4f4caca4351d75af42c57a18138fee8a6"
CACHE_VERIFY_GUARD_RELATIVE = Path(
    "research/experiment_drafts/b300-fast-local-kaggle-20261023/CACHE_VERIFY_GUARD_R1.json"
)
CACHE_VERIFY_GUARD_SHA256 = "f5fec8727cdbfc449995dc3e50fc04af70432941870257ec8f7646d5a0e13e12"
EXPERIMENT_ID = "E-AGE-READOUT-20261023-R1"
ROWS = 20_000
ARMS = ("A", "G", "F")
EXPECTED_CELLS = (
    "age_readout_f0_s1001_a_r1",
    "age_readout_f0_s1001_g_r1",
    "age_readout_f0_s1001_f_r1",
    "age_readout_f1_s1002_a_r1",
    "age_readout_f1_s1002_g_r1",
    "age_readout_f1_s1002_f_r1",
)
CORE_DEPENDENCIES = {
    "research/experiment_drafts/online-relationships-20261022/online_relationship_grade_core.py": "b8480f1ac9f9068e076363486076f1965eeb65c364099c891fa25b2dd03533da",
    "scripts/grade_prediction_bank.py": "fb7a85a5837f61fa6c65cd10a799768006067e87d4c18e92d4b813ba83b9d82e",
    "src/wnn/evaluation.py": "96cebf2c8d564493f5a3a866b8d879964247f885587a455ebb1449e937230d0e",
    "src/wnn/gmetric.py": "2cf10e740a831379f62b7d1269456053f5c1f9efa2dbab03156e366972bed5aa",
}
READ_CONTRACT = {
    "raw_inputs": True,
    "cached_parent_predictions": True,
    "targets": False,
    "row_weights": False,
    "scoring_masks": False,
    "prediction_request_flags": False,
}
CONTRASTS = (("F", "A"), ("F", "G"), ("A", "parent"), ("G", "parent"), ("F", "parent"))


def sha256_file(path: Path) -> str:
    """Hash a file in bounded memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path, *, max_bytes: int = 8_000_000) -> dict[str, Any]:
    """Read bounded JSON metadata and require an object root."""
    size = path.stat().st_size
    if size <= 0 or size > max_bytes:
        raise ValueError(f"metadata size is invalid: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"metadata root must be an object: {path}")
    return value


def row_refs_sha256(refs: list[dict[str, Any]]) -> str:
    """Hash ordered, normalized source-qualified row references."""
    normalized = [{"split": str(ref["split"]), "row_group": int(ref["row_group"])} for ref in refs]
    raw = json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(raw).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _require_source_qualified_disjoint(
    fit_groups: dict[str, list[int]], evaluation_refs: list[dict[str, Any]], message: str
) -> None:
    """Compare fit and evaluation identities as (split, group), never bare group IDs."""
    fit_refs = {(split, int(group)) for split, groups in fit_groups.items() for group in groups}
    evaluation = {(str(ref["split"]), int(ref["row_group"])) for ref in evaluation_refs}
    _require(not (fit_refs & evaluation), message)


def _transported_file_identity(
    path: Path, expected_sha256: str, expected_bytes: int
) -> dict[str, Any]:
    """Validate transported content while recording, never forging, its mtime."""
    actual_bytes = path.stat().st_size
    _require(actual_bytes == expected_bytes, "transported file byte count differs from source pin")
    actual_sha = sha256_file(path)
    _require(actual_sha == expected_sha256, "transported file SHA256 differs from source pin")
    return {
        "path": str(path),
        "sha256": actual_sha,
        "bytes": actual_bytes,
        "transport_mtime_ns": path.stat().st_mtime_ns,
    }


def _require_prediction_header(
    path: Path, expected_shape: list[int], expected_dtype: str, label: str
) -> dict[str, Any]:
    """Require complete C-order NPY coverage and the exact endpoint header."""
    header = _npy_header(path)
    _require(
        header["shape"] == expected_shape and header["descr"] == expected_dtype,
        f"{label}: NPY shape or dtype differs",
    )
    _require(header["fortran_order"] is False, f"{label}: Fortran-order NPY is forbidden")
    return header


def _candidate_paths(bundle: Bundle, output: Path) -> dict[str, Path]:
    """Resolve all four required cell artifacts before inspecting prediction values."""
    return {
        "prediction": bundle.path_for(output / "prediction.npy"),
        "sidecar": bundle.path_for(output / "prediction.json"),
        "checkpoint": bundle.path_for(output / "model_ep2.pt"),
        "provenance": bundle.path_for(output / "provenance.json"),
    }


def _require_source_hash_map(
    actual: dict[str, Any], expected: dict[str, Any], bundle: Bundle
) -> None:
    """Bind the candidate's exact source map to admission and packaged source bytes."""
    _require(actual == expected, "candidate source hash map differs from root admission")
    for name, digest in actual.items():
        _require(
            sha256_file(bundle.path_for(name)) == digest,
            f"candidate training source changed: {name}",
        )


class Bundle:
    """Resolve and verify the root-authored original-path to bundle-path map."""

    def __init__(self, root: Path, manifest: dict[str, Any]) -> None:
        self.root = root.resolve(strict=True)
        self.manifest = manifest
        _require(
            manifest.get("schema") == "wnn.age-readout-grade-bundle.v1",
            "unsupported bundle manifest schema",
        )
        source_root = Path(str(manifest.get("source_root", "")))
        _require(source_root.is_absolute(), "bundle source_root must be absolute")
        self.source_root = source_root
        entries = manifest.get("files")
        _require(isinstance(entries, list) and entries, "bundle file map is empty")
        self.entries: dict[str, dict[str, Any]] = {}
        seen_paths: set[Path] = set()
        for entry in entries:
            _require(isinstance(entry, dict), "bundle file entry is malformed")
            source = self.normalize_source(str(entry.get("source_path", "")))
            relative = Path(str(entry.get("bundle_path", "")))
            _require(
                not relative.is_absolute()
                and all(part not in ("", ".", "..") for part in relative.parts),
                f"unsafe bundle path: {relative}",
            )
            probe = self.root
            for part in relative.parts:
                probe = probe / part
                _require(not probe.is_symlink(), f"bundle symlinks are forbidden: {relative}")
            target = (self.root / relative).resolve(strict=True)
            _require(target.is_relative_to(self.root), f"bundle path escapes root: {relative}")
            _require(source not in self.entries, f"duplicate source alias: {source}")
            _require(target not in seen_paths, f"duplicate bundle path alias: {relative}")
            seen_paths.add(target)
            self.entries[source] = {**entry, "bundle_target": target}

    @staticmethod
    def normalize_source(value: str) -> str:
        """Normalize an original absolute source path without touching Kaggle paths."""
        path = Path(value)
        _require(path.is_absolute(), f"source path must be absolute: {value}")
        return path.as_posix()

    def source_path(self, value: str | Path) -> str:
        """Resolve a repository-relative path against the bundle's original source root."""
        path = Path(value)
        if not path.is_absolute():
            path = self.source_root / path
        return self.normalize_source(path.as_posix())

    def path_for(self, source: str | Path) -> Path:
        """Return a manifested packaged file for its original path."""
        key = self.source_path(source)
        if key not in self.entries:
            raise ValueError(f"bundle has no mapping for source path: {key}")
        return Path(self.entries[key]["bundle_target"])

    def verify_all(self) -> dict[str, dict[str, Any]]:
        """Hash every transported file before prediction or target arrays are opened."""
        verified: dict[str, dict[str, Any]] = {}
        for source, entry in self.entries.items():
            path = Path(entry["bundle_target"])
            expected_bytes = int(entry["bytes"])
            expected_sha = str(entry["sha256"])
            actual_bytes = path.stat().st_size
            _require(actual_bytes == expected_bytes, f"transport byte count differs: {source}")
            actual_sha = sha256_file(path)
            _require(actual_sha == expected_sha, f"transport SHA256 differs: {source}")
            verified[source] = {
                "path": str(path),
                "bytes": actual_bytes,
                "sha256": actual_sha,
                "transport_mtime_ns": path.stat().st_mtime_ns,
            }
        return verified


def _source_sha(bundle: Bundle, hashes: dict[str, Any], name: str) -> str:
    """Return a source digest from either a source-root-relative or absolute key."""
    if name in hashes:
        return str(hashes[name])
    absolute = bundle.source_path(name)
    if absolute in hashes:
        return str(hashes[absolute])
    raise ValueError(f"source hash is absent from the frozen receipt: {name}")


def validate_admission(
    admission_path: Path,
    admission_sha: str,
    recipe: dict[str, Any],
    recipe_sha: str,
    bundle: Bundle,
    bundle_manifest_path: Path,
) -> dict[str, Any]:
    """Require the exact root-issued grade admission before inspecting predictions."""
    _require(sha256_file(admission_path) == admission_sha, "admission receipt SHA256 differs")
    receipt = read_json(admission_path)
    _require(
        receipt.get("schema") == "wnn.age-readout-grade-admission.v1",
        "unsupported grade-admission schema",
    )
    _require(receipt.get("status") == "admitted", "root has not admitted quality grading")
    _require(receipt.get("allow_partial") is False, "partial grading is forbidden")
    _require(receipt.get("experiment_id") == EXPERIMENT_ID, "admission experiment differs")
    _require(receipt.get("recipe_sha256") == recipe_sha == RECIPE_SHA256, "recipe SHA differs")
    _require(receipt.get("metric") == "global_wp_clip2_v1", "metric differs")
    _require(
        receipt.get("required_cells") == list(EXPECTED_CELLS), "admitted cell set/order differs"
    )
    artifact = receipt.get("artifact_manifest", {})
    _require(isinstance(artifact, dict), "admission has no artifact manifest identity")
    _require(
        str(artifact.get("sha256", "")) == sha256_file(bundle_manifest_path),
        "bundle manifest SHA differs from root admission",
    )
    _require(bool(artifact.get("path")), "admission artifact manifest path is missing")
    manifest_source_path = bundle.manifest.get("source_path")
    if manifest_source_path is not None:
        _require(
            str(manifest_source_path) == str(artifact["path"]),
            "bundle manifest original path differs from root admission",
        )
    cache_record = receipt.get("cache_verification", {})
    _require(isinstance(cache_record, dict), "admission has no cache verification identity")
    cache_path = bundle.path_for(str(cache_record.get("path", "")))
    _require(
        sha256_file(cache_path) == str(cache_record.get("sha256", "")),
        "cache verification receipt differs from root admission",
    )
    limits = receipt.get("limits", {})
    _require(
        int(limits.get("cpu_cores", 0)) <= 2
        and int(limits.get("ram_bytes", 0)) <= 4_000_000_000
        and int(limits.get("wall_seconds", 0)) <= 1_800,
        "grade resource limits exceed the frozen allocation",
    )
    shared_device = str(receipt.get("shared_readout_device", ""))
    _require(
        shared_device == "cpu" or re.fullmatch(r"cuda:[0-9]+", shared_device) is not None,
        "admission does not pin a shared readout device",
    )
    fit_hashes = receipt.get("fit_source_sha256", {})
    grader_hashes = receipt.get("grader_source_sha256", {})
    _require(isinstance(fit_hashes, dict) and fit_hashes, "admission has no fit source hashes")
    _require(
        isinstance(grader_hashes, dict) and grader_hashes, "admission has no grader source hashes"
    )
    adapter_key = "grader/age_readout_grade.py"
    _require(
        grader_hashes.get(adapter_key) == sha256_file(Path(__file__)),
        "grader adapter source hash differs from the admitted source",
    )
    for name, expected in CORE_DEPENDENCIES.items():
        _require(
            _source_sha(bundle, grader_hashes, name) == expected,
            f"frozen score dependency hash differs: {name}",
        )
        _require(
            sha256_file(bundle.path_for(name)) == expected,
            f"transported score dependency hash differs: {name}",
        )
    return receipt


def _recipe_source_path(bundle: Bundle, relative: str | Path) -> Path:
    """Resolve a recipe path to its packaged file."""
    return bundle.path_for(relative)


def validate_recipe(bundle: Bundle) -> tuple[dict[str, Any], str, dict[int, dict[str, Any]]]:
    """Validate the frozen six-cell recipe and its source-qualified splits."""
    recipe_path = _recipe_source_path(bundle, RECIPE_RELATIVE)
    recipe_sha = sha256_file(recipe_path)
    _require(recipe_sha == RECIPE_SHA256, "frozen recipe bytes differ")
    recipe = read_json(recipe_path)
    _require(recipe.get("experiment_id") == EXPERIMENT_ID, "unexpected experiment identity")
    _require(int(recipe.get("new_readout_fits", -1)) == 6, "recipe is not the six-fit screen")
    _require(int(recipe.get("new_gru_fits", -1)) == 0, "recipe contains new GRU fits")
    _require(int(recipe.get("new_teacher_fits", -1)) == 0, "recipe contains new teacher fits")
    _require(
        recipe.get("evaluation", {}).get("metric") == "global_wp_clip2_v1", "recipe metric differs"
    )
    cells = recipe.get("cells", [])
    _require(
        [str(cell.get("run")) for cell in cells] == list(EXPECTED_CELLS),
        "recipe cell order differs",
    )
    folds = {int(record["fold"]): record for record in recipe.get("folds", [])}
    _require(set(folds) == {0, 1}, "recipe must have exactly F0 and F1")
    for fold_id, fold in folds.items():
        split_path = _recipe_source_path(bundle, fold["split_manifest"])
        split_sha = sha256_file(split_path)
        _require(split_sha == fold["split_manifest_sha256"], f"F{fold_id} split SHA differs")
        split = read_json(split_path)
        eval_groups = [int(value) for value in fold["evaluation_groups"]]
        _require(
            split.get("evaluation", {}).get("valid") == eval_groups,
            f"F{fold_id} evaluation order differs from split",
        )
        _require(len(eval_groups) == (468 if fold_id == 0 else 457), f"F{fold_id} is incomplete")
        refs = [{"split": "valid", "row_group": group} for group in eval_groups]
        _require(
            len(refs) == len(set((r["split"], r["row_group"]) for r in refs)),
            f"F{fold_id} has duplicate evaluation refs",
        )
        fold["_split"] = split
        fold["_expected_eval_refs"] = refs
        subset_path = _recipe_source_path(bundle, fold["subset"]["path"])
        _require(
            sha256_file(subset_path) == fold["subset"]["sha256"],
            f"F{fold_id} subset manifest SHA differs",
        )
        fold["_subset"] = read_json(subset_path)
        subset = fold["_subset"]
        _require(
            subset.get("split_manifest_sha256") == fold["split_manifest_sha256"],
            f"F{fold_id} subset split identity differs",
        )
        _require(
            subset.get("evaluation_groups") == sorted(eval_groups),
            f"F{fold_id} subset excludes a different evaluation fold",
        )
        _require(
            subset.get("evaluation_membership_sha256") == fold["evaluation_membership_sha256"],
            f"F{fold_id} recipe evaluation membership digest differs from subset",
        )
        _require_source_qualified_disjoint(
            subset.get("selected_groups", {}),
            refs,
            f"F{fold_id} fit subset overlaps evaluation groups",
        )
    return recipe, recipe_sha, folds


def _cache_entries(
    receipt: dict[str, Any],
    recipe: dict[str, Any],
    folds: dict[int, dict[str, Any]],
    bundle: Bundle,
) -> dict[tuple[int, str], dict[str, Any]]:
    """Validate root's local cache-hash attestation and six producer sidecars."""
    _require(
        receipt.get("schema") == "wnn.age-readout-local-cache-verification.v1",
        "unsupported local cache-verification schema",
    )
    _require(
        receipt.get("status") == "PASS_PAYLOAD_HASH_AND_HEADER",
        "local cache payload/hash verification did not pass",
    )
    _require(receipt.get("recipe_sha256") == RECIPE_SHA256, "cache receipt age recipe differs")
    verify_admission_path = bundle.path_for(CACHE_VERIFY_ADMISSION_RELATIVE)
    verify_spec_path = bundle.path_for(CACHE_VERIFY_SPEC_RELATIVE)
    verify_guard_path = bundle.path_for(CACHE_VERIFY_GUARD_RELATIVE)
    _require(
        sha256_file(verify_admission_path) == CACHE_VERIFY_ADMISSION_SHA256,
        "source-bound cache-verification admission differs",
    )
    _require(
        sha256_file(verify_spec_path) == CACHE_VERIFY_SPEC_SHA256,
        "cache-verification specification differs",
    )
    _require(
        sha256_file(verify_guard_path) == CACHE_VERIFY_GUARD_SHA256,
        "cache-verification execution guard differs",
    )
    verify_admission = read_json(verify_admission_path)
    _require(
        verify_admission.get("status") == "admitted_cache_hash_verification_only"
        and verify_admission.get("label_values_decoded") is False
        and int(verify_admission.get("new_fits", -1)) == 0
        and int(verify_admission.get("swap_bytes", -1)) == 0,
        "cache-verification admission exceeds its read-only scope",
    )
    _require(
        verify_admission.get("spec_sha256") == CACHE_VERIFY_SPEC_SHA256,
        "cache-verification admission is not bound to its spec",
    )
    for name, digest in verify_admission.get("source_sha256", {}).items():
        _require(
            sha256_file(bundle.path_for(name)) == digest,
            f"cache-verification source differs: {name}",
        )
    verify_guard = read_json(verify_guard_path)
    _require(
        verify_guard.get("status") == "complete"
        and verify_guard.get("returncode") == 0
        and 0
        <= float(verify_guard.get("wall_seconds", -1))
        <= float(verify_admission["max_wall_seconds"]),
        "cache-verification execution was not complete within its bound",
    )
    source_recipe = receipt.get("source_cache_recipe", {})
    _require(
        source_recipe.get("path") == recipe["source_cache_recipe"]["path"]
        and source_recipe.get("sha256") == SOURCE_CACHE_SHA256,
        "cache receipt producer recipe differs",
    )
    source_recipe_path = bundle.path_for(source_recipe["path"])
    _require(
        sha256_file(source_recipe_path) == SOURCE_CACHE_SHA256, "source cache recipe hash differs"
    )
    milestone_path = bundle.path_for(PARENT_MILESTONE_RELATIVE)
    _require(
        sha256_file(milestone_path) == PARENT_MILESTONE_SHA256,
        "frozen parent-cache milestone differs",
    )
    milestone = read_json(milestone_path)
    _require(
        milestone.get("schema") == "wnn.online-relationships.parent-cache-milestone.v1"
        and milestone.get("status") == "both_full_parent_caches_published_fit_panel_started",
        "parent-cache milestone identity differs",
    )
    producer_sources = receipt.get("producer_source_sha256", {})
    _require(isinstance(producer_sources, dict), "cache receipt producer source map is malformed")
    for name, digest in producer_sources.items():
        _require(
            sha256_file(bundle.path_for(name)) == digest, f"cache producer source changed: {name}"
        )
    entries = receipt.get("files", [])
    for _fold_id, fold in folds.items():
        for kind in ("fit_sequences", "evaluation_sequences"):
            omitted_payload = Bundle.normalize_source(
                str(Path(fold["parent_cache"]) / f"{kind}.npy")
            )
            _require(
                omitted_payload not in bundle.entries,
                f"raw {kind} payload must not be transferred to Kaggle",
            )
    expected_keys = {
        (fold, kind)
        for fold in (0, 1)
        for kind in ("fit_sequences", "evaluation_sequences", "evaluation_predictions")
    }
    mapped: dict[tuple[int, str], dict[str, Any]] = {}
    for entry in entries:
        key = (int(entry["fold"]), str(entry["kind"]))
        _require(key not in mapped, f"duplicate local cache receipt entry: {key}")
        _require(key in expected_keys, f"unexpected local cache receipt entry: {key}")
        payload_source = Bundle.normalize_source(str(entry["source_path"]))
        sidecar_path = bundle.path_for(str(entry["sidecar_path"]))
        _require(
            sha256_file(sidecar_path) == str(entry["sidecar_sha256"]),
            f"cached sidecar attestation differs: {key}",
        )
        if key[1] == "evaluation_predictions":
            expected_name = "evaluation_predictions.npy"
        else:
            expected_name = f"{key[1]}.npy"
        expected_source = str(Path(folds[key[0]]["parent_cache"]) / expected_name)
        _require(
            payload_source == Bundle.normalize_source(expected_source),
            f"cached payload path differs from recipe: {key}",
        )
        expected_sidecar = str(Path(folds[key[0]]["parent_cache"]) / f"{key[1]}.json")
        _require(
            Bundle.normalize_source(str(entry["sidecar_path"]))
            == Bundle.normalize_source(expected_sidecar),
            f"cached sidecar path differs from recipe: {key}",
        )
        header = entry.get("header", {})
        _require(
            int(header.get("file_bytes", -1)) == int(entry["identity"]["bytes"]),
            "cache byte header differs",
        )
        _require(
            header.get("descr") == "<f4" and header.get("fortran_order") is False,
            "cache dtype/order differs",
        )
        _require(
            int(header.get("payload_bytes", -1)) == math.prod(header.get("shape", [])) * 4
            and int(header.get("file_bytes", -1))
            == int(header.get("header_bytes", -1)) + int(header.get("payload_bytes", -1)),
            f"local cache receipt has incomplete NPY coverage: {key}",
        )
        metadata = read_json(sidecar_path)
        milestone_entry = milestone["parent_caches"][str(key[0])]["sidecars"][f"{key[1]}.json"]
        _require(
            milestone_entry.get("path") == str(entry["sidecar_path"])
            and milestone_entry.get("sha256") == entry["sidecar_sha256"],
            f"original producer sidecar differs from the frozen milestone: {key}",
        )
        _require(
            metadata.get("data_sha256", metadata.get("prediction_sha256")) == entry["sha256"],
            "cache sidecar payload hash differs",
        )
        _require(metadata.get("shape") == header.get("shape"), "cache sidecar shape differs")
        _require(metadata.get("dtype") == header.get("descr"), "cache sidecar dtype differs")
        _require(
            metadata.get("recipe_sha256") == SOURCE_CACHE_SHA256,
            "cache sidecar producer recipe differs",
        )
        _require(int(metadata.get("fold", -1)) == key[0], "cache sidecar fold differs")
        _require(
            int(metadata.get("seed", -1)) == int(folds[key[0]]["seed"]),
            "cache sidecar seed differs",
        )
        _require(
            metadata.get("split_manifest_sha256") == folds[key[0]]["split_manifest_sha256"],
            "cache sidecar split differs",
        )
        if key[1] == "fit_sequences":
            _require(metadata.get("scope") == "in_sample_fit", "fit cache scope differs")
            _require(
                metadata.get("read_contract") == {"targets": True, "scored_mask": True},
                "fit cache read contract differs",
            )
            subset = folds[key[0]]["_subset"]
            expected_fit_refs = [
                {"split": split, "row_group": int(group)}
                for split in ("train", "valid")
                for group in subset["selected_groups"][split]
            ]
            _require(
                metadata.get("row_refs") == expected_fit_refs,
                "fit cache refs differ from frozen subset",
            )
            _require(
                metadata.get("subset_manifest_sha256") == folds[key[0]]["subset"]["sha256"],
                "fit cache subset identity differs",
            )
            _require(
                metadata.get("subset_manifest") == folds[key[0]]["subset"]["path"],
                "fit cache subset path differs",
            )
        elif key[1] == "evaluation_sequences":
            _require(metadata.get("scope") == "feature_only_eval", "evaluation cache scope differs")
            _require(
                metadata.get("read_contract") == {"targets": False, "scored_mask": False},
                "evaluation cache may contain target access",
            )
            _require(
                metadata.get("row_refs") == folds[key[0]]["_expected_eval_refs"],
                "evaluation cache row order differs",
            )
            _require(
                metadata.get("membership_sha256")
                == row_refs_sha256(folds[key[0]]["_expected_eval_refs"]),
                "evaluation cache membership differs",
            )
            _require(
                metadata.get("subset_manifest_sha256") == folds[key[0]]["subset"]["sha256"],
                "evaluation cache subset identity differs",
            )
        payload_path: Path | None = None
        if key[1] == "evaluation_predictions":
            payload_path = bundle.path_for(payload_source)
            _require(
                sha256_file(payload_path) == str(entry["sha256"]),
                "parent prediction payload attestation differs",
            )
            _require(
                _npy_header(payload_path) == header, "transported parent prediction header differs"
            )
        mapped[key] = {
            "record": entry,
            "metadata": metadata,
            "path": payload_path,
            "sidecar_path": sidecar_path,
        }
    _require(
        set(mapped) == expected_keys, "local cache receipt does not contain exactly six entries"
    )
    return mapped


def _npy_header(path: Path) -> dict[str, Any]:
    """Parse a complete C-order NPY header without loading its payload."""
    with path.open("rb") as stream:
        version = np.lib.format.read_magic(stream)
        if version == (1, 0):
            shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(stream)
        elif version in ((2, 0), (3, 0)):
            shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(stream)
        else:
            raise ValueError(f"unsupported NPY version {version}: {path}")
        header_bytes = stream.tell()
    dtype = np.dtype(dtype)
    payload_bytes = math.prod(shape) * dtype.itemsize
    file_bytes = path.stat().st_size
    _require(file_bytes == header_bytes + payload_bytes, f"NPY payload coverage differs: {path}")
    return {
        "version": list(version),
        "descr": dtype.str,
        "shape": list(shape),
        "fortran_order": fortran_order,
        "header_bytes": header_bytes,
        "file_bytes": file_bytes,
        "payload_bytes": payload_bytes,
    }


def _validate_parent(
    recipe: dict[str, Any],
    fold_id: int,
    fold: dict[str, Any],
    caches: dict[tuple[int, str], dict[str, Any]],
    bundle: Bundle,
) -> dict[str, Any]:
    """Check the frozen parent metadata and exact validation row order."""
    record = caches[(fold_id, "evaluation_predictions")]
    meta = record["metadata"]
    parent = fold["parent"]
    _require(
        meta.get("schema") == "wnn.online-relationships.parent-predictions.v1",
        "parent sidecar schema differs",
    )
    _require(
        meta.get("fold") == fold_id and meta.get("seed") == fold["seed"],
        "parent sidecar cell differs",
    )
    _require(meta.get("recipe_sha256") == SOURCE_CACHE_SHA256, "parent cache recipe differs")
    _require(meta.get("row_refs") == fold["_expected_eval_refs"], "parent row order differs")
    _require(
        meta.get("shape") == [len(fold["_expected_eval_refs"]), ROWS, 2], "parent shape differs"
    )
    _require(meta.get("dtype") == "<f4", "parent dtype differs")
    endpoint = meta.get("endpoint", {})
    _require(
        endpoint.get("precision") == "strict_fp32" and endpoint.get("backend") == "cpu",
        "parent endpoint is not frozen CPU FP32",
    )
    _require(meta.get("scope") == "feature_only_eval", "parent predictions are not feature-only")
    _require(
        meta.get("membership_sha256") == row_refs_sha256(fold["_expected_eval_refs"]),
        "parent membership digest differs",
    )
    identity = meta.get("parent", {})
    for field in (
        "run",
        "checkpoint_sha256",
        "provenance_sha256",
        "initial_ancestor",
        "initial_ancestor_checkpoint_sha256",
        "mean_identity",
        "std_identity",
        "historical_precision",
        "history_fit_membership",
        "input_width",
        "output_rows",
        "epoch",
    ):
        _require(
            identity.get(field) == parent.get(field),
            f"F{fold_id} parent {field} differs from recipe",
        )
    for kind in ("fit_sequences", "evaluation_sequences"):
        cache_parent = caches[(fold_id, kind)]["metadata"].get("parent")
        _require(
            cache_parent.get("run") == identity.get("run"), f"F{fold_id} {kind} parent run differs"
        )
        _require(
            cache_parent.get("checkpoint_sha256") == identity.get("checkpoint_sha256"),
            f"F{fold_id} {kind} parent checkpoint differs",
        )
        _require(
            cache_parent.get("provenance_sha256") == identity.get("provenance_sha256"),
            f"F{fold_id} {kind} parent provenance differs",
        )
        _require(
            cache_parent.get("epoch") == identity.get("epoch"),
            f"F{fold_id} {kind} parent epoch differs",
        )
        _require(
            cache_parent.get("input_width") == identity.get("input_width"),
            f"F{fold_id} {kind} parent input width differs",
        )
        _require(
            cache_parent.get("mean_sha256") == identity.get("mean_identity"),
            f"F{fold_id} {kind} parent mean identity differs",
        )
        _require(
            cache_parent.get("std_sha256") == identity.get("std_identity"),
            f"F{fold_id} {kind} parent std identity differs",
        )
        _require(
            cache_parent.get("endpoint_backend") == "cpu",
            f"F{fold_id} {kind} parent backend differs",
        )
        _require(
            cache_parent.get("endpoint_precision") == "strict_fp32",
            f"F{fold_id} {kind} parent precision differs",
        )
    pred_path = record["path"]
    _require(
        meta.get("prediction_sha256") == record["record"]["sha256"],
        "parent prediction hash differs",
    )
    _require(
        meta.get("prediction_path")
        == str(Path(fold["parent_cache"]) / "evaluation_predictions.npy"),
        "parent prediction path differs",
    )
    provenance_path = bundle.path_for(parent["provenance"])
    provenance_sha = sha256_file(provenance_path)
    _require(
        provenance_sha == parent["provenance_sha256"],
        f"F{fold_id} original parent provenance hash differs",
    )
    provenance = read_json(provenance_path)
    _require(
        provenance.get("split_manifest_sha256") == fold["split_manifest_sha256"],
        f"F{fold_id} parent provenance split differs",
    )
    _require(
        provenance.get("selected_epoch") == parent["epoch"],
        f"F{fold_id} parent selected epoch differs",
    )
    checkpoint_name = Path(parent["checkpoint"]).name
    _require(
        provenance.get("checkpoint_sha256", {}).get(checkpoint_name) == parent["checkpoint_sha256"],
        f"F{fold_id} parent checkpoint hash provenance differs",
    )
    _require(
        provenance.get("initial_checkpoint_sha256") == parent["initial_ancestor_checkpoint_sha256"],
        f"F{fold_id} parent initializer ancestry differs",
    )
    normalization = provenance.get("normalization_identity", {})
    _require(
        normalization.get("mean", {}).get("bytes_sha256") == parent["mean_identity"],
        f"F{fold_id} parent normalization mean differs",
    )
    _require(
        normalization.get("std", {}).get("bytes_sha256") == parent["std_identity"],
        f"F{fold_id} parent normalization std differs",
    )
    expected_eval_groups = {int(ref["row_group"]) for ref in fold["_expected_eval_refs"]}
    provenance_eval_groups = {int(group) for group in provenance.get("evaluation_row_groups", [])}
    _require(
        provenance_eval_groups == expected_eval_groups,
        f"F{fold_id} parent provenance evaluation groups differ",
    )
    fit_groups = provenance.get("fit_row_groups", {})
    _require_source_qualified_disjoint(
        fit_groups,
        fold["_expected_eval_refs"],
        f"F{fold_id} parent provenance fit/evaluation groups overlap",
    )
    expected_shape = [len(fold["_expected_eval_refs"]), ROWS, 2]
    _require_prediction_header(pred_path, expected_shape, "<f4", f"F{fold_id} parent")
    prediction = np.load(pred_path, mmap_mode="r", allow_pickle=False)
    _require(
        prediction.shape == tuple(expected_shape) and prediction.dtype.str == "<f4",
        f"F{fold_id}: parent loaded shape/dtype differs",
    )
    for index in range(expected_shape[0]):
        _require(
            bool(np.isfinite(prediction[index]).all()), f"F{fold_id}: nonfinite parent predictions"
        )
    return {
        "prediction": pred_path,
        "prediction_path": pred_path,
        "prediction_array": prediction,
        "prediction_shape": expected_shape,
        "sidecar": record["sidecar_path"],
        "sidecar_path": record["sidecar_path"],
        "sidecar_sha256": record["record"]["sidecar_sha256"],
        "prediction_sha256": record["record"]["sha256"],
        "provenance_path": provenance_path,
        "provenance_sha256": provenance_sha,
        "parent_checkpoint_sha256": parent["checkpoint_sha256"],
        "meta": meta,
    }


def _validate_candidate(
    recipe: dict[str, Any],
    cell: dict[str, Any],
    fold: dict[str, Any],
    fold_id: int,
    cache: dict[str, Any],
    parent: dict[str, Any],
    bundle: Bundle,
    fit_hashes: dict[str, Any],
) -> dict[str, Any]:
    """Validate a candidate sidecar, provenance, checkpoint and cache ancestry."""
    run = str(cell["run"])
    output = Path(str(cell["output"]))
    paths = _candidate_paths(bundle, output)
    prediction_path = paths["prediction"]
    sidecar_path = paths["sidecar"]
    checkpoint_path = paths["checkpoint"]
    provenance_path = paths["provenance"]
    sidecar = read_json(sidecar_path)
    provenance = read_json(provenance_path)
    fold_refs = fold["_expected_eval_refs"]
    _require(
        sidecar.get("schema") == "wnn.age-readout-prediction.v1",
        f"{run}: prediction schema differs",
    )
    for key, expected in {
        "experiment_id": EXPERIMENT_ID,
        "run": run,
        "recipe_sha256": RECIPE_SHA256,
        "source_cache_recipe_sha256": SOURCE_CACHE_SHA256,
        "fold": fold_id,
        "seed": int(cell["seed"]),
        "arm": str(cell["arm"]),
        "prediction_path": str(output / "prediction.npy"),
        "prediction_shape": [len(fold_refs), ROWS, 2],
        "prediction_dtype": "<f4",
        "evaluation_row_refs": fold_refs,
        "evaluation_membership_sha256": row_refs_sha256(fold_refs),
        "checkpoint_path": str(output / "model_ep2.pt"),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "provenance_path": str(output / "provenance.json"),
        "provenance_sha256": sha256_file(provenance_path),
        "parent_prediction_path": str(Path(fold["parent_cache"]) / "evaluation_predictions.npy"),
        "parent_prediction_sha256": parent["prediction_sha256"],
        "parent_manifest_path": str(Path(fold["parent_cache"]) / "evaluation_predictions.json"),
        "parent_manifest_sha256": parent["sidecar_sha256"],
        "evaluation_read_contract": READ_CONTRACT,
    }.items():
        _require(sidecar.get(key) == expected, f"{run}: prediction sidecar {key} differs")
    _require(
        provenance.get("schema") == "wnn.age-readout-provenance.v1",
        f"{run}: provenance schema differs",
    )
    for key, expected in {
        "experiment_id": EXPERIMENT_ID,
        "run": run,
        "recipe_sha256": RECIPE_SHA256,
        "fold": fold_id,
        "seed": int(cell["seed"]),
        "arm": str(cell["arm"]),
        "evaluation_read_contract": READ_CONTRACT,
    }.items():
        _require(provenance.get(key) == expected, f"{run}: provenance {key} differs")
    source_cache = provenance.get("source_cache_recipe", {})
    _require(
        source_cache.get("path") == str(bundle.source_path(recipe["source_cache_recipe"]["path"]))
        and source_cache.get("sha256") == SOURCE_CACHE_SHA256,
        f"{run}: producer cache recipe identity differs",
    )
    _require(
        provenance.get("target_cache_opened") is False,
        f"{run}: evaluation lineage reports target access",
    )
    _require(
        provenance.get("cache_payload_hash_verified") is True,
        f"{run}: full cache validation is not recorded",
    )
    source_hashes = provenance.get("source_sha256", {})
    _require_source_hash_map(source_hashes, fit_hashes, bundle)
    _require(
        provenance.get("fit_source_sha256") == source_hashes,
        f"{run}: explicit fit source hash map differs",
    )
    _require(
        provenance.get("features") == recipe["features"], f"{run}: frozen feature contract differs"
    )
    _require(
        provenance.get("readout") == recipe["readout"], f"{run}: frozen readout contract differs"
    )
    runtime = provenance.get("runtime", {})
    _require(runtime.get("precision") == "strict_fp32", f"{run}: readout precision differs")
    _require(
        runtime.get("tf32_matmul") is False
        and runtime.get("tf32_cudnn") is False
        and runtime.get("autocast") is False
        and runtime.get("deterministic_algorithms") is True,
        f"{run}: strict FP32/determinism runtime flags differ",
    )
    _require(
        1 <= int(runtime.get("torch_threads", 0)) <= 2
        and 1 <= int(runtime.get("interop_threads", 0)) <= 2,
        f"{run}: candidate CPU thread counts exceed the frozen task budget",
    )
    device = str(runtime.get("device", ""))
    _require(
        device == "cpu" or re.fullmatch(r"cuda:[0-9]+", device) is not None,
        f"{run}: unsupported readout device",
    )
    cuda_memory = runtime.get("cuda_memory", {})
    _require(isinstance(cuda_memory, dict), f"{run}: CUDA memory receipt is malformed")
    if device == "cpu":
        _require(
            all(
                cuda_memory.get(key) is None
                for key in (
                    "allocator_cap_bytes",
                    "peak_allocated_bytes",
                    "peak_reserved_bytes",
                )
            ),
            f"{run}: CPU runtime reports fabricated CUDA memory counters",
        )
    else:
        allocator_cap = int(cuda_memory.get("allocator_cap_bytes", -1))
        peak_allocated = int(cuda_memory.get("peak_allocated_bytes", -1))
        peak_reserved = int(cuda_memory.get("peak_reserved_bytes", -1))
        _require(
            0 < allocator_cap <= 4 * 1024**3
            and 0 <= peak_allocated <= allocator_cap
            and 0 <= peak_reserved <= allocator_cap,
            f"{run}: CUDA allocator/peak counters exceed the frozen memory cap",
        )
    fit_record = cache[(fold_id, "fit_sequences")]
    eval_record = cache[(fold_id, "evaluation_sequences")]
    for field, record in (("fit_cache", fit_record), ("evaluation_cache", eval_record)):
        cached = provenance.get(field, {})
        meta = record["metadata"]
        expected_refs = meta.get("row_refs")
        _require(
            cached.get("path") == record["record"]["source_path"], f"{run}: {field} path differs"
        )
        _require(
            cached.get("sidecar_path") == record["record"]["sidecar_path"],
            f"{run}: {field} sidecar path differs",
        )
        _require(
            cached.get("sidecar_sha256") == record["record"]["sidecar_sha256"],
            f"{run}: {field} sidecar hash differs",
        )
        _require(
            cached.get("payload_sha256") == record["record"]["sha256"],
            f"{run}: {field} payload hash differs",
        )
        _require(
            cached.get("membership_sha256") == meta.get("membership_sha256"),
            f"{run}: {field} membership differs",
        )
        _require(cached.get("row_refs") == expected_refs, f"{run}: {field} row order differs")
        _require(
            cached.get("shape") == meta.get("shape") and cached.get("dtype") == meta.get("dtype"),
            f"{run}: {field} header differs",
        )
    parent_contract = provenance.get("parent_endpoint", {})
    parent_meta = parent["meta"].get("parent", {})
    parent_endpoint = parent["meta"].get("endpoint", {})
    _require(parent_contract.get("backend") == "cpu", f"{run}: parent prediction backend differs")
    _require(
        parent_contract.get("precision") == "strict_fp32",
        f"{run}: parent prediction precision differs",
    )
    _require(
        parent_contract.get("checkpoint_sha256") == fold["parent"]["checkpoint_sha256"],
        f"{run}: parent checkpoint ancestry differs",
    )
    _require(
        parent_contract.get("provenance_sha256") == fold["parent"]["provenance_sha256"],
        f"{run}: parent provenance ancestry differs",
    )
    _require(
        parent_contract.get("prediction_path")
        == str(Path(fold["parent_cache"]) / "evaluation_predictions.npy"),
        f"{run}: parent prediction path differs",
    )
    _require(
        parent_contract.get("prediction_sha256") == parent["prediction_sha256"],
        f"{run}: parent prediction ancestry differs",
    )
    _require(
        parent_contract.get("manifest_path")
        == str(Path(fold["parent_cache"]) / "evaluation_predictions.json"),
        f"{run}: parent manifest path differs",
    )
    _require(
        parent_contract.get("manifest_sha256") == parent["sidecar_sha256"],
        f"{run}: parent manifest ancestry differs",
    )
    _require(
        parent_contract.get("ancestry") == fold["parent"],
        f"{run}: full parent ancestry differs from recipe",
    )
    normalizer = provenance.get("parent_normalization", {})
    _require(
        normalizer.get("mean_identity") == parent_meta.get("mean_identity"),
        f"{run}: parent mean identity differs",
    )
    _require(
        normalizer.get("std_identity") == parent_meta.get("std_identity"),
        f"{run}: parent std identity differs",
    )
    eval_parent_meta = cache[(fold_id, "evaluation_sequences")]["metadata"]
    _require(
        normalizer.get("mean") == eval_parent_meta.get("parent_mean"),
        f"{run}: parent mean values differ",
    )
    _require(
        normalizer.get("std") == eval_parent_meta.get("parent_std"),
        f"{run}: parent std values differ",
    )
    _require(
        parent_endpoint.get("precision") == "strict_fp32"
        and parent_endpoint.get("backend") == "cpu",
        f"{run}: cached parent endpoint identity differs",
    )
    _require(
        provenance.get("checkpoint", {}).get("sha256") == sidecar["checkpoint_sha256"],
        f"{run}: checkpoint provenance differs",
    )
    _require(
        provenance.get("checkpoint", {}).get("path") == str(output / "model_ep2.pt"),
        f"{run}: checkpoint path differs",
    )
    _require(
        provenance.get("prediction", {}).get("sha256") == sidecar.get("prediction_sha256"),
        f"{run}: prediction provenance differs",
    )
    _require(
        provenance.get("prediction", {}).get("path") == str(output / "prediction.npy"),
        f"{run}: prediction path differs",
    )
    prediction_sha = str(sidecar.get("prediction_sha256", ""))
    _require(
        prediction_sha == sha256_file(prediction_path), f"{run}: prediction payload hash differs"
    )
    _require_prediction_header(
        prediction_path,
        sidecar["prediction_shape"],
        sidecar["prediction_dtype"],
        f"{run} prediction",
    )
    prediction = np.load(prediction_path, mmap_mode="r", allow_pickle=False)
    _require(
        prediction.shape == (len(fold_refs), ROWS, 2) and prediction.dtype.str == "<f4",
        f"{run}: loaded prediction shape/dtype differs",
    )
    for index in range(len(fold_refs)):
        _require(bool(np.isfinite(prediction[index]).all()), f"{run}: nonfinite prediction values")
    return {
        "run": run,
        "arm": str(cell["arm"]),
        "seed": int(cell["seed"]),
        "prediction": prediction,
        "prediction_path": prediction_path,
        "prediction_sha256": prediction_sha,
        "sidecar_path": sidecar_path,
        "sidecar_sha256": sha256_file(sidecar_path),
        "checkpoint_path": checkpoint_path,
        "checkpoint_sha256": sidecar["checkpoint_sha256"],
        "provenance_path": provenance_path,
        "provenance_sha256": sidecar["provenance_sha256"],
        "evaluation_membership_sha256": row_refs_sha256(fold_refs),
        "device": runtime["device"],
    }


def _load_targets(
    bundle: Bundle, recipe: dict[str, Any], folds: dict[int, dict[str, Any]]
) -> tuple[dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]], dict[str, Any]]:
    """Open labels only after every prediction has passed the full artifact gate."""
    target = recipe["target_cache"]
    target_path = bundle.path_for(target["path"])
    expected_bytes = int(folds[0]["_split"]["sources"][target["path"]]["bytes"])
    _require(target["sha256"] == TARGET_SHA256, "target cache recipe content pin differs")
    transported_target = _transported_file_identity(target_path, TARGET_SHA256, expected_bytes)
    size = int(transported_target["bytes"])
    result: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    original_source_stats: dict[str, dict[str, int]] = {}
    for fold_id, fold in folds.items():
        source_stat = fold["_split"]["sources"].get(target["path"], {})
        _require(
            int(source_stat.get("bytes", -1)) == size,
            f"F{fold_id} target source byte count differs",
        )
        original_source_stats[str(fold_id)] = {
            "bytes": int(source_stat["bytes"]),
            "mtime_ns": int(source_stat["mtime_ns"]),
        }
    with np.load(target_path, allow_pickle=False) as archive:
        _require(
            {"row_group", "y", "mask"}.issubset(archive.files),
            "target archive lacks required arrays",
        )
        groups = np.asarray(archive["row_group"])
        targets = np.asarray(archive["y"])
        mask = np.asarray(archive["mask"])
    _require(groups.ndim == 1 and groups.dtype.kind in "iu", "target group index is malformed")
    _require(len(np.unique(groups)) == len(groups), "target groups are duplicated")
    _require(targets.shape == (len(groups), ROWS, 2), "target array shape differs")
    _require(
        targets.dtype.str == "<f4" and mask.shape == (len(groups), ROWS) and mask.dtype == np.bool_,
        "target dtype or mask shape differs",
    )
    _require(bool(np.isfinite(targets[mask]).all()), "scored target labels are nonfinite")
    position = {int(group): index for index, group in enumerate(groups)}
    for fold_id, fold in folds.items():
        eval_groups = [int(ref["row_group"]) for ref in fold["_expected_eval_refs"]]
        _require(
            all(group in position for group in eval_groups),
            f"F{fold_id} target groups are incomplete",
        )
        indices = np.asarray([position[group] for group in eval_groups], dtype=np.int64)
        component_by_group = {
            int(record["row_group"]): int(record["component"])
            for record in fold["_split"]["sequences"]["valid"]
        }
        _require(
            all(group in component_by_group for group in eval_groups),
            f"F{fold_id} component map is incomplete",
        )
        components = np.asarray(
            [component_by_group[group] for group in eval_groups], dtype=np.int64
        )
        result[fold_id] = targets[indices], mask[indices], components
    _require(sha256_file(target_path) == TARGET_SHA256, "target archive changed during read")
    return result, {
        "path": str(target_path),
        "sha256": transported_target["sha256"],
        "bytes": transported_target["bytes"],
        "original_source_stats_by_fold": original_source_stats,
        "transport_mtime_ns": transported_target["transport_mtime_ns"],
    }


def _load_scoring_helpers(bundle_root: Path) -> Any:
    """Import only the frozen official metric and paired-component helpers."""
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(bundle_root / "scripts"))
    sys.path.insert(0, str(bundle_root / "src"))
    sys.path.insert(
        0,
        str(bundle_root / "research/experiment_drafts/online-relationships-20261022"),
    )
    import online_relationship_grade_core  # type: ignore[import-not-found]

    return online_relationship_grade_core


def _recheck_artifacts(
    parents: dict[int, dict[str, Any]], candidates: dict[tuple[int, str], dict[str, Any]]
) -> None:
    """Refuse scoring if any validated prediction or provenance file changed."""
    for fold_id, parent in parents.items():
        for path_key, sha_key in (
            ("prediction_path", "prediction_sha256"),
            ("sidecar_path", "sidecar_sha256"),
            ("provenance_path", "provenance_sha256"),
        ):
            _require(
                sha256_file(parent[path_key]) == parent[sha_key],
                f"F{fold_id} parent artifact changed after validation",
            )
    for artifact in candidates.values():
        for path_key, sha_key in (
            ("prediction_path", "prediction_sha256"),
            ("sidecar_path", "sidecar_sha256"),
            ("checkpoint_path", "checkpoint_sha256"),
            ("provenance_path", "provenance_sha256"),
        ):
            _require(
                sha256_file(artifact[path_key]) == artifact[sha_key],
                f"{artifact['run']}: candidate artifact changed after validation",
            )


def _score_fold(
    helpers: Any,
    fold_id: int,
    parent: dict[str, Any],
    candidates: dict[str, dict[str, Any]],
    targets: np.ndarray,
    mask: np.ndarray,
    components: np.ndarray,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Score fixed endpoints and their five predeclared within-fold contrasts."""
    endpoints: dict[str, np.ndarray] = {"parent": parent["prediction_array"]}
    endpoints.update({arm: candidates[arm]["prediction"] for arm in ARMS})
    scores = {
        name: helpers.official_scores(targets, mask, prediction)
        for name, prediction in endpoints.items()
    }
    moments = {
        name: helpers.paired_endpoint_moments(
            targets,
            prediction,
            mask & (np.arange(ROWS)[None, :] >= 99),
        )
        for name, prediction in endpoints.items()
    }
    contrasts: list[dict[str, Any]] = []
    for candidate_name, reference_name in CONTRASTS:
        result = helpers.paired_component_contrast(
            targets,
            mask,
            endpoints[candidate_name],
            endpoints[reference_name],
            components,
            scores[candidate_name],
            scores[reference_name],
            moments[candidate_name],
            moments[reference_name],
        )
        contrasts.append(
            {
                "contrast": f"{candidate_name}-{reference_name}",
                **result,
            }
        )
    endpoint_report = {
        "fold": fold_id,
        "evaluation_sequences": len(targets),
        "evaluation_components": int(len(np.unique(components))),
        "endpoint_identity": {
            "parent": {
                "run": parent["meta"]["parent"]["run"],
                "checkpoint_sha256": parent["meta"]["parent"]["checkpoint_sha256"],
                "provenance_sha256": parent["provenance_sha256"],
                "historical_precision": parent["meta"]["parent"]["historical_precision"],
                "endpoint_precision": parent["meta"]["endpoint"]["precision"],
                "endpoint_backend": parent["meta"]["endpoint"]["backend"],
                "prediction_sha256": parent["prediction_sha256"],
                "prediction_sidecar_sha256": parent["sidecar_sha256"],
                "membership_sha256": parent["meta"]["membership_sha256"],
            },
            **{
                arm: {
                    "run": candidates[arm]["run"],
                    "seed": candidates[arm]["seed"],
                    "checkpoint_sha256": candidates[arm]["checkpoint_sha256"],
                    "provenance_sha256": candidates[arm]["provenance_sha256"],
                    "prediction_sha256": candidates[arm]["prediction_sha256"],
                    "prediction_sidecar_sha256": candidates[arm]["sidecar_sha256"],
                    "evaluation_membership_sha256": candidates[arm]["evaluation_membership_sha256"],
                    "device": candidates[arm]["device"],
                }
                for arm in ARMS
            },
        },
        "endpoints": scores,
    }
    return endpoint_report, contrasts


def _mean_fold_points(contrasts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Average fixed point deltas equally across folds without pooling intervals."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for report in contrasts:
        grouped.setdefault(str(report["contrast"]), []).append(report["point_deltas"])
    result = []
    for name, values in sorted(grouped.items()):
        bands = values[0].keys()
        mean = {
            band: {
                metric: float(np.mean([value[band][metric] for value in values]))
                for metric in ("wp", "t0", "t1")
            }
            for band in bands
        }
        result.append({"contrast": name, "equal_fold_mean_point_deltas": mean})
    return result


def main() -> int:
    """Run the exact six-cell grade only after root admission and prelabel checks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--bundle-manifest", type=Path, required=True)
    parser.add_argument("--admission", type=Path, required=True)
    parser.add_argument("--admission-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    bundle_root = args.bundle_root.resolve(strict=True)
    bundle_manifest_path = args.bundle_manifest.resolve(strict=True)
    manifest = read_json(bundle_manifest_path)
    bundle = Bundle(bundle_root, manifest)
    recipe, recipe_sha, folds = validate_recipe(bundle)
    admission = validate_admission(
        args.admission.resolve(strict=True),
        args.admission_sha256,
        recipe,
        recipe_sha,
        bundle,
        bundle_manifest_path,
    )
    # This pass hashes all transported bytes, including every prediction and the target archive,
    # before any NPY payload or target label is decoded.
    verified_files = bundle.verify_all()
    cache_path = bundle.path_for(str(admission["cache_verification"]["path"]))
    cache_receipt = read_json(cache_path)
    caches = _cache_entries(cache_receipt, recipe, folds, bundle)
    parents = {
        fold_id: _validate_parent(recipe, fold_id, fold, caches, bundle)
        for fold_id, fold in folds.items()
    }
    candidates: dict[tuple[int, str], dict[str, Any]] = {}
    for cell in recipe["cells"]:
        fold_id = int(cell["fold"])
        parent = parents[fold_id]
        candidates[(fold_id, str(cell["arm"]))] = _validate_candidate(
            recipe,
            cell,
            folds[fold_id],
            fold_id,
            caches,
            parent,
            bundle,
            admission["fit_source_sha256"],
        )
    _require(len(candidates) == 6, "not all six candidate predictions were validated")
    devices = {artifact["device"] for artifact in candidates.values()}
    _require(len(devices) == 1, "the six readout fits used different devices")
    _require(
        devices == {str(admission["shared_readout_device"])},
        "candidate device differs from root's profile-selected shared device",
    )
    _recheck_artifacts(parents, candidates)
    # This is the first function that decodes target labels. Every candidate and parent
    # payload, header, row-reference order, source identity and ancestry passed above.
    target_folds, target_identity = _load_targets(bundle, recipe, folds)
    helpers = _load_scoring_helpers(bundle_root)
    fold_reports: list[dict[str, Any]] = []
    contrast_reports: list[dict[str, Any]] = []
    for fold_id in sorted(folds):
        targets, mask, components = target_folds[fold_id]
        candidate_map = {arm: candidates[(fold_id, arm)] for arm in ARMS}
        endpoint, contrasts = _score_fold(
            helpers,
            fold_id,
            parents[fold_id],
            candidate_map,
            targets,
            mask,
            components,
        )
        fold_reports.append(endpoint)
        contrast_reports.extend({"fold": fold_id, **entry} for entry in contrasts)
    _require(len(contrast_reports) == 10, "grader did not produce the ten frozen contrasts")
    _require(
        sha256_file(bundle.path_for(RECIPE_RELATIVE)) == RECIPE_SHA256,
        "recipe changed during grading",
    )
    _require(
        sha256_file(bundle.path_for(recipe["target_cache"]["path"])) == TARGET_SHA256,
        "target cache changed during grading",
    )
    _recheck_artifacts(parents, candidates)
    report = {
        "schema": "wnn.age-readout-grade.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "experiment_id": EXPERIMENT_ID,
        "recipe": {"path": str(RECIPE_RELATIVE), "sha256": recipe_sha},
        "admission": {"sha256": args.admission_sha256, "status": admission["status"]},
        "shared_readout_device": admission["shared_readout_device"],
        "bundle_manifest_sha256": sha256_file(bundle_manifest_path),
        "metric": "global_wp_clip2_v1",
        "grader_source_sha256": admission["grader_source_sha256"],
        "fit_source_sha256": admission["fit_source_sha256"],
        "target_cache": target_identity,
        "verified_transport_files": len(verified_files),
        "cache_verification": {
            "schema": cache_receipt["schema"],
            "sha256": sha256_file(cache_path),
            "authorization_sha256": CACHE_VERIFY_ADMISSION_SHA256,
            "spec_sha256": CACHE_VERIFY_SPEC_SHA256,
            "execution_guard_sha256": CACHE_VERIFY_GUARD_SHA256,
            "mode": "local_attestation; original cache payloads were not transferred or rehashed on Kaggle",
            "parent_cache_milestone_sha256": PARENT_MILESTONE_SHA256,
        },
        "folds": fold_reports,
        "contrasts": contrast_reports,
        "equal_fold_mean_points": _mean_fold_points(contrast_reports),
        "bootstrap": {
            "draws": 2_000,
            "seed": 271828,
            "unit": "source-qualified dependency component",
        },
        "interpretation": {
            "exposure": "Both full validation folds are exposed development evidence.",
            "fold_seed": "F0/seed1001 and F1/seed1002 are confounded; no training-seed variance is estimated.",
            "interval_scope": "Intervals condition on the fixed fitted endpoints and omit training variation and transfer uncertainty.",
            "selection": "Ten fold-local paired intervals are reported without multiplicity correction; this is a bounded screen.",
            "parent_fit_predictions": "The readout heads were fit against in-sample parent predictions on selected fit groups, not OOF parent endpoints.",
            "parent_weights": "Parent checkpoints were not transferred or reloaded; their hashes and ancestry are recipe- and provenance-pinned.",
            "mechanism": "F-G changes feature rank/clock granularity; this screen does not establish a physical clock interpretation.",
            "serving": "The readout screen alone does not establish native serving compatibility or release readiness.",
            "expected_public_score": None,
        },
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"grade": str(output), "folds": 2, "endpoints": 8, "contrasts": 10}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
