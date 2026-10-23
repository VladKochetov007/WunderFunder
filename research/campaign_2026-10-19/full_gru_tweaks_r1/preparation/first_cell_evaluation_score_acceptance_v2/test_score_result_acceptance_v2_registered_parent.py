"""Focused metadata fixtures for the V2 registered-parent observation gate."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import stat
import unittest
from pathlib import Path, PurePosixPath

import score_result_acceptance_v2 as acceptance

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]
BINDING_PATH = BASE / "preparation/first_cell_evaluation_export_v4/EXPORT_BINDING_V4.json"
ADMISSION_PATH = BASE / "records/first_cell_evaluation_v1/score_v2/ADMISSION.json"
PARENT_DIAGNOSTIC_PATH = (
    BASE / "records/first_cell_evaluation_v1/score_v2/alias_parent_diagnostic_v1/STDOUT.raw.json"
)
BINDING_PIN = (
    15_796,
    "4a8ae22204971ac07d6e5cdfd9f65a62e9f8440831035db7271e4e0da3d83abd",
)
ADMISSION_PIN = (
    69_650,
    "1252b2f1d39ef255fac83c722cfc1a380705963a0b2f9dede35dcabed45ba631",
)
PARENT_DIAGNOSTIC_PIN = (
    3_218,
    "e6c08c71185d34c8763ec4fc983d72a91003d3626e6dc2da6a53926c49cb8380",
)


def _read_pinned_metadata(path: Path, expected: tuple[int, str]) -> dict[str, object]:
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or before.st_size != expected[0]:
        raise AssertionError(f"metadata fixture has an unexpected file identity: {path}")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        opened = os.fstat(descriptor)
        identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        if (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns) != identity:
            raise AssertionError(f"metadata fixture changed before read: {path}")
        raw = os.read(descriptor, expected[0] + 1)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if (
        len(raw) != expected[0]
        or (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) != identity
    ):
        raise AssertionError(f"metadata fixture changed during read: {path}")
    if len(raw) != expected[0] or hashlib.sha256(raw).hexdigest() != expected[1]:
        raise AssertionError(f"metadata fixture pin changed: {path}")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise AssertionError(f"metadata fixture is not an object: {path}")
    return value


def _fixture() -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    binding = _read_pinned_metadata(BINDING_PATH, BINDING_PIN)
    admission = _read_pinned_metadata(ADMISSION_PATH, ADMISSION_PIN)
    parent_diagnostic = _read_pinned_metadata(PARENT_DIAGNOSTIC_PATH, PARENT_DIAGNOSTIC_PIN)
    registered = binding["registered_data"]
    identities = registered["source_identities"]
    admission_sources = admission["checks"]["registered_six_sources_stat_only"]["sources_stat_only"]
    diagnostic_leaves = parent_diagnostic["leaves"]
    field_names = ("path", "resolved", "islink", "device", "inode", "bytes", "mtime_ns")
    if set(identities) != set(admission_sources) or set(identities) != set(diagnostic_leaves):
        raise AssertionError("binding, admission, and parent-diagnostic source sets differ")
    parents = parent_diagnostic["parents"]
    expected_parent_facts = {
        "cache": {
            "path": "/home/grinchenkoeg/computer-vision-homework-20261018/data/cache",
            "resolved": "/home/grinchenkoeg/computer-vision-homework-20261018/data/cache",
            "islink": False,
            "device": 64_512,
            "inode": 725_819_756,
        },
        "datasets": {
            "path": "/home/grinchenkoeg/computer-vision-homework-20261018/data/datasets",
            "resolved": "/home/grinchenkoeg/WunderFunder/datasets",
            "islink": True,
            "device": 64_512,
            "inode": 743_836_258,
        },
    }
    for parent_name, expected in expected_parent_facts.items():
        parent = parents[parent_name]
        if any(parent.get(key) != value for key, value in expected.items()):
            raise AssertionError(f"actual parent diagnostic differs: {parent_name}")
        if parent != parent_diagnostic["parents_after"][parent_name]:
            raise AssertionError(f"parent identity changed during diagnostic: {parent_name}")

    observations: dict[str, dict[str, object]] = {}
    for relative, identity in identities.items():
        expected_leaf = {key: identity[key] for key in field_names}
        admission_leaf = admission_sources[relative]
        diagnostic_leaf = diagnostic_leaves[relative]
        if {key: admission_leaf[key] for key in field_names} != expected_leaf:
            raise AssertionError(f"fresh admission differs from bound source identity: {relative}")
        if {key: diagnostic_leaf[key] for key in field_names} != expected_leaf:
            raise AssertionError(f"parent diagnostic differs from bound leaf identity: {relative}")
        parent_group = "cache" if relative.startswith("cache/") else "datasets"
        parent_stat = parents[parent_group]
        parent = {
            "path": parent_stat["path"],
            "resolved": parent_stat["resolved"],
            "device": parent_stat["device"],
            "inode": parent_stat["inode"],
        }
        observations[relative] = dict(expected_leaf, parent=parent)

    registered_data = {
        "admission": registered["admission"],
        "portable_manifest": registered["portable_manifest"],
        "registered_source_identities": identities,
    }
    binding_data = {
        "admission": registered["admission"],
        "portable_manifest": registered["portable_manifest"],
        "source_identities": identities,
    }
    for phase in (
        "observed_before_split",
        "observed_after_split",
        "observed_before_scoring",
        "observed_after_scoring",
    ):
        registered_data[phase] = copy.deepcopy(observations)
    return registered_data, binding_data, parent_diagnostic


class RegisteredParentV2Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.report_data, self.binding_data, self.parent_diagnostic = _fixture()

    def test_actual_physical_cache_and_symlinked_datasets_parents_pass(self) -> None:
        returned = acceptance._validate_registered_observations(self.report_data, self.binding_data)
        self.assertIs(returned, self.report_data)
        cache_source = "cache/chain_folds.npz"
        cache_parent = self.report_data["observed_before_split"][cache_source]["parent"]
        self.assertEqual(
            cache_parent["resolved"],
            "/home/grinchenkoeg/computer-vision-homework-20261018/data/cache",
        )
        self.assertNotEqual(
            cache_parent["resolved"],
            str(
                PurePosixPath(
                    self.binding_data["source_identities"][cache_source]["resolved"]
                ).parent
            ),
        )
        dataset_parent = self.report_data["observed_before_split"]["datasets/valid.parquet"][
            "parent"
        ]
        self.assertEqual(dataset_parent["resolved"], "/home/grinchenkoeg/WunderFunder/datasets")

    def test_rejects_noncanonical_parent_resolution(self) -> None:
        cases = ("relative/cache", "/", "//", "/home/data/../cache", "/home//cache")
        for invalid in cases:
            with self.subTest(invalid=invalid):
                changed = copy.deepcopy(self.report_data)
                changed["observed_before_split"]["cache/chain_folds.npz"]["parent"]["resolved"] = (
                    invalid
                )
                with self.assertRaises(ValueError):
                    acceptance._validate_registered_observations(changed, self.binding_data)

    def test_rejects_parent_instability_between_phases(self) -> None:
        changed = copy.deepcopy(self.report_data)
        changed["observed_after_scoring"]["datasets/train.parquet"]["parent"]["inode"] += 1
        with self.assertRaisesRegex(ValueError, "observations changed across scorer phases"):
            acceptance._validate_registered_observations(changed, self.binding_data)

    def test_keeps_exact_registered_leaf_target_gate(self) -> None:
        changed = copy.deepcopy(self.report_data)
        changed["observed_before_split"]["cache/chain_folds.npz"]["resolved"] = (
            "/home/grinchenkoeg/WunderFunder/cache/chain_folds.npz"
        )
        with self.assertRaisesRegex(ValueError, "accepted identity"):
            acceptance._validate_registered_observations(changed, self.binding_data)


if __name__ == "__main__":
    unittest.main()
