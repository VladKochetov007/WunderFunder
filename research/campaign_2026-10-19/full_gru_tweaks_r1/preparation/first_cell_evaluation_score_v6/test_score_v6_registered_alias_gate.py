"""Exercise the exact registered-leaf check with pinned records and fake stats."""

from __future__ import annotations

import hashlib
import json
import stat
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

PACKET = Path(__file__).resolve().parent
PREPARATION = PACKET.parent
CAMPAIGN = PREPARATION.parent
EXPORT_V4 = PREPARATION / "first_cell_evaluation_export_v4"
SCORE_V2_RECORD = CAMPAIGN / "records/first_cell_evaluation_v1/score_v2"
FIT_ADMISSION = CAMPAIGN / "records/child_allthree_fit_v1/ADMISSION.json"
PARENT_DIAGNOSTIC = SCORE_V2_RECORD / "alias_parent_diagnostic_v1/STDOUT.raw.json"
DATA_ROOT = Path("/home/grinchenkoeg/computer-vision-homework-20261018/data")

sys.path.insert(0, str(PACKET))

import first_cell_score_core as core  # noqa: E402


def _pinned_json(path: Path, expected_bytes: int, expected_sha256: str) -> dict[str, object]:
    raw = path.read_bytes()
    if len(raw) != expected_bytes or hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise AssertionError(f"pinned metadata bytes differ: {path}")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise AssertionError(f"expected a JSON object: {path}")
    return value


class SyntheticAliasFilesystem:
    """Supply bounded stat and resolution observations without opening source files."""

    def __init__(
        self,
        identities: dict[str, dict[str, object]],
        parent_observations: dict[str, dict[str, object]],
        *,
        wrong_leaf_resolution: str | None = None,
        changing_parent: str | None = None,
    ) -> None:
        self.identities = identities
        self.parent_observations = parent_observations
        self.sources_by_path = {str(item["path"]): item for item in identities.values()}
        self.parents_by_path = {str(item["path"]): item for item in parent_observations.values()}
        self.wrong_leaf_resolution = wrong_leaf_resolution
        self.changing_parent = changing_parent
        self.parent_resolve_calls: dict[str, int] = {}

    def stat(self, path: Path) -> SimpleNamespace:
        source = self.sources_by_path.get(str(path))
        if source is not None:
            return SimpleNamespace(
                st_mode=stat.S_IFREG | 0o644,
                st_dev=source["device"],
                st_ino=source["inode"],
                st_size=source["bytes"],
                st_mtime_ns=source["mtime_ns"],
            )
        parent = self.parents_by_path.get(str(path))
        if parent is not None:
            return SimpleNamespace(
                st_mode=stat.S_IFDIR | 0o755,
                st_dev=parent["device"],
                st_ino=parent["inode"],
                st_size=parent["bytes"],
                st_mtime_ns=parent["mtime_ns"],
            )
        raise AssertionError(f"unexpected stat path: {path}")

    def lstat(self, path: Path) -> SimpleNamespace:
        source = self.sources_by_path.get(str(path))
        if source is None:
            raise AssertionError(f"unexpected lstat path: {path}")
        link_mode = stat.S_IFLNK if source["islink"] else stat.S_IFREG
        return SimpleNamespace(
            st_mode=link_mode | (0o777 if source["islink"] else 0o644),
            st_dev=source["device"],
            st_ino=source["inode"],
            st_size=source["bytes"],
            st_mtime_ns=source["mtime_ns"],
        )

    def is_symlink(self, path: Path) -> bool:
        source = self.sources_by_path.get(str(path))
        if source is None:
            raise AssertionError(f"unexpected symlink query: {path}")
        return bool(source["islink"])

    def resolve(self, path: Path, strict: bool = False) -> Path:
        del strict
        text = str(path)
        source = self.sources_by_path.get(text)
        if source is not None:
            if text == self.wrong_leaf_resolution:
                return Path("/unregistered/wrong-target")
            return Path(str(source["resolved"]))
        parent = self.parents_by_path.get(text)
        if parent is not None:
            calls = self.parent_resolve_calls.get(text, 0) + 1
            self.parent_resolve_calls[text] = calls
            if text == self.changing_parent and calls > 1:
                return Path("/changed/parent/alias")
            return Path(str(parent["resolved"]))
        raise AssertionError(f"unexpected resolve path: {path}")

    def patched(self):
        """Patch only pathlib metadata operations for this synthetic fixture."""
        return (
            patch.object(Path, "stat", lambda path, **kwargs: self.stat(path)),
            patch.object(Path, "lstat", lambda path: self.lstat(path)),
            patch.object(Path, "is_symlink", lambda path: self.is_symlink(path)),
            patch.object(Path, "resolve", lambda path, strict=False: self.resolve(path, strict)),
        )


def _actual_inputs() -> tuple[dict[str, dict[str, object]], dict[str, dict[str, object]]]:
    binding = _pinned_json(
        EXPORT_V4 / "EXPORT_BINDING_V4.json",
        15796,
        "4a8ae22204971ac07d6e5cdfd9f65a62e9f8440831035db7271e4e0da3d83abd",
    )
    fit = _pinned_json(
        FIT_ADMISSION,
        12912,
        "5312ae5feffc67d7a33051379afabb9c134dd946f1f97315d839c7d6f10d5540",
    )
    score = _pinned_json(
        SCORE_V2_RECORD / "ADMISSION.json",
        69650,
        "1252b2f1d39ef255fac83c722cfc1a380705963a0b2f9dede35dcabed45ba631",
    )
    diagnostic = _pinned_json(
        PARENT_DIAGNOSTIC,
        3218,
        "e6c08c71185d34c8763ec4fc983d72a91003d3626e6dc2da6a53926c49cb8380",
    )
    registered = binding["registered_data"]["source_identities"]
    fit_sources = fit["data_identity_stats_only"]
    fresh_sources = score["checks"]["registered_six_sources_stat_only"]["sources_stat_only"]
    if registered != fit_sources or set(registered) != set(core.EXPECTED_PORTABLE_SOURCE_NAMES):
        raise AssertionError("actual export binding and fit admission source identities differ")
    for relative, identity in registered.items():
        fresh = fresh_sources[relative]
        observed_fields = {
            "bytes",
            "device",
            "inode",
            "islink",
            "mtime_ns",
            "path",
            "resolved",
        }
        if {key: identity[key] for key in observed_fields} != fresh:
            raise AssertionError(
                f"fresh score admission differs from registered source: {relative}"
            )
    parents = diagnostic["parents"]
    if diagnostic["parents_after"] != parents:
        raise AssertionError("root parent diagnostic changed during its bounded observation")
    return registered, parents


class ScoreV6RegisteredAliasTests(unittest.TestCase):
    """Keep exact leaf registration while treating parent resolution as fresh state."""

    def setUp(self) -> None:
        self.identities, self.parents = _actual_inputs()

    def _run_observer(
        self,
        relative: str,
        *,
        wrong_leaf_resolution: str | None = None,
        changing_parent: str | None = None,
    ) -> dict[str, object]:
        filesystem = SyntheticAliasFilesystem(
            self.identities,
            self.parents,
            wrong_leaf_resolution=wrong_leaf_resolution,
            changing_parent=changing_parent,
        )
        patches = filesystem.patched()
        with patches[0], patches[1], patches[2], patches[3]:
            return core.observe_registered_source(DATA_ROOT, relative, self.identities[relative])

    def test_all_actual_leaves_pass_with_the_captured_parent_layout(self) -> None:
        observed = {
            relative: self._run_observer(relative)
            for relative in core.EXPECTED_PORTABLE_SOURCE_NAMES
        }
        self.assertEqual(set(observed), set(self.identities))
        for relative, identity in self.identities.items():
            item = observed[relative]
            for field in ("path", "resolved", "bytes", "device", "inode", "islink", "mtime_ns"):
                self.assertEqual(item[field], identity[field], (relative, field))
            parent = self.parents[relative.split("/", 1)[0]]
            self.assertEqual(
                item["parent"],
                {
                    "path": parent["path"],
                    "resolved": parent["resolved"],
                    "device": parent["device"],
                    "inode": parent["inode"],
                },
            )
        self.assertNotEqual(
            observed["cache/chain_folds.npz"]["parent"]["resolved"],
            core.EXPECTED_PARENT_RESOLUTIONS["cache"],
        )
        self.assertEqual(
            observed["datasets/valid.parquet"]["parent"]["resolved"],
            core.EXPECTED_PARENT_RESOLUTIONS["datasets"],
        )

    def test_leaf_target_must_remain_the_exact_registered_resolution(self) -> None:
        relative = "cache/chain_folds.npz"
        with self.assertRaisesRegex(ValueError, "exact alias target"):
            self._run_observer(relative, wrong_leaf_resolution=self.identities[relative]["path"])

    def test_parent_resolution_must_be_stable_during_each_observation(self) -> None:
        relative = "cache/chain_folds.npz"
        parent_path = self.parents["cache"]["path"]
        with self.assertRaisesRegex(ValueError, "identity changed or differs"):
            self._run_observer(relative, changing_parent=parent_path)


if __name__ == "__main__":
    unittest.main()
