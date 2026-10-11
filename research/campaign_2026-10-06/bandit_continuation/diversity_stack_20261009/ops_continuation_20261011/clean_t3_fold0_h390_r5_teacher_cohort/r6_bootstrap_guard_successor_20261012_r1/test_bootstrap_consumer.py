from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from typing import NoReturn
from unittest import mock

PACKET = Path(__file__).resolve().parent
KERNEL_SOURCE = PACKET / "kernel/kernel.py"
DATASET_SOURCE = PACKET / "private_dataset"
EXPECTED_ARCHIVE_SHA256 = "88641b3c9fc629a7a2ce8b11cfd98ecf8b53fe943ffb2d7f632e1dfb997fa549"
KERNEL_ID = "vladkochetov07/wnn-clean-t3-f0-teachers-20261011-r5-runner"


class DriverLaunchPrevented(RuntimeError):
    """Mark the boundary immediately before the real cohort driver would start."""


class BootstrapConsumerTests(unittest.TestCase):
    """Exercise the frozen bootstrap over the actual R5 dataset bundle."""

    def _load_kernel(self) -> ModuleType:
        """Import the exact successor bootstrap without launching its entry point."""
        specification = importlib.util.spec_from_file_location(
            "r6_bootstrap_under_test", KERNEL_SOURCE
        )
        if specification is None or specification.loader is None:
            raise AssertionError("could not load the frozen R6 bootstrap")
        module = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(module)
        return module

    def _prepare_fixture(
        self,
        *,
        expected_mounts: dict[str, str | dict[str, str]] | None = None,
        manifest_mounts: dict[str, str | dict[str, str]] | None = None,
    ) -> tuple[ModuleType, Path, Path, dict[str, Path]]:
        """Create Kaggle's nested input layout around the actual sealed archive."""
        module = self._load_kernel()
        if expected_mounts is not None:
            module.EXPECTED_EXTERNAL_MOUNTS = expected_mounts
        mounts = module.EXPECTED_EXTERNAL_MOUNTS
        temporary = tempfile.TemporaryDirectory(prefix="wnn-r6-bootstrap-")
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        input_root = root / "input"
        working_root = root / "working"
        working_root.mkdir()
        bundle_root = input_root / module.BUNDLE_SLUG
        bundle_root.mkdir(parents=True)
        manifest = json.loads((DATASET_SOURCE / "bundle-manifest.json").read_text())
        if manifest_mounts is not None:
            manifest["external_mounts"] = manifest_mounts
        (bundle_root / "bundle-manifest.json").write_text(json.dumps(manifest, sort_keys=True))
        shutil.copyfile(DATASET_SOURCE / "bundle.wnn", bundle_root / "bundle.wnn")
        source_paths: dict[str, Path] = {}
        for logical, mapping in mounts.items():
            dataset = (
                mapping["dataset"] if isinstance(mapping, dict) else "vladkochetov07/wunder-funder"
            )
            filename = mapping["filename"] if isinstance(mapping, dict) else mapping
            candidate = input_root / dataset.split("/")[1] / filename
            candidate.parent.mkdir(parents=True, exist_ok=True)
            candidate.write_bytes(b"bootstrap-only mount presence fixture\n")
            source_paths[logical] = candidate

        original_path_factory = Path

        def kaggle_path(value: str | os.PathLike[str]) -> Path:
            rendered = os.fspath(value)
            if rendered == "/kaggle/input":
                return input_root
            if rendered == "/kaggle/working/experiment":
                return working_root / "experiment"
            return original_path_factory(rendered)

        module.Path = kaggle_path
        return module, input_root, working_root, source_paths

    def _run_to_driver_boundary(
        self,
        module: ModuleType,
        working_root: Path,
        source_paths: dict[str, Path],
    ) -> None:
        """Stop before subprocess execution after checking all mounted links."""

        def stop_before_driver(*args: object, **kwargs: object) -> NoReturn:
            experiment = working_root / "experiment"
            for logical, source in source_paths.items():
                mounted = experiment / logical
                if not mounted.is_symlink() or mounted.resolve() != source.resolve():
                    raise AssertionError(f"unexpected external mount for {logical}")
            raise DriverLaunchPrevented

        with (
            mock.patch.dict(os.environ),
            mock.patch.object(module.subprocess, "Popen", side_effect=stop_before_driver) as launch,
        ):
            with self.assertRaises(DriverLaunchPrevented):
                module.main()
            launch.assert_called_once()

    def test_actual_bundle_valid_mapping_reaches_driver_boundary(self) -> None:
        """The unchanged external identities mount after real archive extraction."""
        archive = DATASET_SOURCE / "bundle.wnn"
        self.assertEqual(hashlib.sha256(archive.read_bytes()).hexdigest(), EXPECTED_ARCHIVE_SHA256)
        module, _input_root, working_root, sources = self._prepare_fixture()
        manifest = json.loads((DATASET_SOURCE / "bundle-manifest.json").read_text())
        self.assertEqual(manifest["external_mounts"], module.EXPECTED_EXTERNAL_MOUNTS)
        self._run_to_driver_boundary(module, working_root, sources)

    def test_manifest_extra_key_is_rejected_before_mounts(self) -> None:
        """An input manifest cannot add keys outside the source-bound identity."""
        module = self._load_kernel()
        mounts = dict(module.EXPECTED_EXTERNAL_MOUNTS)
        mounts["cache/unexpected.npy"] = "train.parquet"
        module, _input_root, _working_root, _sources = self._prepare_fixture(manifest_mounts=mounts)
        with mock.patch.object(module.subprocess, "Popen") as launch:
            with self.assertRaisesRegex(ValueError, "External mappings differ"):
                module.main()
            launch.assert_not_called()

    def test_manifest_missing_key_is_rejected_before_mounts(self) -> None:
        """The manifest must retain every source-bound external input."""
        module = self._load_kernel()
        mounts = dict(module.EXPECTED_EXTERNAL_MOUNTS)
        mounts.pop("cache/train_targets.npz")
        module, _input_root, _working_root, _sources = self._prepare_fixture(manifest_mounts=mounts)
        with mock.patch.object(module.subprocess, "Popen") as launch:
            with self.assertRaisesRegex(ValueError, "External mappings differ"):
                module.main()
            launch.assert_not_called()

    def test_manifest_wrong_value_is_rejected_before_mounts(self) -> None:
        """The manifest cannot redirect an allowed key to another input."""
        module = self._load_kernel()
        mounts = json.loads(json.dumps(module.EXPECTED_EXTERNAL_MOUNTS))
        mounts["cache/train_targets.npz"]["filename"] = "valid.parquet"
        module, _input_root, _working_root, _sources = self._prepare_fixture(manifest_mounts=mounts)
        with mock.patch.object(module.subprocess, "Popen") as launch:
            with self.assertRaisesRegex(ValueError, "External mappings differ"):
                module.main()
            launch.assert_not_called()

    def test_actual_archive_member_outside_manifest_allowlist_is_rejected(self) -> None:
        """Extraction still rejects a real archive member omitted from its manifest."""
        module, input_root, _working_root, _sources = self._prepare_fixture()
        manifest_path = input_root / module.BUNDLE_SLUG / "bundle-manifest.json"
        manifest = json.loads(manifest_path.read_text())
        spec_paths = set(manifest["spec_paths"])
        omitted_path = next(
            item["path"] for item in manifest["files"] if item["path"] not in spec_paths
        )
        manifest["files"] = [item for item in manifest["files"] if item["path"] != omitted_path]
        manifest_path.write_text(json.dumps(manifest, sort_keys=True))
        with mock.patch.object(module.subprocess, "Popen") as launch:
            with self.assertRaisesRegex(
                ValueError, "Unsupported, undeclared or duplicate archive member"
            ):
                module.main()
            launch.assert_not_called()

    def _assert_unsafe_key_rejected(self, logical: str) -> None:
        """Reject unsafe declared keys even if a test mutates both identity sides."""
        module = self._load_kernel()
        mounts = dict(module.EXPECTED_EXTERNAL_MOUNTS)
        mounts[logical] = "train.parquet"
        module, _input_root, _working_root, _sources = self._prepare_fixture(
            expected_mounts=mounts,
            manifest_mounts=mounts,
        )
        with mock.patch.object(module.subprocess, "Popen") as launch:
            with self.assertRaisesRegex(ValueError, "Unsafe external input mapping"):
                module.main()
            launch.assert_not_called()

    def test_parent_path_component_is_rejected(self) -> None:
        """The exact-key guard does not replace the existing traversal check."""
        self._assert_unsafe_key_rejected("../escape.parquet")

    def test_absolute_path_is_rejected(self) -> None:
        """An absolute logical path cannot redirect a symlink outside the root."""
        self._assert_unsafe_key_rejected("/tmp/escape.parquet")

    def test_backslash_path_is_rejected(self) -> None:
        """Windows-style separators are rejected consistently on POSIX."""
        self._assert_unsafe_key_rejected("..\\escape.parquet")


if __name__ == "__main__":
    unittest.main(verbosity=2)
