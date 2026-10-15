from __future__ import annotations

import sys
import unittest
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import collect_terminal_r1 as collector


class FakeHelper:
    @staticmethod
    def relative_result_path(name: str) -> PurePosixPath:
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or not name.startswith("results/"):
            raise ValueError("unsafe results path")
        return path


class CollectionInventoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.helper = FakeHelper()
        self.files = [
            SimpleNamespace(file_name=name, url=f"https://storage.invalid/{index}")
            for index, name in enumerate(sorted(collector.expected_remote_paths()))
        ]

    def test_exact_complete_inventory_is_bounded(self) -> None:
        with patch.object(collector, "declared_length", return_value=1024):
            rows, missing = collector.inventory(self.files, 2048, self.helper, complete=True)
        self.assertEqual(len(rows), 14)
        self.assertEqual(missing, [])
        self.assertEqual({row["path"] for row in rows}, collector.expected_remote_paths())

    def test_terminal_partial_inventory_is_preserved_as_missing(self) -> None:
        partial = self.files[:-1]
        with patch.object(collector, "declared_length", return_value=256):
            rows, missing = collector.inventory(partial, 512, self.helper, complete=True)
        self.assertEqual(len(rows), 13)
        self.assertEqual(len(missing), 1)

    def test_unexpected_path_fails_before_size_requests(self) -> None:
        unexpected = [SimpleNamespace(file_name="results/unexpected.bin", url="https://invalid")]
        with (
            patch.object(collector, "declared_length", side_effect=AssertionError),
            self.assertRaisesRegex(ValueError, "undeclared"),
        ):
            collector.inventory(unexpected, 64, self.helper, complete=False)

    def test_inventory_cap_is_enforced_before_download(self) -> None:
        with (
            patch.object(collector, "declared_length", return_value=collector.MAX_BYTES),
            self.assertRaisesRegex(ValueError, "2.5 GB"),
        ):
            collector.inventory([self.files[0]], 1, self.helper, complete=False)


if __name__ == "__main__":
    unittest.main()
