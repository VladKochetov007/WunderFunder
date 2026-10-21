from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


def load_reader() -> object:
    """Load the new stdlib metadata reader without invoking its main."""
    path = Path(__file__).parent / "READ_EXPORT_TEMPLATE.py"
    spec = importlib.util.spec_from_file_location("export_metadata_reader_fixture", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ReaderBoundaryTests(unittest.TestCase):
    """Check boundaries that prevent accidental payload reads or unbound B queries."""

    def test_unbound_template_fails_before_remote_metadata_access(self) -> None:
        reader = load_reader()
        with (
            patch.object(reader, "read_regular", side_effect=AssertionError("unexpected read")),
            patch.object(reader, "show", side_effect=AssertionError("unexpected query")),
            self.assertRaisesRegex(ValueError, "unbound"),
        ):
            reader.main()

    def test_empty_own_cgroup_never_reads_host_root(self) -> None:
        reader = load_reader()
        with patch.object(Path, "read_text", side_effect=AssertionError("host-root read")):
            self.assertIsNone(reader.cgroup_snapshot(""))
            self.assertIsNone(reader.cgroup_snapshot("/"))

    def test_inventory_stats_predictions_without_opening_them(self) -> None:
        reader = load_reader()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            predictions = root / "exports/control/predictions.npy"
            predictions.parent.mkdir(parents=True)
            predictions.write_bytes(b"synthetic-unread-payload")
            with (
                patch.object(reader, "D", root),
                patch.object(reader.os, "open", side_effect=AssertionError("payload open")),
            ):
                inventory = reader.output_inventory()
            row = next(item for item in inventory if item["path"] == str(predictions))
            self.assertTrue(row["exists"])
            self.assertEqual(row["bytes"], 24)

    def test_metadata_symlink_and_cap_fail_closed(self) -> None:
        reader = load_reader()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            target = root / "metadata.json"
            target.write_bytes(b"12345")
            alias = root / "alias.json"
            alias.symlink_to(target)
            with self.assertRaisesRegex(ValueError, "unsafe metadata"):
                reader.read_regular(alias, 10)
            with self.assertRaisesRegex(ValueError, "exceeds cap"):
                reader.read_regular(target, 4)


if __name__ == "__main__":
    unittest.main()
