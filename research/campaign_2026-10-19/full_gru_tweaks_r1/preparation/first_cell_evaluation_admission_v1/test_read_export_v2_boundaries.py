from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


def load_reader_v2() -> object:
    """Load only the new stdlib metadata reader, without main execution."""
    spec = importlib.util.spec_from_file_location(
        "export_reader_v2_fixture", Path(__file__).parent / "READ_EXPORT_TEMPLATE_V2.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ChangedReaderBoundaries(unittest.TestCase):
    """Exercise only the two boundaries corrected after independent source review."""

    def test_dangling_prediction_symlink_is_rejected(self) -> None:
        reader = load_reader_v2()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            target = root / "exports/control/predictions.npy"
            target.parent.mkdir(parents=True)
            target.symlink_to(root / "absent")
            with patch.object(reader, "D", root), self.assertRaisesRegex(ValueError, "symlink"):
                reader.output_inventory()

    def test_cgroup_inventory_and_file_bytes_are_bounded(self) -> None:
        reader = load_reader_v2()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "cgroup.procs").write_bytes(b"1" * 32769)
            with self.assertRaisesRegex(RuntimeError, "exceeded"):
                reader.read_cgroup_tree(root)
            (root / "cgroup.procs").write_bytes(b"")
            for index in range(129):
                (root / str(index)).mkdir()
            with self.assertRaisesRegex(ValueError, "exceeds 128"):
                reader.read_cgroup_tree(root)


if __name__ == "__main__":
    unittest.main()
