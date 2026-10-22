"""Bounded tests for terminal metadata collection."""

from __future__ import annotations

import struct
import tempfile
import unittest
from pathlib import Path

from scripts.collect_online_panel_terminal import read_npy_header


class NpyHeaderTests(unittest.TestCase):
    """Check NPY metadata parsing without touching payload values."""

    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)

    def test_reads_v1_header_and_reports_payload_bytes(self) -> None:
        header = self._write_npy(
            1, b"{'descr': '<f4', 'fortran_order': False, 'shape': (2, 3),}", b"1234"
        )
        result = read_npy_header(header)
        self.assertEqual(result["version"], [1, 0])
        self.assertEqual(result["descr"], "<f4")
        self.assertEqual(result["shape"], [2, 3])
        self.assertEqual(result["payload_bytes"], 4)

    def test_reads_v2_header(self) -> None:
        header = self._write_npy(
            2, b"{'descr': '<f4', 'fortran_order': False, 'shape': (1,),}", b"x"
        )
        result = read_npy_header(header)
        self.assertEqual(result["version"], [2, 0])
        self.assertEqual(result["shape"], [1])
        self.assertEqual(result["payload_bytes"], 1)

    def _write_npy(self, major: int, header: bytes, payload: bytes) -> Path:
        prefix_bytes = 10 if major == 1 else 12
        alignment = 16
        padding = (-(prefix_bytes + len(header) + 1)) % alignment
        padded_header = header + b" " * padding + b"\n"
        length = struct.pack("<H" if major == 1 else "<I", len(padded_header))
        path = Path(self.temporary_directory.name) / f"v{major}.npy"
        with path.open("wb") as output:
            output.write(b"\x93NUMPY" + bytes((major, 0)) + length + padded_header + payload)
        return path


if __name__ == "__main__":
    unittest.main()
