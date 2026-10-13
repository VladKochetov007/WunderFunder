"""Regression tests for bounded parsing of NPY prediction headers."""

from __future__ import annotations

import importlib.util
import struct
import tempfile
import unittest
from pathlib import Path
from typing import BinaryIO
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/alias_grade_prediction.py"
SPEC = importlib.util.spec_from_file_location("alias_grade_prediction", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot import alias helper at {SCRIPT}")
ALIAS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ALIAS)


class ReadCounter:
    """Count bytes requested from a wrapped binary stream."""

    def __init__(self, stream: BinaryIO) -> None:
        self._stream = stream
        self.bytes_read = 0

    def __enter__(self) -> ReadCounter:
        self._stream.__enter__()
        return self

    def __exit__(self, *args: object) -> None:
        self._stream.__exit__(*args)

    def read(self, size: int = -1) -> bytes:
        value = self._stream.read(size)
        self.bytes_read += len(value)
        return value


class BoundedNpyHeaderTests(unittest.TestCase):
    """Verify oversized declarations are rejected before their body is read."""

    def test_oversized_declared_header_is_rejected_after_fixed_prefix(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temp_dir:
            path = Path(temp_dir) / "oversized.npy"
            declared_length = ALIAS.MAX_NPY_HEADER_BYTES + 1
            path.write_bytes(
                b"\x93NUMPY\x02\x00"
                + struct.pack("<I", declared_length)
                + b"H" * declared_length
                + b"PAYLOAD_SENTINEL"
            )

            original_open = Path.open
            counter: ReadCounter | None = None

            def tracked_open(
                candidate: Path, *args: object, **kwargs: object
            ) -> BinaryIO | ReadCounter:
                nonlocal counter
                stream = original_open(candidate, *args, **kwargs)
                if candidate == path:
                    counter = ReadCounter(stream)
                    return counter
                return stream

            with (
                patch.object(Path, "open", tracked_open),
                self.assertRaisesRegex(ValueError, "NPY header exceeds"),
            ):
                ALIAS._npy_header(path)

            self.assertIsNotNone(counter)
            self.assertEqual(counter.bytes_read, 12)


if __name__ == "__main__":
    unittest.main()
