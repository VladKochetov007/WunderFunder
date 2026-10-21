"""Tiny local metadata fixture for caller V8 empty stderr records."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path("/Users/nb/Documents/wunder/WunderFunder-maintenance")
CALLER = (
    ROOT
    / "research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/first_cell_evaluation_score_acceptance_v1/ROOT_ACCEPT_SCORE_RESULT_V8.py"
)


def _load_read_function():
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location("score_acceptance_v8_fixture", CALLER)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load the frozen caller source")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.read


class EmptyStderrFixture(unittest.TestCase):
    def test_only_explicit_empty_stderr_reads_are_permitted(self) -> None:
        read = _load_read_function()
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            status_stderr = folder / "status-STDERR.raw.txt"
            launch_stderr = folder / "launch-STDERR.raw.txt"
            nonempty = folder / "TRANSPORT.json"
            status_stderr.write_bytes(b"")
            launch_stderr.write_bytes(b"")
            nonempty.write_bytes(b"{}\n")

            self.assertEqual(read(nonempty), b"{}\n")
            with self.assertRaises(ValueError):
                read(status_stderr, 4096)
            self.assertEqual(read(status_stderr, 4096, allow_empty=True), b"")
            self.assertEqual(read(launch_stderr, 4096, allow_empty=True), b"")


if __name__ == "__main__":
    unittest.main()
