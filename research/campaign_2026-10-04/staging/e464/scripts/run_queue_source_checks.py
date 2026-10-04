"""Execute lightweight E464 queue-controller negative fixtures under the CPU draft."""

from __future__ import annotations

import io
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
TESTS = ROOT / "research/campaign_2026-10-04/staging/e464/tests"
OUTPUT = ROOT / "research/campaign_2026-10-04/checks/e464_queue_source_checks_v1.json"
LOG = ROOT / "research/campaign_2026-10-04/checks/e464_queue_source_checks_v1.log"


def main() -> int:
    if OUTPUT.exists() or LOG.exists():
        raise FileExistsError("refusing to overwrite E464 queue source-check outputs")
    stream = io.StringIO()
    suite = unittest.defaultTestLoader.discover(
        start_dir=str(TESTS), pattern="test_e464_queue_controller.py"
    )
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    log = stream.getvalue()
    LOG.write_text(log, encoding="utf-8")
    if not result.wasSuccessful() or result.testsRun != 11:
        raise RuntimeError(f"E464 queue source checks failed; see {LOG}")
    OUTPUT.write_text(
        json.dumps({"status": "pass", "tests_run": result.testsRun,
                    "failures": len(result.failures), "errors": len(result.errors)},
                   indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "pass", "tests_run": result.testsRun}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
