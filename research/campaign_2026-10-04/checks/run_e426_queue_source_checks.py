"""Run the pure-standard-library safety fixtures for the E426 queue."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
TEST_PATH = ROOT / "research/campaign_2026-10-04/staging/e426_clock_comparison/tests/test_queue_v2.py"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output = output.resolve()
    if not output.is_relative_to(ROOT) or output.exists():
        raise ValueError(f"source-check output must be new and inside the repository: {output}")
    spec = importlib.util.spec_from_file_location("e426_queue_source_tests", TEST_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load fixture module: {TEST_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    suite = unittest.defaultTestLoader.loadTestsFromModule(module)
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    report = {
        "schema": "e426-queue-source-checks-v1",
        "status": "passed" if result.wasSuccessful() else "failed",
        "tests_run": result.testsRun,
        "failures": [error for _test, error in result.failures],
        "errors": [error for _test, error in result.errors],
        "test_sha256": sha256_file(TEST_PATH),
        "stdout": stream.getvalue(),
        "imports_torch": False,
        "launches_workers": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as destination:
        destination.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": report["status"], "tests_run": report["tests_run"],
                      "report": output.relative_to(ROOT).as_posix()}, sort_keys=True))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
