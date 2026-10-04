"""Run source-only consistency tests for the timing-window v2 drafts."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
TEST_PATH = (
    ROOT / "research/campaign_2026-10-04/staging/timing_window/v3/tests/test_timing_window_v3.py"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output = output.resolve()
    if not output.is_relative_to(ROOT):
        raise ValueError("fixture report output must stay inside the repository")
    if output.exists():
        raise FileExistsError(output)

    spec = importlib.util.spec_from_file_location("timing_window_v3_tests", TEST_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load test module: {TEST_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    suite = unittest.defaultTestLoader.loadTestsFromModule(module)
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    report = {
        "schema": "wnn.timing-window-v3-source-check-v1",
        "status": "pass" if result.wasSuccessful() else "fail",
        "tests_run": result.testsRun,
        "failures": [str(error) for _, error in result.failures],
        "errors": [str(error) for _, error in result.errors],
        "test_source_sha256": sha256(TEST_PATH),
        "stdout": stream.getvalue(),
        "launches_workers": False,
        "imports_torch": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as destination:
        destination.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "status": report["status"],
                "tests_run": report["tests_run"],
                "report": output.relative_to(ROOT).as_posix(),
            },
            sort_keys=True,
        )
    )
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
