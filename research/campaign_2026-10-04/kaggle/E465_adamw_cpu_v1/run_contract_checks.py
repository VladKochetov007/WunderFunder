"""Record source-only transport checks without importing a machine-learning runtime."""

from __future__ import annotations

import argparse
import io
import json
import unittest
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    suite = unittest.defaultTestLoader.discover(str(Path(__file__).parent / "tests"))
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    report = {
        "status": "passed" if result.wasSuccessful() else "failed",
        "tests_run": result.testsRun,
        "output": stream.getvalue(),
        "scope": "stdlib worker-report and source-identity contracts; no JAX/Torch execution",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": report["status"], "tests_run": result.testsRun}))
    return 0 if result.wasSuccessful() and result.testsRun == 5 else 1


if __name__ == "__main__":
    raise SystemExit(main())
