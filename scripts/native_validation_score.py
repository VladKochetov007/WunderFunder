"""Replay a fixed submission callback over the whole supplied validation file."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import tempfile
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils import ScorerStepByStep  # noqa: E402


def digest(path: Path) -> str:
    """Hash a fixed archive without retaining its contents."""
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    """Score actual callbacks, retaining official moments and source identity."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    arguments = parser.parse_args()
    if arguments.output.exists():
        raise FileExistsError(arguments.output)
    if digest(arguments.submission) != arguments.expected_sha256:
        raise ValueError("Submission bytes differ from the prescribed public archive")
    start = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="wnn-native-validation-") as temporary:
        directory = Path(temporary)
        with zipfile.ZipFile(arguments.submission) as archive:
            archive.extractall(directory)
        spec = importlib.util.spec_from_file_location(
            "full_validation_solution", directory / "solution.py"
        )
        if spec is None or spec.loader is None:
            raise ImportError("Submission is missing its callback module")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        model = module.PredictionModel()
        score = ScorerStepByStep(arguments.data).score(model)
    if digest(arguments.submission) != arguments.expected_sha256:
        raise ValueError("Submission changed during its whole-file replay")
    result = {
        "submission": str(arguments.submission),
        "zip_sha256": arguments.expected_sha256,
        "data": str(arguments.data),
        "score": score,
        "elapsed_seconds": time.monotonic() - start,
        "scope": (
            "Actual fixed native callback over all supplied validation sequences. "
            "This file includes sequences used in final fitting; it is not independent CV. "
            "No checkpoint, weight or predictor selection."
        ),
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
