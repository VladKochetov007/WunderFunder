from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from multistage_native_fit_finalize import finalize_native_multistage_fit


def parse_args() -> argparse.Namespace:
    """Parse explicit native fit, source pin, data, and output paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--expected-config-sha256")
    parser.add_argument("--source-request", type=Path)
    parser.add_argument("--profile-receipt", type=Path, required=True)
    parser.add_argument("--approved-profile", type=Path, required=True)
    completion = parser.add_mutually_exclusive_group(required=True)
    completion.add_argument("--child-receipt", type=Path)
    completion.add_argument("--legacy-terminal-attestation", type=Path)
    parser.add_argument("--training-source-root", type=Path, required=True)
    parser.add_argument("--training-source-revision")
    parser.add_argument("--evaluation-source-root", type=Path, required=True)
    parser.add_argument("--evaluation-source-sha256", required=True)
    parser.add_argument("--evaluation-files-sha256", required=True)
    parser.add_argument("--evaluation-score-script-sha256", required=True)
    parser.add_argument("--evaluation-source-revision", default="source6-eval")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--portable-data-manifest", type=Path, required=True)
    parser.add_argument("--primary-split-manifest", type=Path, required=True)
    parser.add_argument("--auxiliary-split-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", type=int, default=1)
    parser.add_argument("--trial-number", type=int, required=True)
    parser.add_argument("--family", required=True)
    parser.add_argument("--parent-config", type=Path)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--timeout-seconds", type=int, default=600)
    parser.add_argument("--python")
    parser.add_argument("--cuda-visible-devices")
    return parser.parse_args()


def main() -> None:
    """Verify and postprocess an existing native auxiliary-trained fit without retraining."""
    args = parse_args()
    try:
        result: dict[str, Any] = finalize_native_multistage_fit(args)
    except (OSError, RuntimeError, TimeoutError, ValueError) as error:
        raise SystemExit(f"native fit postprocessing refused: {error}") from error
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
