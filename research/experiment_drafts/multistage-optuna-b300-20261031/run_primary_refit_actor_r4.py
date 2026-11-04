"""Launch one root-approved Stage4 refit from an existing R5-reviewed native profile."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main() -> None:
    """Admit fitting only; the global campaign profile budget is already exhausted."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("fit",), required=True)
    parser.add_argument("--fit-id", required=True)
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--operator-root", type=Path, required=True)
    parser.add_argument("--device-inventory", type=Path)
    parser.add_argument("--native-approval", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.operator_root.resolve() / "src"))
    try:
        from primary_refit_actor_fit_r4 import run_fit

        result = run_fit(args)
    except (
        OSError,
        RuntimeError,
        ValueError,
        TypeError,
        KeyError,
        subprocess.SubprocessError,
    ) as error:
        raise SystemExit(f"Primary refit R4 refused: {type(error).__name__}: {error}") from error
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
