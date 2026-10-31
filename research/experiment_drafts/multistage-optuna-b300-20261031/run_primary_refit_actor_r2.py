"""Run one bounded Stage 4 profile or explicitly approved native refit."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main() -> None:
    """Dispatch an exact profile or approved fit without selecting primary-valid results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("profile", "fit"), required=True)
    parser.add_argument("--fit-id", required=True)
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--operator-root", type=Path, required=True)
    parser.add_argument("--device-inventory", type=Path)
    parser.add_argument("--gpu-key")
    parser.add_argument("--native-approval", type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.operator_root.resolve() / "src"))
    try:
        if args.action == "profile":
            if not args.gpu_key:
                raise ValueError("profile action requires an admitted --gpu-key")
            from primary_refit_actor_profile import run_profile

            result = run_profile(args)
        else:
            if args.native_approval is None:
                raise ValueError("fit action requires an explicit --native-approval receipt")
            from primary_refit_actor_fit import run_fit

            result = run_fit(args)
    except (
        OSError,
        RuntimeError,
        ValueError,
        TypeError,
        KeyError,
        subprocess.SubprocessError,
    ) as error:
        raise SystemExit(f"Primary refit actor refused: {type(error).__name__}: {error}") from error
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
