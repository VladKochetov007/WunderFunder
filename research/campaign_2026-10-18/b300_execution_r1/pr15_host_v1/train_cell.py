"""Run the frozen PR15 trainer once inside the B300 process envelope."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


def _request(path: Path) -> dict[str, Any]:
    """Load an object-valued worker request."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("worker request must be a JSON object")
    return value


def main() -> None:
    """Pass unchanged PR15 command arguments to its canonical modular trainer."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, type=Path)
    args = parser.parse_args()
    request = _request(args.request.resolve(strict=True))
    source_root = Path(request["source_root"]).resolve(strict=True)
    work = Path(request["work"]).resolve(strict=True)
    train_args = request["train_argv"]
    if request.get("mode") not in {"profile", "fit"} or not isinstance(train_args, list):
        raise ValueError("unsupported PR15 training request")
    if os.environ.get("WNN_PROJECT_ROOT") != str(work):
        raise ValueError("trainer project root differs from its isolated work directory")
    sys.path.insert(0, str(source_root / "src"))
    from wnn.training.environment import TrainingEnvironment
    from wnn.training.runner import run_training

    environment = TrainingEnvironment(
        source_root=source_root,
        root=work,
        argv=[str(value) for value in train_args],
    )
    run_training(environment.argv, environment=environment)
    print(
        "PR15_TRAIN_RETURNED "
        + json.dumps({"mode": request["mode"], "name": request["name"]}, sort_keys=True),
        flush=True,
    )


if __name__ == "__main__":
    main()
