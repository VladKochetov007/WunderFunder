"""Export evaluation-only epoch-2 predictions with the original PR15 predictor."""

from __future__ import annotations

import argparse
import json
import runpy
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
    """Run the pinned predict_run.py with its exact held-out-fold command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, type=Path)
    args = parser.parse_args()
    request = _request(args.request.resolve(strict=True))
    source_root = Path(request["source_root"]).resolve(strict=True)
    script = source_root / "scripts" / "predict_run.py"
    if not script.is_file() or request.get("mode") != "fit":
        raise ValueError("PR15 predictor request or source is missing")
    sys.path[:0] = [str(script.parent), str(source_root / "src")]
    sys.argv = [str(script), *[str(value) for value in request["predict_argv"]]]
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
