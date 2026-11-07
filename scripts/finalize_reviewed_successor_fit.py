from __future__ import annotations

import argparse
import json
from pathlib import Path

from wnn.multistage_search_successor_finalizer import finalize_successor_fit


def main() -> None:
    """Finalize one completed reviewed successor fit without repeating training."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--stage", required=True, type=int, choices=(2, 3))
    parser.add_argument("--trial-number", required=True, type=int)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--timeout-seconds", type=int, default=600)
    parser.add_argument("--python")
    args = parser.parse_args()
    result = finalize_successor_fit(
        args.run_dir,
        args.stage,
        args.trial_number,
        batch_size=args.batch_size,
        timeout_seconds=args.timeout_seconds,
        python=args.python,
    )
    print(json.dumps(result, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
