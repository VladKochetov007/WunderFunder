from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path

from plot_results import draw
from reference_panel import compare
from sample_data import write_json


def main() -> None:
    """Add matched-panel context and figures without any model execution."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--eda", type=Path, required=True)
    parser.add_argument("--ridge", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    args.output.mkdir(exist_ok=False)
    compare(args.sample, args.output / "REFERENCE.json")
    draw(
        json.loads(args.eda.read_bytes()),
        json.loads(args.ridge.read_bytes()),
        args.output / "plots",
    )
    usage = resource.getrusage(resource.RUSAGE_SELF)
    write_json(
        args.output / "DONE.json",
        {
            "elapsed_s": time.monotonic() - started,
            "user_cpu_s": usage.ru_utime,
            "system_cpu_s": usage.ru_stime,
            "maxrss_kib": usage.ru_maxrss,
            "scope": "existing fixed-prediction context and result plotting only",
        },
    )


if __name__ == "__main__":
    main()
