from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib
import platform
import resource
import sys
import time
from pathlib import Path

from sample_data import load_sample, write_json


def main() -> None:
    """Run one reviewed analysis on the immutable bounded remote sample."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--operation", choices=("eda", "ridge"), required=True)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    packet = Path(__file__).resolve().parent
    source = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(packet.glob("*.py"))
    }
    before = time.monotonic()
    write_json(
        args.output / "START.json",
        {
            "utc": dt.datetime.now(dt.UTC).isoformat(),
            "operation": args.operation,
            "sample_manifest_sha256": hashlib.sha256(
                (args.sample / "SAMPLE.json").read_bytes()
            ).hexdigest(),
            "source_sha256": source,
            "python": platform.python_version(),
        },
    )
    sequences, manifest = load_sample(args.sample)
    sys.path.insert(0, str(packet.parent))
    package = packet.name
    module = importlib.import_module(
        f"{package}.{'eda_mechanisms' if args.operation == 'eda' else 'ridge_probe'}"
    )
    module.run(sequences, args.output / "RESULT.json")
    usage = resource.getrusage(resource.RUSAGE_SELF)
    write_json(
        args.output / "DONE.json",
        {
            "utc": dt.datetime.now(dt.UTC).isoformat(),
            "operation": args.operation,
            "elapsed_s": time.monotonic() - before,
            "user_cpu_s": usage.ru_utime,
            "system_cpu_s": usage.ru_stime,
            "maxrss_kib": usage.ru_maxrss,
            "sequences": len(manifest["records"]),
            "result_sha256": hashlib.sha256((args.output / "RESULT.json").read_bytes()).hexdigest(),
            "cuda_initialized": "torch.cuda" in sys.modules,
        },
    )


if __name__ == "__main__":
    main()
