"""Run all sixteen immutable E470 recipes at an explicitly registered new seed."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import fit_activation_optimizer as experiment


def main() -> None:
    """Override only the documented seed, keeping the original numerical code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=(271829, 271830), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    experiment.SEED = args.seed
    sys.argv = [str(Path(experiment.__file__)), "--output", str(args.output), "--device", args.device]
    experiment.main()
    result_path = args.output / "result.json"
    result = json.loads(result_path.read_text())
    if result["seed"] != args.seed or len(result["arms"]) != 16:
        raise ValueError("replication did not complete its fixed seed and all sixteen arms")
    result["replication_wrapper_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result["replication_design_sha256"] = hashlib.sha256(
        (Path(__file__).parent / "SEED_REPLICATION_PREREG.md").read_bytes()
    ).hexdigest()
    result_path.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
