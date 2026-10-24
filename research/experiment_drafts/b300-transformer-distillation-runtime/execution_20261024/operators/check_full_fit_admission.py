"""Exercise real full-fit admission, then exit before the first scientific update."""

from __future__ import annotations

import argparse
import json
import os
import runpy
import sys
from pathlib import Path


class AdmissionVerified(Exception):
    """Stop the engineering check after canonical receipt and metadata admission."""


def main() -> None:
    """Use the unchanged full-fit CLI and intercept only the subsequent fit dispatch."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--proof-root", type=Path, required=True)
    args, remaining = parser.parse_known_args()
    source = args.source_root.resolve(strict=True)
    sys.path.insert(0, str(source / "src"))
    from wnn.training import sequence_fit

    def stop_before_fit(*_args: object, **_kwargs: object) -> None:
        """Canonical profile validation already passed when this dispatch is reached."""
        raise AdmissionVerified()

    sequence_fit.run_full_fit = stop_before_fit
    entrypoint = source / "scripts/train_sequence.py"
    sys.argv = [str(entrypoint), *remaining]
    try:
        runpy.run_path(str(entrypoint), run_name="__main__")
    except AdmissionVerified:
        args.proof_root.mkdir(parents=True, exist_ok=True)
        rank = int(os.environ.get("RANK", "0"))
        receipt = {
            "status": "full_fit_admission_verified",
            "rank": rank,
            "world_size": int(os.environ.get("WORLD_SIZE", "1")),
            "scientific_updates": 0,
            "source_commit": json.loads(
                (source / "TRANSPORT_MANIFEST.json").read_text()
            )["commit"],
            "method": "Unchanged canonical profile admission and metadata setup; intercept run_full_fit before training.",
        }
        with (args.proof_root / f"rank{rank}.json").open("x") as stream:
            json.dump(receipt, stream, indent=2)
        print(json.dumps(receipt))
    else:
        raise RuntimeError("full-fit dispatch was not reached")


if __name__ == "__main__":
    main()
