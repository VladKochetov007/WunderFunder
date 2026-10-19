"""CPU-only official grading for one exported native-feature pair."""

from __future__ import annotations

import argparse
from pathlib import Path

from feature_common import read, verify_config, verify_source
from feature_outputs import grade_pair


def main() -> None:
    """Grade one exact pair; a failure never reopens the corresponding fit cells."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--control", required=True)
    args = parser.parse_args()
    config = verify_config(args.config, args.config_sha256)
    root = Path(config["root"])
    verify_source(root)
    state = root / "state/parallel-feature-execution-v1"
    ids = {job["id"] for job in config["panel"]["jobs"]}
    if args.candidate not in ids or args.control not in ids:
        raise ValueError("Grade endpoint belongs outside the frozen panel")
    candidate = read(state / "work" / args.candidate / "EXPORTED.json")
    control = read(state / "work" / args.control / "EXPORTED.json")
    grade_pair(root, state, candidate, control)


if __name__ == "__main__":
    main()
