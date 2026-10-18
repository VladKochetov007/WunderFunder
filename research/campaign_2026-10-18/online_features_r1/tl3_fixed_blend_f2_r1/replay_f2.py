"""Replay one exact package on source-qualified F2 inputs without labels."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PACKET = Path(__file__).resolve().parent
ROOT = PACKET.parents[3]
sys.path.insert(0, str(PACKET.parent / "tl3_replay"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from f2_candidates import CANDIDATES  # noqa: E402
from f2_receipts import write_receipts  # noqa: E402
from f2_replay_engine import replay  # noqa: E402
from f2_source import validate_f2_source  # noqa: E402


def main() -> None:
    """Validate frozen F2 sources and run an exact, feature-only replay."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--candidate", choices=tuple(CANDIDATES), required=True)
    parser.add_argument("--zip", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--output-predictions", type=Path, required=True)
    parser.add_argument("--output-sidecar", type=Path, required=True)
    parser.add_argument("--output-receipt", type=Path, required=True)
    parser.add_argument("--max-groups", type=int)
    parser.add_argument("--reset-check", action="store_true")
    parser.add_argument("--resource-profile", required=True)
    args = parser.parse_args()
    root = args.repo_root.resolve(strict=True)
    archive_path = args.zip.resolve(strict=True)
    split_path = args.split.resolve(strict=True)
    scratch = args.scratch.resolve(strict=True)
    outputs = (
        args.output_predictions.resolve(),
        args.output_sidecar.resolve(),
        args.output_receipt.resolve(),
    )
    if any(path.exists() for path in outputs):
        raise FileExistsError("refusing to overwrite a prior replay artifact")
    for path in (outputs[0].parent, outputs[1].parent, outputs[2].parent):
        if not path.is_dir():
            raise NotADirectoryError(path)
    source = validate_f2_source(root, split_path, args.max_groups)
    result = replay(source, args.candidate, archive_path, scratch, outputs[0], args.reset_check)
    report = write_receipts(
        source,
        result,
        args.candidate,
        archive_path,
        split_path,
        outputs[0],
        outputs[1],
        outputs[2],
        args.resource_profile,
    )
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "candidate": args.candidate,
                "groups": report["evaluation_groups_replayed"],
                "rows": report["rows_total"],
                "reset_aba_equal": report["reset_aba_equal"],
                "callback_seconds": report["timing"]["callback_only_seconds"],
                "memory_max_rss_kib": report["memory_max_rss_kib"],
                "receipt": str(outputs[2]),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
