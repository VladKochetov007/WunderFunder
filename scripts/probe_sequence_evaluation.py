"""Measure a bounded official-valid evaluation using fresh untrained weights."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from wnn.training.sequence_evaluation_probe import (
    MAX_PROBE_SEQUENCES,
    run_evaluation_probe,
)


def _parser() -> argparse.ArgumentParser:
    """Build the explicit bounded CLI for a one-device evaluation timing probe."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--source-audit", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, choices=(128, 384), default=128)
    parser.add_argument("--chunk-rows", type=int, choices=(64, 256), default=64)
    parser.add_argument("--max-sequences", type=int, default=MAX_PROBE_SEQUENCES)
    parser.add_argument("--output-json", type=Path)
    return parser


def main() -> int:
    """Run the bounded probe and write its complete engineering-only receipt."""
    args = _parser().parse_args()
    receipt = run_evaluation_probe(
        args.config,
        args.data_root,
        args.source_audit,
        device_name=args.device,
        batch_size=args.batch_size,
        chunk_rows=args.chunk_rows,
        max_sequences=args.max_sequences,
    )
    rendered = json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
