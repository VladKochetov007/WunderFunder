"""Run a synthetic-only shape and throughput probe through the sequence runtime."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path

from wnn.training.sequence_shape_probe import run_probe, source_identity


def build_parser() -> argparse.ArgumentParser:
    """Create the bounded synthetic probe CLI; no data or training output is accepted."""
    parser = argparse.ArgumentParser(
        description="Measure a few synthetic FP32 sequence updates; makes no quality claim."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, help="override the config batch size")
    parser.add_argument("--warmup-updates", type=int, default=2)
    parser.add_argument("--measure-updates", type=int, default=3)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--output-json", type=Path)
    parser.add_argument(
        "--profiler-trace",
        type=Path,
        help="optionally capture one extra update in a Chrome trace, outside timing",
    )
    return parser


def _emit(result: Mapping[str, object], output_path: Path | None) -> None:
    """Write one finite JSON receipt to stdout and optionally to a file."""
    rendered = json.dumps(result, sort_keys=True, allow_nan=False)
    print(rendered)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")


def main() -> int:
    """Run the bounded shape probe and preserve errors as machine-readable receipts."""
    args = build_parser().parse_args()
    try:
        result = run_probe(args)
        checks = (
            "predictions_and_losses_finite",
            "all_trainable_parameters_have_gradients",
            "gradients_finite",
            "parameters_finite",
            "optimizer_fp32_state_finite",
        )
        result["checks_passed"] = all(result.get(name) is True for name in checks)
        _emit(result, args.output_json)
        return 0 if result["checks_passed"] else 1
    except Exception as error:
        failure = {
            "status": "error",
            "label": "synthetic_engineering_only",
            "quality_claim": "no_quality_claim",
            "source_identity": source_identity(args.config),
            "error_type": type(error).__name__,
            "error": str(error),
        }
        _emit(failure, args.output_json)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
