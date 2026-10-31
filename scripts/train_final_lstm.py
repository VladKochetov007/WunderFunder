from __future__ import annotations

import argparse
from pathlib import Path

from wnn.training.sequence_fit import run_sequence_fit


def build_parser() -> argparse.ArgumentParser:
    """Create the explicit Source4 LSTM all-labeled final-fit entrypoint."""
    parser = argparse.ArgumentParser(
        description=(
            "Train the fixed Source4 LSTM192x1 on all labeled whole sequences; "
            "this run has no independent evaluation set."
        )
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--final-fit-manifest", required=True, type=Path)
    parser.add_argument("--final-fit-source-validation-receipt", required=True, type=Path)
    parser.add_argument("--final-fit-source-validation-sha256", required=True)
    parser.add_argument("--profile-updates", type=int, default=0)
    parser.add_argument("--profile-receipt", type=Path)
    return parser


def main() -> None:
    """Validate the exact final-fit data identity and dispatch the shared trainer."""
    parser = build_parser()
    args = parser.parse_args()
    if args.profile_updates < 0:
        parser.error("--profile-updates cannot be negative")
    if args.profile_updates and args.profile_receipt is not None:
        parser.error("engineering profiles cannot consume a profile receipt")
    if not args.profile_updates and args.profile_receipt is None:
        parser.error("full fits require --profile-receipt from the matching final-fit profile")
    run_sequence_fit(
        config_path=args.config,
        data_root=args.data_root,
        split_manifest=None,
        output=args.output,
        entrypoint=Path(__file__).resolve(),
        profile_updates=args.profile_updates,
        profile_receipt=args.profile_receipt,
        final_fit_manifest=args.final_fit_manifest,
        final_fit_source_validation_receipt=args.final_fit_source_validation_receipt,
        final_fit_source_validation_sha256=args.final_fit_source_validation_sha256,
    )


if __name__ == "__main__":
    main()
