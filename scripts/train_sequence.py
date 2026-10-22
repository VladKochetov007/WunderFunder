from __future__ import annotations

import argparse
from pathlib import Path

from wnn.training.sequence_fit import run_sequence_fit


def build_parser() -> argparse.ArgumentParser:
    """Create the thin torchrun entrypoint for fixed-final sequence fitting."""
    parser = argparse.ArgumentParser(
        description="Train the source-qualified causal sequence model on one CUDA GPU per rank."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--split-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--teacher-cache", type=Path)
    parser.add_argument("--input-artifact", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--profile-updates", type=int, default=0)
    parser.add_argument("--profile-receipt", type=Path)
    return parser


def main() -> None:
    """Validate profile/resume mode and dispatch the modular training runtime."""
    parser = build_parser()
    args = parser.parse_args()
    if args.profile_updates < 0:
        parser.error("--profile-updates cannot be negative")
    if args.profile_updates and (args.resume is not None or args.profile_receipt is not None):
        parser.error("profile runs cannot resume or consume a profile receipt")
    if not args.profile_updates and args.profile_receipt is None:
        parser.error("full fits require --profile-receipt with explicit approval")
    run_sequence_fit(
        config_path=args.config,
        data_root=args.data_root,
        split_manifest=args.split_manifest,
        output=args.output,
        entrypoint=Path(__file__).resolve(),
        teacher_cache=args.teacher_cache,
        resume=args.resume,
        profile_updates=args.profile_updates,
        profile_receipt=args.profile_receipt,
        input_artifact=args.input_artifact,
    )


if __name__ == "__main__":
    main()
