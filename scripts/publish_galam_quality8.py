from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT / "scripts")]


def build_parser() -> argparse.ArgumentParser:
    """Build the explicit publisher command without accepting endpoint overrides."""
    parser = argparse.ArgumentParser(
        description="Publish fixed-eight GALAM run updates to issue 46"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    publish = commands.add_parser("publish", help="poll one launch receipt and post updates")
    publish.add_argument("--receipt", required=True, type=Path)
    publish.add_argument("--token-file", required=True, type=Path)
    publish.add_argument(
        "--output", required=True, type=Path, help="private publisher state directory"
    )
    publish.add_argument(
        "--cleanup-owned-token",
        action="store_true",
        help="remove the token only if it is still the exact inode opened by this process",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the requested local publisher operation."""
    from galam_l4_quality8.publisher import PublisherError, run_publisher

    args = build_parser().parse_args(argv)
    if args.command != "publish":
        return 2
    try:
        run_publisher(
            args.receipt,
            args.token_file,
            args.output,
            cleanup_owned_token=args.cleanup_owned_token,
        )
    except PublisherError as error:
        print(f"publisher stopped: {error}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("publisher interrupted", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
