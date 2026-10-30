from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .admission import preflight
from .common import LaunchError
from .dashboard import start_dashboard
from .monitor import refresh_summary
from .supervisor import reserve_and_detach, run_supervisor

REPO_ROOT = Path(__file__).resolve().parents[2]


def build_parser() -> argparse.ArgumentParser:
    """Build commands for preflight, detached launch, monitoring, and dashboard."""
    parser = argparse.ArgumentParser(description="Fixed eight-fit GALAM L4 launch supervisor.")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("preflight", "launch"):
        subparser = commands.add_parser(name)
        subparser.add_argument("--panel", required=True, type=Path)
        subparser.add_argument("--profile", required=True, action="append", metavar="CELL=PATH")
        subparser.add_argument("--data-root", required=True, type=Path)
        subparser.add_argument("--split-manifest", required=True, type=Path)
        subparser.add_argument("--output-root", required=True, type=Path)
        subparser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
        if name == "launch":
            subparser.add_argument("--python", required=True, type=Path)
    supervise = commands.add_parser("supervise", help=argparse.SUPPRESS)
    supervise.add_argument("--receipt", required=True, type=Path)
    monitor = commands.add_parser("monitor")
    monitor.add_argument("--receipt", required=True, type=Path)
    dashboard = commands.add_parser("dashboard")
    dashboard.add_argument("--launch-root", required=True, type=Path)
    dashboard.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Validate command inputs and dispatch only the fixed-eight operations."""
    args = build_parser().parse_args(argv)
    try:
        result = _dispatch(args)
    except (LaunchError, OSError, ValueError, KeyError) as error:
        print(f"launch rejected: {error}", file=sys.stderr)
        return 2
    if isinstance(result, int):
        return result
    if result is not None:
        print(json.dumps(result, sort_keys=True, indent=2, ensure_ascii=False))
    return 0


def _dispatch(args: argparse.Namespace) -> dict[str, Any] | int | None:
    if args.command == "preflight":
        manifest = preflight(
            args.panel,
            args.profile,
            args.data_root,
            args.split_manifest,
            args.output_root,
            args.repo_root,
        )
        return {
            "status": "preflight_passed",
            "panel_sha256": manifest["panel_sha256"],
            "source_sha256": manifest["source_sha256"],
            "split_sha256": manifest["split_sha256"],
            "gpus": manifest["runtime"]["gpu_uuids"],
            "cell_ids": list(manifest["cells"]),
        }
    if args.command == "launch":
        return reserve_and_detach(
            python=args.python,
            script=Path(__file__).resolve().parents[1] / "launch_galam_quality8.py",
            repo_root=args.repo_root,
            panel=args.panel,
            profiles=args.profile,
            data_root=args.data_root,
            split_manifest=args.split_manifest,
            output_root=args.output_root,
        )
    if args.command == "supervise":
        return run_supervisor(args.receipt)
    if args.command == "monitor":
        return refresh_summary(args.receipt)
    if args.command == "dashboard":
        return start_dashboard(args.launch_root, args.repo_root)
    raise LaunchError("unknown launch command")
