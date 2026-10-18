"""Command-line entrypoint for the bounded R01 density4 takeover."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from authority import load_authority
from frozen_panel import read_object, validate_panel, verify_frozen_packet
from lifecycle import freeze_old_controller, prepare_state, reconcile_authorized
from successor_runner import run_successor


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authority", type=Path, required=True)
    parser.add_argument("command", choices=("preview", "prepare", "freeze", "run"))
    return parser


def _preview(authority: dict[str, Any]) -> dict[str, Any]:
    packet = Path(authority["packet_root"])
    manifest = verify_frozen_packet(packet)
    jobs = validate_panel(read_object(packet / "PANEL.json"), Path(authority["root"]))
    snapshot = reconcile_authorized(authority, stopped=False)
    return {
        "schema": "tabdata-r01-density4-preview.v1",
        "status": "read_only_preview",
        "execute_enabled": authority["execute_enabled"],
        "panel_jobs": len(jobs),
        "claimed": snapshot["claimed"],
        "pending": snapshot["pending"],
        "old_main": snapshot["main"],
        "source_manifest_sha256": manifest.get("source_manifest_sha256"),
    }


def main() -> int:
    args = _parser().parse_args()
    authority = load_authority(args.authority, require_execute=args.command != "preview")
    if args.command == "preview":
        result = _preview(authority)
    elif args.command == "prepare":
        result = prepare_state(authority)
    elif args.command == "freeze":
        result = freeze_old_controller(authority)
    else:
        result = run_successor(authority)
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
