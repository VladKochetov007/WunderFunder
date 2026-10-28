from __future__ import annotations

import argparse
import logging
from pathlib import Path

from wnn.dashboard.server import create_server


def _parser() -> argparse.ArgumentParser:
    """Build the small command line interface for the local dashboard."""
    parser = argparse.ArgumentParser(description="Serve a local read-only run dashboard.")
    parser.add_argument(
        "--runs-root", required=True, type=Path, help="directory containing run folders"
    )
    parser.add_argument("--host", default="127.0.0.1", help="local bind host (must be 127.0.0.1)")
    parser.add_argument("--port", default=8766, type=int, help="loopback HTTP port (default: 8766)")
    return parser


def main() -> None:
    """Start the dashboard on 127.0.0.1 without exposing a write endpoint."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = _parser().parse_args()
    try:
        server = create_server(args.port, args.runs_root, args.host)
    except (OSError, ValueError) as exc:
        raise SystemExit(f"Cannot start dashboard: {exc}") from exc
    host, port = server.server_address[:2]
    logging.getLogger(__name__).info("Dashboard listening at http://%s:%s/", host, port)
    logging.getLogger(__name__).info("Run root is read-only; press Ctrl+C to stop.")
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        logging.getLogger(__name__).info("Dashboard stopped.")
    finally:
        server.server_close()
