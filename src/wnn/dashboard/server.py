from __future__ import annotations

import json
import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import cast
from urllib.parse import parse_qs, urlsplit

from wnn.dashboard.reader import DashboardReader

LOGGER = logging.getLogger(__name__)
MAX_QUERY_LENGTH = 4_096
STATIC_FILES = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/dashboard.css": ("dashboard.css", "text/css; charset=utf-8"),
    "/dashboard.js": ("dashboard.js", "text/javascript; charset=utf-8"),
}
SECURITY_HEADERS = {
    "Cache-Control": "no-store",
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "Content-Security-Policy": "default-src 'self'; script-src 'self'; style-src 'self'; "
    "connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'",
}


class DashboardHTTPServer(ThreadingHTTPServer):
    """A bounded-thread local server that exposes read-only dashboard routes."""

    allow_reuse_address = True
    daemon_threads = True
    request_queue_size = 8

    def __init__(self, port: int, run_root: Path | str, host: str = "127.0.0.1") -> None:
        """Bind only to IPv4 loopback and prepare the validated run reader."""
        if host != "127.0.0.1":
            raise ValueError("dashboard host must be 127.0.0.1")
        if not 0 <= port <= 65_535:
            raise ValueError("port must be between 0 and 65535")
        self.reader = DashboardReader(run_root)
        super().__init__((host, port), DashboardRequestHandler)


class DashboardRequestHandler(BaseHTTPRequestHandler):
    """Serve fixed static assets and safe JSON snapshots from the run reader."""

    server_version = "WNNLocalDashboard/1.0"
    sys_version = ""

    def do_GET(self) -> None:
        """Handle the small allowlist of local read-only routes."""
        if len(self.path) > MAX_QUERY_LENGTH:
            self._send_json(414, {"error": "request is too long"})
            return
        request = urlsplit(self.path)
        if request.path in STATIC_FILES:
            self._send_static(*STATIC_FILES[request.path])
            return
        if request.path == "/api/runs":
            self._get_run_list(parse_qs(request.query, keep_blank_values=True))
            return
        if request.path == "/api/run":
            self._get_run_events(parse_qs(request.query, keep_blank_values=True))
            return
        self._send_json(404, {"error": "route not found"})

    def do_POST(self) -> None:
        """Reject all state-changing requests."""
        self._send_json(405, {"error": "read-only endpoint"})

    def do_PUT(self) -> None:
        """Reject all state-changing requests."""
        self._send_json(405, {"error": "read-only endpoint"})

    def do_DELETE(self) -> None:
        """Reject all state-changing requests."""
        self._send_json(405, {"error": "read-only endpoint"})

    def log_message(self, format: str, *args: object) -> None:
        """Avoid echoing query strings or metadata into the local terminal log."""
        LOGGER.info("served dashboard request")

    def _get_run_list(self, query: dict[str, list[str]]) -> None:
        try:
            source_filter = self._single_query(query, "source", 200)
            config_filter = self._single_query(query, "config", 200)
            reader = cast(DashboardHTTPServer, self.server).reader
            result = reader.list_runs(source_filter, config_filter)
        except (ValueError, OSError):
            self._send_json(400, {"error": "run filters are invalid"})
            return
        self._send_json(200, result)

    def _get_run_events(self, query: dict[str, list[str]]) -> None:
        raw_ids = self._single_query(query, "ids", 800)
        run_ids = [run_id for run_id in raw_ids.split(",") if run_id]
        limit_value = self._single_query(query, "limit", 10)
        try:
            limit = int(limit_value) if limit_value else 1_200
            reader = cast(DashboardHTTPServer, self.server).reader
            result = reader.get_runs(run_ids, limit)
        except (ValueError, OSError):
            self._send_json(400, {"error": "run selection is invalid or unavailable"})
            return
        self._send_json(200, result)

    @staticmethod
    def _single_query(query: dict[str, list[str]], name: str, max_length: int) -> str:
        values = query.get(name, [""])
        if len(values) != 1 or len(values[0]) > max_length:
            raise ValueError("invalid query")
        return values[0]

    def _send_static(self, filename: str, content_type: str) -> None:
        asset = Path(__file__).with_name("static") / filename
        try:
            payload = asset.read_bytes()
        except OSError:
            self._send_json(500, {"error": "dashboard asset is unavailable"})
            return
        self._send_bytes(200, payload, content_type)

    def _send_json(self, status: int, value: object) -> None:
        try:
            payload = json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")
        except (TypeError, ValueError):
            status = 500
            payload = b'{"error":"response could not be encoded"}'
        self._send_bytes(status, payload, "application/json; charset=utf-8")

    def _send_bytes(self, status: int, payload: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        for name, value in SECURITY_HEADERS.items():
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(payload)


def create_server(port: int, run_root: Path | str, host: str = "127.0.0.1") -> DashboardHTTPServer:
    """Construct the loopback-only dashboard server for CLI or local checks."""
    return DashboardHTTPServer(port, run_root, host)
