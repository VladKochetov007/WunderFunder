from __future__ import annotations

import base64
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from datetime import UTC, datetime
from http.client import HTTPMessage
from typing import BinaryIO

REPOSITORY = "VladKochetov007/WunderFunder"
ACCOUNT = "SemyonEpanov"
API = "https://api.github.com"


class RejectRedirect(urllib.request.HTTPRedirectHandler):
    """Keep credentials on the single declared API origin by rejecting redirects."""

    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: BinaryIO,
        code: int,
        msg: str,
        headers: HTTPMessage,
        newurl: str,
    ) -> None:
        raise RuntimeError("GitHub redirect rejected")


class ProgressGitHub:
    """Use only the explicitly supplied environment credential and fixed repository."""

    def __init__(self, pull_requests: set[int], deadline: float | None = None) -> None:
        """Bind the two ROOT-selected targets without accessing credentials or network."""
        if not 1 <= len(pull_requests) <= 2 or any(
            type(number) is not int or number <= 0 for number in pull_requests
        ):
            raise ValueError("positive ROOT-selected PR numbers are required")
        self.pull_requests = frozenset(pull_requests)
        self.deadline = deadline
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), RejectRedirect())

    def _headers(self) -> dict[str, str]:
        """Read the credential only when an authorized live request is made."""
        value = os.environ.get("WNN_GITHUB_BASIC", "")
        try:
            decoded = base64.b64decode(value, validate=True)
        except (ValueError, TypeError):
            raise RuntimeError("WNN_GITHUB_BASIC is invalid") from None
        if not value or len(value) > 4096 or b":" not in decoded or not all(decoded.split(b":", 1)):
            raise RuntimeError("WNN_GITHUB_BASIC must encode username and PAT")
        return {
            "Authorization": f"Basic {value}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2026-03-10",
            "User-Agent": "wnn-bounded-progress-reporter",
        }

    def request(self, method: str, path: str, payload: object = None) -> object:
        """Call a narrowly allowlisted endpoint, reporting no credential/error body."""
        parsed = urllib.parse.urlsplit(path)
        match = re.fullmatch(
            rf"/repos/{re.escape(REPOSITORY)}/(issues|pulls)/(\d+)(/comments)?", parsed.path
        )
        allowed = method == "GET" and path == "/user"
        if match and int(match.group(2)) in self.pull_requests:
            allowed = (
                method == "GET"
                and (
                    (match.group(1) == "pulls" and match.group(3) is None)
                    or (match.group(1) == "issues" and match.group(3) == "/comments")
                )
            ) or (
                method == "POST"
                and match.group(1) == "issues"
                and match.group(3) == "/comments"
                and not parsed.query
            )
        if not allowed or parsed.scheme or parsed.netloc or parsed.fragment:
            raise ValueError("GitHub endpoint is outside the declared reporting scope")
        data = None if payload is None else json.dumps(payload, allow_nan=False).encode()
        remaining = 30 if self.deadline is None else min(30, self.deadline - time.monotonic())
        if remaining <= 0:
            raise RuntimeError("reporting wall limit reached")
        request = urllib.request.Request(
            API + path, data=data, headers=self._headers(), method=method
        )
        if data is not None:
            request.add_header("Content-Type", "application/json")
        try:
            with self.opener.open(request, timeout=remaining) as response:
                if urllib.parse.urlsplit(response.geturl()).netloc != "api.github.com":
                    raise RuntimeError("GitHub response origin rejected")
                raw = response.read(8 * 1_048_576 + 1)
                if len(raw) > 8 * 1_048_576:
                    raise RuntimeError("GitHub response exceeded its bound")
                return json.loads(raw)
        except urllib.error.HTTPError as error:
            raise RuntimeError(f"GitHub {method} failed with HTTP {error.code}") from None
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError):
            raise RuntimeError(
                f"GitHub {method} request failed; response outcome may be uncertain"
            ) from None

    def verify_account(self) -> None:
        """Require the actual authenticated login before reading/writing PR reports."""
        user = self.request("GET", "/user")
        if not isinstance(user, dict) or user.get("login") != ACCOUNT:
            raise RuntimeError("GitHub authenticated login is not SemyonEpanov")

    def find_report(self, number: int, marker: str, prepared_at: float) -> dict[str, object] | None:
        """Reconcile a pending report marker before any possible duplicate POST."""
        self.verify_account()
        since = datetime.fromtimestamp(prepared_at - 300, UTC).isoformat().replace("+00:00", "Z")
        for page in range(1, 11):
            query = urllib.parse.urlencode({"per_page": 100, "page": page, "since": since})
            comments = self.request("GET", f"/repos/{REPOSITORY}/issues/{number}/comments?{query}")
            if not isinstance(comments, list):
                raise RuntimeError("GitHub comment listing is malformed")
            for comment in comments:
                if (
                    isinstance(comment, dict)
                    and marker in str(comment.get("body", ""))
                    and comment.get("user", {}).get("login") == ACCOUNT
                ):
                    return comment
            if len(comments) < 100:
                return None
        raise RuntimeError("GitHub deduplication scan reached its bound; no POST made")

    def post_report(
        self, number: int, body: str, *, before_post: Callable[[], None]
    ) -> dict[str, object]:
        """Verify account and PR target immediately before creating one comment."""
        self.verify_account()
        target = self.request("GET", f"/repos/{REPOSITORY}/pulls/{number}")
        if (
            not isinstance(target, dict)
            or target.get("number") != number
            or target.get("base", {}).get("repo", {}).get("full_name") != REPOSITORY
        ):
            raise RuntimeError("GitHub target is not the declared repository PR")
        self.verify_account()
        before_post()
        comment = self.request(
            "POST", f"/repos/{REPOSITORY}/issues/{number}/comments", {"body": body}
        )
        if (
            not isinstance(comment, dict)
            or type(comment.get("id")) is not int
            or comment.get("user", {}).get("login") != ACCOUNT
        ):
            raise RuntimeError("GitHub comment outcome is unconfirmed")
        return comment
