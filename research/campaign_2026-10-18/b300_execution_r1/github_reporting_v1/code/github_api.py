"""Bounded GitHub access for the explicitly authorized experiment updates."""

from __future__ import annotations

import json
import os
import stat
import urllib.error
import urllib.request
from pathlib import Path

REPOSITORY = "VladKochetov007/WunderFunder"


class GitHubError(RuntimeError):
    """Expose HTTP status without credentials, request headers or response bodies."""

    def __init__(self, status: int, retry_after: int = 0) -> None:
        self.status = status
        self.retry_after = retry_after
        super().__init__(f"GitHub HTTP {status}")


def read_token(path: Path) -> str:
    """Read a private regular credential file; never put its value in diagnostics."""
    if path.is_symlink() or not path.is_file():
        raise ValueError("GitHub credential is not a regular private file")
    if os.name == "posix" and stat.S_IMODE(path.stat().st_mode) & 0o077:
        raise ValueError("GitHub credential must not be accessible to other users")
    token = path.read_text(encoding="ascii").strip()
    if not token or any(character.isspace() for character in token):
        raise ValueError("GitHub credential has an invalid format")
    return token


class GitHub:
    """Access one fixed repository with a bounded number of requests per run."""

    def __init__(self, token: str, *, request_limit: int = 32) -> None:
        self._token = token
        self._remaining = request_limit

    def request(self, method: str, suffix: str, body: dict | None = None) -> dict | list:
        """Make one request without retrying an ambiguously accepted write."""
        if self._remaining <= 0:
            raise RuntimeError("GitHub request budget exhausted; defer to the next timer")
        if suffix != "/user" and not suffix.startswith(f"/repos/{REPOSITORY}/"):
            raise ValueError("GitHub target is outside the authorized repository")
        self._remaining -= 1
        request = urllib.request.Request(
            "https://api.github.com" + suffix,
            data=None if body is None else json.dumps(body).encode(),
            headers={
                "Authorization": "Bearer " + self._token,
                "Accept": "application/vnd.github+json",
                "Content-Type": "application/json",
                "User-Agent": "tabdata-contract-reporter-v1",
                "X-GitHub-Api-Version": "2022-11-28",
            },
            method=method,
        )
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            retry = error.headers.get("Retry-After", "0")
            raise GitHubError(error.code, int(retry) if retry.isdigit() else 0) from None
        except (urllib.error.URLError, TimeoutError):
            raise RuntimeError(
                "GitHub transport failed; reconcile markers before retrying"
            ) from None

    def comments(self, number: int) -> list[dict]:
        """Read bounded comment pages, including earlier acknowledged result markers."""
        comments = []
        for page in range(1, 6):
            result = self.request(
                "GET", f"/repos/{REPOSITORY}/issues/{number}/comments?per_page=100&page={page}"
            )
            if not isinstance(result, list):
                raise ValueError("GitHub comments response is not a list")
            comments.extend(result)
            if len(result) < 100:
                return comments
        raise RuntimeError("Comment history exceeds the bounded reconciliation budget")

    def post(self, number: int, body: str) -> dict:
        """Publish a marked result/status only to its configured experiment PR."""
        if len(body) > 50000:
            raise ValueError("Report exceeds the bounded GitHub comment size")
        result = self.request(
            "POST", f"/repos/{REPOSITORY}/issues/{number}/comments", {"body": body}
        )
        if not isinstance(result, dict) or not result.get("id") or not result.get("html_url"):
            raise ValueError("GitHub did not return an identifiable published comment")
        return result

    def patch(self, comment_id: int, body: str) -> dict:
        """Update an already reconciled comment owned by the authenticated user."""
        if len(body) > 50000:
            raise ValueError("Report exceeds the bounded GitHub comment size")
        result = self.request(
            "PATCH", f"/repos/{REPOSITORY}/issues/comments/{comment_id}", {"body": body}
        )
        if not isinstance(result, dict) or result.get("id") != comment_id:
            raise ValueError("GitHub comment update identity differs")
        return result
