from __future__ import annotations

import json
import os
import stat
import time
import urllib.error
import urllib.request
from pathlib import Path

if __package__:
    from scripts.heterogeneous_results_report import REPO, atomic_json, sha256
else:
    from heterogeneous_results_report import REPO, atomic_json, sha256


class PublicationError(RuntimeError):
    """One sanitized GitHub failure; publication is blocked without automatic retries."""


class NoRedirect(urllib.request.HTTPRedirectHandler):
    """Never forward the private token to a redirected host."""

    def redirect_request(
        self, req: object, fp: object, code: int, msg: str, headers: object, newurl: str
    ) -> None:
        return None


class GitHub:
    """Make one request per action to the authorized account and fixed PR52 only."""

    def __init__(self, path: Path) -> None:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
        try:
            item = os.fstat(descriptor)
            if (
                not stat.S_ISREG(item.st_mode)
                or item.st_uid != os.getuid()
                or stat.S_IMODE(item.st_mode) != 0o600
                or not 0 < item.st_size <= 16384
            ):
                raise PublicationError("token file must be bounded, owned and mode0600")
            self.token = os.read(descriptor, 16385).decode().strip()
            if not self.token or any(char.isspace() for char in self.token):
                raise PublicationError("token file must contain one token")
        finally:
            os.close(descriptor)
        self.opener = urllib.request.build_opener(NoRedirect())
        if self.request("GET", "/user").get("login") != "SemyonEpanov":
            raise PublicationError("GitHub token does not identify SemyonEpanov")

    def request(self, method: str, path: str, body: str | None = None) -> dict:
        if (method, path) not in {("GET", "/user"), ("POST", f"/repos/{REPO}/issues/52/comments")}:
            raise PublicationError("GitHub action is outside the fixed PR52 authorization")
        data = None if body is None else json.dumps({"body": body}).encode()
        request = urllib.request.Request(
            "https://api.github.com" + path,
            data=data,
            method=method,
            headers={
                "Authorization": "Bearer " + self.token,
                "Accept": "application/vnd.github+json",
                "Content-Type": "application/json",
                "X-GitHub-Api-Version": "2022-11-28",
                "User-Agent": "tabdata-PR52-results",
            },
        )
        try:
            with self.opener.open(request, timeout=20) as response:
                value = json.loads(response.read(1 << 20))
        except urllib.error.HTTPError as error:
            raise PublicationError(
                f"GitHub {method} failed with HTTP{error.code}; not retried"
            ) from None
        except (
            urllib.error.URLError,
            TimeoutError,
            OSError,
            json.JSONDecodeError,
            UnicodeDecodeError,
        ):
            raise PublicationError("GitHub network/response failure; not retried") from None
        if not isinstance(value, dict):
            raise PublicationError("GitHub response was not an object")
        return value

    def post(self, body: str) -> str:
        result = self.request("POST", f"/repos/{REPO}/issues/52/comments", body)
        url = result.get("html_url")
        if (
            result.get("user", {}).get("login") != "SemyonEpanov"
            or not isinstance(url, str)
            or not url.startswith(f"https://github.com/{REPO}/pull/52#issuecomment-")
        ):
            raise PublicationError("GitHub returned an unexpected comment author or URL")
        return url


def publish(client: GitHub | None, state: dict, state_path: Path, event: str, body: str) -> bool:
    """Journal an exact comment before POST, preventing duplicate uncertain submissions."""
    if event in state["posts"]:
        return True
    if client is None or state.get("publication_blocked") or state["posts_attempted"] >= 24:
        return False
    if state.get("pending_post"):
        state["publication_blocked"] = {"kind": "unconfirmed_prior_POST", "epoch": time.time()}
        atomic_json(state_path, state)
        return False
    directory = state_path.parent / "results-publisher-comments"
    directory.mkdir(exist_ok=True)
    path = directory / f"{event}.md"
    if path.exists():
        if path.read_text(encoding="utf-8") != body:
            state["publication_blocked"] = {
                "kind": "prepared_comment_body_changed",
                "epoch": time.time(),
            }
            atomic_json(state_path, state)
            return False
    else:
        with path.open("x", encoding="utf-8") as stream:
            stream.write(body)
    state["pending_post"] = {
        "event": event,
        "body_sha256": sha256(path),
        "body_file": str(path),
        "epoch": time.time(),
    }
    state["posts_attempted"] += 1
    atomic_json(state_path, state)
    try:
        url = client.post(body)
    except PublicationError as error:
        state["publication_blocked"] = {
            "kind": "GitHub_POST_failure",
            "message": str(error),
            "epoch": time.time(),
        }
        atomic_json(state_path, state)
        return False
    state["posts"][event] = {"url": url, **state.pop("pending_post")}
    state["last_post_epoch"] = time.time()
    atomic_json(state_path, state)
    return True
