"""Fixed-origin GitHub operations for the detached B300 snapshot publisher."""

from __future__ import annotations

import base64
import hashlib
import json
import urllib.error
import urllib.parse
import urllib.request
from http.client import HTTPMessage
from typing import BinaryIO

from multistage_github_common import (
    MANIFEST_PATH,
    PARENT_CODE_BRANCH,
    REPOSITORY,
    RESULTS_BRANCH,
    SNAPSHOT_PATH,
    canonical_json,
    is_sha256,
)

API = "https://api.github.com"
LOGIN = "SemyonEpanov"
PULL_REQUEST = 53


class APIError(RuntimeError):
    """A fixed GitHub API request failed without exposing response contents."""

    def __init__(self, status: int, path: str) -> None:
        super().__init__(f"GitHub API returned HTTP {status}: {path}")
        self.status = status


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Refuse redirects so the staged token remains bound to api.github.com."""

    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: BinaryIO,
        code: int,
        msg: str,
        headers: HTTPMessage,
        newurl: str,
    ) -> urllib.request.Request | None:
        _ = (req, fp, msg, headers, newurl)
        raise APIError(code, "redirect refused")


class GitHubAPI:
    """Use one private token only for fixed-origin GitHub REST requests."""

    def __init__(self, token: str) -> None:
        self._token = token
        self._opener = urllib.request.build_opener(_NoRedirect())

    def request(self, method: str, path: str, payload: object = None) -> object:
        """Make one bounded JSON request to the fixed API origin."""
        if not path.startswith("/") or "\r" in path or "\n" in path:
            raise ValueError("GitHub API path is invalid")
        data = None if payload is None else json.dumps(payload, allow_nan=False).encode("utf-8")
        request = urllib.request.Request(
            API + path,
            data=data,
            headers={
                "Authorization": f"Bearer {self._token}",
                "Accept": "application/vnd.github+json",
                "Content-Type": "application/json; charset=utf-8",
                "User-Agent": "WunderFunder-multistage-dashboard-snapshot",
                "X-GitHub-Api-Version": "2022-11-28",
            },
            method=method,
        )
        try:
            with self._opener.open(request, timeout=45) as response:
                raw = response.read(4 * 1024 * 1024 + 1)
        except urllib.error.HTTPError as error:
            raise APIError(error.code, path) from None
        except (urllib.error.URLError, TimeoutError, OSError):
            raise APIError(503, path) from None
        if len(raw) > 4 * 1024 * 1024:
            raise ValueError("GitHub API response exceeded its bounded JSON size")
        if not raw:
            return None
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise ValueError("GitHub API returned invalid JSON") from None


def _api_path(path: str) -> str:
    """Quote a branch or blob path while keeping the API route separators."""
    return urllib.parse.quote(path, safe="/.-")


def remote_snapshot_sha(api: GitHubAPI) -> str | None:
    """Return the committed snapshot identity, if the results branch has one."""
    query = urllib.parse.urlencode({"ref": RESULTS_BRANCH})
    endpoint = f"/repos/{REPOSITORY}/contents/{_api_path(MANIFEST_PATH)}?{query}"
    try:
        record = api.request("GET", endpoint)
    except APIError as error:
        if error.status == 404:
            return None
        raise
    if (
        not isinstance(record, dict)
        or record.get("type") != "file"
        or record.get("path") != MANIFEST_PATH
    ):
        raise ValueError("GitHub returned an unexpected results-branch manifest")
    content = record.get("content")
    size = record.get("size")
    blob_sha = record.get("sha")
    if (
        not isinstance(content, str)
        or type(size) is not int
        or not 0 <= size <= 1_048_576
        or not isinstance(blob_sha, str)
        or len(blob_sha) != 40
    ):
        raise ValueError("results manifest has no small JSON Contents payload")
    raw = base64.b64decode(content, validate=False)
    git_sha = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
    if len(raw) != size or git_sha != blob_sha:
        raise ValueError("results manifest bytes differ from their Git blob identity")
    manifest = json.loads(raw)
    value = manifest.get("snapshot_sha256") if isinstance(manifest, dict) else None
    if not is_sha256(value):
        raise ValueError("results manifest has no valid snapshot identity")
    return value


def results_branch_head(api: GitHubAPI) -> str | None:
    """Return the current dedicated results-branch commit, if it exists."""
    endpoint = f"/repos/{REPOSITORY}/git/ref/heads/{_api_path(RESULTS_BRANCH)}"
    try:
        value = api.request("GET", endpoint)
    except APIError as error:
        if error.status == 404:
            return None
        raise
    branch_object = value.get("object", {}) if isinstance(value, dict) else {}
    sha = branch_object.get("sha") if isinstance(branch_object, dict) else None
    if not isinstance(sha, str) or len(sha) != 40:
        raise ValueError("results branch head is malformed")
    return sha


def verify_publication_context(api: GitHubAPI) -> str:
    """Require the verified account, repository permission, and exact PR head branch."""
    user = api.request("GET", "/user")
    repo = api.request("GET", f"/repos/{REPOSITORY}")
    pull = api.request("GET", f"/repos/{REPOSITORY}/pulls/{PULL_REQUEST}")
    if not isinstance(user, dict) or user.get("login") != LOGIN:
        raise ValueError("GitHub credential does not identify the verified publisher")
    permissions = repo.get("permissions") if isinstance(repo, dict) else None
    if (
        not isinstance(repo, dict)
        or repo.get("full_name", "").lower() != REPOSITORY.lower()
        or not isinstance(permissions, dict)
        or permissions.get("push") is not True
    ):
        raise ValueError("GitHub repository identity or push permission is unavailable")
    head = pull.get("head", {}) if isinstance(pull, dict) else {}
    head_repo = head.get("repo") if isinstance(head, dict) else None
    code_sha = head.get("sha") if isinstance(head, dict) else None
    if (
        not isinstance(head, dict)
        or head.get("ref") != PARENT_CODE_BRANCH
        or not isinstance(head_repo, dict)
        or head_repo.get("full_name", "").lower() != REPOSITORY.lower()
        or not isinstance(code_sha, str)
        or len(code_sha) != 40
    ):
        raise ValueError("PR53 head is not the authorized multistage code branch")
    ref_path = _api_path(PARENT_CODE_BRANCH)
    ref = api.request("GET", f"/repos/{REPOSITORY}/git/ref/heads/{ref_path}")
    code_object = ref.get("object", {}) if isinstance(ref, dict) else {}
    if not isinstance(code_object, dict) or code_object.get("sha") != code_sha:
        raise ValueError("PR53 head and authorized code branch have different commits")
    return code_sha


def _branch_ref(api: GitHubAPI, code_sha: str) -> tuple[str, str]:
    """Create or verify results branch ancestry, returning its current commit and tree."""
    endpoint = f"/repos/{REPOSITORY}/git/ref/heads/{_api_path(RESULTS_BRANCH)}"
    try:
        reference = api.request("GET", endpoint)
    except APIError as error:
        if error.status != 404:
            raise
        reference = api.request(
            "POST",
            f"/repos/{REPOSITORY}/git/refs",
            {"ref": f"refs/heads/{RESULTS_BRANCH}", "sha": code_sha},
        )
    branch_object = reference.get("object", {}) if isinstance(reference, dict) else {}
    commit_sha = branch_object.get("sha") if isinstance(branch_object, dict) else None
    if not isinstance(commit_sha, str):
        raise ValueError("results branch reference is malformed")
    commit = api.request("GET", f"/repos/{REPOSITORY}/git/commits/{commit_sha}")
    tree = commit.get("tree", {}) if isinstance(commit, dict) else {}
    tree_sha = tree.get("sha") if isinstance(tree, dict) else None
    comparison = urllib.parse.quote(f"{code_sha}...{commit_sha}", safe=".")
    relation = api.request("GET", f"/repos/{REPOSITORY}/compare/{comparison}")
    merge_base = relation.get("merge_base_commit", {}) if isinstance(relation, dict) else {}
    if (
        not isinstance(tree_sha, str)
        or not isinstance(merge_base, dict)
        or merge_base.get("sha") != code_sha
    ):
        raise ValueError("results branch is not based on the current authorized code branch")
    return commit_sha, tree_sha


def commit_snapshot(
    api: GitHubAPI, code_sha: str, compressed: bytes, manifest: dict[str, object]
) -> str:
    """Add only the two allowlisted snapshot blobs to the dedicated results branch."""
    parent_sha, parent_tree = _branch_ref(api, code_sha)
    files = {SNAPSHOT_PATH: compressed, MANIFEST_PATH: canonical_json(manifest) + b"\n"}
    tree_entries = []
    for path, contents in files.items():
        if path not in {SNAPSHOT_PATH, MANIFEST_PATH}:
            raise ValueError("snapshot writer path is not allowlisted")
        blob = api.request(
            "POST",
            f"/repos/{REPOSITORY}/git/blobs",
            {"content": base64.b64encode(contents).decode("ascii"), "encoding": "base64"},
        )
        if not isinstance(blob, dict) or not isinstance(blob.get("sha"), str):
            raise ValueError("GitHub did not return a snapshot blob identity")
        tree_entries.append({"path": path, "mode": "100644", "type": "blob", "sha": blob["sha"]})
    tree = api.request(
        "POST", f"/repos/{REPOSITORY}/git/trees", {"base_tree": parent_tree, "tree": tree_entries}
    )
    if not isinstance(tree, dict) or not isinstance(tree.get("sha"), str):
        raise ValueError("GitHub did not return a snapshot tree identity")
    commit = api.request(
        "POST",
        f"/repos/{REPOSITORY}/git/commits",
        {
            "message": "Publish live multistage dashboard snapshot",
            "tree": tree["sha"],
            "parents": [parent_sha],
        },
    )
    if not isinstance(commit, dict) or not isinstance(commit.get("sha"), str):
        raise ValueError("GitHub did not return a snapshot commit identity")
    ref = urllib.parse.quote(RESULTS_BRANCH, safe="/")
    updated = api.request(
        "PATCH",
        f"/repos/{REPOSITORY}/git/refs/heads/{ref}",
        {"sha": commit["sha"], "force": False},
    )
    obj = updated.get("object", {}) if isinstance(updated, dict) else {}
    if not isinstance(obj, dict) or obj.get("sha") != commit["sha"]:
        raise ValueError("GitHub results branch did not advance to the snapshot commit")
    return commit["sha"]


def post_comment(api: GitHubAPI, body: str, marker: str) -> None:
    """Post one idempotent progress comment to the authorized PR."""
    endpoint = f"/repos/{REPOSITORY}/issues/{PULL_REQUEST}/comments"
    comments = api.request("GET", endpoint + "?per_page=100&sort=created&direction=desc")
    if not isinstance(comments, list):
        raise ValueError("GitHub returned a malformed PR comment list")
    if any(isinstance(item, dict) and marker in str(item.get("body", "")) for item in comments):
        return
    result = api.request("POST", endpoint, {"body": f"{body}\n\n<!-- {marker} -->"})
    if not isinstance(result, dict) or not isinstance(result.get("html_url"), str):
        raise ValueError("GitHub did not return a PR comment URL")


def comment_text(commit_sha: str, run_count: int, completed_count: int, suffix: str) -> str:
    """Describe snapshot contents while distinguishing inner proxy from primary reporting."""
    snapshot_url = f"https://github.com/{REPOSITORY}/tree/{RESULTS_BRANCH}"
    commit_url = f"https://github.com/{REPOSITORY}/commit/{commit_sha}"
    return (
        f"Multistage dashboard snapshot {suffix}: {run_count} native dashboard views; "
        f"{completed_count} distinct fits with completed or declared early-stop status. "
        f"[Snapshot branch]({snapshot_url}) | [snapshot commit]({commit_url}).\n\n"
        "Training monitor evaluation is the source-bound inner train-control proxy. "
        "Any official primary-valid report is separate and development-exposed; it is not an Optuna objective."
    )
