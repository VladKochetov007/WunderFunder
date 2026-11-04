from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from scripts import w18_l4_quality_finalize_20261104 as publisher
from scripts.l4_teacher_capacity_20261104.pr61_observer import (
    COMMENT_MARKER,
    HEAD_REF,
    PR_NUMBER,
    PUBLISHER,
    REPOSITORY,
    _credential_identity,
    _phase_receipt,
    _require_shared_identity,
    _verify_api_scope,
)
from scripts.w18_l4_quality_finalize_20261104 import FinalizeError


class _FakeGitHub:
    def __init__(self, login: str = PUBLISHER, ref: str = HEAD_REF) -> None:
        self.login = login
        self.ref = ref
        self.calls: list[tuple[str, str]] = []

    def request(self, method: str, path: str, payload: object = None) -> object:
        self.calls.append((method, path))
        if path == "/user":
            return {"login": self.login}
        if path == f"/repos/{REPOSITORY}/pulls/{PR_NUMBER}":
            return {
                "number": PR_NUMBER,
                "state": "open",
                "head": {
                    "ref": self.ref,
                    "sha": "a" * 40,
                    "repo": {"full_name": REPOSITORY},
                },
            }
        raise AssertionError(f"out-of-scope API call: {method} {path} {payload}")


def test_scope_verifier_binds_account_repo_pr_and_head() -> None:
    client = _FakeGitHub()

    assert _verify_api_scope(client) == "a" * 40
    assert client.calls == [
        ("GET", "/user"),
        ("GET", f"/repos/{REPOSITORY}/pulls/{PR_NUMBER}"),
    ]


@pytest.mark.parametrize("login,ref", [("someone-else", HEAD_REF), (PUBLISHER, "other-branch")])
def test_scope_verifier_rejects_wrong_account_or_branch(login: str, ref: str) -> None:
    with pytest.raises(FinalizeError):
        _verify_api_scope(_FakeGitHub(login, ref))


def test_transformer_receipt_fallback_stays_in_transformer_namespace(tmp_path: Path) -> None:
    panel = {
        "schema": "wnn.l4-teacher-capacity-transformer-panel.v1",
        "paths": {"runtime_root": str(tmp_path)},
    }
    cell = {"cell_id": "tf0_teacher_s1001"}

    assert _phase_receipt(panel, cell, "fit") == (
        tmp_path / "receipts" / "transformer" / "fits" / "tf0_teacher_s1001.json"
    )


def test_family_source_bindings_may_differ_when_data_membership_matches() -> None:
    gru_identity = {
        "source_binding_sha256": "a" * 64,
        "data_binding_sha256": "c" * 64,
        "split_sha256": "d" * 64,
        "fit_members_sha256": "e" * 64,
    }
    tf_identity = {
        "source_binding_sha256": "b" * 64,
        "data_binding_sha256": "c" * 64,
        "split_sha256": "d" * 64,
        "fit_members_sha256": "e" * 64,
    }

    _require_shared_identity(gru_identity, tf_identity, "teacher panels")
    assert gru_identity["source_binding_sha256"] != tf_identity["source_binding_sha256"]


def test_credential_must_be_external_and_inode_bound(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    inside = runtime / "secret.token"
    inside.write_text("not-a-real-token", encoding="utf-8")
    with pytest.raises(ValueError):
        _credential_identity(inside, runtime)

    external = tmp_path / "credential" / "publish.token"
    external.parent.mkdir()
    external.write_text("not-a-real-token", encoding="utf-8")
    identity = _credential_identity(external, runtime)
    assert isinstance(identity[0], int) and isinstance(identity[1], int)


def test_shared_release_helper_keeps_two_argument_upload_client_compatibility(
    tmp_path: Path,
) -> None:
    asset_path = tmp_path / "evidence.zip"
    asset_path.write_bytes(b"evidence")
    digest = hashlib.sha256(asset_path.read_bytes()).hexdigest()
    release_tag = "tabdata-pr61-l4-teacher-capacity-20261104-r1"

    class FakePublisher:
        def __init__(self) -> None:
            self.assets: list[dict[str, object]] = []

        def request(self, method: str, path: str, payload: object = None) -> object:
            if path == "/user":
                return {"login": PUBLISHER}
            if path == f"/repos/{REPOSITORY}":
                return {
                    "full_name": REPOSITORY,
                    "private": True,
                    "permissions": {"push": True},
                }
            if path.endswith(f"/releases/tags/{release_tag}"):
                raise FinalizeError("GitHub GET request returned HTTP 404")
            if method == "POST" and path.endswith("/releases"):
                return {
                    "id": 8,
                    "tag_name": release_tag,
                    "target_commitish": "b" * 40,
                    "draft": True,
                    "author": {"login": PUBLISHER},
                }
            if method == "GET" and path.endswith("/assets?per_page=100"):
                return self.assets
            if method == "PATCH" and path.endswith("/releases/8"):
                return {
                    "id": 8,
                    "tag_name": release_tag,
                    "target_commitish": "b" * 40,
                    "draft": False,
                    "prerelease": True,
                    "html_url": "https://github.com/example/release",
                    "author": {"login": PUBLISHER},
                }
            raise AssertionError(f"unexpected API request: {method} {path} {payload}")

        def upload(self, release_id: int, path: Path) -> dict[str, object]:
            assert release_id == 8
            assert path == asset_path
            asset = {
                "id": 9,
                "name": path.name,
                "state": "uploaded",
                "size": path.stat().st_size,
                "digest": f"sha256:{digest}",
                "browser_download_url": "https://github.com/example/evidence.zip",
            }
            self.assets = [asset]
            return asset

    result = publisher._publish_release(
        FakePublisher(),
        [asset_path],
        "b" * 40,
        "bounded result",
        repository_name=REPOSITORY,
        publisher_login=PUBLISHER,
        release_tag=release_tag,
        release_name="PR61 result",
    )

    assert result["status"] == "published"


def test_pr61_comment_marker_updates_existing_owned_comment() -> None:
    class FakeCommentClient:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str, object]] = []

        def request(self, method: str, path: str, payload: object = None) -> object:
            self.calls.append((method, path, payload))
            if method == "GET":
                return [
                    {
                        "id": 41,
                        "body": f"old status\n\n{COMMENT_MARKER}",
                        "user": {"login": PUBLISHER},
                    }
                ]
            return {
                "html_url": "https://github.com/example/pull/61#issuecomment-41",
                "user": {"login": PUBLISHER},
            }

    client = FakeCommentClient()
    result = publisher._post_comment(
        client,
        "new status",
        repository_name=REPOSITORY,
        pr_number=PR_NUMBER,
        publisher_login=PUBLISHER,
        idempotency_marker=COMMENT_MARKER,
    )

    assert result["status"] == "posted_or_updated"
    assert len(client.calls) == 2
    assert client.calls[1][0] == "PATCH"
    assert client.calls[1][1] == f"/repos/{REPOSITORY}/issues/comments/41"
    assert COMMENT_MARKER in client.calls[1][2]["body"]
