from __future__ import annotations

import hashlib
import json
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
    _recovery_completion,
    _recovery_progress,
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


def test_recovery_completion_requires_source_admission_and_original_failed_attempt(
    tmp_path: Path,
) -> None:
    runtime = tmp_path / "runtime"
    receipts = runtime / "receipts" / "continuation"
    control = runtime / "control"
    receipts.mkdir(parents=True)
    control.mkdir(parents=True)
    panel_sha = "a" * 64
    identity = {
        "source_binding_sha256": "b" * 64,
        "data_binding_sha256": "c" * 64,
        "split_sha256": "d" * 64,
        "fit_members_sha256": "e" * 64,
        "baseline_ancestry_sha256": "f" * 64,
    }
    recovery_source = "1" * 64
    deadline = 1_800_000_000
    child = {"pid": 123, "pgid": 123, "birth": "123.456"}
    panel_path = runtime / "STUDENT_PANEL_R1.json"
    project_root = runtime / "project"
    python_path = runtime / "venv" / "bin" / "python"
    entrypoint = (
        runtime / "source-r3" / "scripts" / "l4_teacher_capacity_20261104" / "recover_students.py"
    )
    project_root.mkdir(parents=True)
    python_path.parent.mkdir(parents=True)
    python_path.write_text("python", encoding="utf-8")
    entrypoint.parent.mkdir(parents=True)
    entrypoint.write_text("# frozen R3 recovery entrypoint fixture\n", encoding="utf-8")
    panel_path.write_text("{}", encoding="utf-8")
    finalize_log = runtime / "logs" / "stage2" / "wide_gru_stage2_06.log"
    finalize_log.parent.mkdir(parents=True)
    finalize_log.write_text(
        "ValueError: metadata exceeds its 512000-byte read bound\n", encoding="utf-8"
    )

    def write_json(path: Path, value: dict[str, object]) -> str:
        path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
        return hashlib.sha256(path.read_bytes()).hexdigest()

    r1_start = receipts / "STUDENT_CONTINUATION_START_R1.json"
    r1_start_sha = write_json(
        r1_start,
        {
            "schema": "wnn.l4-teacher-capacity-continuation-state.v1",
            "status": "running",
            "panel_sha256": panel_sha,
            **{
                key: identity[key]
                for key in ("source_binding_sha256", "data_binding_sha256", "split_sha256")
            },
        },
    )
    r1_failure = receipts / "STUDENT_CONTINUATION_FAILED_R1.json"
    r1_failure_sha = write_json(
        r1_failure,
        {
            "schema": "wnn.l4-teacher-capacity-continuation-failure.v1",
            "status": "failed",
            "panel_sha256": panel_sha,
            "family": "wide_gru",
            "phase": "stage2_target",
            "error_type": "RuntimeError",
            "error": "wide_gru_stage2_06 stage2 failed (1, timeout=False); preserve receipt",
            "source_binding_sha256": identity["source_binding_sha256"],
            "data_binding_sha256": identity["data_binding_sha256"],
        },
    )
    admission_path = control / "STUDENT_RECOVERY_SOURCE_ADMISSION_R2.json"
    admission = {
        "schema": "wnn.l4-student-recovery-source-admission.v1",
        "status": "approved",
        "source_binding_sha256": recovery_source,
        "continuation_panel_sha256": panel_sha,
        **{
            key: identity[key]
            for key in (
                "data_binding_sha256",
                "split_sha256",
                "fit_members_sha256",
                "baseline_ancestry_sha256",
            )
        },
    }
    admission_sha = write_json(admission_path, admission)
    controller_path = receipts / "STUDENT_CONTINUATION_RECOVERY_R3_CONTROLLER.json"
    controller_command = [
        "taskset",
        "-c",
        ",".join(str(index) for index in range(64)),
        str(python_path.resolve()),
        str(entrypoint.resolve()),
        "--controller-mode",
        "continue-students",
        "--panel",
        str(panel_path.resolve()),
        "--recovery-admission",
        str(admission_path.resolve()),
        "--controller-receipt",
        str(controller_path.resolve()),
    ]
    controller_command_sha = hashlib.sha256(
        json.dumps(controller_command, separators=(",", ":")).encode()
    ).hexdigest()
    write_json(
        controller_path,
        {
            "schema": "wnn.multistage-owned-process.v1",
            "status": "completed",
            "exit_code": 0,
            "timed_out": False,
            "child": child,
            "command": controller_command,
            "command_sha256": controller_command_sha,
            "owner": {
                "continuation_panel_sha256": panel_sha,
                "source_binding_sha256": identity["source_binding_sha256"],
                "recovery_source_binding_sha256": recovery_source,
                "data_binding_sha256": identity["data_binding_sha256"],
                "split_sha256": identity["split_sha256"],
                "fit_members_sha256": identity["fit_members_sha256"],
                "baseline_ancestry_sha256": identity["baseline_ancestry_sha256"],
                "recovery_admission_receipt_path": str(admission_path.resolve()),
                "recovery_admission_receipt_sha256": admission_sha,
                "recovery_entrypoint_path": str(entrypoint.resolve()),
                "recovery_entrypoint_sha256": hashlib.sha256(entrypoint.read_bytes()).hexdigest(),
                "cell_id": "student_continuation_recovery_r3_controller",
                "run_name": "student_continuation_recovery_r3_controller",
                "family": "continuation",
                "command_sha256": controller_command_sha,
            },
            "output_dir": str(project_root.resolve()),
            "gpu_key": "cpu-stage2",
        },
    )
    start = {
        "schema": "wnn.l4-teacher-capacity-continuation-recovery.v1",
        "status": "running",
        "panel_sha256": panel_sha,
        **identity,
        "recovery_source_binding_sha256": recovery_source,
        "recovery_admission_receipt_path": str(admission_path),
        "recovery_admission_receipt_sha256": admission_sha,
        "controller_receipt_path": str(controller_path),
        "controller_child": child,
        "controller_command_sha256": controller_command_sha,
        "r1_start_receipt_path": str(r1_start),
        "r1_start_receipt_sha256": r1_start_sha,
        "r1_failure_receipt_path": str(r1_failure),
        "r1_failure_receipt_sha256": r1_failure_sha,
        "hard_deadline_epoch": deadline,
    }
    start_path = receipts / "STUDENT_CONTINUATION_RECOVERY_R3_START.json"
    write_json(start_path, start)
    write_json(runtime / "GRU_STATE_R1.json", {"hard_deadline_epoch": deadline})
    complete = {
        **start,
        "status": "complete",
        "finished_epoch": deadline - 1,
        "checkpoint_endpoint_epoch": 1,
        "canonical_fixed_weights": {"TL3": 0.9, "candidate": 0.1},
        "all_family_grades_complete": True,
        "families": {"wide_gru": {}, "causal_transformer": {}},
    }
    complete_path = receipts / "STUDENT_CONTINUATION_RECOVERY_R3_COMPLETE.json"
    write_json(complete_path, complete)
    panel = {
        "identity": identity,
        "paths": {
            "python": str(python_path.resolve()),
            "project_root": str(project_root.resolve()),
        },
    }

    assert _recovery_completion(runtime, panel_path, panel, panel_sha) is not None
    status, issue = _recovery_progress(runtime, panel_sha)
    assert status == "R3 completion receipt recorded; canonical grade gate pending"
    assert issue == (
        "R1 stopped at wide_gru / stage2_target: "
        "wide_gru_stage2_06 stage2 failed (1, timeout=False); preserve receipt"
    )

    finalize_log.write_text("different failure detail\n", encoding="utf-8")
    assert _recovery_completion(runtime, panel_path, panel, panel_sha) is None
    finalize_log.write_text(
        "ValueError: metadata exceeds its 512000-byte read bound\n", encoding="utf-8"
    )

    admission["source_binding_sha256"] = "2" * 64
    admission_sha = write_json(admission_path, admission)
    start["recovery_admission_receipt_sha256"] = admission_sha
    write_json(receipts / "STUDENT_CONTINUATION_RECOVERY_R3_START.json", start)
    complete["recovery_admission_receipt_sha256"] = admission_sha
    complete_path.write_text(json.dumps(complete, sort_keys=True), encoding="utf-8")
    assert _recovery_completion(runtime, panel_path, panel, panel_sha) is None
