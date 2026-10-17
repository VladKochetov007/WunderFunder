"""Offline safety tests for the bounded DataSphere profile controller."""

from __future__ import annotations

import fcntl
import json
import os
import subprocess
import tempfile
import unittest
from collections.abc import Sequence
from pathlib import Path

from controller import (
    CANCEL_AFTER_SECONDS,
    EXPECTED_JOB_NAME,
    PROJECT_ID,
    FailClosed,
    ProfileController,
)


class FakeClock:
    """Advance wall time only when the controller sleeps."""

    def __init__(self) -> None:
        self.value = 1000.0

    def now(self) -> float:
        return self.value

    def sleep(self, seconds: float) -> None:
        self.value += seconds


class FakeRunner:
    """Model the DataSphere CLI's JSON files and exact-ID command effects."""

    def __init__(self, statuses: list[str] | None = None) -> None:
        self.statuses = list(statuses or ["SUCCESS"])
        self.calls: list[list[str]] = []
        self.submissions = 0
        self.cancelled = False
        self.observed_status: str | None = None
        self.downloaded_after: str | None = None
        self.fail_submit = False
        self.fail_first_get = False
        self.mismatch_status_identity = False
        self.saw_iam_token = False
        self.saw_legacy_oauth_token = False

    def __call__(
        self,
        command: Sequence[str],
        *,
        cwd: Path,
        env: dict[str, str],
        timeout: int,
    ) -> subprocess.CompletedProcess[str]:
        argv = list(command)
        self.calls.append(argv)
        self.saw_iam_token |= bool(env.get("YC_IAM_TOKEN"))
        self.saw_legacy_oauth_token |= bool(env.get("YC_TOKEN") or env.get("YC_OAUTH_TOKEN"))
        self.assert_no_credentials(argv)
        if "execute" in argv:
            self.submissions += 1
            if self.fail_submit:
                return subprocess.CompletedProcess(argv, 1)
            self.output_json(argv, {"job_id": "job-123", "operation_id": "op-456"})
        elif "get" in argv:
            if self.fail_first_get:
                self.fail_first_get = False
                return subprocess.CompletedProcess(argv, 1)
            status = self.next_status()
            self.observed_status = status
            job_id = "different-job" if self.mismatch_status_identity else "job-123"
            self.output_json(
                argv,
                {"id": job_id, "name": EXPECTED_JOB_NAME, "status": status},
            )
        elif "cancel" in argv:
            self.cancelled = True
        elif "download-files" in argv:
            self.downloaded_after = self.observed_status
        return subprocess.CompletedProcess(argv, 0)

    def next_status(self) -> str:
        if self.cancelled and self.statuses and self.statuses[0] not in {"CANCELLING", "CANCELLED"}:
            return "CANCELLING"
        if self.statuses:
            status = self.statuses.pop(0)
            if status == "CANCELLING":
                self.cancelled = True
            return status
        return "CANCELLED" if self.cancelled else "SUCCESS"

    @staticmethod
    def output_json(command: list[str], value: dict[str, str]) -> None:
        output_path = Path(command[command.index("-o") + 1])
        output_path.write_text(json.dumps(value), encoding="utf-8")

    @staticmethod
    def assert_no_credentials(command: list[str]) -> None:
        serialized = " ".join(command)
        for secret in ("test-iam-secret", "test-oauth-secret"):
            if secret in serialized:
                raise AssertionError("credential leaked to provider command arguments")


class ControllerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.job_dir = self.root / "private-job"
        self.job_dir.mkdir(mode=0o700)
        self.config = self.job_dir / "config.yaml"
        self.config.write_text(
            f"name: {EXPECTED_JOB_NAME}\ncloud-instance-type: g2.1\n",
            encoding="utf-8",
        )
        self.cli = self.root / "datasphere"
        self.cli.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        self.cli.chmod(0o700)
        self.clock = FakeClock()

    def controller(
        self,
        runner: FakeRunner,
        *,
        cancel_after_seconds: int = CANCEL_AFTER_SECONDS,
        poll_seconds: int = 1,
    ) -> ProfileController:
        return ProfileController(
            cli=self.cli,
            profile="offline-test-profile",
            job_dir=self.job_dir,
            config=self.config,
            runner=runner,
            clock=self.clock.now,
            sleep=self.clock.sleep,
            poll_seconds=poll_seconds,
            cancel_after_seconds=cancel_after_seconds,
        )

    def test_terminal_job_downloads_only_after_success(self) -> None:
        runner = FakeRunner(["SUCCESS"])

        self.assertEqual(self.controller(runner).run(), 0)

        self.assertEqual(runner.submissions, 1)
        self.assertEqual(runner.downloaded_after, "SUCCESS")
        state = json.loads((self.job_dir / "controller-state.json").read_text())
        self.assertEqual(state["phase"], "COMPLETE")
        self.assertEqual(state["project_id"], PROJECT_ID)
        self.assertEqual(state["job_id"], "job-123")
        self.assertEqual((self.job_dir / "controller-state.json").stat().st_mode & 0o777, 0o600)

    def test_cancel_is_forced_by_exact_id_at_deadline_then_polls_terminal(self) -> None:
        runner = FakeRunner(
            ["EXECUTING", "EXECUTING", "EXECUTING", "EXECUTING", "CANCELLING", "CANCELLED"]
        )

        self.assertEqual(self.controller(runner, cancel_after_seconds=3).run(), 0)

        cancel = next(call for call in runner.calls if "cancel" in call)
        self.assertEqual(cancel[-2:], ["--id", "job-123"])
        self.assertNotIn("--graceful", cancel)
        self.assertGreaterEqual(self.clock.now(), 1003.0)
        self.assertEqual(runner.downloaded_after, "CANCELLED")

    def test_external_deadline_is_fixed_at_3480_seconds(self) -> None:
        self.assertEqual(CANCEL_AFTER_SECONDS, 3480)

    def test_unknown_submission_never_retries_or_cancels_by_name(self) -> None:
        runner = FakeRunner()
        runner.fail_submit = True
        controller = self.controller(runner)

        with self.assertRaises(FailClosed):
            controller.run()
        with self.assertRaises(FailClosed):
            controller.run()

        state = json.loads((self.job_dir / "controller-state.json").read_text())
        self.assertEqual(state["phase"], "AMBIGUOUS_SUBMISSION")
        self.assertEqual(runner.submissions, 1)
        self.assertFalse(any("cancel" in call for call in runner.calls))

    def test_interrupted_submitting_state_without_id_fails_closed(self) -> None:
        state = {
            "schema_version": 1,
            "phase": "SUBMITTING",
            "project_id": PROJECT_ID,
            "job_name": EXPECTED_JOB_NAME,
            "cancel_at_epoch": 5000.0,
        }
        (self.job_dir / "controller-state.json").write_text(json.dumps(state), encoding="utf-8")
        runner = FakeRunner()

        with self.assertRaises(FailClosed):
            self.controller(runner).run()

        self.assertEqual(runner.submissions, 0)
        self.assertEqual(runner.calls, [])
        persisted = json.loads((self.job_dir / "controller-state.json").read_text())
        self.assertEqual(persisted["phase"], "AMBIGUOUS_SUBMISSION")

    def test_saved_exact_job_id_resumes_without_submitting_again(self) -> None:
        state = {
            "schema_version": 1,
            "phase": "MONITORING",
            "project_id": PROJECT_ID,
            "job_name": EXPECTED_JOB_NAME,
            "job_id": "job-123",
            "operation_id": "op-456",
            "cancel_at_epoch": 5000.0,
        }
        (self.job_dir / "controller-state.json").write_text(json.dumps(state), encoding="utf-8")
        runner = FakeRunner(["SUCCESS"])

        self.assertEqual(self.controller(runner).run(), 0)

        self.assertEqual(runner.submissions, 0)
        self.assertEqual(runner.downloaded_after, "SUCCESS")

    def test_terminal_receipt_restart_retries_download_without_poll_or_submit(self) -> None:
        state = {
            "schema_version": 1,
            "phase": "TERMINAL",
            "project_id": PROJECT_ID,
            "job_name": EXPECTED_JOB_NAME,
            "job_id": "job-123",
            "operation_id": "op-456",
            "cancel_at_epoch": 5000.0,
            "provider_status": "CANCELLED",
        }
        (self.job_dir / "controller-state.json").write_text(json.dumps(state), encoding="utf-8")
        runner = FakeRunner(["SUCCESS"])

        self.assertEqual(self.controller(runner).run(), 0)

        self.assertEqual(runner.submissions, 0)
        self.assertFalse(any("get" in call for call in runner.calls))
        self.assertTrue(any("download-files" in call for call in runner.calls))

    def test_mismatched_provider_identity_is_not_retried_or_cancelled(self) -> None:
        runner = FakeRunner(["EXECUTING"])
        runner.mismatch_status_identity = True

        with self.assertRaises(FailClosed):
            self.controller(runner).run()

        self.assertEqual(runner.submissions, 1)
        self.assertFalse(any("cancel" in call or "download-files" in call for call in runner.calls))

    def test_transient_poll_error_is_retried_without_duplicate_submission(self) -> None:
        runner = FakeRunner(["SUCCESS"])
        runner.fail_first_get = True

        self.assertEqual(self.controller(runner).run(), 0)

        self.assertEqual(runner.submissions, 1)
        self.assertEqual(sum("get" in call for call in runner.calls), 2)
        self.assertEqual(runner.downloaded_after, "SUCCESS")

    def test_existing_lock_prevents_a_second_controller(self) -> None:
        lock_path = self.job_dir / "controller.lock"
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        runner = FakeRunner()
        try:
            with self.assertRaisesRegex(RuntimeError, "already owns"):
                self.controller(runner).run()
        finally:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            os.close(lock_fd)
        self.assertEqual(runner.calls, [])

    def test_cli_environment_does_not_forward_ambient_token_when_profile_selected(self) -> None:
        runner = FakeRunner(["SUCCESS"])
        old_iam = os.environ.get("YC_IAM_TOKEN")
        old_oauth = os.environ.get("YC_OAUTH_TOKEN")
        os.environ["YC_IAM_TOKEN"] = "test-iam-secret"
        os.environ["YC_OAUTH_TOKEN"] = "test-oauth-secret"
        try:
            self.assertEqual(self.controller(runner).run(), 0)
        finally:
            if old_iam is None:
                os.environ.pop("YC_IAM_TOKEN", None)
            else:
                os.environ["YC_IAM_TOKEN"] = old_iam
            if old_oauth is None:
                os.environ.pop("YC_OAUTH_TOKEN", None)
            else:
                os.environ["YC_OAUTH_TOKEN"] = old_oauth

        execute = next(call for call in runner.calls if "execute" in call)
        self.assertFalse(runner.saw_iam_token)
        self.assertFalse(runner.saw_legacy_oauth_token)
        self.assertNotIn("test-iam-secret", execute)
        self.assertNotIn("test-oauth-secret", execute)

    def test_short_lived_iam_token_is_inherited_but_never_added_to_arguments(self) -> None:
        runner = FakeRunner(["SUCCESS"])
        old_iam = os.environ.get("YC_IAM_TOKEN")
        old_token = os.environ.get("YC_TOKEN")
        old_oauth = os.environ.get("YC_OAUTH_TOKEN")
        os.environ["YC_IAM_TOKEN"] = "test-iam-secret"
        os.environ["YC_TOKEN"] = "test-oauth-secret"
        os.environ["YC_OAUTH_TOKEN"] = "test-oauth-secret"
        controller = ProfileController(
            cli=self.cli,
            profile=None,
            job_dir=self.job_dir,
            config=self.config,
            runner=runner,
            clock=self.clock.now,
            sleep=self.clock.sleep,
        )
        try:
            self.assertEqual(controller.run(), 0)
        finally:
            for key, old_value in (
                ("YC_IAM_TOKEN", old_iam),
                ("YC_TOKEN", old_token),
                ("YC_OAUTH_TOKEN", old_oauth),
            ):
                if old_value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = old_value

        self.assertTrue(runner.saw_iam_token)
        self.assertFalse(runner.saw_legacy_oauth_token)
        self.assertFalse(any("test-iam-secret" in " ".join(call) for call in runner.calls))


if __name__ == "__main__":
    unittest.main()
