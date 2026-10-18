#!/usr/bin/env python3
"""Submit one sealed F1 pair and verify its terminal DataSphere output."""

from __future__ import annotations

import argparse
import json
import os
import re
import tarfile
import time
from collections.abc import Callable
from pathlib import Path

from controller import (
    CLI_TIMEOUT_SECONDS,
    DOWNLOAD_TIMEOUT_SECONDS,
    POLL_SECONDS,
    PROJECT_ID,
    TERMINAL_STATUSES,
    ControllerError,
    FailClosed,
    ProfileController,
    Runner,
    atomic_write,
    read_single_json,
    run_quietly,
    sha256_file,
    utc_now,
)
from f1_fit_admission import (
    F1_JOB_NAME,
    F1_OUTPUT_PATH,
    F1Admission,
    validate_admission,
)
from f1_fit_verification import (
    ProviderOutputPending,
    validate_provider_file,
    verify_bundle_file,
)

MAX_DOWNLOAD_ATTEMPTS = 3
MAX_MANIFEST_ATTEMPTS = 3
PROVIDER_GRACE_SECONDS = 60


class F1FitController(ProfileController):
    """Run one sealed F1 submission using profile-controller polling and cancellation."""

    def __init__(
        self,
        *,
        sdk_python: Path,
        admission: Path,
        profile_spec: Path,
        fit_spec: Path,
        code_archive: Path,
        supplemental_archive: Path,
        cli: Path,
        profile: str | None,
        job_dir: Path,
        config: Path,
        runner: Runner = run_quietly,
        clock: Callable[[], float] = time.time,
        sleep: Callable[[float], None] = time.sleep,
        poll_seconds: int = POLL_SECONDS,
    ) -> None:
        super().__init__(
            cli=cli,
            profile=profile,
            job_dir=job_dir,
            config=config,
            runner=runner,
            clock=clock,
            sleep=sleep,
            poll_seconds=poll_seconds,
            # Replaced by the sealed finite F1 allocation before any submission.
            cancel_after_seconds=1,
        )
        self.sdk_python = Path(os.path.abspath(sdk_python))
        self.admission_path = Path(os.path.abspath(admission))
        self.profile_spec_path = Path(os.path.abspath(profile_spec))
        self.fit_spec_path = Path(os.path.abspath(fit_spec))
        self.code_archive_path = Path(os.path.abspath(code_archive))
        self.supplemental_archive_path = Path(os.path.abspath(supplemental_archive))
        self.state_path = self.job_dir / "f1-controller-state.json"
        self.execution_path = self.job_dir / "f1-execution.json"
        self.status_path = self.job_dir / "f1-provider-status.json"
        self.download_dir = self.job_dir / "f1-downloads"
        self.manifest_path = self.job_dir / "f1-provider-manifest.json"
        self.admission: F1Admission | None = None

    def run(self) -> int:
        """Validate admission, submit or resume one job, then verify terminal artifacts."""
        self.admission = validate_admission(
            self.admission_path,
            self.config,
            self.profile_spec_path,
            self.fit_spec_path,
            self.code_archive_path,
            self.supplemental_archive_path,
        )
        self.cancel_after_seconds = self.admission.cancel_after_seconds
        if not self.sdk_python.is_file() or not os.access(self.sdk_python, os.X_OK):
            raise FailClosed("DataSphere SDK Python interpreter is missing or not executable")
        self._prepare_private_dir()
        lock_path = self.job_dir / "f1-controller.lock"
        if lock_path.is_symlink():
            raise FailClosed("F1 controller lock path is a symlink")
        descriptor = os.open(
            lock_path, os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0), 0o600
        )
        try:
            import fcntl

            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise ControllerError(
                    "another F1 controller already owns this run directory"
                ) from error
            state = self._load_or_submit_f1()
            if state.get("phase") == "UNVERIFIED_TERMINAL":
                raise FailClosed("terminal bundle could not be verified; inspect the saved receipt")
            if state.get("phase") not in {
                "TERMINAL",
                "DOWNLOAD_ATTEMPTED",
                "VERIFIED",
                "VERIFIED_PARTIAL",
            }:
                state = self._monitor(state)
            return self._download_and_verify(state)
        finally:
            os.close(descriptor)

    def _required_admission(self) -> F1Admission:
        if self.admission is None:
            raise FailClosed("F1 admission was not validated")
        return self.admission

    def _load_or_submit_f1(self) -> dict[str, object]:
        admission = self._required_admission()
        if self.state_path.exists():
            state = read_single_json(self.state_path, "F1 controller state")
            if (
                state.get("project_id") != PROJECT_ID
                or state.get("job_name") != F1_JOB_NAME
                or state.get("admission_sha256") != admission.sha256
                or state.get("config_sha256") != admission.config_sha256
            ):
                raise FailClosed("saved F1 state differs from the sealed admission identity")
            phase = state.get("phase")
            if phase == "AMBIGUOUS_SUBMISSION":
                raise FailClosed(
                    "F1 submission identity is ambiguous; no retry or name-based cancel"
                )
            if phase == "SUBMITTING":
                self._recover_f1_submission(state)
            elif phase not in {
                "MONITORING",
                "PROVIDER_CANCELLING",
                "CANCEL_REQUESTED",
                "TERMINAL",
                "DOWNLOAD_ATTEMPTED",
                "VERIFIED",
                "VERIFIED_PARTIAL",
                "UNVERIFIED_TERMINAL",
            }:
                raise FailClosed("saved F1 controller phase is unknown")
            if phase != "SUBMITTING" and not self._valid_job_id(state.get("job_id")):
                raise FailClosed("saved F1 state lacks one exact provider job ID")
            if not isinstance(state.get("cancel_at_epoch"), (int, float)):
                raise FailClosed("saved F1 state lacks its durable cancel deadline")
            return state
        if self.execution_path.exists():
            raise FailClosed("F1 execute response exists without controller state")
        now = self.clock()
        state: dict[str, object] = {
            "schema_version": 1,
            "phase": "SUBMITTING",
            "project_id": PROJECT_ID,
            "job_name": F1_JOB_NAME,
            "admission_sha256": admission.sha256,
            "config_sha256": admission.config_sha256,
            "submit_started_epoch": now,
            "submit_started_utc": utc_now(),
            "startup_deadline_epoch": now + admission.startup_reserve_seconds,
            "cancel_at_epoch": now + admission.cancel_after_seconds,
            "startup_reserve_seconds": admission.startup_reserve_seconds,
            "internal_deadline_seconds": admission.internal_deadline_seconds,
            "collection_reserve_seconds": admission.collection_reserve_seconds,
            "max_cost_currency": admission.currency,
            "max_cost_minor_units": admission.cost_limit_minor_units,
            "estimated_max_cost_minor_units": admission.estimated_max_cost_minor_units,
        }
        atomic_write(self.state_path, state)
        command = self._base_command() + [
            "project",
            "job",
            "execute",
            "-p",
            PROJECT_ID,
            "-c",
            str(self.config),
            "--async",
            "-o",
            str(self.execution_path),
        ]
        try:
            self._invoke(command, 300)
        except ControllerError as error:
            if not self.execution_path.exists():
                state.update(
                    phase="AMBIGUOUS_SUBMISSION",
                    submission_error=str(error),
                    updated_at_utc=utc_now(),
                )
                atomic_write(self.state_path, state)
                raise FailClosed(
                    "F1 submit outcome is unknown; refusing duplicate submission"
                ) from error
        self._recover_f1_submission(state)
        return state

    @staticmethod
    def _valid_job_id(value: object) -> bool:
        return isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_-]{1,200}", value) is not None

    def _recover_f1_submission(self, state: dict[str, object]) -> None:
        try:
            response = read_single_json(self.execution_path, "F1 execute response")
        except FailClosed as error:
            state.update(
                phase="AMBIGUOUS_SUBMISSION", submission_error=str(error), updated_at_utc=utc_now()
            )
            atomic_write(self.state_path, state)
            raise
        job_id = response.get("job_id")
        if not self._valid_job_id(job_id):
            state.update(
                phase="AMBIGUOUS_SUBMISSION",
                submission_error="execute response has no exact F1 job ID",
                updated_at_utc=utc_now(),
            )
            atomic_write(self.state_path, state)
            raise FailClosed("F1 execute response has no exact provider job ID")
        operation_id = response.get("operation_id")
        if not self._valid_job_id(operation_id):
            operation_id = None
        state.update(
            phase="MONITORING",
            job_id=job_id,
            operation_id=operation_id,
            submitted_at_utc=utc_now(),
            updated_at_utc=utc_now(),
        )
        atomic_write(self.state_path, state)

    def _read_provider_status(self, state: dict[str, object]) -> str:
        command = self._base_command() + [
            "project",
            "job",
            "get",
            "--id",
            str(state["job_id"]),
            "--format",
            "json",
            "-o",
            str(self.status_path),
        ]
        self._invoke(command, CLI_TIMEOUT_SECONDS)
        response = read_single_json(self.status_path, "F1 provider status")
        if response.get("id") != state["job_id"] or response.get("name") != F1_JOB_NAME:
            raise FailClosed("provider status does not match the exact F1 job identity")
        status = response.get("status")
        if not isinstance(status, str) or status not in {
            "CREATING",
            "PREPARING",
            "EXECUTING",
            "UPLOADING_OUTPUT",
            "CANCELLING",
            *TERMINAL_STATUSES,
        }:
            raise FailClosed("provider returned an unknown F1 status")
        if status == "UPLOADING_OUTPUT" and not state.get("fit_started_seen"):
            # This may be the first observation after execution already finished.
            # Do not mistake delayed polling for a startup breach or cancel output upload.
            state.update(
                fit_started_seen=True,
                fit_started_unobserved=True,
                first_uploading_output_at_utc=utc_now(),
            )
        if status == "EXECUTING" and not state.get("fit_started_seen"):
            now = self.clock()
            submit_started = float(state["submit_started_epoch"])
            elapsed = max(0.0, now - submit_started)
            allocated = (
                self._required_admission().internal_deadline_seconds
                + self._required_admission().collection_reserve_seconds
                + PROVIDER_GRACE_SECONDS
            )
            state.update(
                fit_started_seen=True,
                first_executing_at_epoch=now,
                first_executing_at_utc=utc_now(),
                submit_to_execution_seconds=round(elapsed, 3),
                fit_and_collection_deadline_epoch=(
                    now
                    + self._required_admission().internal_deadline_seconds
                    + self._required_admission().collection_reserve_seconds
                ),
            )
            if (
                elapsed > self._required_admission().startup_reserve_seconds
                or elapsed + allocated > self._required_admission().cancel_after_seconds
            ):
                state.update(
                    startup_reserve_breached=True,
                    cancel_reason="observed submit-to-execution delay leaves insufficient fit/collection time",
                )
        return status

    def _cancel_due(self, state: dict[str, object]) -> bool:
        startup_breached = bool(state.get("startup_reserve_breached")) or (
            not state.get("fit_started_seen")
            and self.clock() >= float(state["startup_deadline_epoch"])
        )
        if startup_breached:
            state["startup_reserve_breached"] = True
            state.setdefault("cancel_reason", "submit-to-execution reserve expired")
            last_attempt = state.get("last_cancel_attempt_epoch")
            return last_attempt is None or self.clock() - float(last_attempt) >= 60
        return super()._cancel_due(state)

    def _read_manifest(self, state: dict[str, object], attempt: int) -> dict[str, object]:
        if self.manifest_path.is_symlink():
            raise FailClosed("provider manifest path is a symlink")
        self.manifest_path.unlink(missing_ok=True)
        probe = Path(__file__).with_name("f1_provider_manifest.py")
        log_dir = self.job_dir / "provider-logs"
        command = [
            str(self.sdk_python),
            str(probe),
            "--job-id",
            str(state["job_id"]),
            "--project-id",
            PROJECT_ID,
            "--job-name",
            F1_JOB_NAME,
            "--output",
            str(self.manifest_path),
            "--log-dir",
            str(log_dir),
        ]
        if self.profile is not None:
            command.extend(["--profile", self.profile])
        self._invoke(command, CLI_TIMEOUT_SECONDS)
        return read_single_json(self.manifest_path, f"F1 provider manifest attempt {attempt}")

    def _validated_provider_file(
        self, state: dict[str, object], provider: dict[str, object]
    ) -> dict[str, object]:
        return validate_provider_file(state, provider, self._required_admission())

    def _download_and_verify(self, state: dict[str, object]) -> int:
        if state.get("phase") == "UNVERIFIED_TERMINAL":
            raise FailClosed("terminal F1 output remains unverified")
        if state.get("phase") not in {
            "TERMINAL",
            "DOWNLOAD_ATTEMPTED",
            "VERIFIED",
            "VERIFIED_PARTIAL",
        }:
            raise FailClosed("F1 bundle download was requested before terminal provider status")
        provider_file = None
        pending_attempts = int(state.get("pending_output_manifest_attempts", 0))
        while pending_attempts < MAX_MANIFEST_ATTEMPTS:
            manifest_attempts = int(state.get("manifest_query_attempts", 0)) + 1
            state["manifest_query_attempts"] = manifest_attempts
            atomic_write(self.state_path, state)
            try:
                provider = self._read_manifest(state, manifest_attempts)
            except ControllerError as error:
                failures = int(state.get("manifest_query_failures", 0)) + 1
                state["manifest_query_failures"] = failures
                state.update(last_manifest_error=str(error), updated_at_utc=utc_now())
                if failures >= MAX_MANIFEST_ATTEMPTS:
                    state.update(
                        phase="UNVERIFIED_TERMINAL",
                        verification_error="provider manifest query limit reached",
                    )
                    atomic_write(self.state_path, state)
                    raise FailClosed(
                        "provider manifest could not be reconciled after bounded retries"
                    ) from error
                atomic_write(self.state_path, state)
                raise
            state["manifest_query_failures"] = 0
            try:
                provider_file = self._validated_provider_file(state, provider)
            except ProviderOutputPending as error:
                pending_attempts += 1
                state.update(
                    pending_output_manifest_attempts=pending_attempts,
                    last_manifest_pending_error=str(error),
                    last_manifest_pending_at_utc=utc_now(),
                    updated_at_utc=utc_now(),
                )
                if pending_attempts >= MAX_MANIFEST_ATTEMPTS:
                    state.update(
                        phase="UNVERIFIED_TERMINAL",
                        verification_error="terminal output remained absent after bounded retries",
                    )
                    atomic_write(self.state_path, state)
                    raise FailClosed(
                        "terminal provider manifest did not list the expected output after bounded retries"
                    ) from error
                atomic_write(self.state_path, state)
                self.sleep(10 * pending_attempts)
                continue
            except FailClosed as error:
                state.update(
                    phase="UNVERIFIED_TERMINAL",
                    verification_error=f"{type(error).__name__}: {error}",
                    updated_at_utc=utc_now(),
                )
                atomic_write(self.state_path, state)
                raise
            state["pending_output_manifest_attempts"] = pending_attempts
            break
        if provider_file is None:
            raise FailClosed("terminal provider output manifest could not be reconciled")
        prior_provider_file = state.get("provider_output_file")
        state["provider_output_file"] = provider_file
        state["provider_manifest_sha256"] = sha256_file(self.manifest_path)

        if state.get("phase") in {"VERIFIED", "VERIFIED_PARTIAL"}:
            prior_digest = state.get("verified_output_sha256")
            prior_size = state.get("verified_output_bytes")
            if (
                not isinstance(prior_provider_file, dict)
                or prior_provider_file != provider_file
                or prior_digest != provider_file.get("sha256")
                or prior_size != provider_file.get("size_bytes")
            ):
                state.update(
                    phase="UNVERIFIED_TERMINAL",
                    verification_error="provider output manifest changed after local verification",
                    updated_at_utc=utc_now(),
                )
                atomic_write(self.state_path, state)
                raise FailClosed("provider output manifest changed after local verification")
            verified_attempt = state.get("download_attempts")
            if (
                type(verified_attempt) is not int
                or not 1 <= verified_attempt <= MAX_DOWNLOAD_ATTEMPTS
            ):
                raise FailClosed("verified F1 state has no valid download attempt identity")
            local_path = self.job_dir / f"f1-download-{verified_attempt:02d}" / F1_OUTPUT_PATH
            if state.get("verified_local_path") != str(local_path):
                raise FailClosed("verified F1 state local path differs from its attempt identity")
            try:
                return self._verify_local_bundle(local_path, state, provider_file)
            except (OSError, ValueError, FailClosed) as error:
                if isinstance(error, FailClosed):
                    state.update(
                        phase="UNVERIFIED_TERMINAL",
                        verification_error=f"{type(error).__name__}: {error}",
                        updated_at_utc=utc_now(),
                    )
                    atomic_write(self.state_path, state)
                    raise
                state.update(
                    phase="DOWNLOAD_ATTEMPTED",
                    last_verification_error=f"{type(error).__name__}: {error}",
                    updated_at_utc=utc_now(),
                )
                atomic_write(self.state_path, state)

        attempts = int(state.get("download_attempts", 0))
        if attempts >= MAX_DOWNLOAD_ATTEMPTS:
            state.update(
                phase="UNVERIFIED_TERMINAL", verification_error="download attempt limit reached"
            )
            atomic_write(self.state_path, state)
            raise FailClosed("F1 output could not be verified after bounded download attempts")
        attempts += 1
        attempt_dir = self.job_dir / f"f1-download-{attempts:02d}"
        if attempt_dir.exists() and (attempt_dir.is_symlink() or not attempt_dir.is_dir()):
            raise FailClosed("F1 download attempt path is not a real directory")
        attempt_dir.mkdir(mode=0o700, exist_ok=True)
        command = self._base_command() + [
            "project",
            "job",
            "download-files",
            "--id",
            str(state["job_id"]),
            "--output-dir",
            str(attempt_dir),
        ]
        state.update(
            phase="DOWNLOAD_ATTEMPTED",
            download_attempts=attempts,
            download_attempt_dir=str(attempt_dir),
            download_attempt_at_utc=utc_now(),
        )
        atomic_write(self.state_path, state)
        try:
            self._invoke(command, DOWNLOAD_TIMEOUT_SECONDS)
        except ControllerError as error:
            state.update(download_error=str(error), download_error_at_utc=utc_now())
            if attempts >= MAX_DOWNLOAD_ATTEMPTS:
                state.update(
                    phase="UNVERIFIED_TERMINAL",
                    verification_error="download command retry limit reached",
                )
                atomic_write(self.state_path, state)
                raise FailClosed("F1 output download failed after bounded retries") from error
            atomic_write(self.state_path, state)
            raise
        local_path = attempt_dir / F1_OUTPUT_PATH
        try:
            return self._verify_local_bundle(local_path, state, provider_file)
        except (OSError, ValueError, tarfile.TarError, json.JSONDecodeError, FailClosed) as error:
            state.update(
                last_verification_error=f"{type(error).__name__}: {error}", updated_at_utc=utc_now()
            )
            if attempts >= MAX_DOWNLOAD_ATTEMPTS:
                state.update(
                    phase="UNVERIFIED_TERMINAL",
                    verification_error="download verification retry limit reached",
                )
                atomic_write(self.state_path, state)
                raise FailClosed(
                    "F1 output failed bounded provider or bundle verification"
                ) from error
            atomic_write(self.state_path, state)
            raise ControllerError(
                "F1 download was attempted but the bundle is not verified; retry is bounded"
            ) from error

    def _verify_local_bundle(
        self, path: Path, state: dict[str, object], provider_file: dict[str, object]
    ) -> int:
        complete, bundle_manifest = verify_bundle_file(
            path, provider_file, self._required_admission()
        )
        phase = (
            "VERIFIED"
            if complete and state.get("provider_status") == "SUCCESS"
            else "VERIFIED_PARTIAL"
        )
        state.update(
            phase=phase,
            verified_at_utc=utc_now(),
            verified_local_path=str(path),
            verified_output_bytes=path.stat().st_size,
            verified_output_sha256=sha256_file(path),
            embedded_manifest_file_count=bundle_manifest["file_count"],
            embedded_manifest_total_bytes=bundle_manifest["total_bytes"],
            fit_completion_status="complete" if complete else "partial_or_failed",
        )
        atomic_write(self.state_path, state)
        return 0


def parse_args() -> argparse.Namespace:
    """Parse explicit packet, interpreter, and private run-state paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cli", required=True, type=Path)
    parser.add_argument("--sdk-python", required=True, type=Path)
    parser.add_argument("--profile")
    parser.add_argument("--job-dir", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--admission", required=True, type=Path)
    parser.add_argument("--profile-spec", required=True, type=Path)
    parser.add_argument("--fit-spec", required=True, type=Path)
    parser.add_argument("--code-archive", required=True, type=Path)
    parser.add_argument("--supplemental-archive", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    """Run the one F1 controller without printing credentials or provider payloads."""
    args = parse_args()
    try:
        return F1FitController(
            cli=args.cli,
            sdk_python=args.sdk_python,
            profile=args.profile,
            job_dir=args.job_dir,
            config=args.config,
            admission=args.admission,
            profile_spec=args.profile_spec,
            fit_spec=args.fit_spec,
            code_archive=args.code_archive,
            supplemental_archive=args.supplemental_archive,
            runner=run_quietly,
            clock=time.time,
            sleep=time.sleep,
            poll_seconds=POLL_SECONDS,
        ).run()
    except FailClosed as error:
        print(f"FAIL_CLOSED: {error}")
        return 78
    except ControllerError as error:
        print(f"CONTROLLER_ERROR: {error}")
        return 75


if __name__ == "__main__":
    raise SystemExit(main())
