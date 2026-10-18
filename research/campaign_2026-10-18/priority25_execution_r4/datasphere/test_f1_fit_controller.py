"""CPU-only checks for the exact F1 DataSphere admission and output verifier."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from controller import PROJECT_ID, ControllerError, FailClosed
from f1_fit_admission import (
    F1_JOB_NAME,
    F1_OUTPUT_PATH,
    F1_OUTPUT_VARIABLE,
    HOST_CONTROLLER_FILES,
    UPLOADED_INPUT_FILES,
    F1Admission,
    _validate_cost_budget,
    _validate_hash_map,
    _validate_runtime_budget,
)
from f1_fit_bundle import write_bundle
from f1_fit_controller import F1FitController
from f1_fit_recipe import ARMS, FOLD1_SHA256, SOURCE_COMMIT
from f1_fit_verification import verify_fit_receipt

HASHES = {
    "profile": "1" * 64,
    "fit": "2" * 64,
    "code": "3" * 64,
    "supplemental": "4" * 64,
}


def admission() -> F1Admission:
    """Return a synthetic, finite allocation for isolated verification tests."""
    return F1Admission(
        sha256="a" * 64,
        config_sha256="b" * 64,
        profile_spec_sha256=HASHES["profile"],
        fit_spec_sha256=HASHES["fit"],
        code_archive_sha256=HASHES["code"],
        supplemental_archive_sha256=HASHES["supplemental"],
        startup_reserve_seconds=120,
        internal_deadline_seconds=3000,
        collection_reserve_seconds=300,
        cancel_after_seconds=3480,
        external_deadline_seconds=3600,
        currency="RUB",
        cost_limit_minor_units=50000,
        estimated_max_cost_minor_units=42000,
        output_max_bytes=768 << 20,
    )


def _file_record(path: Path) -> dict[str, object]:
    payload = path.read_bytes()
    return {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def make_bundle(bundle_path: Path, *, inputs: dict[str, object] | None = None) -> dict[str, object]:
    """Write a producer-shaped pair bundle with independently hashed manifest entries."""
    output = bundle_path.parent / "fit-results"
    output.mkdir(parents=True)
    sample = admission()
    receipt_inputs = (
        inputs
        if inputs is not None
        else {
            "profile_spec_sha256": sample.profile_spec_sha256,
            "fit_spec_sha256": sample.fit_spec_sha256,
            "code_archive_sha256": sample.code_archive_sha256,
            "supplemental_archive_sha256": sample.supplemental_archive_sha256,
            "source_commit": SOURCE_COMMIT,
            "fold_manifest_sha256": FOLD1_SHA256,
            "internal_deadline_seconds": sample.internal_deadline_seconds,
            "collection_reserve_seconds": sample.collection_reserve_seconds,
            "external_cancel_deadline_seconds_from_submit": sample.external_deadline_seconds,
        }
    )
    receipt: dict[str, object] = {
        "status": "PAIR_FIT_AND_EXPORT_COMPLETE",
        "quality_result": None,
        "inputs": receipt_inputs,
        "split_sha256": FOLD1_SHA256,
        "collection_status": "COMPLETE",
        "collection_errors": [],
        "arms": {},
    }
    arms: dict[str, object] = {}
    for arm in ARMS:
        name = str(arm["name"])
        artifacts: dict[str, object] = {}
        for artifact in (
            "model_ep5.pt",
            "model.pt",
            "training_state.pt",
            "history.json",
            "provenance.json",
        ):
            path = output / "artifacts/runs" / name / artifact
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"fixture:{name}:{artifact}".encode())
            artifacts[artifact] = _file_record(path)
        training_log = output / f"{name}.train.log"
        training_log.write_text("train log\n", encoding="utf-8")
        prediction = output / "cache" / f"pred_{name}_valid.npy"
        prediction.parent.mkdir(parents=True, exist_ok=True)
        prediction.write_bytes(f"prediction:{name}".encode())
        sidecar = prediction.with_suffix(".json")
        sidecar.write_text("{}\n", encoding="utf-8")
        prediction_log = output / f"{name}.predict.log"
        prediction_log.write_text("prediction log\n", encoding="utf-8")
        arms[name] = {
            "status": "COMPLETE",
            "fit_artifacts": artifacts,
            "training_log": training_log.name,
            "prediction_export": {
                "prediction": _file_record(prediction),
                "sidecar": _file_record(sidecar),
                "log": prediction_log.name,
            },
        }
    receipt["arms"] = arms
    (output / "fit-pair-receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
    write_bundle(bundle_path, output)
    return json.loads((output / "fit-artifact-manifest.json").read_text(encoding="utf-8"))


class FakeClock:
    """Controllable wall clock for provider deadline checks."""

    def __init__(self, value: float) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value


class F1AdmissionTests(unittest.TestCase):
    def test_template_hash_maps_cover_every_uploaded_and_host_module(self) -> None:
        template_path = Path(__file__).with_name("f1-fit-admission.template.json")
        template = json.loads(template_path.read_text(encoding="utf-8"))
        self.assertEqual(set(template["uploaded_input_sha256"]), set(UPLOADED_INPUT_FILES))
        self.assertEqual(set(template["host_controller_sha256"]), set(HOST_CONTROLLER_FILES))

    def test_runtime_includes_startup_reserve_and_accepts_explicit_multihour_fit(self) -> None:
        runtime = {
            "submit_to_execution_reserve_seconds": 600,
            "internal_deadline_seconds": 14400,
            "collection_reserve_seconds": 600,
            "cancel_after_seconds_from_submit": 15660,
            "external_deadline_seconds_from_submit": 15720,
        }
        self.assertEqual(
            _validate_runtime_budget(runtime, (14400, 600, 15720)),
            (600, 14400, 600, 15660, 15720),
        )
        runtime["submit_to_execution_reserve_seconds"] = 601
        with self.assertRaisesRegex(FailClosed, "startup, fit, collection, and provider grace"):
            _validate_runtime_budget(runtime, (14400, 600, 15720))

    def test_cost_gate_requires_estimate_beneath_finite_limit_and_evidence(self) -> None:
        cost = {
            "currency": "RUB",
            "max_cost_minor_units": 50000,
            "estimated_max_cost_minor_units": 42000,
            "balance_and_rate_evidence_sha256": "5" * 64,
        }
        self.assertEqual(_validate_cost_budget(cost), ("RUB", 50000, 42000))
        cost["estimated_max_cost_minor_units"] = 50001
        with self.assertRaisesRegex(FailClosed, "exceeds the explicit cost allocation"):
            _validate_cost_budget(cost)

    def test_all_uploaded_and_host_files_must_match_the_exact_admission_map(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "entry.py").write_text("pass\n", encoding="utf-8")
            (root / "helper.py").write_text("pass\n", encoding="utf-8")
            digests = {
                filename: _sha256_file(root / filename) for filename in ("entry.py", "helper.py")
            }
            _validate_hash_map(root, digests, ("entry.py", "helper.py"), "test input")
            digests.pop("helper.py")
            with self.assertRaisesRegex(FailClosed, "missing, extra, or malformed"):
                _validate_hash_map(root, digests, ("entry.py", "helper.py"), "test input")


class F1StartupDeadlineTests(unittest.TestCase):
    def _controller(self, root: Path, clock: FakeClock, status: str) -> F1FitController:
        def runner(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
            output_index = command.index("-o") + 1
            Path(command[output_index]).write_text(
                json.dumps({"id": "exact-job-id", "name": F1_JOB_NAME, "status": status}),
                encoding="utf-8",
            )
            return subprocess.CompletedProcess(command, 0, "", "")

        controller = F1FitController(
            sdk_python=Path("/bin/true"),
            admission=root / "unused-admission.json",
            profile_spec=root / "unused-profile.json",
            fit_spec=root / "unused-fit.json",
            code_archive=root / "unused-code.tar.gz",
            supplemental_archive=root / "unused-inputs.tar.gz",
            cli=Path("/bin/true"),
            profile=None,
            job_dir=root / "controller",
            config=root / "unused-config.yaml",
            runner=runner,
            clock=clock,
            sleep=lambda _seconds: None,
            poll_seconds=1,
        )
        controller.job_dir.mkdir()
        controller.admission = admission()
        return controller

    def test_observed_start_delay_can_breach_reserve_before_fixed_cancel_time(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            clock = FakeClock(225)
            controller = self._controller(root, clock, "EXECUTING")
            state: dict[str, object] = {
                "job_id": "exact-job-id",
                "submit_started_epoch": 100,
                "startup_deadline_epoch": 220,
                "cancel_at_epoch": 3580,
            }
            self.assertEqual(controller._read_provider_status(state), "EXECUTING")
            self.assertEqual(state["submit_to_execution_seconds"], 125)
            self.assertTrue(state["startup_reserve_breached"])
            self.assertTrue(controller._cancel_due(state))

    def test_tight_but_admitted_start_delay_does_not_cancel_early(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            clock = FakeClock(200)
            controller = self._controller(root, clock, "EXECUTING")
            state: dict[str, object] = {
                "job_id": "exact-job-id",
                "submit_started_epoch": 100,
                "startup_deadline_epoch": 220,
                "cancel_at_epoch": 3580,
            }
            self.assertEqual(controller._read_provider_status(state), "EXECUTING")
            self.assertNotIn("startup_reserve_breached", state)
            self.assertFalse(controller._cancel_due(state))

    def test_no_execution_at_startup_deadline_cancels_exact_job(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            controller = self._controller(Path(temporary), FakeClock(220), "PREPARING")
            state = {
                "job_id": "exact-job-id",
                "submit_started_epoch": 100,
                "startup_deadline_epoch": 220,
                "cancel_at_epoch": 3580,
            }
            self.assertTrue(controller._cancel_due(state))
            self.assertTrue(state["startup_reserve_breached"])

    def test_first_uploading_poll_does_not_mistake_unobserved_start_for_breach(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            clock = FakeClock(1000)
            controller = self._controller(Path(temporary), clock, "UPLOADING_OUTPUT")
            state: dict[str, object] = {
                "job_id": "exact-job-id",
                "submit_started_epoch": 100,
                "startup_deadline_epoch": 220,
                "cancel_at_epoch": 3580,
            }
            self.assertEqual(controller._read_provider_status(state), "UPLOADING_OUTPUT")
            self.assertTrue(state["fit_started_seen"])
            self.assertTrue(state["fit_started_unobserved"])
            self.assertNotIn("startup_reserve_breached", state)
            self.assertFalse(controller._cancel_due(state))


class F1OutputVerificationTests(unittest.TestCase):
    def test_complete_bundle_receipt_is_bound_to_source_and_all_manifest_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / F1_OUTPUT_PATH
            manifest = make_bundle(archive)
            self.assertTrue(verify_fit_receipt(archive, manifest, admission()))

    def test_missing_receipt_ancestry_is_not_treated_as_partial_or_verified(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / F1_OUTPUT_PATH
            manifest = make_bundle(archive, inputs={})
            with self.assertRaisesRegex(FailClosed, "ancestry"):
                verify_fit_receipt(archive, manifest, admission())

    def test_download_attempt_is_not_verified_until_provider_hash_and_bundle_pass(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_bundle = root / "source" / F1_OUTPUT_PATH
            source_bundle.parent.mkdir()
            make_bundle(source_bundle)
            expected = {
                "path": F1_OUTPUT_PATH,
                "variable": F1_OUTPUT_VARIABLE,
                "size_bytes": source_bundle.stat().st_size,
                "sha256": _sha256_file(source_bundle),
            }
            controller = F1FitController(
                sdk_python=Path("/bin/true"),
                admission=root / "unused-admission.json",
                profile_spec=root / "unused-profile.json",
                fit_spec=root / "unused-fit.json",
                code_archive=root / "unused-code.tar.gz",
                supplemental_archive=root / "unused-inputs.tar.gz",
                cli=Path("/bin/true"),
                profile=None,
                job_dir=root / "controller",
                config=root / "unused-config.yaml",
            )
            controller.job_dir.mkdir()
            controller.admission = admission()
            provider = {
                "id": "exact-job-id",
                "project_id": PROJECT_ID,
                "name": F1_JOB_NAME,
                "status": "SUCCESS",
                "output_files": [expected],
                "output_file_errors": [],
            }

            def read_manifest(state: dict[str, object], attempt: int) -> dict[str, object]:
                controller.manifest_path.write_text(json.dumps(provider), encoding="utf-8")
                return provider

            def runner(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
                destination = Path(command[command.index("--output-dir") + 1])
                shutil.copyfile(source_bundle, destination / F1_OUTPUT_PATH)
                return subprocess.CompletedProcess(command, 0, "", "")

            controller.runner = runner
            state: dict[str, object] = {
                "phase": "TERMINAL",
                "provider_status": "SUCCESS",
                "job_id": "exact-job-id",
            }
            with patch.object(controller, "_read_manifest", side_effect=read_manifest):
                self.assertEqual(controller._download_and_verify(state), 0)
            self.assertEqual(state["phase"], "VERIFIED")
            self.assertEqual(state["verified_output_sha256"], expected["sha256"])

    def test_empty_terminal_manifest_gets_bounded_retry_before_download(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_bundle = root / "source" / F1_OUTPUT_PATH
            source_bundle.parent.mkdir()
            make_bundle(source_bundle)
            expected = {
                "path": F1_OUTPUT_PATH,
                "variable": F1_OUTPUT_VARIABLE,
                "size_bytes": source_bundle.stat().st_size,
                "sha256": _sha256_file(source_bundle),
            }
            controller = F1FitController(
                sdk_python=Path("/bin/true"),
                admission=root / "unused-admission.json",
                profile_spec=root / "unused-profile.json",
                fit_spec=root / "unused-fit.json",
                code_archive=root / "unused-code.tar.gz",
                supplemental_archive=root / "unused-inputs.tar.gz",
                cli=Path("/bin/true"),
                profile=None,
                job_dir=root / "controller",
                config=root / "unused-config.yaml",
                sleep=lambda _seconds: None,
            )
            controller.job_dir.mkdir()
            controller.admission = admission()
            providers = [
                {
                    "id": "exact-job-id",
                    "project_id": PROJECT_ID,
                    "name": F1_JOB_NAME,
                    "status": "SUCCESS",
                    "output_files": [],
                    "output_file_errors": [],
                },
                {
                    "id": "exact-job-id",
                    "project_id": PROJECT_ID,
                    "name": F1_JOB_NAME,
                    "status": "SUCCESS",
                    "output_files": [expected],
                    "output_file_errors": [],
                },
            ]
            calls = 0

            def read_manifest(state: dict[str, object], attempt: int) -> dict[str, object]:
                nonlocal calls
                provider = providers[calls]
                calls += 1
                controller.manifest_path.write_text(json.dumps(provider), encoding="utf-8")
                return provider

            def runner(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
                destination = Path(command[command.index("--output-dir") + 1])
                shutil.copyfile(source_bundle, destination / F1_OUTPUT_PATH)
                return subprocess.CompletedProcess(command, 0, "", "")

            controller.runner = runner
            state: dict[str, object] = {
                "phase": "TERMINAL",
                "provider_status": "SUCCESS",
                "job_id": "exact-job-id",
            }
            with patch.object(controller, "_read_manifest", side_effect=read_manifest):
                self.assertEqual(controller._download_and_verify(state), 0)
            self.assertEqual(calls, 2)
            self.assertEqual(state["pending_output_manifest_attempts"], 1)
            self.assertEqual(state["phase"], "VERIFIED")

    def test_output_identity_mismatch_is_fail_closed_without_retry(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            controller = F1FitController(
                sdk_python=Path("/bin/true"),
                admission=root / "unused-admission.json",
                profile_spec=root / "unused-profile.json",
                fit_spec=root / "unused-fit.json",
                code_archive=root / "unused-code.tar.gz",
                supplemental_archive=root / "unused-inputs.tar.gz",
                cli=Path("/bin/true"),
                profile=None,
                job_dir=root / "controller",
                config=root / "unused-config.yaml",
                sleep=lambda _seconds: None,
            )
            controller.job_dir.mkdir()
            controller.admission = admission()
            wrong_file = {
                "path": "wrong.tar.gz",
                "variable": F1_OUTPUT_VARIABLE,
                "size_bytes": 1,
                "sha256": "a" * 64,
            }
            provider = {
                "id": "exact-job-id",
                "project_id": PROJECT_ID,
                "name": F1_JOB_NAME,
                "status": "SUCCESS",
                "output_files": [wrong_file],
                "output_file_errors": [],
            }
            controller._read_manifest = lambda _state, _attempt: provider
            state: dict[str, object] = {
                "phase": "TERMINAL",
                "provider_status": "SUCCESS",
                "job_id": "exact-job-id",
            }
            with self.assertRaisesRegex(FailClosed, "unexpected F1 output"):
                controller._download_and_verify(state)
            self.assertEqual(state["phase"], "UNVERIFIED_TERMINAL")
            self.assertNotIn("pending_output_manifest_attempts", state)

    def test_terminal_manifest_absence_fails_closed_after_three_reads(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            controller = F1FitController(
                sdk_python=Path("/bin/true"),
                admission=root / "unused-admission.json",
                profile_spec=root / "unused-profile.json",
                fit_spec=root / "unused-fit.json",
                code_archive=root / "unused-code.tar.gz",
                supplemental_archive=root / "unused-inputs.tar.gz",
                cli=Path("/bin/true"),
                profile=None,
                job_dir=root / "controller",
                config=root / "unused-config.yaml",
                sleep=lambda _seconds: None,
            )
            controller.job_dir.mkdir()
            controller.admission = admission()
            provider = {
                "id": "exact-job-id",
                "project_id": PROJECT_ID,
                "name": F1_JOB_NAME,
                "status": "SUCCESS",
                "output_files": [],
                "output_file_errors": [],
            }
            calls = 0

            def read_manifest(state: dict[str, object], attempt: int) -> dict[str, object]:
                nonlocal calls
                calls += 1
                controller.manifest_path.write_text(json.dumps(provider), encoding="utf-8")
                return provider

            controller._read_manifest = read_manifest
            state: dict[str, object] = {
                "phase": "TERMINAL",
                "provider_status": "SUCCESS",
                "job_id": "exact-job-id",
            }
            with self.assertRaisesRegex(FailClosed, "did not list the expected output"):
                controller._download_and_verify(state)
            self.assertEqual(calls, 3)
            self.assertEqual(state["pending_output_manifest_attempts"], 3)
            self.assertEqual(state["phase"], "UNVERIFIED_TERMINAL")

    def test_zero_exit_without_bundle_remains_download_attempted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_bundle = root / "source" / F1_OUTPUT_PATH
            source_bundle.parent.mkdir()
            make_bundle(source_bundle)
            expected = {
                "path": F1_OUTPUT_PATH,
                "variable": F1_OUTPUT_VARIABLE,
                "size_bytes": source_bundle.stat().st_size,
                "sha256": _sha256_file(source_bundle),
            }
            controller = F1FitController(
                sdk_python=Path("/bin/true"),
                admission=root / "unused-admission.json",
                profile_spec=root / "unused-profile.json",
                fit_spec=root / "unused-fit.json",
                code_archive=root / "unused-code.tar.gz",
                supplemental_archive=root / "unused-inputs.tar.gz",
                cli=Path("/bin/true"),
                profile=None,
                job_dir=root / "controller",
                config=root / "unused-config.yaml",
            )
            controller.job_dir.mkdir()
            controller.admission = admission()
            provider = {
                "id": "exact-job-id",
                "project_id": PROJECT_ID,
                "name": F1_JOB_NAME,
                "status": "SUCCESS",
                "output_files": [expected],
                "output_file_errors": [],
            }

            def read_manifest(state: dict[str, object], attempt: int) -> dict[str, object]:
                controller.manifest_path.write_text(json.dumps(provider), encoding="utf-8")
                return provider

            controller.runner = lambda command, **kwargs: subprocess.CompletedProcess(
                command, 0, "", ""
            )
            state: dict[str, object] = {
                "phase": "TERMINAL",
                "provider_status": "SUCCESS",
                "job_id": "exact-job-id",
            }
            with (
                patch.object(controller, "_read_manifest", side_effect=read_manifest),
                self.assertRaisesRegex(ControllerError, "not verified"),
            ):
                controller._download_and_verify(state)
            self.assertEqual(state["phase"], "DOWNLOAD_ATTEMPTED")
            self.assertNotIn("verified_output_sha256", state)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    unittest.main()
