from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
POLL_PROVIDER_PATH = (
    REPOSITORY_ROOT / "research/campaign_2026-10-06/cloud_preflight/poll_provider.py"
)
SPEC = importlib.util.spec_from_file_location("e529_poll_provider_test", POLL_PROVIDER_PATH)
assert SPEC is not None and SPEC.loader is not None
poll_provider = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(poll_provider)

CAMPAIGN_ROOT = REPOSITORY_ROOT / "research/campaign_2026-10-06"
NO_JOB_RECEIPT = (
    "cloud_preflight/e529_v4_rejected_longslug_no_provider_job_receipt_20261008_R1.json"
)
EVIDENCE_PATHS = {
    "push_attempt": (
        "bandit_continuation/continuous_execution_E529/"
        "gru512_768_muon_capacity_v4_root_candidate_20261008/"
        "GRU_CAPACITY_V4_PROVIDER_PUSH_20261008_R1.json"
    ),
    "read_only_absence": (
        "bandit_continuation/continuous_execution_E529/"
        "gru512_768_muon_capacity_v4_shortslug_root_candidate_20261008/"
        "GRU_CAPACITY_V4_SHORTSLUG_PREPUSH_GATE_20261008_R1.json"
    ),
    "hold_transfer": (
        "bandit_continuation/continuous_execution_E529/"
        "gru512_768_muon_capacity_v4_shortslug_root_candidate_20261008/"
        "GRU_CAPACITY_V4_SHORTSLUG_HOLD_TRANSFER_20261008_R1.json"
    ),
}
OLD_KERNEL = "vladkochetov07/e529-gru512-768-muon-capacity-v4-profile-20261008-runner"
NEW_KERNEL = "vladkochetov07/gru-capacity-v4-manifest-profile-20261008"
HOLD_TRANSFER = (
    "../bandit_continuation/continuous_execution_E529/"
    "gru512_768_muon_capacity_v4_shortslug_root_candidate_20261008/"
    "GRU_CAPACITY_V4_SHORTSLUG_HOLD_TRANSFER_20261008_R1.json"
)
ROOT_AUTHORIZATION_SHA256 = "5fe98b4b24e3a2d49ecdf4f8ae7879502622edd2e9d39832c805c918114181a0"


def _write_fixture(tmp_path: Path, damage: str | None = None) -> Path:
    campaign_root = tmp_path / "campaign"
    config_dir = campaign_root / "cloud_preflight"
    config_dir.mkdir(parents=True)
    receipt_source = CAMPAIGN_ROOT / NO_JOB_RECEIPT
    receipt_raw = receipt_source.read_bytes()
    receipt = json.loads(receipt_raw)
    (config_dir / Path(NO_JOB_RECEIPT).name).write_bytes(receipt_raw)

    for entry in receipt["evidence"].values():
        relative = entry["path"]
        source = CAMPAIGN_ROOT / relative
        destination = campaign_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    if damage == "receipt":
        receipt_path = config_dir / Path(NO_JOB_RECEIPT).name
        receipt_path.write_bytes(receipt_path.read_bytes() + b" ")

    target = {
        "kernel": OLD_KERNEL,
        "resource": "gpu",
        "authorized_metadata_id": OLD_KERNEL,
        "root_one_push_authorization_sha256": ROOT_AUTHORIZATION_SHA256,
        "fit_authorized": False,
        "provisional_reservation_seconds": 0,
        "status": "push_rejected_http_400_readonly_not_found_hold_transferred",
        "hold_transferred_to_kernel": NEW_KERNEL,
        "hold_transfer_receipt": HOLD_TRANSFER,
        "no_provider_job_receipt": Path(NO_JOB_RECEIPT).name,
        "no_provider_job_receipt_sha256": hashlib.sha256(receipt_raw).hexdigest(),
    }
    if damage == "kernel":
        target["kernel"] = "vladkochetov07/unrelated-kernel"
    targets_path = config_dir / "targets.json"
    targets_path.write_text(
        json.dumps(
            {
                "targets": [
                    target,
                    {"kernel": NEW_KERNEL, "resource": "gpu"},
                ]
            }
        )
    )
    return targets_path


class _FakeClient:
    def __init__(self) -> None:
        quota = SimpleNamespace(
            time_used=timedelta(seconds=10),
            time_reserved=timedelta(0),
            total_time_allowed=timedelta(seconds=100),
        )
        self.kernels = SimpleNamespace(
            kernels_api_client=SimpleNamespace(
                get_accelerator_quota_statistics=lambda _request: SimpleNamespace(
                    gpu_quota=quota,
                    tpu_quota=quota,
                    quota_refresh_time=None,
                )
            )
        )

    def __exit__(self, *_args: object) -> None:
        return None


class _FakeApi:
    def authenticate(self) -> None:
        return None


def _collect(
    monkeypatch: pytest.MonkeyPatch, targets_path: Path
) -> tuple[dict[str, object], list[str]]:
    called: list[str] = []
    monkeypatch.setattr(poll_provider, "KaggleApi", _FakeApi)
    monkeypatch.setattr(poll_provider, "_bounded_client", lambda _api: _FakeClient())

    def status(_client: object, kernel: str) -> dict[str, str]:
        called.append(kernel)
        return {"kernel": kernel, "status": "KernelWorkerStatus.RUNNING", "failure_message": ""}

    monkeypatch.setattr(poll_provider, "_status", status)
    result = poll_provider.collect(targets_path)
    return result, called


def test_rejected_push_with_bound_no_job_evidence_is_not_a_terminal_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    targets_path = _write_fixture(tmp_path)
    result, called = _collect(monkeypatch, targets_path)

    rows = {row["kernel"]: row for row in result["status"]}
    no_job = rows[OLD_KERNEL]
    assert no_job["status"] == "NO_PROVIDER_JOB_CREATED"
    assert no_job["monitoring_state"] == "stopped_confirmed_no_provider_job"
    assert no_job["provider_job_created"] is False
    assert no_job["no_provider_job_evidence"]["outputs_downloaded"] is False
    assert no_job["no_provider_job_evidence"]["terminal_job_status"] is None
    assert "terminal_evidence" not in no_job
    assert rows[NEW_KERNEL]["status"] == "KernelWorkerStatus.RUNNING"
    assert called == [NEW_KERNEL]


def test_no_job_receipt_hashes_the_actual_rejection_absence_and_hold_receipts() -> None:
    receipt = json.loads((CAMPAIGN_ROOT / NO_JOB_RECEIPT).read_text())
    push = json.loads((CAMPAIGN_ROOT / EVIDENCE_PATHS["push_attempt"]).read_text())
    absence = json.loads((CAMPAIGN_ROOT / EVIDENCE_PATHS["read_only_absence"]).read_text())
    transfer = json.loads((CAMPAIGN_ROOT / EVIDENCE_PATHS["hold_transfer"]).read_text())

    for evidence in receipt["evidence"].values():
        raw = (CAMPAIGN_ROOT / evidence["path"]).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == evidence["sha256"]
    assert push["status"] == "push_failed"
    assert push["attempt_limit"] == 1
    assert any("400 Client Error: Bad Request" in line for line in push["stderr_tail"])
    assert any(
        item["slug"] == OLD_KERNEL.split("/", 1)[1] and item["result"] == "Not found"
        for item in absence["own_kernel_list_searches"]
    )
    assert transfer["old_kernel_id"] == OLD_KERNEL
    assert transfer["old_row_provisional_reservation_after"] == 0
    assert transfer["new_kernel_id"] == NEW_KERNEL
    assert receipt["provider_job_created"] is False
    assert receipt["terminal_job_status"] is None
    assert receipt["outputs_downloaded"] is False


@pytest.mark.parametrize(
    ("damage", "expected_reason"),
    [("receipt", "no_job_receipt_sha256_mismatch"), ("kernel", "no_job_receipt_fields_mismatch")],
)
def test_invalid_no_job_evidence_does_not_suppress_provider_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    damage: str,
    expected_reason: str,
) -> None:
    targets_path = _write_fixture(tmp_path, damage)
    result, called = _collect(monkeypatch, targets_path)

    rows = {row["kernel"]: row for row in result["status"]}
    old_row = (
        rows[OLD_KERNEL]
        if damage != "kernel"
        else next(
            row for row in result["status"] if row["kernel"] == "vladkochetov07/unrelated-kernel"
        )
    )
    assert old_row["status"] == "KernelWorkerStatus.RUNNING"
    assert old_row["no_provider_job_receipt_validation"] == "invalid"
    assert old_row["no_provider_job_receipt_reason"] == expected_reason
    assert len(called) == 2
