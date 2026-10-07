from __future__ import annotations

import base64
import hashlib
import io
import json
import re
import tarfile
import urllib.error
import urllib.parse
import urllib.request
from datetime import UTC, datetime, timedelta
from http.client import HTTPMessage
from pathlib import Path
from types import SimpleNamespace
from typing import Never

import pytest

from scripts import vast_campaign as vast


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_create_payload_cannot_switch_authenticated_client_scope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    intent_path, _, _, _ = make_attempt(tmp_path, monkeypatch)
    intent = vast._read_object(intent_path)
    intent["create_payload"]["client_id"] = "unrelated-account"
    with pytest.raises(vast.CampaignError, match="authenticated account scope"):
        vast._validate_intent(intent, intent_path)


def make_attempt(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
    attempt_id: str = "vast-attempt-001",
    *,
    stage: str = "first_cohort",
    ceiling: str = "4.00",
    total_ceiling: str = "20.00",
    released_usd: str | None = None,
    expected_artifacts: list[str] | None = None,
    worker_inputs: list[dict[str, object]] | None = None,
    upload_timeout_seconds: int = 600,
    worker_setup_timeout_seconds: int = 0,
    offer_id: int = 12345,
    machine_id: int = 777,
) -> tuple[Path, Path, Path, dict[str, object]]:
    monkeypatch.setattr(vast, "ROOT", root)
    root.mkdir(parents=True, exist_ok=True)
    ssh_dir = root / ".ssh"
    ssh_dir.mkdir(exist_ok=True)
    public_key_path = ssh_dir / "id_ed25519.pub"
    public_key_path.write_text(
        "ssh-ed25519 bW9jay1lZDI1NTE5LWtleQ== campaign-test\n", encoding="utf-8"
    )
    private_key_path = ssh_dir / "id_ed25519"
    private_key_path.write_text("mock-private-key\n", encoding="utf-8")
    private_key_path.chmod(0o600)
    monkeypatch.setattr(vast, "DEFAULT_SSH_PUBLIC_KEY", public_key_path)
    packet = root / "packet.tar"
    run_script = b"#!/bin/sh\nexit 0\n"
    with tarfile.open(packet, "w:") as archive:
        member = tarfile.TarInfo("run.sh")
        member.size = len(run_script)
        member.mode = 0o700
        archive.addfile(member, io.BytesIO(run_script))
    source = root / "frozen_source.py"
    source.write_text("frozen source\n", encoding="utf-8")
    scope = vast._account_scope({"id": "test-account"})
    label = f"wnn-e529-{attempt_id.removeprefix('vast-attempt-')}-run"
    offer: dict[str, object] = {
        "id": offer_id,
        "machine_id": machine_id,
        "gpu_name": "RTX 3090",
        "gpu_ram": 24576,
        "num_gpus": 1,
        "cpu_cores": 4,
        "cpu_cores_effective": 10.7,
        "cpu_ram": 42885,
        "cuda_max_good": 12.0,
        "disk_space": 120,
        "dph_total": "0.10",
        "dph_gpu": "0.08",
        "dph_cpu": "0.01",
        "dph_ram": "0.005",
        "dph_disk": "0.005",
        "reliability": 0.95,
        "reliability2": 0.95,
        "verified": None,
        "verification": "verified",
        "is_vm_deverified": False,
        "rentable": True,
        "rented": False,
        "external": False,
        "type": None,
        "query_type": "on-demand",
        "geolocation": "test-zone",
        "inet_down": 200,
        "inet_down_cost": 0.0026041666666666665,
        "inet_up": 100,
        "inet_up_cost": 0.00390625,
        "storage_cost": 0.3666666666666667,
    }
    payload: dict[str, object] = {
        "image": f"example/worker@sha256:{'a' * 64}",
        "disk": 40,
        "runtype": "ssh_direc ssh_proxy",
        "label": label,
        "onstart": "sleep infinity",
        "cancel_unavail": True,
    }
    intent_path = root / f"{attempt_id}.intent.json"
    write_json(
        intent_path,
        {
            "schema_version": 1,
            "attempt_id": attempt_id,
            "campaign_label": label,
            "account_scope_sha256": scope,
            "packet_path": str(packet),
            "packet_sha256": digest(packet.read_bytes()),
            "worker_inputs": worker_inputs or [],
            "max_upload_bytes": 1_000_000,
            "upload_timeout_seconds": upload_timeout_seconds,
            "worker_setup_timeout_seconds": worker_setup_timeout_seconds,
            "offer": offer,
            "create_payload": payload,
            "expected_artifacts": expected_artifacts or ["profile.json"],
            "remote_output_dir": "/workspace/results",
            "max_runtime_seconds": 3600,
            "max_artifact_bytes": 1_000_000,
            "max_worker_host_ram_gb": "16.00",
        },
    )
    normalized = vast._validate_intent(vast._read_object(intent_path), intent_path)
    now = datetime.now(UTC)
    job_sha = "b" * 64
    dispatch_path = root / "runtime" / "attempts" / f"{attempt_id}.json"
    write_json(
        dispatch_path,
        {
            "attempt_id": attempt_id,
            "job_sha256": job_sha,
            "lane": "vast",
            "status": "launch_uncertain",
            "created_utc": now.isoformat(),
        },
    )
    amount = vast._money(ceiling, "test ceiling")
    quote_share = (amount / vast.Decimal(4)).quantize(vast.Decimal("0.01"))
    cost_breakdown = {
        "compute_storage_usd": vast._money_text(quote_share),
        "upload_usd": vast._money_text(quote_share),
        "download_usd": vast._money_text(quote_share),
        "cleanup_usd": vast._money_text(amount - quote_share * vast.Decimal(3)),
    }
    vast_admission: dict[str, object] = {
        "decision": "admit_one_vast_instance",
        "authorized_actions": ["create_one_instance"],
        "attempt_id": attempt_id,
        "campaign_label": label,
        "intent_sha256": normalized["intent_sha256"],
        "packet_sha256": normalized["packet_sha256"],
        "request_sha256": normalized["request_sha256"],
        "account_scope_sha256": scope,
        "offer_snapshot_sha256": normalized["offer_snapshot_sha256"],
        "offer_id": normalized["offer_id"],
        "payment_policy_receipt_sha256": "c" * 64,
        "cohort_stage": stage,
        "max_all_in_usd": ceiling,
        "minimum_live_balance_usd": "30.00",
        "total_campaign_ceiling_usd": total_ceiling,
        "max_runtime_seconds": 3600,
        "worker_setup_timeout_seconds": worker_setup_timeout_seconds,
        "max_upload_bytes": normalized["max_upload_bytes"],
        "upload_timeout_seconds": normalized["upload_timeout_seconds"],
        "worker_inputs_sha256": normalized["worker_inputs_sha256"],
        "max_disk_gb": 40,
        "max_worker_host_ram_gb": "16.00",
        "hard_stop_utc": (now + timedelta(seconds=3600)).isoformat(),
        "cleanup_deadline_utc": (now + timedelta(seconds=5400)).isoformat(),
        "cost_breakdown_usd": cost_breakdown,
    }
    if stage == "evidence_released":
        assert released_usd is not None
        vast_admission.update(
            evidence_release_receipt_sha256="d" * 64,
            evidence_released_usd=released_usd,
        )
    approval_path = root / f"{attempt_id}.admission.json"
    write_json(
        approval_path,
        {
            "owner": "root",
            "authorized": True,
            "attempt_id": attempt_id,
            "job_sha256": job_sha,
            "created_utc": now.isoformat(),
            "expires_utc": (now + timedelta(seconds=90)).isoformat(),
            "source_hashes": {"frozen_source.py": digest(source.read_bytes())},
            "vast_admission": vast_admission,
        },
    )
    return intent_path, approval_path, dispatch_path, offer


def test_load_api_key_reads_only_lowercase_vastai_from_dotenv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text("VAST_API_KEY=wrong\nvastai=test-secret-token\n", encoding="utf-8")
    monkeypatch.setattr(vast, "ROOT", tmp_path)
    monkeypatch.delenv("VAST_API_KEY", raising=False)
    monkeypatch.delenv("VASTAI_API_KEY", raising=False)

    assert vast.load_api_key() == "test-secret-token"

    dotenv_path.write_text("VAST_API_KEY=wrong\n", encoding="utf-8")
    with pytest.raises(vast.CampaignError, match="no `vastai` key"):
        vast.load_api_key()


def test_available_credit_prefers_live_credit_and_falls_back_only_when_absent() -> None:
    assert vast._balance({"balance": 0, "credit": "19.648636682581355"}) == vast.Decimal(
        "19.648636682581355"
    )
    assert vast._balance({"balance": "7.50"}) == vast.Decimal("7.50")
    with pytest.raises(vast.CampaignError, match="Vast available credit"):
        vast._balance({"balance": "50.00", "credit": None})


def test_inspect_reports_only_allowlisted_credit_fields() -> None:
    class AccountProvider(FakeProvider):
        def current_user(self) -> dict[str, object]:
            return {
                "id": "test-account",
                "balance": 0,
                "credit": "19.64863668",
                "billing_creditonly": 1,
                "api_key": "must-not-appear-in-output",
            }

    provider = AccountProvider({})
    result = vast.inspect_account(provider)
    assert result["available_credit_usd"] == "19.65"
    assert result["available_credit_field"] == "credit"
    assert "api_key" not in json.dumps(result)
    assert "must-not-appear-in-output" not in json.dumps(result)


def test_frozen_worker_inputs_bind_size_hash_and_upload_ceiling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "data" / "valid.parquet"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"small fixture")
    inputs = [
        {
            "source_path": "data/valid.parquet",
            "source_sha256": digest(source.read_bytes()),
            "size_bytes": source.stat().st_size,
            "remote_path": "valid.parquet",
        }
    ]
    intent_path, _, _, _ = make_attempt(tmp_path, monkeypatch, worker_inputs=inputs)
    normalized = vast._validate_intent(vast._read_object(intent_path), intent_path)
    assert normalized["worker_inputs_sha256"] == vast.canonical_sha256(inputs)
    assert normalized["max_upload_bytes"] == 1_000_000

    source.write_bytes(b"different size")
    with pytest.raises(vast.CampaignError, match="size differs"):
        vast._validate_intent(vast._read_object(intent_path), intent_path)


def test_upload_resume_window_is_bounded_and_root_admission_binds_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    intent_path, admission_path, dispatch_path, _ = make_attempt(
        tmp_path, monkeypatch, upload_timeout_seconds=5400
    )
    intent = vast._read_object(intent_path)
    normalized = vast._validate_intent(intent, intent_path)
    vast._validate_admission(
        vast._read_object(admission_path),
        intent,
        normalized,
        vast._read_object(dispatch_path),
        admission_path,
    )
    assert normalized["upload_timeout_seconds"] == 5400

    intent["upload_timeout_seconds"] = 5401
    with pytest.raises(vast.CampaignError, match=r"must be in \(0, 5400\]"):
        vast._validate_intent(intent, intent_path)


def test_root_admission_accepts_only_fresh_successful_submitted_dispatch_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    intent_path, admission_path, dispatch_path, _ = make_attempt(tmp_path, monkeypatch)
    intent = vast._read_object(intent_path)
    normalized = vast._validate_intent(intent, intent_path)
    approval = vast._read_object(admission_path)
    dispatch = vast._read_object(dispatch_path)
    dispatch.update(
        status="submitted",
        returncode=0,
        observed_utc=datetime.now(UTC).isoformat(),
    )
    vast._validate_admission(approval, intent, normalized, dispatch, admission_path)

    dispatch["returncode"] = 1
    with pytest.raises(vast.CampaignError, match="successful return code"):
        vast._validate_admission(approval, intent, normalized, dispatch, admission_path)

    dispatch["returncode"] = 0
    dispatch["observed_utc"] = (datetime.now(UTC) - timedelta(seconds=121)).isoformat()
    with pytest.raises(vast.CampaignError, match="fresh 120-second window"):
        vast._validate_admission(approval, intent, normalized, dispatch, admission_path)

    dispatch["observed_utc"] = datetime.now(UTC).isoformat()
    dispatch["job_sha256"] = "e" * 64
    with pytest.raises(vast.CampaignError, match="not bound to root admission job hash"):
        vast._validate_admission(approval, intent, normalized, dispatch, admission_path)

    dispatch["job_sha256"] = approval["job_sha256"]
    dispatch["status"] = "needs_review"
    with pytest.raises(vast.CampaignError, match="persisted launch intent"):
        vast._validate_admission(approval, intent, normalized, dispatch, admission_path)


def test_worker_setup_timeout_is_frozen_and_root_admission_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    intent_path, admission_path, dispatch_path, _ = make_attempt(
        tmp_path, monkeypatch, worker_setup_timeout_seconds=450
    )
    intent = vast._read_object(intent_path)
    normalized = vast._validate_intent(intent, intent_path)
    assert normalized["worker_setup_timeout_seconds"] == 450
    vast._validate_admission(
        vast._read_object(admission_path),
        intent,
        normalized,
        vast._read_object(dispatch_path),
        admission_path,
    )

    approval = vast._read_object(admission_path)
    approval["vast_admission"]["worker_setup_timeout_seconds"] = 449
    write_json(admission_path, approval)
    with pytest.raises(vast.CampaignError, match="worker_setup_timeout_seconds"):
        vast._validate_admission(
            approval, intent, normalized, vast._read_object(dispatch_path), admission_path
        )

    intent["worker_setup_timeout_seconds"] = 3601
    with pytest.raises(vast.CampaignError, match="worker_setup_timeout_seconds"):
        vast._validate_intent(intent, intent_path)


def test_upload_calls_never_outlive_the_receipted_hard_stop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    intent, admission, dispatch, offer = make_attempt(
        tmp_path, monkeypatch, upload_timeout_seconds=5400
    )
    _, controller = controller_for(tmp_path, offer)
    controller._create_one_unarmed(intent, admission, dispatch)
    mock_campaign_host_key(monkeypatch, tmp_path)
    hard_stop = datetime.now(UTC) + timedelta(seconds=20)
    controller._update_attempt("vast-attempt-001", hard_stop_utc=hard_stop.isoformat())
    observed_timeouts: list[int] = []

    def stop_after_first_owned_ssh(
        host: str,
        port: int,
        command: str,
        timeout: int,
        *,
        known_hosts_path: Path,
        allowed_exit_codes: tuple[int, ...] = (),
    ) -> int:
        del host, port, command, known_hosts_path, allowed_exit_codes
        observed_timeouts.append(timeout)
        raise vast.CampaignError("test stop after bounded first SSH call")

    monkeypatch.setattr(vast, "_ssh_command", stop_after_first_owned_ssh)
    with pytest.raises(vast.CampaignError, match="test stop"):
        controller._stage_and_launch("vast-attempt-001")

    assert observed_timeouts
    assert all(0 < timeout <= 20 for timeout in observed_timeouts)


def test_frozen_worker_input_paths_cannot_include_credentials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    credential_file = tmp_path / ".env"
    credential_file.write_text("vastai=not-for-worker", encoding="utf-8")
    inputs = [
        {
            "source_path": ".env",
            "source_sha256": digest(credential_file.read_bytes()),
            "size_bytes": credential_file.stat().st_size,
            "remote_path": "worker-env.txt",
        }
    ]
    with pytest.raises(vast.CampaignError, match="credential-like"):
        make_attempt(tmp_path, monkeypatch, worker_inputs=inputs)


def test_worker_bundle_requires_a_safe_root_run_script(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, _, _, _ = make_attempt(tmp_path, monkeypatch)
    packet = tmp_path / "packet.tar"
    vast._validate_worker_bundle(packet, 1_000_000)

    secret_packet = tmp_path / "secret.tar"
    secret_data = b"never upload credentials"
    with tarfile.open(secret_packet, "w:") as archive:
        run_script = tarfile.TarInfo("run.sh")
        run_script.size = 0
        archive.addfile(run_script, io.BytesIO())
        secret = tarfile.TarInfo(".env")
        secret.size = len(secret_data)
        archive.addfile(secret, io.BytesIO(secret_data))
    with pytest.raises(vast.CampaignError, match="credential-named"):
        vast._validate_worker_bundle(secret_packet, 1_000_000)


def test_live_offer_memory_fields_are_mib_and_worker_limit_uses_decimal_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    intent_path, _, _, offer = make_attempt(tmp_path, monkeypatch)
    snapshot = vast._offer_snapshot(offer)
    assert snapshot["gpu_ram_gib"] == "24.000"
    assert snapshot["cpu_ram_gib"] == "41.880"
    offer["gpu_ram"] = 6144
    with pytest.raises(vast.CampaignError, match="24 GiB GPU"):
        vast._offer_snapshot(offer)

    intent = vast._read_object(intent_path)
    intent["max_worker_host_ram_gb"] = "35.92"
    vast._validate_intent(intent, intent_path)
    intent["max_worker_host_ram_gb"] = "36.00"
    with pytest.raises(vast.CampaignError, match="20% selected-host headroom"):
        vast._validate_intent(intent, intent_path)


def test_cli_create_is_blocked_before_credentials_or_provider_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_key_load(_path: Path) -> str:
        raise AssertionError("disabled create must not load credentials")

    monkeypatch.setattr(vast, "load_api_key", unexpected_key_load)
    assert (
        vast.main(
            [
                "create",
                "--intent",
                "missing-intent.json",
                "--admission",
                "missing-admission.json",
                "--dispatch-receipt",
                "missing-dispatch.json",
                "--confirm-create",
                "attempt",
            ]
        )
        == 2
    )


class FakeProvider:
    def __init__(self, offer: dict[str, object]) -> None:
        self.offer = offer
        self.instances: list[dict[str, object]] = []
        self.create_calls = 0
        self.destroy_calls: list[str] = []
        self.keep_after_destroy = False
        self.timeout_after_create = False
        self.search_calls: list[dict[str, object]] = []
        self.id_counter = 9000
        self.current_user_calls = 0
        self.list_calls = 0
        self.response_id_override: int | None = None
        self.omit_created_instance = False
        self.created_label_override: str | None = None
        self.attach_calls: list[tuple[str, str]] = []

    def current_user(self) -> dict[str, object]:
        self.current_user_calls += 1
        return {"id": "test-account", "balance": "50.00", "credit": "50.00"}

    def list_instances(self) -> list[dict[str, object]]:
        self.list_calls += 1
        return [dict(row) for row in self.instances]

    def search_offers(self, criteria: dict[str, object]) -> list[dict[str, object]]:
        self.search_calls.append(dict(criteria))
        return [dict(self.offer)]

    def search_offer_for_revalidation(
        self, offer: dict[str, object], allocated_storage: int
    ) -> list[dict[str, object]]:
        assert allocated_storage > 0
        if offer.get("machine_id") == self.offer["machine_id"]:
            return [dict(self.offer)]
        return []

    def create_instance(self, offer_id: int, payload: dict[str, object]) -> dict[str, object]:
        self.create_calls += 1
        self.id_counter += 1
        instance = {
            "id": self.id_counter,
            "label": self.created_label_override or payload["label"],
            "machine_id": self.offer["machine_id"],
            "start_date": "2099-01-01T00:00:00+00:00",
            "actual_status": "running",
            "public_ipaddr": "192.0.2.1",
            "ssh_host": "ssh-proxy.example",
            "ssh_port": 2200,
        }
        if not self.omit_created_instance:
            self.instances.append(instance)
        if self.timeout_after_create:
            raise vast.ApiError("mock timeout after provider accepted create")
        return {"new_contract": self.response_id_override or self.id_counter}

    def get_instance(self, instance_id: str) -> dict[str, object]:
        for row in self.instances:
            if str(row["id"]) == instance_id:
                return dict(row)
        raise vast.ApiError("instance missing")

    def destroy_instance(self, instance_id: str) -> dict[str, object]:
        self.destroy_calls.append(instance_id)
        if not self.keep_after_destroy:
            self.instances = [row for row in self.instances if str(row["id"]) != instance_id]
        return {}

    def attach_ssh_key(self, instance_id: str, public_key: str) -> dict[str, object]:
        self.attach_calls.append((instance_id, digest(public_key.encode("utf-8"))))
        return {"attached": True}


def controller_for(
    tmp_path: Path, offer: dict[str, object]
) -> tuple[FakeProvider, vast.VastCampaign]:
    provider = FakeProvider(offer)
    return provider, vast.VastCampaign(provider, tmp_path / "state" / "ledger.json")


def test_public_create_method_is_blocked_without_provider_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    intent, admission, dispatch, offer = make_attempt(tmp_path, monkeypatch)
    provider, controller = controller_for(tmp_path, offer)

    with pytest.raises(vast.CampaignError, match="systemd-supervised run"):
        controller.create_one(intent, admission, dispatch)
    assert provider.current_user_calls == 0
    assert provider.list_calls == 0
    assert provider.create_calls == 0


def test_create_response_reusing_pre_submit_id_remains_unowned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    intent, admission, dispatch, offer = make_attempt(tmp_path, monkeypatch)
    provider, controller = controller_for(tmp_path, offer)
    provider.instances.append(
        {
            "id": 77,
            "label": "unrelated-preexisting",
            "machine_id": offer["machine_id"],
            "start_date": "2099-01-01T00:00:00+00:00",
        }
    )
    provider.response_id_override = 77

    with pytest.raises(vast.CampaignError, match="reused a pre-submit instance ID"):
        controller._create_one_unarmed(intent, admission, dispatch)
    record = controller._get_attempt("vast-attempt-001")
    assert record["status"] == "launch_uncertain"
    assert record["instance_id"] is None
    assert record["reported_create_instance_id"] == "77"
    assert provider.create_calls == 1


def test_false_success_create_response_remains_unowned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    intent, admission, dispatch, offer = make_attempt(tmp_path, monkeypatch)
    provider, controller = controller_for(tmp_path, offer)
    provider.omit_created_instance = True

    with pytest.raises(vast.CampaignError, match="could not be verified"):
        controller._create_one_unarmed(intent, admission, dispatch)
    record = controller._get_attempt("vast-attempt-001")
    assert record["status"] == "launch_uncertain"
    assert record["instance_id"] is None
    assert record["reported_create_instance_id"] == "9001"
    assert provider.create_calls == 1


def test_create_response_with_mismatched_provider_label_remains_unowned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    intent, admission, dispatch, offer = make_attempt(tmp_path, monkeypatch)
    provider, controller = controller_for(tmp_path, offer)
    provider.created_label_override = "someone-elses-instance"

    with pytest.raises(vast.CampaignError, match="does not prove a new matching instance"):
        controller._create_one_unarmed(intent, admission, dispatch)
    record = controller._get_attempt("vast-attempt-001")
    assert record["status"] == "launch_uncertain"
    assert record["instance_id"] is None
    assert record["reported_create_instance_id"] == "9001"


def mock_campaign_host_key(monkeypatch: pytest.MonkeyPatch, root: Path) -> Path:
    """Keep SSH mocked while providing the attempt-local known-hosts argument."""
    path = root / ".runtime" / "vast" / "known_hosts" / "test.known_hosts"
    monkeypatch.setattr(
        vast.VastCampaign,
        "_campaign_known_hosts",
        lambda _self, *_args, **_kwargs: path,
    )
    return path


def collect_valid(
    controller: vast.VastCampaign,
    root: Path,
    attempt_id: str,
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, object]:
    mock_campaign_host_key(monkeypatch, root)
    record = controller._get_attempt(attempt_id)
    contents = b"profile contents\n"
    manifest = {
        "attempt_id": attempt_id,
        "packet_sha256": record["packet_sha256"],
        "worker_terminal": True,
        "files": [
            {
                "path": "profile.json",
                "size_bytes": len(contents),
                "sha256": digest(contents),
            }
        ],
    }

    def downloader(
        host: str,
        port: int,
        remote: str,
        destination: Path,
        timeout: int,
        max_bytes: int,
        known_hosts_path: Path,
    ) -> None:
        del host, port, timeout, known_hosts_path
        assert max_bytes >= 0
        if remote.endswith("artifact_manifest.json"):
            write_json(destination, manifest)
        else:
            destination.write_bytes(contents)

    return controller.collect(
        attempt_id,
        root / f"{attempt_id}-artifacts",
        downloader=downloader,
    )


def test_create_timeout_is_durable_and_same_attempt_never_posts_twice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    intent, admission, dispatch, offer = make_attempt(tmp_path, monkeypatch)
    provider, controller = controller_for(tmp_path, offer)
    provider.timeout_after_create = True

    with pytest.raises(vast.CampaignError, match="never resubmit"):
        controller._create_one_unarmed(intent, admission, dispatch)
    assert controller._get_attempt("vast-attempt-001")["status"] == "launch_uncertain"
    with pytest.raises(vast.CampaignError, match="never POST"):
        controller._create_one_unarmed(intent, admission, dispatch)
    assert provider.create_calls == 1


def test_matching_preexisting_label_is_never_adopted_or_created(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    intent, admission, dispatch, offer = make_attempt(tmp_path, monkeypatch)
    provider, controller = controller_for(tmp_path, offer)
    provider.instances.append(
        {
            "id": 77,
            "label": "wnn-e529-001-run",
            "machine_id": offer["machine_id"],
            "start_date": "2099-01-01T00:00:00+00:00",
        }
    )

    with pytest.raises(vast.CampaignError, match="never adopted"):
        controller._create_one_unarmed(intent, admission, dispatch)
    assert provider.create_calls == 0
    assert vast._attempts(vast._read_object(controller.ledger_path)) == {}


@pytest.mark.parametrize(
    "machine_id,start_date",
    [(999, "2099-01-01T00:00:00+00:00"), (777, "2000-01-01T00:00:00+00:00")],
)
def test_ambiguous_create_does_not_adopt_label_spoof_or_old_instance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    machine_id: int,
    start_date: str,
) -> None:
    intent, admission, dispatch, offer = make_attempt(tmp_path, monkeypatch)
    provider, controller = controller_for(tmp_path, offer)
    provider.timeout_after_create = True
    with pytest.raises(vast.CampaignError):
        controller._create_one_unarmed(intent, admission, dispatch)
    provider.instances[0]["machine_id"] = machine_id
    provider.instances[0]["start_date"] = start_date

    status = controller.reconcile("vast-attempt-001")
    assert status["status"] == "needs_review"
    assert status["instance_id"] is None
    assert provider.create_calls == 1
    assert provider.destroy_calls == []


def test_ambiguous_create_reconciles_only_one_new_exact_instance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    intent, admission, dispatch, offer = make_attempt(tmp_path, monkeypatch)
    provider, controller = controller_for(tmp_path, offer)
    provider.timeout_after_create = True
    with pytest.raises(vast.CampaignError):
        controller._create_one_unarmed(intent, admission, dispatch)

    status = controller.reconcile("vast-attempt-001")
    assert status["status"] == "created"
    assert status["instance_id"] == str(provider.instances[0]["id"])
    assert provider.create_calls == 1


@pytest.mark.parametrize(
    "manifest_files,contents",
    [([], b"ok"), ([{"path": "profile.json", "size_bytes": 2, "sha256": "0" * 64}], b"ok")],
)
def test_missing_output_or_hash_mismatch_stays_needs_review(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    manifest_files: list[dict[str, object]],
    contents: bytes,
) -> None:
    intent, admission, dispatch, offer = make_attempt(tmp_path, monkeypatch)
    provider, controller = controller_for(tmp_path, offer)
    controller._create_one_unarmed(intent, admission, dispatch)
    mock_campaign_host_key(monkeypatch, tmp_path)

    manifest = {
        "attempt_id": "vast-attempt-001",
        "packet_sha256": vast._validate_intent(vast._read_object(intent), intent)["packet_sha256"],
        "worker_terminal": True,
        "files": manifest_files,
    }

    def downloader(
        host: str,
        port: int,
        remote: str,
        destination: Path,
        timeout: int,
        max_bytes: int,
        known_hosts_path: Path,
    ) -> None:
        del host, port, timeout, known_hosts_path
        assert max_bytes >= 0
        if remote.endswith("artifact_manifest.json"):
            write_json(destination, manifest)
        else:
            destination.write_bytes(contents)

    with pytest.raises(vast.CampaignError):
        controller.collect("vast-attempt-001", tmp_path / "collected", downloader=downloader)
    assert controller._get_attempt("vast-attempt-001")["status"] == "needs_review"


def test_destroy_rejects_unrelated_id_before_any_provider_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    intent, admission, dispatch, offer = make_attempt(tmp_path, monkeypatch)
    provider, controller = controller_for(tmp_path, offer)
    controller._create_one_unarmed(intent, admission, dispatch)
    before = (provider.current_user_calls, provider.list_calls, list(provider.destroy_calls))

    with pytest.raises(vast.CampaignError, match="not in this attempt"):
        controller.destroy_owned("vast-attempt-001", "812345")
    assert (provider.current_user_calls, provider.list_calls, provider.destroy_calls) == before


def test_destroy_requires_collection_and_unverified_delete_remains_pending(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    intent, admission, dispatch, offer = make_attempt(tmp_path, monkeypatch)
    provider, controller = controller_for(tmp_path, offer)
    created = controller._create_one_unarmed(intent, admission, dispatch)
    instance_id = str(created["instance_id"])
    before = len(provider.destroy_calls)

    with pytest.raises(vast.CampaignError, match="collect and verify"):
        controller.destroy_owned("vast-attempt-001", instance_id)
    assert len(provider.destroy_calls) == before

    provider.keep_after_destroy = True
    assert (
        collect_valid(controller, tmp_path, "vast-attempt-001", monkeypatch)["status"]
        == "collected"
    )
    status = controller.destroy_owned("vast-attempt-001", instance_id)
    assert status["status"] == "destroy_pending"
    assert (
        status["destroy_verification"] == "instance remains in account listing after delete request"
    )
    assert provider.destroy_calls == [instance_id]


def test_ssh_endpoint_uses_paired_proxy_endpoint_without_mixing_public_ip_or_proxy_port() -> None:
    assert vast._ssh_endpoint(
        {
            "ssh_host": "proxy.vast.ai",
            "ssh_port": 2200,
            "public_ipaddr": "192.0.2.20",
            "direct_port_start": 44000,
        }
    ) == ("proxy.vast.ai", 2200)


def test_ssh_endpoint_requires_explicit_direct_22_tcp_mapping() -> None:
    assert vast._ssh_endpoint(
        {
            "public_ipaddr": "192.0.2.20",
            "ports": {"22/tcp": [{"HostIp": "0.0.0.0", "HostPort": "44022"}]},
        }
    ) == ("192.0.2.20", 44022)

    for instance in (
        {"public_ipaddr": "192.0.2.20", "ssh_port": 2200},
        {"public_ipaddr": "192.0.2.20", "direct_port_start": 44022},
        {"public_ipaddr": "192.0.2.20", "ports": {"8080/tcp": [{"HostPort": 44022}]}},
        {
            "public_ipaddr": "192.0.2.20",
            "ports": {"22/tcp": [{"HostPort": 44022}, {"HostPort": 44023}]},
        },
    ):
        with pytest.raises(vast.CampaignError):
            vast._ssh_endpoint(instance)


def test_worker_waits_for_provider_running_before_consuming_stage_attempt_or_ssh(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    intent, admission, dispatch, offer = make_attempt(tmp_path, monkeypatch)
    provider, controller = controller_for(tmp_path, offer)
    controller._create_one_unarmed(intent, admission, dispatch)
    provider.instances[0]["actual_status"] = "loading"

    def unexpected_scan(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("SSH must not be touched while provider status is loading")

    monkeypatch.setattr(vast.subprocess, "run", unexpected_scan)
    waiting = controller._stage_and_launch("vast-attempt-001")
    assert waiting["worker_stage_attempts"] in (None, 0)
    assert waiting["worker_startup_status"] == "loading"
    assert provider.attach_calls == []

    provider.instances[0]["actual_status"] = "offline"
    with pytest.raises(vast.WorkerStartupFailure, match="failed to reach running"):
        controller._stage_and_launch("vast-attempt-001")
    failed = controller._get_attempt("vast-attempt-001")
    assert failed["worker_stage_attempts"] in (None, 0)
    assert failed["worker_terminal_failure"] is True
    assert provider.attach_calls == []


def test_campaign_host_key_is_pinned_per_owned_endpoint_and_used_strictly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    intent, admission, dispatch, offer = make_attempt(tmp_path, monkeypatch)
    provider, controller = controller_for(tmp_path, offer)
    created = controller._create_one_unarmed(intent, admission, dispatch)
    instance = provider.get_instance(str(created["instance_id"]))
    controller._attach_owned_ssh_key("vast-attempt-001", instance)
    key_type = b"ssh-ed25519"
    key_blob = len(key_type).to_bytes(4, "big") + key_type + b"campaign-host-test"
    encoded = base64.b64encode(key_blob).decode("ascii")
    host_line = f"[ssh-proxy.example]:2200 ssh-ed25519 {encoded}\n"
    scan_call: list[list[str]] = []

    def keyscan(argv: list[str], **kwargs: object) -> SimpleNamespace:
        scan_call.append(argv)
        assert kwargs["timeout"] == 10
        return SimpleNamespace(returncode=0, stdout=host_line, stderr="")

    monkeypatch.setattr(vast.subprocess, "run", keyscan)
    known_hosts = controller._campaign_known_hosts(
        "vast-attempt-001", instance, allow_first_use=True
    )
    record = controller._get_attempt("vast-attempt-001")
    assert scan_call == [["ssh-keyscan", "-T", "5", "-p", "2200", "ssh-proxy.example"]]
    assert known_hosts == Path(record["known_hosts_path"])
    assert known_hosts.stat().st_mode & 0o777 == 0o600
    assert len(record["known_hosts_fingerprints"]) == 1
    assert provider.attach_calls == [(str(created["instance_id"]), record["ssh_key_sha256"])]

    commands: list[list[str]] = []

    def ssh(argv: list[str], **kwargs: object) -> SimpleNamespace:
        commands.append(argv)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(vast.subprocess, "run", ssh)
    vast._ssh_command("ssh-proxy.example", 2200, "true", 10, known_hosts_path=known_hosts)
    assert "StrictHostKeyChecking=yes" in commands[0]
    assert f"UserKnownHostsFile={known_hosts}" in commands[0]
    assert "GlobalKnownHostsFile=/dev/null" in commands[0]
    assert "IdentitiesOnly=yes" in commands[0]

    changed_blob = key_blob[:-1] + bytes([key_blob[-1] ^ 1])
    changed = base64.b64encode(changed_blob).decode("ascii")
    known_hosts.write_text(host_line.replace(encoded, changed), encoding="ascii")
    with pytest.raises(vast.CampaignError, match="host key|known-hosts"):
        controller._campaign_known_hosts("vast-attempt-001", instance, allow_first_use=True)


def test_stage_uploads_exact_inputs_verifies_remote_hashes_and_launches_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "data" / "valid.parquet"
    source.parent.mkdir(parents=True)
    contents = b"frozen input bytes"
    source.write_bytes(contents)
    inputs = [
        {
            "source_path": "data/valid.parquet",
            "source_sha256": digest(contents),
            "size_bytes": len(contents),
            "remote_path": "valid.parquet",
        }
    ]
    intent, admission, dispatch, offer = make_attempt(
        tmp_path,
        monkeypatch,
        worker_inputs=inputs,
        worker_setup_timeout_seconds=450,
    )
    provider, controller = controller_for(tmp_path, offer)
    controller._create_one_unarmed(intent, admission, dispatch)
    mock_path = mock_campaign_host_key(monkeypatch, tmp_path)
    commands: list[str] = []
    command_timeouts: list[tuple[str, int]] = []
    uploads: list[tuple[Path, str, int]] = []

    def ssh_command(
        host: str,
        port: int,
        command: str,
        timeout: int,
        *,
        known_hosts_path: Path,
        allowed_exit_codes: tuple[int, ...] = (),
    ) -> int:
        assert host == "ssh-proxy.example"
        assert port == 2200
        assert timeout > 0
        assert known_hosts_path == mock_path
        commands.append(command)
        command_timeouts.append((command, timeout))
        if "sha256sum --check --status" in command and "mv -f" not in command:
            return 1  # Receipt-matching remote file is absent; upload the frozen local file.
        return 0

    def rsync_push(
        host: str,
        port: int,
        upload_source: Path,
        remote_path: str,
        timeout: int,
        max_bytes: int,
        known_hosts_path: Path,
    ) -> None:
        assert host == "ssh-proxy.example"
        assert port == 2200
        assert timeout > 0
        assert known_hosts_path == mock_path
        assert upload_source.stat().st_size <= max_bytes
        uploads.append((upload_source, remote_path, max_bytes))

    monkeypatch.setattr(vast, "_ssh_command", ssh_command)
    monkeypatch.setattr(vast, "_rsync_push", rsync_push)

    staged = controller._stage_and_launch("vast-attempt-001")
    assert staged["worker_launch_status"] == "worker_started"
    verified_paths = staged["upload_verified_files"]
    assert isinstance(verified_paths, list) and len(verified_paths) == 2
    assert verified_paths[0].endswith("/packet.tar")
    assert verified_paths[1].endswith("/inputs/valid.parquet")
    assert len(uploads) == 2
    assert uploads[1][0] == source
    assert uploads[1][1].endswith("/inputs/valid.parquet.part")
    checksum_timeouts = [
        timeout for command, timeout in command_timeouts if "sha256sum --check --status" in command
    ]
    assert len(checksum_timeouts) == 4  # Packet/input: existing check and post-upload verification.
    assert all(timeout > 30 for timeout in checksum_timeouts)
    launch_commands = [command for command in commands if "nohup env" in command]
    assert len(launch_commands) == 1
    assert "WNN_ATTEMPT_ID=vast-attempt-001" in launch_commands[0]
    assert "WNN_INPUT_DIR=" in launch_commands[0]
    assert "WNN_INPUT_MANIFEST=" in launch_commands[0]
    runtime_match = re.search(r"WNN_MAX_RUNTIME_SECONDS=(\d+)", launch_commands[0])
    assert runtime_match is not None
    worker_runtime = int(runtime_match.group(1))
    assert 2700 <= worker_runtime < 2850
    assert staged["worker_runtime_seconds"] == worker_runtime
    assert staged["worker_setup_timeout_seconds"] == 450

    controller._stage_and_launch("vast-attempt-001")
    assert len([command for command in commands if "nohup env" in command]) == 1
    assert len(uploads) == 2


def test_stage_refuses_launch_without_worker_and_collection_runway(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    intent, admission, dispatch, offer = make_attempt(tmp_path, monkeypatch)
    provider, controller = controller_for(tmp_path, offer)
    controller._create_one_unarmed(intent, admission, dispatch)
    mock_path = mock_campaign_host_key(monkeypatch, tmp_path)
    hard_stop = datetime.now(UTC) + timedelta(seconds=240)
    controller._update_attempt("vast-attempt-001", hard_stop_utc=hard_stop.isoformat())
    commands: list[str] = []

    def ssh_command(
        host: str,
        port: int,
        command: str,
        timeout: int,
        *,
        known_hosts_path: Path,
        allowed_exit_codes: tuple[int, ...] = (),
    ) -> int:
        del host, port, timeout, allowed_exit_codes
        assert known_hosts_path == mock_path
        commands.append(command)
        if "sha256sum --check --status" in command and "mv -f" not in command:
            return 1
        return 0

    monkeypatch.setattr(vast, "_ssh_command", ssh_command)
    monkeypatch.setattr(vast, "_rsync_push", lambda *_args, **_kwargs: None)

    with pytest.raises(vast.CampaignError, match="insufficient hard-stop runway"):
        controller._stage_and_launch("vast-attempt-001")

    assert not any("nohup env" in command for command in commands)
    assert controller._get_attempt("vast-attempt-001").get("worker_runtime_seconds") is None


def test_rsync_push_resumes_partial_and_keeps_strict_host_pinning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    make_attempt(tmp_path, monkeypatch)
    source = tmp_path / "data" / "train.parquet"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"immutable staged input")
    known_hosts = tmp_path / ".runtime" / "vast" / "known_hosts"
    invocations: list[tuple[list[str], dict[str, object]]] = []

    def fake_run(command: list[str], **kwargs: object) -> SimpleNamespace:
        invocations.append((command, kwargs))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(vast.subprocess, "run", fake_run)
    vast._rsync_push(
        "ssh-proxy.example",
        2200,
        source,
        "/workspace/inputs/train.parquet.part",
        5400,
        source.stat().st_size,
        known_hosts,
    )

    assert len(invocations) == 1
    command, kwargs = invocations[0]
    assert "--partial" in command
    assert "--append-verify" in command
    assert "--inplace" not in command
    assert f"--max-size={source.stat().st_size}" in command
    ssh_options = command[command.index("-e") + 1]
    assert "StrictHostKeyChecking=yes" in ssh_options
    assert f"UserKnownHostsFile={known_hosts}" in ssh_options
    assert "GlobalKnownHostsFile=/dev/null" in ssh_options
    assert kwargs["timeout"] == 5400
    assert command[-1] == "root@ssh-proxy.example:/workspace/inputs/train.parquet.part"


def test_repeated_remote_integrity_failure_destroys_only_the_owned_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    intent, admission, dispatch, offer = make_attempt(tmp_path, monkeypatch)
    provider, controller = controller_for(tmp_path, offer)
    created = controller._create_one_unarmed(intent, admission, dispatch)
    mock_campaign_host_key(monkeypatch, tmp_path)
    launch_commands: list[str] = []

    def ssh_command(
        host: str,
        port: int,
        command: str,
        timeout: int,
        *,
        known_hosts_path: Path,
        allowed_exit_codes: tuple[int, ...] = (),
    ) -> int:
        del host, port, timeout, known_hosts_path, allowed_exit_codes
        launch_commands.append(command)
        if "sha256sum --check --status" in command and "mv -f" not in command:
            return 1
        if "mv -f" in command:
            return 1  # Remote bytes did not match the persisted source receipt.
        return 0

    monkeypatch.setattr(vast, "_ssh_command", ssh_command)
    monkeypatch.setattr(vast, "_rsync_push", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        controller,
        "_worker_manifest_is_terminal",
        lambda *args, **kwargs: False,
    )

    with pytest.raises(vast.WorkerStagingIntegrityError):
        controller._stage_and_launch("vast-attempt-001")
    result = controller.supervise(
        "vast-attempt-001",
        tmp_path / "results" / "vast-attempt-001",
        poll_interval_seconds=1,
        sleep_fn=lambda _seconds: None,
    )

    assert result["status"] == "destroyed_verified"
    assert result["worker_terminal_failure"] is True
    assert provider.destroy_calls == [str(created["instance_id"])]
    assert not any("nohup env" in command for command in launch_commands)


def test_supervisor_collects_failure_archive_result_and_log_then_tears_down_owned_instance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact_bytes = {
        "artifact_bundle.tar": b"partial output archive",
        "result.json": b'{"exit_code":1,"worker_success":false}\n',
        "worker.log": b"training failed after startup\n",
    }
    intent, admission, dispatch, offer = make_attempt(
        tmp_path, monkeypatch, expected_artifacts=list(artifact_bytes)
    )
    provider, controller = controller_for(tmp_path, offer)
    created = controller._create_one_unarmed(intent, admission, dispatch)
    mock_campaign_host_key(monkeypatch, tmp_path)
    record = controller._get_attempt("vast-attempt-001")
    manifest = {
        "attempt_id": "vast-attempt-001",
        "packet_sha256": record["packet_sha256"],
        "worker_terminal": True,
        "files": [
            {
                "path": relative,
                "size_bytes": len(contents),
                "sha256": digest(contents),
            }
            for relative, contents in artifact_bytes.items()
        ],
    }

    def downloader(
        host: str,
        port: int,
        remote: str,
        destination: Path,
        timeout: int,
        max_bytes: int,
        known_hosts_path: Path,
    ) -> None:
        del known_hosts_path
        assert host == "ssh-proxy.example"
        assert port == 2200
        assert timeout > 0
        if remote.endswith("artifact_manifest.json"):
            write_json(destination, manifest)
        else:
            relative = remote.rsplit("/", maxsplit=1)[-1]
            contents = artifact_bytes[relative]
            assert max_bytes == len(contents)
            destination.write_bytes(contents)

    controller._update_attempt(
        "vast-attempt-001",
        status="worker_running",
        worker_launch_status="worker_started",
        worker_started_at_utc=vast.utc_now(),
    )
    result = controller.supervise(
        "vast-attempt-001",
        tmp_path / "results" / "vast-attempt-001",
        poll_interval_seconds=1,
        downloader=downloader,
        sleep_fn=lambda _seconds: None,
    )

    assert result["status"] == "destroyed_verified"
    assert result["collected_files"] == [
        {
            "path": relative,
            "size_bytes": len(contents),
            "sha256": digest(contents),
        }
        for relative, contents in artifact_bytes.items()
    ]
    output = tmp_path / "results" / "vast-attempt-001"
    assert (output / "artifact_bundle.tar").read_bytes() == artifact_bytes["artifact_bundle.tar"]
    assert (output / "result.json").read_bytes() == artifact_bytes["result.json"]
    assert (output / "worker.log").read_bytes() == artifact_bytes["worker.log"]
    assert provider.destroy_calls == [str(created["instance_id"])]


def test_supervisor_quarantines_partial_collection_and_retries_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact_bytes = {
        "artifact_bundle.tar": b"complete result archive",
        "result.json": b'{"status":"delivered_ungraded"}\n',
        "worker.log": b"worker completed\n",
    }
    intent, admission, dispatch, offer = make_attempt(
        tmp_path, monkeypatch, expected_artifacts=list(artifact_bytes)
    )
    provider, controller = controller_for(tmp_path, offer)
    created = controller._create_one_unarmed(intent, admission, dispatch)
    mock_campaign_host_key(monkeypatch, tmp_path)
    record = controller._get_attempt("vast-attempt-001")
    manifest = {
        "attempt_id": "vast-attempt-001",
        "packet_sha256": record["packet_sha256"],
        "worker_terminal": True,
        "files": [
            {
                "path": relative,
                "size_bytes": len(contents),
                "sha256": digest(contents),
            }
            for relative, contents in artifact_bytes.items()
        ],
    }
    output = tmp_path / "results" / "vast-attempt-001"
    failed_first_collection = False

    def downloader(
        host: str,
        port: int,
        remote: str,
        destination: Path,
        timeout: int,
        max_bytes: int,
        known_hosts_path: Path,
    ) -> None:
        nonlocal failed_first_collection
        del host, port, timeout, known_hosts_path
        if remote.endswith("artifact_manifest.json"):
            if destination == output / "artifact_manifest.json" and not failed_first_collection:
                failed_first_collection = True
                destination.write_text("partial manifest", encoding="utf-8")
                raise OSError("simulated first manifest transfer interruption")
            write_json(destination, manifest)
            return
        relative = remote.rsplit("/", maxsplit=1)[-1]
        contents = artifact_bytes[relative]
        assert max_bytes == len(contents)
        destination.write_bytes(contents)

    controller._update_attempt(
        "vast-attempt-001",
        status="worker_running",
        worker_launch_status="worker_started",
        worker_started_at_utc=vast.utc_now(),
    )
    result = controller.supervise(
        "vast-attempt-001",
        output,
        poll_interval_seconds=1,
        downloader=downloader,
        sleep_fn=lambda _seconds: None,
    )

    quarantine = output.with_name(f"{output.name}.collection-failed-1")
    assert result["status"] == "destroyed_verified"
    assert result["collection_attempts"] == 2
    assert result["collection_terminal_failure"] is False
    assert quarantine.is_dir()
    assert (quarantine / "artifact_manifest.json").read_text() == "partial manifest"
    assert (output / "result.json").read_bytes() == artifact_bytes["result.json"]
    assert provider.destroy_calls == [str(created["instance_id"])]


def test_supervisor_hard_stop_destroys_only_the_receipt_owned_instance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    intent, admission, dispatch, offer = make_attempt(tmp_path, monkeypatch)
    provider, controller = controller_for(tmp_path, offer)
    created = controller._create_one_unarmed(intent, admission, dispatch)
    now = datetime.now(UTC)
    controller._update_attempt(
        "vast-attempt-001",
        hard_stop_utc=(now - timedelta(seconds=1)).isoformat(),
        cleanup_deadline_utc=(now + timedelta(seconds=30)).isoformat(),
    )

    def unexpected_worker_access(*_args: object, **_kwargs: object) -> Never:
        raise AssertionError("hard-stop path must teardown without staging or collection")

    monkeypatch.setattr(controller, "_stage_and_launch", unexpected_worker_access)
    monkeypatch.setattr(controller, "_worker_manifest_is_terminal", unexpected_worker_access)
    result = controller.supervise(
        "vast-attempt-001",
        tmp_path / "results" / "vast-attempt-001",
        poll_interval_seconds=1,
        sleep_fn=lambda _seconds: None,
    )

    assert result["status"] == "destroyed_verified"
    assert provider.destroy_calls == [str(created["instance_id"])]


def authorize_parallel_spending(admission: Path) -> None:
    """Bind the new root spending decision without changing historical receipts."""
    relative = "specs/adr/0015-flexible-credit-bounded-parallel-compute.md"
    original = Path(__file__).resolve().parents[1] / relative
    authority = vast.ROOT / relative
    authority.parent.mkdir(parents=True, exist_ok=True)
    authority.write_bytes(original.read_bytes())
    approval = vast._read_object(admission)
    approval["source_hashes"][relative] = digest(authority.read_bytes())
    approval["vast_admission"]["spending_authority_sha256"] = digest(authority.read_bytes())
    write_json(admission, approval)


def test_parallel_cohort_uses_root_authority_without_evidence_release(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    intent, admission, dispatch, offer = make_attempt(
        tmp_path, monkeypatch, stage="parallel_authorized", ceiling="6.00"
    )
    authorize_parallel_spending(admission)
    provider, controller = controller_for(tmp_path, offer)
    created = controller._create_one_unarmed(intent, admission, dispatch)
    assert created["reserved_all_in_usd"] == "6.00"
    assert provider.create_calls == 1


@pytest.mark.parametrize("missing", ["authority", "source"])
def test_parallel_cohort_requires_exact_source_bound_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, missing: str
) -> None:
    intent, admission, dispatch, offer = make_attempt(
        tmp_path, monkeypatch, stage="parallel_authorized", ceiling="6.00"
    )
    authorize_parallel_spending(admission)
    approval = vast._read_object(admission)
    if missing == "authority":
        approval["vast_admission"]["spending_authority_sha256"] = "0" * 64
    else:
        del approval["source_hashes"]["specs/adr/0015-flexible-credit-bounded-parallel-compute.md"]
    write_json(admission, approval)
    provider, controller = controller_for(tmp_path, offer)
    with pytest.raises(vast.CampaignError, match="spending authority"):
        controller._create_one_unarmed(intent, admission, dispatch)
    assert provider.create_calls == 0


def test_parallel_reservations_still_cannot_exceed_twenty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    intent, admission, dispatch, offer = make_attempt(
        tmp_path, monkeypatch, stage="parallel_authorized", ceiling="12.00"
    )
    authorize_parallel_spending(admission)
    provider, controller = controller_for(tmp_path, offer)
    controller._create_one_unarmed(intent, admission, dispatch)
    intent2, admission2, dispatch2, offer2 = make_attempt(
        tmp_path,
        monkeypatch,
        "vast-attempt-002",
        stage="parallel_authorized",
        ceiling="12.00",
        offer_id=12346,
        machine_id=778,
    )
    authorize_parallel_spending(admission2)
    provider.offer = offer2
    with pytest.raises(vast.CampaignError, match="cumulative Vast credit ceiling"):
        controller._create_one_unarmed(intent2, admission2, dispatch2)
    assert provider.create_calls == 1


def test_first_cohort_allows_exactly_four_but_not_over_four(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    intent, admission, dispatch, offer = make_attempt(tmp_path, monkeypatch)
    provider, controller = controller_for(tmp_path, offer)
    created = controller._create_one_unarmed(intent, admission, dispatch)
    assert created["reserved_all_in_usd"] == "4.00"
    assert provider.create_calls == 1

    intent2, admission2, dispatch2, offer2 = make_attempt(
        tmp_path / "over", monkeypatch, "vast-attempt-overfour", ceiling="4.01"
    )
    provider2, controller2 = controller_for(tmp_path / "over", offer2)
    with pytest.raises(vast.CampaignError, match=r"limited to \$4"):
        controller2._create_one_unarmed(intent2, admission2, dispatch2)
    assert provider2.create_calls == 0


def test_multiple_owned_first_cohort_workers_share_the_four_dollar_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = tmp_path / "first"
    intent1, admission1, dispatch1, offer1 = make_attempt(
        first, monkeypatch, "vast-attempt-001", ceiling="2.00", offer_id=12345, machine_id=777
    )
    provider, controller = controller_for(first, offer1)
    controller._create_one_unarmed(intent1, admission1, dispatch1)

    intent2, admission2, dispatch2, offer2 = make_attempt(
        first, monkeypatch, "vast-attempt-002", ceiling="2.00", offer_id=12346, machine_id=778
    )
    provider.offer = offer2
    controller2 = vast.VastCampaign(provider, controller.ledger_path)
    controller2._create_one_unarmed(intent2, admission2, dispatch2)
    assert len(provider.instances) == 2
    assert vast._committed_usd(
        vast._attempts(vast._read_object(controller.ledger_path))
    ) == vast.Decimal("4.00")

    intent3, admission3, dispatch3, offer3 = make_attempt(
        first, monkeypatch, "vast-attempt-003", ceiling="2.00", offer_id=12347, machine_id=779
    )
    provider.offer = offer3
    controller3 = vast.VastCampaign(provider, controller.ledger_path)
    before = provider.create_calls
    with pytest.raises(vast.CampaignError, match="first-cohort reserved and settled spend"):
        controller3._create_one_unarmed(intent3, admission3, dispatch3)
    assert provider.create_calls == before


def test_root_cost_quote_must_cover_admitted_transfer_caps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    intent, admission, dispatch, offer = make_attempt(tmp_path, monkeypatch)
    approval = vast._read_object(admission)
    vast_admission = approval["vast_admission"]
    assert isinstance(vast_admission, dict)
    costs = vast_admission["cost_breakdown_usd"]
    assert isinstance(costs, dict)
    costs["compute_storage_usd"] = "2.00"
    costs["upload_usd"] = "0.00"
    write_json(admission, approval)
    provider, controller = controller_for(tmp_path, offer)

    with pytest.raises(vast.CampaignError, match="maximum upload bytes"):
        controller._create_one_unarmed(intent, admission, dispatch)
    assert provider.create_calls == 0


def test_first_four_plus_evidence_released_sixteen_equals_twenty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = tmp_path / "first"
    intent, admission, dispatch, offer = make_attempt(first, monkeypatch)
    provider, controller = controller_for(first, offer)
    created = controller._create_one_unarmed(intent, admission, dispatch)
    assert (
        collect_valid(controller, first, "vast-attempt-001", monkeypatch)["status"] == "collected"
    )
    instance_id = str(created["instance_id"])
    controller.destroy_owned("vast-attempt-001", instance_id)
    billing = first / "root-billing-receipt.json"
    write_json(
        billing,
        {
            "schema_version": 1,
            "authorized_by": "root_controller",
            "decision": "settle_vast_attempt",
            "attempt_id": "vast-attempt-001",
            "instance_id": instance_id,
            "account_scope_sha256": vast._account_scope({"id": "test-account"}),
            "job_sha256": "b" * 64,
            "billing_source_sha256": "e" * 64,
            "actual_all_in_usd": "4.00",
        },
    )
    assert controller.settle("vast-attempt-001", billing)["status"] == "settled"

    intent2, admission2, dispatch2, _ = make_attempt(
        first,
        monkeypatch,
        "vast-attempt-002",
        stage="evidence_released",
        ceiling="16.00",
        released_usd="16.00",
    )
    second_created = controller._create_one_unarmed(intent2, admission2, dispatch2)
    assert second_created["reserved_all_in_usd"] == "16.00"
    assert vast._committed_usd(
        vast._attempts(vast._read_object(controller.ledger_path))
    ) == vast.Decimal("20.00")


def test_campaign_ceiling_above_twenty_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    intent, admission, dispatch, offer = make_attempt(tmp_path, monkeypatch, total_ceiling="20.01")
    provider, controller = controller_for(tmp_path, offer)
    with pytest.raises(vast.CampaignError, match=r"no greater than \$20"):
        controller._create_one_unarmed(intent, admission, dispatch)
    assert provider.create_calls == 0


def test_worker_ram_limit_preserves_twenty_percent_of_advertised_host_memory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    intent_path, _, _, _ = make_attempt(tmp_path, monkeypatch)
    intent = vast._read_object(intent_path)
    intent["max_worker_host_ram_gb"] = "36.00"
    write_json(intent_path, intent)

    with pytest.raises(vast.CampaignError, match="20% selected-host headroom"):
        vast._validate_intent(intent, intent_path)


def test_no_instance_ambiguous_create_can_settle_only_from_root_absence_and_billing_receipts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    intent, admission, dispatch, offer = make_attempt(tmp_path, monkeypatch)
    provider, controller = controller_for(tmp_path, offer)
    provider.timeout_after_create = True
    with pytest.raises(vast.CampaignError):
        controller._create_one_unarmed(intent, admission, dispatch)
    provider.instances.clear()
    first = controller.reconcile("vast-attempt-001")
    assert first["status"] == "needs_review"
    assert controller.reconcile("vast-attempt-001")["status"] == "needs_review"
    assert provider.create_calls == 1

    receipt = tmp_path / "root-no-instance-billing.json"
    write_json(
        receipt,
        {
            "schema_version": 1,
            "authorized_by": "root_controller",
            "decision": "settle_vast_attempt",
            "attempt_id": "vast-attempt-001",
            "instance_id": None,
            "account_scope_sha256": vast._account_scope({"id": "test-account"}),
            "job_sha256": "b" * 64,
            "provider_absence_verified": True,
            "provider_terminal_confirmed": True,
            "billing_source_sha256": "e" * 64,
            "actual_all_in_usd": "0.00",
        },
    )
    assert controller.settle("vast-attempt-001", receipt)["status"] == "settled"


def test_legacy_approved_status_without_exact_root_action_cannot_create(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    intent, admission, dispatch, offer = make_attempt(tmp_path, monkeypatch)
    approval = vast._read_object(admission)
    approval["status"] = "approved"
    approval["vast_admission"] = {"status": "approved", "scope": "profile-only"}
    write_json(admission, approval)
    provider, controller = controller_for(tmp_path, offer)

    with pytest.raises(vast.CampaignError, match="missing"):
        controller._create_one_unarmed(intent, admission, dispatch)
    assert provider.current_user_calls == 0
    assert provider.create_calls == 0


def test_search_cannot_relax_verified_provider_filters(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, _, _, offer = make_attempt(tmp_path, monkeypatch)
    provider = FakeProvider(offer)
    result = vast.search_verified_offers(
        provider, {"gpu_ram": {"gte": 24}, "allocated_storage": 40, "limit": 50}
    )
    assert len(result) == 1
    assert provider.search_calls[0]["verified"] == {"eq": True}
    assert provider.search_calls[0]["rentable"] == {"eq": True}
    assert provider.search_calls[0]["type"] == "on-demand"
    assert provider.search_calls[0]["allocated_storage"] == 40
    assert provider.search_calls[0]["gpu_ram"] == {"gte": 24576}
    assert provider.search_calls[0]["cpu_ram"] == {"gte": 24576}
    assert provider.search_calls[0]["limit"] == 20
    with pytest.raises(vast.CampaignError, match="may not override"):
        vast.search_verified_offers(provider, {"verified": {"eq": False}})


def test_search_skips_explicitly_deverified_rows_from_live_bundles_response(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, _, _, offer = make_attempt(tmp_path, monkeypatch)

    class ShapedProvider(FakeProvider):
        def search_offers(self, criteria: dict[str, object]) -> list[dict[str, object]]:
            self.search_calls.append(dict(criteria))
            return [
                {**self.offer, "id": 54321, "is_vm_deverified": True},
                {**self.offer, "id": 54322, "is_vm_deverified": False},
            ]

    provider = ShapedProvider(offer)
    results = vast.search_verified_offers(
        provider, {"gpu_ram": {"gte": 24576}, "allocated_storage": 100}
    )

    assert [row["offer"]["id"] for row in results] == [54322]
    assert provider.search_calls[0]["verified"] == {"eq": True}
    assert provider.search_calls[0]["type"] == "on-demand"


def test_offer_memory_uses_live_mib_units_and_preserves_fee_quote_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, _, _, offer = make_attempt(tmp_path, monkeypatch)
    snapshot = vast._offer_snapshot(offer)

    assert snapshot["gpu_ram_gib"] == "24.000"
    assert snapshot["cpu_ram_gib"] == "41.880"
    assert snapshot["inet_down_cost"] == 0.0026041666666666665
    assert snapshot["inet_up_cost"] == 0.00390625
    assert snapshot["storage_cost"] == 0.3666666666666667
    assert snapshot["cpu_cores_effective"] == 10.7
    assert snapshot["cuda_max_good"] == 12.0

    with pytest.raises(vast.CampaignError, match="GPU memory"):
        vast._offer_snapshot({**offer, "gpu_ram": 6144})
    with pytest.raises(vast.CampaignError, match="at least 24 GiB host memory"):
        vast._offer_snapshot({**offer, "cpu_ram": 8192})


def test_unverified_offer_is_rejected_regardless_of_offer_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, _, _, offer = make_attempt(tmp_path, monkeypatch)
    unverified = {**offer, "verification": "unverified"}
    with pytest.raises(vast.CampaignError, match="verified and rentable"):
        vast._offer_snapshot(unverified)


def test_api_redirect_is_refused_without_logging_bearer_or_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class RedirectingOpener:
        def __init__(self) -> None:
            self.calls = 0
            self.request: object | None = None

        def open(self, request: urllib.request.Request, timeout: float) -> Never:
            del timeout
            self.calls += 1
            self.request = request
            raise urllib.error.HTTPError(
                "https://console.vast.ai/api/v0/users/current/",
                302,
                "redirect",
                HTTPMessage(),
                io.BytesIO(b"private response body"),
            )

    opener = RedirectingOpener()
    monkeypatch.setattr(vast.urllib.request, "build_opener", lambda *_handlers: opener)
    api = vast.VastApi("test-bearer-secret")

    with pytest.raises(vast.ApiError) as error:
        api.current_user()
    assert opener.calls == 1
    assert "test-bearer-secret" not in str(error.value)
    assert "private response body" not in str(error.value)
    assert (
        vast.NoRedirectHandler().redirect_request(
            urllib.request.Request("https://console.vast.ai/"),
            io.BytesIO(),
            302,
            "redirect",
            HTTPMessage(),
            "https://other.example/",
        )
        is None
    )


def test_instance_inventory_uses_paginated_official_v1_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pages = [
        {"instances": [{"id": 101, "label": "owned-a"}], "next_token": "page 2/+"},
        {"instances": [{"id": 202, "label": "owned-b"}], "next_token": None},
    ]

    class JsonResponse(io.BytesIO):
        def __init__(self, url: str, body: dict[str, object]) -> None:
            super().__init__(json.dumps(body).encode("utf-8"))
            self.url = url

        def geturl(self) -> str:
            return self.url

        def __enter__(self) -> JsonResponse:
            return self

        def __exit__(self, *_args: object) -> None:
            self.close()

    class PagingOpener:
        def __init__(self) -> None:
            self.requests: list[urllib.request.Request] = []

        def open(self, request: urllib.request.Request, timeout: float) -> JsonResponse:
            assert timeout == 15.0
            self.requests.append(request)
            return JsonResponse(request.full_url, pages[len(self.requests) - 1])

    opener = PagingOpener()
    monkeypatch.setattr(vast.urllib.request, "build_opener", lambda *_handlers: opener)
    api = vast.VastApi("private-test-key")

    assert api.list_instances() == [
        {"id": 101, "label": "owned-a"},
        {"id": 202, "label": "owned-b"},
    ]
    assert len(opener.requests) == 2
    first, second = opener.requests
    for request in opener.requests:
        assert request.get_method() == "GET"
        assert request.full_url.startswith(f"{vast.API_V1_BASE}/instances/?")
        assert request.get_header("Authorization") == "Bearer private-test-key"
    first_query = urllib.parse.parse_qs(urllib.parse.urlsplit(first.full_url).query)
    second_query = urllib.parse.parse_qs(urllib.parse.urlsplit(second.full_url).query)
    assert first_query["select_filters"] == ["{}"]
    assert json.loads(first_query["order_by"][0]) == [{"col": "id", "dir": "asc"}]
    assert first_query["limit"] == ["25"]
    assert "after_token" not in first_query
    assert second_query["after_token"] == ["page 2/+"]
    assert "private-test-key" not in first.full_url


def test_v1_instance_transport_cannot_be_used_for_mutating_routes() -> None:
    api = vast.VastApi("private-test-key")
    with pytest.raises(ValueError, match="restricted to read-only instance inventory"):
        api._request("DELETE", "/instances/123/", api_version=1)
