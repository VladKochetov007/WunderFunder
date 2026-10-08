"""Verify final r5 startup and current-ledger admission without provider mutations."""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import sys
import tarfile
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))


class MockCreateReached(BaseException):
    """The admission state machine reached the mocked mutation boundary."""


def write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def main() -> None:
    from scripts import vast_campaign as vast

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sealed", type=Path, required=True)
    args = parser.parse_args()
    sealed = args.sealed.resolve(strict=True)
    sealed.relative_to(ROOT)
    intent_path = sealed / "r5_intent.json"
    intent = json.loads(intent_path.read_text())
    normalized = vast._validate_intent(intent, intent_path)
    vast._validate_staging_plan(
        normalized, sealed / "signed_plan.private.json", sealed / "prefetch_admission.json"
    )
    packet_path = Path(normalized["packet_path"])
    unpacked = vast._validate_worker_bundle(packet_path, normalized["max_upload_bytes"])
    with tempfile.TemporaryDirectory(prefix="r5-root-startup-") as temporary:
        packet = Path(temporary) / "packet"
        with tarfile.open(packet_path) as archive:
            archive.extractall(packet, filter="data")
        sys.path.insert(0, str(packet))
        os.environ["WNN_ATTEMPT_ID"] = intent["attempt_id"]
        import capacity_fit_worker
        import capacity_worker
        import worker

        worker.PACKET = packet
        spec = worker.verify_authority()
        source_hashes = capacity_worker._safe_extract_source(packet, Path(temporary) / "source")
        producer = capacity_fit_worker._verify_profile_evidence(packet)
        assert set(producer["profiles"]) == {"muon512", "muon768"}

    api = vast.VastApi(vast.load_api_key(ROOT / ".env"))
    raw_user = api.current_user()
    instances = api.list_instances()
    assert not instances, "Unexpected live instance: refresh root allocation before admission"
    assert vast._account_scope(raw_user) == normalized["account_scope_sha256"]
    assert vast._balance(raw_user) >= 5
    offers = api.search_offer_for_revalidation(intent["offer"], 120)
    live_offer = next(
        vast._offer_snapshot(row, query_type="on-demand")
        for row in offers
        if int(row["id"]) == int(intent["offer"]["id"])
    )
    allowed_differences = vast._compare_revalidated_offer(
        intent["offer"],
        live_offer,
        reliability_floor=normalized["min_offer_reliability"],
    )
    cloud_bytes = 37_102_250_127
    laptop_bytes = packet_path.stat().st_size + 1_200_214
    assert normalized["worker_setup_timeout_seconds"] == 750
    staging_seconds = (
        laptop_bytes / 4000
        + cloud_bytes * 8 / (float(live_offer["inet_down"]) * 1e6) * 1.35
        + 750
        + 180
    )
    assert laptop_bytes < 100_000_000 and staging_seconds < 1800
    now = datetime.now(UTC)
    approval = json.loads((sealed / "r5_root_admission.DRAFT.json").read_text())
    approval.update(
        authorized=True,
        created_utc=now.isoformat(),
        expires_utc=(now + timedelta(seconds=110)).isoformat(),
    )
    approval["vast_admission"].update(
        hard_stop_utc=(now + timedelta(hours=10)).isoformat(),
        cleanup_deadline_utc=(now + timedelta(hours=10, minutes=15)).isoformat(),
    )
    dispatch = {
        "attempt_id": intent["attempt_id"],
        "lane": "vast",
        "status": "launch_uncertain",
        "job_sha256": approval["job_sha256"],
        "created_utc": now.isoformat(),
    }

    class MockProvider:
        def current_user(self):
            return raw_user

        def list_instances(self):
            return instances

        def search_offer_for_revalidation(self, snapshot, disk):
            return [live_offer]

        def create_instance(self, offer_id, payload):
            raise MockCreateReached

    ledger = Path("/home/vlad/.local/state/wnn-campaign/vast/ledger.json")
    ledger_sha = vast.file_sha256(ledger)
    current = json.loads(ledger.read_text())
    committed = str(vast._committed_usd(vast._attempts(current)))
    checks = {}
    audit_root = sealed / "ledger-audit"
    audit_root.mkdir(mode=0o700, exist_ok=True)
    for case in ("parallel_authorized", "first_cohort_rejected", "campaign_ceiling_rejected"):
        folder = audit_root / case
        folder.mkdir(exist_ok=True)
        private_ledger = folder / "ledger.private.json"
        shutil.copyfile(ledger, private_ledger)
        os.chmod(private_ledger, 0o600)
        test_approval = copy.deepcopy(approval)
        if case == "first_cohort_rejected":
            test_approval["vast_admission"]["cohort_stage"] = "first_cohort"
        if case == "campaign_ceiling_rejected":
            saturated = copy.deepcopy(current)
            vast._attempts(saturated)["mock_other_reservation"] = {
                "status": "launch_uncertain",
                "reserved_all_in_usd": "18.00",
                "cohort_stage": "parallel_authorized",
            }
            write(private_ledger, saturated)
        admission_path = folder / "admission.TEST_ONLY.json"
        dispatch_path = folder / "attempts" / f"{intent['attempt_id']}.json"
        write(admission_path, test_approval)
        write(dispatch_path, dispatch)
        controller = vast.VastCampaign(MockProvider(), private_ledger)
        try:
            controller._create_one_unarmed(intent_path, admission_path, dispatch_path)
        except MockCreateReached:
            assert case == "parallel_authorized"
            checks[case] = "passed: exact current-ledger reservation reached mocked create boundary"
        except vast.CampaignError as error:
            expected = "$4" if case == "first_cohort_rejected" else "$20 cumulative"
            assert case != "parallel_authorized" and expected in str(error), str(error)
            checks[case] = str(error)
        else:
            raise AssertionError("Admission check produced no expected result")
    assert vast.file_sha256(ledger) == ledger_sha, "Actual ledger changed during read-only audit"
    result = {
        "status": "final_sealed_startup_and_private_ledger_admission_pass",
        "created_utc": datetime.now(UTC).isoformat(),
        "packet_sha256": vast.file_sha256(packet_path),
        "intent_sha256": normalized["intent_sha256"],
        "packet_member_count": len(spec["packet_files"]),
        "source_member_count": len(source_hashes),
        "expanded_packet_bytes": unpacked,
        "embedded_authority": "exact attempt/recipe/spec and all source members passed",
        "actual_v4_producer_profiles": "passed",
        "current_credit_usd": str(vast._balance(raw_user)),
        "current_committed_usd": committed,
        "reservation_usd": "5.00",
        "private_ledger_checks": checks,
        "provider_mutations": 0,
        "actual_ledger_unchanged_sha256": ledger_sha,
        "fresh_offer_id": live_offer["id"],
        "fresh_offer_reliability_differences": allowed_differences,
        "laptop_closure_bytes": laptop_bytes,
        "conservative_staging_seconds": staging_seconds,
        "startup_deadline_seconds": 1800,
        "staging_model": "Historical 4000B/s laptop; advertised remote download with35% margin;750s setup/terminal reserve+180sSSH/hash. Actual throughput remains unproven.",
        "collection_timeout_seconds": 1800,
        "deployment_adaptations": [
            "Vast network permitted for four signed inputs and450s pinned dependencies",
            "10h root hardstop supersedes prior Kaggle26000s runtime",
            "Actual3090 nativeBF16 policy is profiled before fresh weights fit",
        ],
    }
    path = Path(__file__).with_name("R5_FINAL_STARTUP_LEDGER_CHECK_R1.json")
    write(path, result)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
