"""Root-run preparation only: seal authorization and draft exact Vast inputs; no create."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tarfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

ROOT = Path("/home/vlad/development/wunder/wnn_connectome_starterpack")
ATTEMPT = "e529-vast-capacity-muon-f0-20261008-r5"
LABEL = "wnn-e529-capacity-muon-f0-r5-20261008"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet-directory", type=Path, required=True)
    parser.add_argument("--offer", type=Path, required=True, help="Exact fresh single offer object")
    parser.add_argument(
        "--destination",
        type=Path,
        required=True,
        help="New physical directory within main repository",
    )
    parser.add_argument("--account-scope-sha256", required=True)
    parser.add_argument(
        "--signed-plan",
        type=Path,
        required=True,
        help="Fresh private four-input signed plan; URLs never printed",
    )
    parser.add_argument(
        "--authorize-packet",
        action="store_true",
        help="Root explicitly enables embedded fit authority; paid admission remains a draft",
    )
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT))
    from scripts import kaggle_vast_prefetch, vast_campaign

    destination = args.destination.resolve()
    destination.relative_to(ROOT)
    destination.mkdir(parents=True, exist_ok=False)
    packet = args.packet_directory.resolve(strict=True)
    specification = json.loads((packet / "worker_spec.json").read_text())
    for name, expected in specification["packet_files"].items():
        if sha(packet / name) != expected:
            raise ValueError("root preparation packet binding changed")
    write(
        packet / "ROOT_AUTHORIZATION.json",
        {
            "authorized": args.authorize_packet,
            "attempt_id": ATTEMPT,
            "recipe_sha256": sha(packet / "CAPACITY_R5_RECIPE.json"),
            "worker_spec_sha256": sha(packet / "worker_spec.json"),
        },
    )
    archive = destination / "r5_capacity_root_sealed.tar.gz"
    with tarfile.open(archive, "w:gz") as stream:
        for path in sorted(packet.rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                stream.add(path, arcname=str(path.relative_to(packet)), recursive=False)
    plan = destination / "signed_plan.private.json"
    descriptor = os.open(plan, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(args.signed_plan.read_bytes())
    signed = json.loads(plan.read_text())
    binding = [
        {key: row[key] for key in ("remote_path", "sha256", "size_bytes")}
        for row in signed["inputs"]
    ]
    prefetch = destination / "prefetch_admission.json"
    write(
        prefetch,
        {
            "schema_version": 1,
            "attempt_id": ATTEMPT,
            "campaign_label": LABEL,
            "private_plan_sha256": sha(plan),
            "source_binding": binding,
        },
    )
    inputs = [
        {
            "remote_path": name,
            "source_path": name,
            "size_bytes": item["bytes"],
            "source_sha256": item["sha256"],
        }
        for name, item in specification["inputs"].items()
    ]
    offer = json.loads(args.offer.read_text())
    if (
        float(offer["dph_total"]) > 0.35
        or float(offer["cpu_ram"]) * 1048576 < 64_000_000_000
        or float(offer["inet_down"]) < 500
    ):
        raise ValueError("offer outside root requested price/RAM/network bounds")
    intent = {
        "schema_version": 1,
        "attempt_id": ATTEMPT,
        "campaign_label": LABEL,
        "account_scope_sha256": args.account_scope_sha256,
        "packet_path": str(archive.relative_to(ROOT)),
        "packet_sha256": sha(archive),
        "purpose": "Own fresh sequential Muon512/768 fold0 seed9, fixedep3 checkpoints and complete teacher exports; fold0 exposed",
        "offer": offer,
        "create_payload": {
            "client_id": "me",
            "cancel_unavail": True,
            "disk": 120,
            "image": specification["expected_image"],
            "label": LABEL,
            "onstart": "sleep infinity",
            "runtype": "ssh_direc ssh_proxy",
        },
        "worker_inputs": inputs,
        "max_upload_bytes": sum(row["size_bytes"] for row in inputs) + archive.stat().st_size,
        "max_laptop_upload_bytes": 99999999,
        "upload_timeout_seconds": 1800,
        "worker_setup_timeout_seconds": 750,
        "max_worker_host_ram_gb": "28",
        "max_runtime_seconds": 36000,
        "max_artifact_bytes": 5000000000,
        "remote_output_dir": "/workspace/results",
        "expected_artifacts": [
            "artifact_bundle.tar",
            "result.json",
            "worker.log",
            "image_preflight.json",
        ],
        "min_offer_reliability": "0.97",
        "reservation_total_usd": "5.00",
        "reservation_usd": {
            "compute_storage_usd": "4.20",
            "upload_usd": "0.20",
            "download_usd": "0.30",
            "cleanup_usd": "0.30",
        },
    }
    intent_path = destination / "r5_intent.json"
    write(intent_path, intent)
    normalized = vast_campaign._validate_intent(intent, intent_path)
    vast_campaign._validate_staging_plan(normalized, plan, prefetch)
    kaggle_vast_prefetch.prepare_prefetch(
        plan,
        prefetch,
        attempt_id=ATTEMPT,
        campaign_label=LABEL,
        frozen_inputs=inputs,
        runtime_root=destination,
    )
    template = (
        ROOT
        / "research/campaign_2026-10-06/bandit_continuation/vast_warm_cohort/WARM_SMOKE_R4_ROOT_ADMISSION_V1.json"
    )
    approval = json.loads(template.read_text())
    now = datetime.now(UTC)
    approval.update(
        owner="root",
        authorized=False,
        attempt_id=ATTEMPT,
        job_sha256=normalized["intent_sha256"],
        created_utc=now.isoformat(),
        expires_utc=(now + timedelta(seconds=110)).isoformat(),
    )
    approval["approval_basis"] = (
        "DRAFT: root must verify fresh quote, credit, ledger reservation and final sealed startup before admission"
    )
    source_paths = [
        intent_path,
        archive,
        prefetch,
        plan,
        args.offer.resolve(),
        ROOT / "scripts/vast_campaign.py",
        ROOT / "scripts/kaggle_vast_prefetch.py",
        ROOT / "scripts/kaggle_vast_prefetch_remote.py",
        ROOT / "specs/adr/0015-flexible-credit-bounded-parallel-compute.md",
    ]
    approval["source_hashes"] = {str(path.relative_to(ROOT)): sha(path) for path in source_paths}
    vast = approval["vast_admission"]
    for key in (
        "attempt_id",
        "campaign_label",
        "intent_sha256",
        "packet_sha256",
        "request_sha256",
        "account_scope_sha256",
        "offer_snapshot_sha256",
        "offer_id",
        "max_runtime_seconds",
        "max_upload_bytes",
        "max_laptop_upload_bytes",
        "upload_timeout_seconds",
        "worker_inputs_sha256",
        "worker_setup_timeout_seconds",
    ):
        vast[key] = normalized[key]
    vast.update(
        max_all_in_usd="5.00",
        minimum_live_balance_usd="5.00",
        total_campaign_ceiling_usd="20.00",
        max_disk_gb=120,
        max_worker_host_ram_gb=28,
        hard_stop_utc=(now + timedelta(hours=10)).isoformat(),
        cleanup_deadline_utc=(now + timedelta(hours=10, minutes=15)).isoformat(),
        cost_breakdown_usd=intent["reservation_usd"],
    )
    write(destination / "r5_root_admission.DRAFT.json", approval)
    print(
        json.dumps(
            {
                "status": "exact_schema_and_staging_plan_pass_no_provider_mutation",
                "packet_sha256": intent["packet_sha256"],
                "intent_sha256": normalized["intent_sha256"],
                "destination": str(destination),
                "root_admission_authorized": False,
            }
        )
    )


if __name__ == "__main__":
    main()
