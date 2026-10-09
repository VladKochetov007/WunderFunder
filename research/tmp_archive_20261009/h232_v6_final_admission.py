from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

lineage = Path("research/campaign_2026-10-06/bandit_continuation/continuous_execution_E529/h232_conditional_v6")
packet = lineage / "packet"
main = Path("/home/vlad/development/wunder/wnn_connectome_starterpack")
review_path = packet / "external_review.json"
meta_path = packet / "packet_meta.json"
review = json.loads(review_path.read_text())
meta = json.loads(meta_path.read_text())
review_sha = hashlib.sha256(review_path.read_bytes()).hexdigest()
meta_sha = hashlib.sha256(meta_path.read_bytes()).hexdigest()
if meta.get("external_review_sha256") != review_sha:
    raise SystemExit("packet metadata does not bind the final external review")
if meta.get("reviewed_hashes") != review.get("reviewed_hashes"):
    raise SystemExit("packet metadata/review hash maps differ")

# Confirm every non-cyclic review hash against the final isolated files.
for name, digest in review["reviewed_hashes"].items():
    if name.startswith("source/"):
        path = lineage / name
    elif name.startswith("builder/"):
        path = lineage / name.removeprefix("builder/")
    elif name.startswith("test/"):
        path = lineage / "tests" / name.removeprefix("test/")
    else:
        path = packet / name
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise SystemExit(f"sealed review hash mismatch: {name}")

code_path = packet / "h232_profile_kernel_bundle.py"
code = code_path.read_text(encoding="utf-8")
match = re.search(r"^PAYLOAD_ZIP_SHA256 = '([0-9a-f]{64})'$", code, re.MULTILINE)
if match is None:
    raise SystemExit("embedded wrapper payload digest missing")
embedded_zip_sha = match.group(1)

quota_log = main / "research/campaign_2026-10-06/cloud_preflight/provider_poll_log.jsonl"
quota_line_number = 1771
quota_lines = quota_log.read_bytes().splitlines(keepends=True)
raw_quota_line = quota_lines[quota_line_number - 1]
quota_row_sha = hashlib.sha256(raw_quota_line).hexdigest()
expected_quota_sha = "dee85c31065cebff9778c0b626db323c5d377324c4cb5cd8a95dd57325799854"
if quota_row_sha != expected_quota_sha:
    raise SystemExit("root-reported quota row changed before final admission receipt")
quota_row = json.loads(raw_quota_line)
if quota_row.get("observed_at_utc") != "2026-10-08T00:42:47+00:00":
    raise SystemExit("quota admission row timestamp mismatch")
gpu = quota_row["quota"]["gpu"]
if gpu["used_seconds"] != 52201.386 or gpu["reserved_seconds"] != 0.0 or gpu["total_allowed_seconds"] != 108000.0:
    raise SystemExit("quota values differ from root admission message")
provider_headroom = gpu["total_allowed_seconds"] - gpu["used_seconds"] - gpu["reserved_seconds"]
campaign_remaining = gpu["campaign_headroom_seconds"]
worker_ceiling = 34000.0
if provider_headroom < worker_ceiling or campaign_remaining < worker_ceiling:
    raise SystemExit("quota no longer covers the V6 worker ceiling")

payload_names = (
    "H232_PROFILE_PLAN.json",
    "H232_AMP4096_SUCCESSOR_RECIPE.json",
    "cache_pressure_recovery.py",
    "external_review.json",
    "gpu_memory_domains.py",
    "packet_meta.json",
    "profile_kernel.py",
    "profile_worker.py",
    "prior_input_identity_v2.json",
    "source_archive_manifest.json",
    "source_archive.tar",
    "verified_digest_handoff.py",
)
payload_hashes = {name: hashlib.sha256((packet / name).read_bytes()).hexdigest() for name in payload_names}
source_hashes = {
    "train_rnn.py": hashlib.sha256((lineage / "source/scripts/train_rnn.py").read_bytes()).hexdigest(),
    "predict_run.py": hashlib.sha256((lineage / "source/scripts/predict_run.py").read_bytes()).hexdigest(),
    "data.py": hashlib.sha256((lineage / "source/src/wnn/data.py").read_bytes()).hexdigest(),
}
wrapper_sha = hashlib.sha256(code_path.read_bytes()).hexdigest()
kernel_metadata_sha = hashlib.sha256((packet / "kernel-metadata.json").read_bytes()).hexdigest()
source_tar_sha = hashlib.sha256((packet / "source_archive.tar").read_bytes()).hexdigest()
source_manifest_sha = hashlib.sha256((packet / "source_archive_manifest.json").read_bytes()).hexdigest()
review_map = review["reviewed_hashes"]
if review_map.get("source_archive.tar") != source_tar_sha or review_map.get("source_archive_manifest.json") != source_manifest_sha:
    raise SystemExit("source archive is not the root-reviewed source")

admission = {
    "schema": "h232-v6-root-final-admission-receipt-v1",
    "packet_id": meta["packet_id"],
    "status": "root_approved_pending_exact_hash_confirmation_before_push",
    "reviewer": review["reviewer"],
    "review_sha256": review_sha,
    "review_status": review["status"],
    "conditional_fit_authorized": review["conditional_fit_authorized"],
    "provider_push_authorized": review["provider_push_authorized"],
    "push_count_before_receipt": 0,
    "root_hash_confirmation": None,
    "root_confirmation_required_before_push": True,
    "exact_hashes": {
        "packet_meta_sha256": meta_sha,
        "packet_metadata_projection_sha256": review["packet_metadata_binding"]["projection_sha256"],
        "external_review_sha256": review_sha,
        "wrapper_code_file_sha256": wrapper_sha,
        "embedded_bundle_zip_sha256": embedded_zip_sha,
        "kernel_metadata_sha256": kernel_metadata_sha,
        "source_archive_sha256": source_tar_sha,
        "source_archive_manifest_sha256": source_manifest_sha,
        "profile_worker_sha256": review_map["profile_worker.py"],
        "cache_pressure_recovery_sha256": review_map["cache_pressure_recovery.py"],
        "source_hashes": source_hashes,
        "payload_hashes": payload_hashes,
        "reviewed_hash_map_sha256": hashlib.sha256(json.dumps(review_map, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
    },
    "quota_admission": {
        "source": "read-only Kaggle SDK quota/status poll; no reserve mutation",
        "log_path": "research/campaign_2026-10-06/cloud_preflight/provider_poll_log.jsonl",
        "line": quota_line_number,
        "observed_at_utc": quota_row["observed_at_utc"],
        "raw_line_sha256": quota_row_sha,
        "provider_gpu_seconds": {
            "used": gpu["used_seconds"],
            "reserved": gpu["reserved_seconds"],
            "total": gpu["total_allowed_seconds"],
            "headroom_before_worker": provider_headroom,
            "worker_ceiling": worker_ceiling,
            "headroom_after_worker": provider_headroom - worker_ceiling,
        },
        "campaign_gpu_seconds": {
            "headroom_before_worker": campaign_remaining,
            "worker_ceiling": worker_ceiling,
            "headroom_after_worker": campaign_remaining - worker_ceiling,
        },
        "active_gpu_kernels_in_poll": sum(
            1 for item in quota_row.get("status", [])
            if item.get("resource") == "gpu" and "RUNNING" in str(item.get("status"))
        ),
        "other_resource_note": "E463 TPU V5 remained QUEUED; no GPU reservation or active GPU worker appeared in this row.",
        "provider_boot_headroom_note": "Root reports the remaining 21,798.614 provider seconds as generous for boot outside the worker cap.",
    },
    "planned_single_push": {
        "command": "kaggle kernels push -p research/campaign_2026-10-06/bandit_continuation/continuous_execution_E529/h232_conditional_v6/packet -t 34000",
        "target_slug": "vladkochetov07/h232-full-feature-gru384-b8-amp4096-conditional-profile-fit-v6",
        "worker_timeout_seconds": 34000,
        "scope": "one push only; one conditional profile-gated fit plus fixed ep3 FP32 valid export; root owns final grading",
    },
    "validation": {
        "tests": "19 passed in 0.22s under the 1 CPU/1 GB wrapper",
        "ruff": "All checks passed",
        "production_source_and_recipe_unchanged_after_seal": True,
        "v5_v6_source_member_delta": ["src/wnn/data.py"],
        "score_or_grade_performed": False,
    },
}
receipt_path = lineage / "V6_ROOT_FINAL_ADMISSION_RECEIPT_V1.json"
receipt_path.write_text(json.dumps(admission, indent=2, sort_keys=True) + "\n")

repair_path = lineage / "V6_RESOURCE_REPAIR_RECEIPT_V1.json"
repair = json.loads(repair_path.read_text())
repair.update(
    status="root_approved_conditional_pending_exact_hash_confirmation",
    fit_admission="single provider attempt; same-worker profile-gated fit only",
    provider_push_authorized=True,
)
repair["validation"]["test_result"] = "19 passed in 0.22s"
repair["validation"]["ruff_result"] = "All checks passed"
repair["hashes"].update(
    external_review_sha256=review_sha,
    packet_meta_sha256=meta_sha,
    transport_bundle_sha256=embedded_zip_sha,
    transport_code_file_sha256=wrapper_sha,
    kernel_metadata_sha256=kernel_metadata_sha,
)
repair["root_review"] = {
    "reviewer": review["reviewer"],
    "status": review["status"],
    "exact_hash_confirmation_pending": True,
    "admission_receipt": receipt_path.name,
}
repair_path.write_text(json.dumps(repair, indent=2, sort_keys=True) + "\n")
print(json.dumps({
    "receipt_path": str(receipt_path),
    "review_sha256": review_sha,
    "packet_meta_sha256": meta_sha,
    "wrapper_sha256": wrapper_sha,
    "embedded_bundle_zip_sha256": embedded_zip_sha,
    "kernel_metadata_sha256": kernel_metadata_sha,
    "quota_row_sha256": quota_row_sha,
    "quota_headroom_after_worker_provider": provider_headroom - worker_ceiling,
    "quota_headroom_after_worker_campaign": campaign_remaining - worker_ceiling,
}, indent=2, sort_keys=True))
