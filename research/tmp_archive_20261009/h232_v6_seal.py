from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

lineage = Path("research/campaign_2026-10-06/bandit_continuation/continuous_execution_E529/h232_conditional_v6")
packet = lineage / "packet"
manifest = json.loads((packet / "source_archive_manifest.json").read_text())
reviewed: dict[str, str] = {}
packet_names = (
    "profile_worker.py",
    "profile_kernel.py",
    "gpu_memory_domains.py",
    "cache_pressure_recovery.py",
    "verified_digest_handoff.py",
    "prior_input_identity_v2.json",
    "H232_PROFILE_PLAN.json",
    "H232_AMP4096_SUCCESSOR_RECIPE.json",
    "source_archive_manifest.json",
    "source_archive.tar",
    "kernel-metadata.json",
)
for name in packet_names:
    reviewed[name] = hashlib.sha256((packet / name).read_bytes()).hexdigest()
for rel in ("build_h232_bundle.py", "build_h232_source_archive.py", "prepare_h232_v6_packet.py"):
    reviewed[f"builder/{rel}"] = hashlib.sha256((lineage / rel).read_bytes()).hexdigest()
for test_path in sorted((lineage / "tests").glob("test_*.py")):
    reviewed[f"test/{test_path.name}"] = hashlib.sha256(test_path.read_bytes()).hexdigest()
for member in manifest["members"]:
    rel = str(member["path"])
    data = (lineage / "source" / rel).read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != member["sha256"]:
        raise SystemExit(f"source member changed after archive seal: {rel}")
    reviewed[f"source/{rel}"] = digest

plan_path = packet / "H232_PROFILE_PLAN.json"
plan = json.loads(plan_path.read_text())
plan["status"] = "v6-root-reviewed-awaiting-exact-final-hash-confirmation"
plan["next_gate"] = (
    "Root confirms the exact final packet metadata, external review, wrapper/bundle and quota "
    "receipt hashes; then the operator may make exactly one Kaggle push. The fit remains "
    "conditional on this worker's profile gate and the original resource caps."
)
plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")

meta_path = packet / "packet_meta.json"
meta = json.loads(meta_path.read_text())
meta["plan_sha256"] = hashlib.sha256((packet / "H232_PROFILE_PLAN.json").read_bytes()).hexdigest()
current_packet_hashes = meta.get("current_packet_hashes")
if not isinstance(current_packet_hashes, dict):
    raise SystemExit("packet_meta current_packet_hashes is missing")
current_packet_hashes["H232_PROFILE_PLAN.json"] = meta["plan_sha256"]
meta.update(
    {
        "status": "root_approved_conditional_pending_final_hash_confirmation",
        "fit_admission": "same-worker-only-after-profile-gate",
        "provider_push_authorized": True,
        "reviewer_id": "root-controller-20261008",
        "reviewed_hashes": reviewed,
    }
)
# The metadata embeds the review SHA, so its independently reviewable projection omits only that self-reference.
projection = dict(meta)
projection.pop("external_review_sha256", None)
projection_bytes = json.dumps(projection, sort_keys=True, separators=(",", ":")).encode()
projection_sha = hashlib.sha256(projection_bytes).hexdigest()

quota = {
    "source": "read-only Kaggle SDK quota/status calls",
    "log_path": "research/campaign_2026-10-06/cloud_preflight/provider_poll_log.jsonl",
    "line": 1771,
    "observed_at_utc": "2026-10-08T00:42:47+00:00",
    "raw_line_sha256": "dee85c31065cebff9778c0b626db323c5d377324c4cb5cd8a95dd57325799854",
    "provider_gpu_seconds": {
        "used": 52201.386,
        "reserved": 0.0,
        "total": 108000.0,
        "headroom_before_attempt": 55798.614,
        "worker_ceiling": 34000.0,
        "headroom_after_worker": 21798.614,
    },
    "campaign_gpu_seconds": {
        "remaining_before_attempt": 41626.45100000001,
        "worker_ceiling": 34000.0,
        "remaining_after_worker": 7626.451000000011,
    },
    "active_gpu_kernels_in_poll": 0,
    "other_resource_note": "E463 TPU V5 remained QUEUED; it is not a GPU worker/reservation.",
}
review = {
    "schema": "h232-independent-review-v1",
    "experiment_id": "E529_H232_FULLFEATURE_GRU384_B8_AMP4096_CONDITIONAL_PROFILE_FIT_V6",
    "scope": "conditional-profile-gated-fit",
    "status": "approved",
    "conditional_fit_authorized": True,
    "fit_admission": "same-worker-only-after-profile-gate",
    "provider_push_authorized": True,
    "provider_push_scope": "exactly one identified V6 Kaggle provider submission after root confirms final metadata, review, wrapper, and quota receipt hashes",
    "reviewer": {"id": "root-controller-20261008", "role": "independent-controller"},
    "reviewed_utc": datetime.now(timezone.utc).isoformat(),
    "actor": "independent_model_data_forensics (packet operator; not the reviewer)",
    "reviewed_hashes": reviewed,
    "packet_metadata_binding": {
        "file": "packet_meta.json",
        "projection_sha256": projection_sha,
        "projection_definition": "canonical sorted compact JSON of the final packet_meta.json with only external_review_sha256 omitted; the exact raw metadata hash is bound by the separate nonembedded final-admission receipt",
    },
    "transport_binding": {
        "note": "The final wrapper embeds this review, so its exact code/bundle digest cannot be included here without a hash cycle. The separate nonembedded final-admission receipt binds those exact final digests and awaits root hash confirmation.",
        "quota_admission": quota,
    },
    "findings": [
        "Root independently reviewed V6 rowgroup range identity, bounded advice budget/rate, eligibility and post-advice hard-gate recheck; root reran the original 18-test V6 suite successfully. The operator updated a stale pending-review test after the conditional approval seal; the current 19-test suite passes, and its three test-file hashes are included in this review map.",
        "The only V6 source-archive member differing from frozen V5 is src/wnn/data.py, an opt-in rowgroup byte-range receipt hook. The V5 trainer and H232 AMP4096 recipe, target/data treatment and fold-0 membership are unchanged.",
        "V5 ended by SIGTERM at epoch0 batch660/1492 after the cgroup 25% headroom gate was crossed by 19,554,304 bytes; abort-time memory.stat and OOM counters are absent. The cause is unknown; the earlier split-load sample is active-file dominated and would veto relief at that boundary.",
        "V6 is a conditional, evidence-gated cache hint with identity-bound rowgroups, same-cgroup memory.stat evidence, unchanged host/cgroup hard gates, post-advice recheck plus 512MB reserve, bounded event/rate/total budget, and fail-closed fallback. It is not evidence that V5's failure was caused by file cache or that a later cache hint will recover memory.",
        "B8 clock confound: the fixed batch size 8 yields 1,492 outer batches per epoch and 5,968 scheduled optimizer batches over four epochs. V5 reached only 660/1,492 in epoch0, so it produced no complete-epoch or score evidence; comparisons against other batch sizes or differently elapsed fits conflate batch size, optimizer-update exposure and wall time. V6 leaves B8 and the four-epoch schedule unchanged.",
        "The fixed epoch3 valid export is an added delivery stage with explicit FP32 inference, shape/order/hash/finiteness checks, a 1,800-second cap and no metric computation. Root retains grading ownership.",
    ],
    "falsifiers_and_limits": [
        "If V6 reaches the cgroup gate without sufficient inactive_file/stable anon and working-set evidence, it must preserve the original abort; this falsifies the proposed repair for that observed state, not H232 model quality.",
        "POSIX_FADV_DONTNEED is a kernel hint; only the measured post-advice cgroup/host snapshot can justify continuation.",
        "The quota poll is a mutable log; its exact source row SHA256 and relevant values are frozen above. Root must confirm the final nonembedded wrapper/metadata hashes before the single push.",
    ],
    "quota_authority": "one 34,000-second worker is within exact root-reported fresh provider and campaign GPU headroom; no further quota grant, cancellation, rental, or quota mutation is included",
}
review_path = packet / "external_review.json"
review_path.write_text(json.dumps(review, indent=2, sort_keys=True) + "\n")
meta["external_review_sha256"] = hashlib.sha256(review_path.read_bytes()).hexdigest()
meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
print(json.dumps({
    "review_sha256": meta["external_review_sha256"],
    "metadata_projection_sha256": projection_sha,
    "metadata_raw_sha256": hashlib.sha256(meta_path.read_bytes()).hexdigest(),
    "reviewed_hash_count": len(reviewed),
    "source_member_count": len(manifest["members"]),
    "quota_raw_line_sha256": quota["raw_line_sha256"],
    "status": review["status"],
}, indent=2, sort_keys=True))
