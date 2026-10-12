from __future__ import annotations

import fcntl
import hashlib
import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path

MAIN = Path("/home/vlad/development/wunder/wnn_connectome_starterpack")
ACTOR = Path("/home/vlad/development/wunder/wnn_connectome_starterpack-clean-t3-profile")
PACKET = MAIN / "research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/ops_continuation_20261011/clean_t3_f0_matched_student_r5_cache_r1/student_fit_r2"
R21 = MAIN / "research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/ops_continuation_20261011/monitor_20261012_1044_r21"
CONTRACT = PACKET / "CONTROL_P1_FIT_CONTRACT_R1.json"
MANIFEST = PACKET / "CONTROL_P1_LAUNCH_MANIFEST_R1.json"
RELEASE = R21 / "ROOT_CONTROL_P1_FIT_RELEASE_R1.json"
BINDING = R21 / "ROOT_CONTROL_P1_FINAL_BINDING_R1.json"
REVIEW = MAIN / "research/campaign_2026-10-12/independent_reviews/clean-t3-control-p1-fit-r21/INDEPENDENT_REVIEW_R1.json"
RECEIPT = ACTOR / "research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/ops_continuation_20261011/clean_t3_f0_matched_student_r5_cache_r1/student_fit_r2/PRELAUNCH_RESOURCE_RECLAIM_R1.json"
RUNTIME = MAIN / "research/campaign_2026-10-06/bandit_continuation/runtime"
RESERVE = 5_500_000_000
ALLOWANCE = 18_000_000_000
MAX_RECLAIM = 3_000_000_000


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def run(argv: list[str]) -> str:
    return subprocess.run(argv, check=True, capture_output=True, text=True).stdout.strip()


def snapshot(cgroup: Path) -> dict[str, object]:
    stat = dict(line.split() for line in (cgroup / "memory.stat").read_text().splitlines())
    events = dict(line.split() for line in (cgroup / "memory.events").read_text().splitlines())
    return {
        "memory_current_bytes": int((cgroup / "memory.current").read_text()),
        "memory_file_bytes": int(stat["file"]),
        "memory_anon_bytes": int(stat["anon"]),
        "memory_swap_current_bytes": int((cgroup / "memory.swap.current").read_text()),
        "memory_events_counts": {key: int(value) for key, value in events.items()},
    }


if RECEIPT.exists():
    raise FileExistsError(RECEIPT)
for path in (CONTRACT, MANIFEST, RELEASE, BINDING, REVIEW):
    if not path.is_file():
        raise FileNotFoundError(path)
contract_sha, manifest_sha = digest(CONTRACT), digest(MANIFEST)
release_sha, review_sha = digest(RELEASE), digest(REVIEW)
contract, manifest = json.loads(CONTRACT.read_text()), json.loads(MANIFEST.read_text())
release, binding, review = (
    json.loads(RELEASE.read_text()), json.loads(BINDING.read_text()), json.loads(REVIEW.read_text())
)
assert contract_sha == "17350c0616debacd3507ad25bc627ba2f7087ddb17e0398cedf7674d696aaf79"
assert manifest_sha == "178e5edc2a60054f0a5291d181359157b655f5ec7aef7e95364a5f8b77fa7c89"
assert binding["fit_authorized"] is True
assert binding["contract_sha256"] == contract_sha and binding["launch_manifest_sha256"] == manifest_sha
assert binding["independent_review_sha256"] == review_sha
assert binding["independent_review_status"] == "PASS_EXACT_PREPARED_P1"
assert binding["release_sha256"] == release_sha
assert release["fit_authorized"] is True and release["contract_sha256"] == contract_sha
assert release["launch_manifest_sha256"] == manifest_sha
assert review.get("status") == "PASS_EXACT_PREPARED_P1"
assert manifest["attempt_id"] == release["attempt_id"]
assert manifest["output_absent_at_creation"] is True
output = MAIN / "artifacts/runs/clean_t3_f0_control_s3_a9_t3r5_r2_p1"
if output.exists():
    raise FileExistsError(output)
# Verify source and small immutable inputs; large NPY/Parquet arrays remain bound by the
# full finite/hash cache-consumer and actual-profile receipts to avoid rereading them.
small_pins = {
    MAIN / "research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/ops_continuation_20261011/clean_t3_fold0_h390_r3_teacher_cohort/source_snapshot/scripts/train_rnn.py": "a2b5440ff4b8e5179db01cc047102ef634522327374fced31ca5fdf8afcd17a6",
    MAIN / "research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/ops_continuation_202611/clean_t3_fold0_h390_r3_teacher_cohort/source_snapshot/scripts/cleanfold_exec.py": "246586bfd4be3a4e3d95ac026860bb835611f09993b75798124c9500455ab",
}
# Wrapper path is validated from the exact packet source field because the wrapper lives in MAIN.
wrapper = Path(contract["source"]["wrapper_path"])
small_pins[wrapper] = contract["source"]["wrapper_sha256"]
small_pins[MAIN / "artifacts/runs/c20261003_combo_s3_f0/model_ep3.pt"] = "d4a1e50b1892b4d9edd7c255908b68073a760c059ca80223e98a63c7e023371e"
small_pins[MAIN / "artifacts/runs/c20260929_prop_f0/model.pt"] = "b243b9d44a71bcaa02589be0a38b19477f448f9da98a02e16495af6730b517a7"
small_pins[MAIN / "research/campaign_2026-09-29/artifacts/splits/fold0.json"] = "09263ad4fe2b18e2b2f3b79f4562678dce801b067b868134ddc79279f9316a3d"
actual_small_pins = {str(path): digest(path) for path in small_pins}
for path, expected in small_pins.items():
    if actual_small_pins[str(path)] != expected:
        raise ValueError(f"small source/checkpoint pin mismatch: {path}")
cache_root = MAIN / "cache/clean_t3_f0_h390_t3r5_a9_r2"
cache_files = {
    cache_root / "main_homogeneous_train.npy": 1_685_600_128,
    cache_root / "main_homogeneous_valid.npy": 224_800_128,
}
cache_stats = {}
for path, expected_bytes in cache_files.items():
    stat = path.stat()
    if stat.st_size != expected_bytes:
        raise ValueError(f"cache artifact size mismatch: {path}: {stat.st_size}")
    cache_stats[str(path)] = {"bytes": stat.st_size, "device": stat.st_dev, "inode": stat.st_ino, "mtime_ns": stat.st_mtime_ns}
metadata = cache_root / "metadata.json"
if digest(metadata) != "1e4c5c4915b643352494558efc17996a11900224231405fa4ab05698f2be3dc8":
    raise ValueError("cache metadata hash mismatch")
props = run(["systemctl", "--user", "show", "wnn-campaign.slice", "--no-pager", "-p", "ControlGroup", "-p", "MemoryCurrent", "-p", "MemoryHigh", "-p", "MemoryMax", "-p", "MemorySwapMax", "-p", "CPUQuotaPerSecUSec"])
parent = dict(line.split("=", 1) for line in props.splitlines() if "=" in line)
cgroup = Path("/sys/fs/cgroup") / parent["ControlGroup"].lstrip("/")
page_size = os.sysconf("SC_PAGE_SIZE")
high = int((cgroup / "memory.high").read_text())
maximum = int((cgroup / "memory.max").read_text())
expected_high = (18_000_000_000 // page_size) * page_size
expected_max = (20_000_000_000 // page_size) * page_size
quota, period = map(int, (cgroup / "cpu.max").read_text().split())
if high != expected_high or maximum != expected_max or (cgroup / "memory.swap.max").read_text().strip() != "0" or quota > 5 * period:
    raise RuntimeError("effective campaign cgroup caps differ from authorized limits")

with (RUNTIME / "gpu.lock").open("a+") as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    compute_apps = run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"])
    gpu_state = run(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader"])
    active = run(["systemctl", "--user", "list-units", "--type=service", "--state=running", "--no-legend", "--no-pager"])
    active_workers = [line for line in active.splitlines() if "wnn-bandit-" in line]
    if compute_apps.strip() or active_workers:
        raise RuntimeError(f"GPU not idle: compute={compute_apps!r}, workers={active_workers!r}")
    before = snapshot(cgroup)
    if before["memory_swap_current_bytes"] != 0:
        raise RuntimeError("campaign slice is using swap")
    needed = max(0, int(before["memory_current_bytes"]) + RESERVE - ALLOWANCE)
    requested = min(MAX_RECLAIM, ((needed + 500_000_000 + page_size - 1) // page_size) * page_size) if needed else 0
    if requested > before["memory_file_bytes"]:
        raise RuntimeError("insufficient charged file cache for bounded reclaim")
    if requested:
        (cgroup / "memory.reclaim").write_text(f"{requested} swappiness=0\n")
    after = snapshot(cgroup)
    host_swap = run(["swapon", "--show", "--bytes", "--noheadings"])
    receipt = {
        "schema": "wnn.clean-t3-control-p1-prelaunch-resource-reclaim.r1",
        "status": "PASS_FRESH_RESOURCE_HEADROOM" if int(after["memory_current_bytes"]) + RESERVE <= ALLOWANCE else "FAILED_RESOURCE_HEADROOM",
        "observed_utc": datetime.now(UTC).isoformat(),
        "attempt_id": manifest["attempt_id"],
        "root_release_path": str(RELEASE),
        "root_release_sha256": release_sha,
        "final_binding_path": str(BINDING),
        "final_binding_sha256": digest(BINDING),
        "contract_sha256": contract_sha,
        "launch_manifest_sha256": manifest_sha,
        "critic_receipt_path": str(REVIEW),
        "critic_receipt_sha256": review_sha,
        "cgroup": str(cgroup),
        "page_size_bytes": page_size,
        "parent_systemd_properties": parent,
        "effective_limits": {"memory_high_bytes": high, "memory_max_bytes": maximum, "memory_swap_max_bytes": 0, "cpu_quota": quota, "cpu_period": period},
        "shared_gpu_lock_acquired": True,
        "gpu_compute_apps": compute_apps,
        "gpu_state": gpu_state,
        "active_durable_workers": active_workers,
        "before": before,
        "after": after,
        "reserve_bytes": RESERVE,
        "working_allowance_bytes": ALLOWANCE,
        "requested_reclaim_bytes": requested,
        "maximum_authorized_reclaim_bytes": MAX_RECLAIM,
        "observed_current_drop_bytes": int(before["memory_current_bytes"]) - int(after["memory_current_bytes"]),
        "gate_current_plus_reserve_bytes": int(after["memory_current_bytes"]) + RESERVE,
        "source_small_input_sha256": actual_small_pins,
        "large_cache_files_stat_only_after_prior_full_finite_hash_validation": cache_stats,
        "cache_metadata_sha256": digest(metadata),
        "prior_cache_consumer_receipt_sha256": contract["inputs"]["consumer_validation"]["sha256"],
        "prior_full_actual_profile_receipt_sha256": contract["inputs"]["profile_terminal"]["sha256"],
        "host_swap_snapshot": host_swap,
        "parent_swap_bytes": int(after["memory_swap_current_bytes"]),
        "memory_events_are_counter_values_not_bytes": True,
        "memory_events_before_after": {"before": before["memory_events_counts"], "after": after["memory_events_counts"]},
        "limits_changed": False,
        "filecache_only_reclaim_swappiness": 0,
        "output_directory_absent": True,
        "full_fit_started": False,
    }
    RECEIPT.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    if receipt["status"] != "PASS_FRESH_RESOURCE_HEADROOM":
        raise RuntimeError("bounded file-cache reclaim did not restore the fit admission gate")
    print(json.dumps(receipt, indent=2, sort_keys=True))
