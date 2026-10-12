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
PROFILE_DIR = (
    ACTOR
    / "research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/ops_continuation_20261011/clean_t3_f0_matched_student_r5_cache_r1/student_resource_profile_r2"
)
OPS = (
    MAIN
    / "research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/ops_continuation_20261011"
)
STATE = MAIN / "research/campaign_2026-10-06/bandit_continuation/runtime"
RECEIPT = PROFILE_DIR / "PROFILE_PRELAUNCH_RECLAIM_R3.json"
RESERVE = 5_500_000_000
ALLOWANCE = 18_000_000_000
MAX_RECLAIM = 3_000_000_000
EXPECTED = {
    "root_release": "cad0783993e16b1bed95f1bd6c64423f86346700424aa4ea0310f1b015f3d666",
    "contract": "cae424ce880651db4b618a757ae9f2979fca40a45b28273e3d2b413b32ed2e76",
    "launch_manifest": "02481fe96b3b968f8e145e6f26a94d3ad63ab02cedd929a83f9b013366f22132",
    "review": "284c92acb71df5db3c77d77e87bc63043700fee46320b00da270f43e17838f21",
    "cache_validation": "90528f221fdca114f506871fed06372bb7580a9d2700632abd156aa03830bedf",
    "train_cache": "6a19f7ac45c6e8a02d3b89c0ac4986ffab4885ab7a4ebc5fb4599b3047f4e7b3",
    "valid_cache": "d1cd78537f3b6aaa830861ac771cf5c4621f1f2c1f19a73f6affb813c33ca3a2",
    "cache_metadata": "1e4c5c4915b643352494558efc17996a11900224231405fa4ab05698f2be3dc8",
    "trainer": "a2b5440ff4b8e5179db01cc047102ef634522327374fced31ca5fdf8afcd17a6",
    "wrapper": "246586bfd4be3a4e3d95ac026860bb835611f09958193b75798124c9500455ab",
    "source_manifest": "41a5452c752bb10bee28af2eb2e06e7a224da8d72f307b4d5bec6fb5b3b236fa",
    "source_delta": "912205eba101c8539c372206e5ae0e56f2d0a39f8997b271189b3d2f7d601406",
    "parent_checkpoint": "d4a1e50b1892b4d9edd7c255908b68073a760c059ca80223e98a63c7e023371e",
    "parent_provenance": "9a9d17eeb4b844d85bb5ca132e0085d0cf6dd1387d3d3de4c56d8e3bb8be7a6f",
    "propensity_checkpoint": "b243b9d44a71bcaa02589be0a38b19477f448f9da98a02e16495af6730b517a7",
    "propensity_train": "811ce4328be739b039bc62ba1f696386e59b168fbe3733be81783a41417da174",
    "propensity_valid": "9f3e09f04b9fadfd142d4ecbaba8b4d17d6d1ae04ea40e2cb8d4b64ecd0a718c",
    "split_manifest": "09263ad4fe2b18e2b2f3b79f4562678dce801b067b868134ddc79279f9316a3d",
    "recipe": "8514e8a88bfecbc2cae9b8950cc88f2a1354f21b36a49dc9a3cd16b634598b32",
    "adapter_contract": "c04fc22b78be9869c4e447316a5bd84a100bd4ef469bb244ce6f591974d9c033",
    "adapter_script": "2e486fd26bacfcdf3cf752c765f8e9b7815027f3a0f451586ca1c209c87b21f2",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def command(args: list[str]) -> str:
    return subprocess.run(args, check=True, capture_output=True, text=True).stdout.strip()


def unit_properties() -> dict[str, str]:
    output = command(
        [
            "systemctl",
            "--user",
            "show",
            "wnn-campaign.slice",
            "--no-pager",
            "-p",
            "ControlGroup",
            "-p",
            "MemoryCurrent",
            "-p",
            "MemoryHigh",
            "-p",
            "MemoryMax",
            "-p",
            "MemorySwapMax",
            "-p",
            "CPUQuotaPerSecUSec",
        ]
    )
    return dict(line.split("=", 1) for line in output.splitlines() if "=" in line)


def snapshot(cgroup: Path) -> dict[str, object]:
    stats = dict(line.split() for line in (cgroup / "memory.stat").read_text().splitlines())
    events = dict(line.split() for line in (cgroup / "memory.events").read_text().splitlines())
    return {
        "current": int((cgroup / "memory.current").read_text()),
        "file": int(stats["file"]),
        "anon": int(stats["anon"]),
        "swap": int((cgroup / "memory.swap.current").read_text()),
        "events": {key: int(value) for key, value in events.items()},
    }


def main() -> None:
    if RECEIPT.exists():
        raise FileExistsError(RECEIPT)
    release = OPS / "monitor_20261012_1001_r20/ROOT_LOCAL_STUDENT_PROFILE_RELEASE_R2.json"
    contract = PROFILE_DIR / "PROFILE_CONTRACT_R2.json"
    manifest = PROFILE_DIR / "PROFILE_LAUNCH_MANIFEST_R2.json"
    review = OPS / "clean_t3_f0_matched_student_r5_cache_r2_review/INDEPENDENT_REVIEW_R2.json"
    validation = (
        PROFILE_DIR.parent / "cache_adapter_r2/CACHE_METADATA_COMPATIBILITY_VALIDATION_R2.json"
    )
    cache_root = MAIN / "cache/clean_t3_f0_h390_t3r5_a9_r2"
    source_root = OPS / "clean_t3_fold0_h390_r3_teacher_cohort"
    files = {
        "root_release": release,
        "contract": contract,
        "launch_manifest": manifest,
        "review": review,
        "cache_validation": validation,
        "train_cache": cache_root / "main_homogeneous_train.npy",
        "valid_cache": cache_root / "main_homogeneous_valid.npy",
        "cache_metadata": cache_root / "metadata.json",
        "trainer": source_root / "source_snapshot/scripts/train_rnn.py",
        "wrapper": source_root / "source_snapshot/scripts/cleanfold_exec.py",
        "source_manifest": source_root / "source_snapshot/SOURCE_MANIFEST_H390.json",
        "source_delta": source_root / "SOURCE_DELTA_R3.json",
        "parent_checkpoint": MAIN / "artifacts/runs/c20261003_combo_s3_f0/model_ep3.pt",
        "parent_provenance": MAIN / "artifacts/runs/c20261003_combo_s3_f0/provenance.json",
        "propensity_checkpoint": MAIN / "artifacts/runs/c20260929_prop_f0/model.pt",
        "propensity_train": MAIN / "cache/propcache_c20260929_prop_f0_train.npy",
        "propensity_valid": MAIN / "cache/propcache_c20260929_prop_f0_valid.npy",
        "split_manifest": MAIN / "research/campaign_2026-09-29/artifacts/splits/fold0.json",
        "recipe": PROFILE_DIR.parent / "STUDENT_RECIPE_PENDING_CACHE_R1.json",
        "adapter_contract": PROFILE_DIR.parent
        / "cache_adapter_r2/CACHE_METADATA_COMPATIBILITY_ADAPTER_R2.json",
        "adapter_script": PROFILE_DIR.parent
        / "cache_adapter_r2/build_cache_metadata_compatibility_r2.py",
    }
    hashes = {key: sha256(path) for key, path in files.items()}
    mismatches = {
        key: (EXPECTED[key], value) for key, value in hashes.items() if EXPECTED[key] != value
    }
    if mismatches:
        raise ValueError(f"pinned input hash mismatch: {mismatches}")
    release_data = json.loads(release.read_text())
    if (
        not release_data.get("profile_authorized")
        or release_data.get("gpu_fit_authorized")
        or release_data.get("profile_contract_sha256") != EXPECTED["contract"]
        or release_data.get("launch_manifest_sha256") != EXPECTED["launch_manifest"]
        or release_data.get("independent_review_sha256") != EXPECTED["review"]
    ):
        raise ValueError("R2 root release does not bind this reviewed profile")
    contract_data = json.loads(contract.read_text())
    manifest_data = json.loads(manifest.read_text())
    if (
        manifest_data.get("attempt_id") != contract_data.get("profile_attempt_id")
        or manifest_data.get("profile_contract_sha256") != EXPECTED["contract"]
        or contract_data.get("student_fit_authorized")
        or not contract_data.get("profile_only")
    ):
        raise ValueError("R2 contract/manifest scope mismatch")
    output_path = Path(contract_data["profile_geometry"]["expected_profile_output_directory"])
    if output_path.exists():
        raise FileExistsError(f"profile output already exists: {output_path}")

    receipt: dict[str, object] = {
        "schema": "wnn.clean-t3-f0-profile-prelaunch-reclaim.r3",
        "observed_utc": datetime.now(UTC).isoformat(),
        "status": "FAILED_PREFLIGHT",
        "pinned_sha256": hashes,
        "output_absent": True,
        "root_release_path": str(release),
        "root_release_sha256": hashes["root_release"],
        "contract_sha256": hashes["contract"],
        "launch_manifest_sha256": hashes["launch_manifest"],
        "review_sha256": hashes["review"],
        "memory_reclaim_authorized_max_bytes": MAX_RECLAIM,
        "reserve_bytes": RESERVE,
        "working_allowance_bytes": ALLOWANCE,
        "limits_changed": False,
    }
    runtime = unit_properties()
    receipt["parent_systemd_properties"] = runtime
    cgroup = Path("/sys/fs/cgroup") / runtime["ControlGroup"].lstrip("/")
    page_size = os.sysconf("SC_PAGE_SIZE")
    high = int((cgroup / "memory.high").read_text())
    maximum = int((cgroup / "memory.max").read_text())
    expected_high = (18_000_000_000 // page_size) * page_size
    expected_max = (20_000_000_000 // page_size) * page_size
    cpu_quota, cpu_period = map(int, (cgroup / "cpu.max").read_text().split())
    if (
        high != expected_high
        or maximum != expected_max
        or (cgroup / "memory.swap.max").read_text().strip() != "0"
        or cpu_quota > 5 * cpu_period
    ):
        raise ValueError("campaign slice limits differ from authorized caps")
    receipt["page_size_bytes"] = page_size
    receipt["effective_limits"] = {
        "memory_high_bytes": high,
        "memory_max_bytes": maximum,
        "memory_swap_max_bytes": 0,
        "cpu_quota": cpu_quota,
        "cpu_period": cpu_period,
    }

    with (STATE / "gpu.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        compute_apps = command(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name,used_memory",
                "--format=csv,noheader",
            ]
        )
        gpu_usage = command(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,utilization.gpu",
                "--format=csv,noheader",
            ]
        )
        active = command(
            [
                "systemctl",
                "--user",
                "list-units",
                "--type=service",
                "--state=running",
                "--no-legend",
                "--no-pager",
            ]
        )
        active_workers = [line for line in active.splitlines() if "wnn-bandit-" in line]
        if compute_apps.strip() or active_workers:
            raise RuntimeError(
                f"GPU not idle: compute_apps={compute_apps!r}, workers={active_workers}"
            )
        before = snapshot(cgroup)
        if before["swap"] != 0:
            raise RuntimeError("campaign swap usage is nonzero")
        needed = max(0, int(before["current"]) + RESERVE - ALLOWANCE)
        requested = min(
            MAX_RECLAIM, ((needed + 500_000_000 + page_size - 1) // page_size) * page_size
        )
        if requested > int(before["file"]):
            raise RuntimeError("charged file cache is below the required bounded reclaim")
        if requested:
            (cgroup / "memory.reclaim").write_text(f"{requested} swappiness=0\n")
        after = snapshot(cgroup)
        receipt.update(
            status="PASS_FRESH_RESOURCE_HEADROOM"
            if int(after["current"]) + RESERVE <= ALLOWANCE
            else "FAILED_RESOURCE_HEADROOM",
            gpu_memory_and_utilization=gpu_usage,
            nvidia_compute_processes=compute_apps,
            active_durable_workers=active_workers,
            shared_gpu_lock_acquired=True,
            before=before,
            after=after,
            requested_file_reclaim_bytes=requested,
            observed_memory_drop_bytes=int(before["current"]) - int(after["current"]),
            gate_current_plus_reserve_bytes=int(after["current"]) + RESERVE,
        )
        RECEIPT.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        if receipt["status"] != "PASS_FRESH_RESOURCE_HEADROOM":
            raise RuntimeError("bounded file-cache reclaim did not restore the aggregate gate")


if __name__ == "__main__":
    main()
