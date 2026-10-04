from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
CAMPAIGN = "research/campaign_2026-10-04"
OLD = f"{CAMPAIGN}/E462_public_q8_build_v2"
PARITY = f"{CAMPAIGN}/E462_public_q8_parity_v2"
CASES = {
    "q8r": (
        f"{CAMPAIGN}/E462_python311_repair_v1/q8r/E462_q8r_python311_repaired_v1.zip",
        f"{PARITY}/q8r_v3.npz",
        ("v2", "v3"),
    ),
    "q8x": (
        f"{CAMPAIGN}/E462_python311_repair_v1/q8x/E462_q8x_python311_repaired_v1.zip",
        f"{PARITY}/q8x_v3.npz",
        ("v2", "v3"),
    ),
    "incumbent": (
        "submissions/c20261004_combo_s3_i16/submission.zip",
        f"{PARITY}/i16_v3.npz",
        ("v2", "avx2", "v3"),
    ),
    "head": (
        "submissions/c20261004_combo_head_s3_i16/c20261004_combo_head_s3_i16.zip",
        f"{CAMPAIGN}/E457_head_transfer/head/i16_v3.npz",
        ("v2", "avx2", "v3"),
    ),
    "parent": (
        "submissions/c20261004_combo_parent_s3_i16/c20261004_combo_parent_s3_i16.zip",
        f"{CAMPAIGN}/E457_head_transfer/parent/i16_v3.npz",
        ("v2", "avx2", "v3"),
    ),
    "seed4": (
        "submissions/c20261004_combo_s4_i16/c20261004_combo_s4_i16.zip",
        f"{CAMPAIGN}/checks/e450_seed4_native_parity_v1/i16_v3.npz",
        ("v2", "avx2", "v3"),
    ),
}


def check_container(label: str, level: str, archive: str, reference: str) -> dict[str, object]:
    """Run one fresh offline interpreter, recording actual Docker exit/OOM state."""
    name = f"wnn-e462-abi-repair-20261004-{label}-{level}"
    command = [
        "docker",
        "run",
        "--name",
        name,
        "--network",
        "none",
        "--cpus",
        "1",
        "--memory",
        "1g",
        "--memory-swap",
        "1g",
        "--pids-limit",
        "64",
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "-v",
        f"{ROOT}:/workspace:ro",
        "-v",
        f"{OUT / 'checks'}:/results",
        "wnn-python311-submission-check",
        "python",
        "/workspace/scripts/check_submission_python311.py",
        "--zip",
        f"/workspace/{archive}",
        "--data",
        "/workspace/cache/valid_subset20.parquet",
        "--level",
        level,
        "--reference",
        f"/workspace/{reference}",
        "--output",
        f"/results/{label}-{level}.json",
    ]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=240)
    log = completed.stdout + completed.stderr
    (OUT / "checks" / f"{label}-{level}.log").write_text(log)
    inspected = subprocess.run(
        ["docker", "inspect", name], capture_output=True, text=True, check=True
    )
    container = json.loads(inspected.stdout)[0]
    (OUT / "checks" / f"{label}-{level}-docker.json").write_text(
        json.dumps(
            {
                "state": container["State"],
                "image": container["Image"],
                "network": container["HostConfig"]["NetworkMode"],
                "memory": container["HostConfig"]["Memory"],
                "memory_swap": container["HostConfig"]["MemorySwap"],
                "nano_cpus": container["HostConfig"]["NanoCpus"],
                "devices": container["HostConfig"]["DeviceRequests"],
                "command": command,
            },
            indent=2,
        )
        + "\n"
    )
    subprocess.run(["docker", "rm", name], capture_output=True, text=True, check=True)
    return {
        "exit_code": completed.returncode,
        "oom_killed": container["State"]["OOMKilled"],
        "log": log,
    }


def main() -> None:
    """Reject old artifacts, then verify all offered ZIPs from their frozen bytes."""
    (OUT / "checks").mkdir(exist_ok=False)
    summary = {
        "scope": "packaging compatibility repair; no fits, kernels or runtime trials",
        "cases": {},
    }
    original = f"{OLD}/q8r/E462_public_q8_build_v2q8r.zip"
    negative = check_container("original-q8r", "v2", original, f"{PARITY}/q8r_v3.npz")
    if negative["exit_code"] == 0 or "libpython3.13.so.1.0" not in negative["log"]:
        raise RuntimeError(f"failed to reproduce original missing-library error: {negative}")
    summary["original_q8r_rejected"] = True
    for label, (archive, reference, levels) in CASES.items():
        static = subprocess.run(
            [
                "python3",
                str(ROOT / "scripts/check_submission_abi.py"),
                str(ROOT / archive),
                "--output",
                str(OUT / "checks" / f"{label}-static.json"),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        del static
        receipts = {}
        for level in levels:
            result = check_container(label, level, archive, reference)
            if result["exit_code"] != 0 or result["oom_killed"]:
                raise RuntimeError(
                    f"{label}/{level} compatibility failed; inspect its persisted log"
                )
            receipts[level] = json.loads((OUT / "checks" / f"{label}-{level}.json").read_text())
        summary["cases"][label] = {
            "zip": archive,
            "sha256": hashlib.sha256((ROOT / archive).read_bytes()).hexdigest(),
            "levels": receipts,
        }
        print(json.dumps({"completed": label, "levels": list(receipts)}), flush=True)
    summary["passed"] = True
    summary["repaired_q8_manual_ready"] = False
    summary["repaired_q8_remaining_gate"] = "competition runtime verification remains paused"
    (OUT / "compatibility_report.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
