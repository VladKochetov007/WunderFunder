"""Detach the two independently reviewed exact B256 anchors on their profiled GPUs."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

from wnn.training.sequence_provenance import source_identity, validate_profile_receipt

ROOT = Path("/data/epanov_s/tabdata/runtime/multistage-optuna-20261031-r1")
SOURCE_SHA = "873a3c07fd88d3eec9ed52b2472b8d7f37ef5f488855a3c753db438e4fdee5aa"
REVIEW_SHA = "3b760b5828c4e6a85a8d4befb59dcda194c7b3e9ca7691c215d0eef1ca2d6a7e"
PYTHON = "/data/epanov_s/tabdata/.venv/bin/python"


def digest(path: Path) -> str:
    """Hash the original bytes of a small frozen receipt."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_new(path: Path, value: object) -> None:
    """Publish an immutable new receipt, preserving an identical predecessor."""
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise FileExistsError(path)
        return
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def main() -> None:
    """Verify the independent review and reserve exactly two bounded scientific fits."""
    root, source = ROOT, ROOT / "source-r3"
    review_path = root / "receipts/INDEPENDENT_ACTUAL_PROFILE_REVIEW_R3.json"
    review = json.loads(review_path.read_text())
    if digest(review_path) != REVIEW_SHA or review["status"] != "pass_for_exact_fit_admission":
        raise ValueError("independent review differs from its original frozen bytes")
    if source_identity(source, source / "scripts/train_sequence.py")["sha256"] != SOURCE_SHA:
        raise ValueError("native training source changed")
    old = root / "receipts/b300-inventory-r2.json"
    allocation = json.loads(old.read_text())
    allocation.update(
        schema="wnn.b300-multistage-fit-allocation.v1",
        parent_inventory_sha256=digest(old), max_workers=8,
        host_rss_limit_per_worker_bytes=24 * 2**30,
    )
    for device in allocation["devices"]:
        device.update(profile_allowed=True, fit_allowed=True)
    allocation_path = root / "receipts/b300-fit-allocation-r3.json"
    write_new(allocation_path, allocation)
    deadline = json.loads((root / "receipts/first-gpu-profile-r2.json").read_text())
    wall = int(min(7190, deadline["reserve_start_epoch"] - time.time() - 20))
    if wall <= 0:
        raise RuntimeError("no campaign time remains before collection reserve")
    owners = []
    for index, entry in enumerate(review["batch256_r5_profiles"]):
        family = entry["family"]
        profile_path = root / "profiles" / f"{family}-b256-r5/profile_receipt.json"
        config = root / "requests" / f"{family}-b256-prefetch1-r5.json"
        if digest(profile_path) != entry["native_profile_sha256"]:
            raise ValueError("original native profile changed")
        if digest(config) != entry["config_sha256_raw"]:
            raise ValueError("profiled config changed")
        profile = json.loads(profile_path.read_text())
        gpu = entry["runtime"]["gpu_index"]
        device = allocation["devices"][gpu]
        approved = profile | {
            "approved": True,
            "external_review": {
                "path": str(review_path), "sha256": REVIEW_SHA,
                "policy_sha256": digest(root / "receipts/PROFILE_REVIEW_POLICY_R3.json"),
                "allocation_sha256": digest(allocation_path),
                "native_profile_sha256": digest(profile_path),
            },
        }
        approved_path = profile_path.with_name("approved_profile_receipt_r3.json")
        write_new(approved_path, approved)
        validate_profile_receipt(approved_path, profile["identity"])
        output = root / "runs" / f"anchor-{family}-b256-r5"
        owner_path = root / "receipts" / f"fit-{family}-b256-r5.owner.json"
        if output.exists() or owner_path.exists():
            raise FileExistsError("preserve existing scientific fit; never duplicate it")
        argv = [
            "/usr/bin/timeout", "--signal=TERM", "--kill-after=5s", f"{wall}s", PYTHON,
            str(source / "scripts/train_sequence.py"), "--config", str(config),
            "--data-root", str(root / "data"), "--split-manifest",
            str(root / "data/cache/multistage/auxiliary-r3.json"),
            "--output", str(output), "--profile-receipt", str(approved_path),
        ]
        env = os.environ.copy()
        env.update(
            PYTHONPATH=f"{source / 'src'}:{root / 'python-deps'}",
            CUDA_VISIBLE_DEVICES=device["uuid"], OMP_NUM_THREADS="4",
            MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4",
            WNN_PORTABLE_DATA_MANIFEST=str(root / "inputs/portable-data-r1.json"),
        )
        for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE"):
            env.pop(key, None)
        started = time.time()
        with (root / "logs" / f"fit-{family}-b256-r5.log").open("xb") as log:
            proc = subprocess.Popen(
                argv, cwd=source, env=env, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
            )
        fields = Path(f"/proc/{proc.pid}/stat").read_text().rsplit(")", 1)[1].split()
        owner = {
            "schema": "wnn.multistage-external-worker.v1", "worker_id": f"root-{family}-b256-r5",
            "attempt_id": f"preexisting-stage1-{family}-b256-r5", "mode": "fit", "stage": 1,
            "gpu_key": str(gpu), "pid": proc.pid, "birth": fields[19], "pgid": int(fields[2]),
            "command": argv,
            "command_sha256": hashlib.sha256(json.dumps(argv, separators=(",", ":")).encode()).hexdigest(),
            "output_dir": str(output), "started_at_epoch": started,
            "deadline_epoch": started + wall + 5, "trial_number": -1001 - index,
            "worker_owner_path": str(owner_path), "review_sha256": REVIEW_SHA,
        }
        write_new(owner_path, owner)
        owners.append(owner)
        print(f"FULL_FIT_DETACHED family={family} gpu={gpu} pid={proc.pid}", flush=True)
    write_new(root / "receipts/first-two-full-fits-r1.json", {
        "scientific_fit_attempts": 2, "owners": owners, "status": "launched",
    })


if __name__ == "__main__":
    main()
