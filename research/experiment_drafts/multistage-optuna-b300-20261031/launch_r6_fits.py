"""Detach only the six independently approved R6 base-model fits."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

from launch_first_two_fits import PYTHON, ROOT, digest, write_new

from wnn.training.sequence_provenance import source_identity, validate_profile_receipt

REVIEW_SHA = "20665ddd5092a255452d7d0a7bb506798d15ac377a98016fe64c6422fe8cb5d5"
SOURCE_SHA = "c520c67ca8d1fb49fc0c452a4606f5ad130ed51e7fc689fdaf9daea2b429dd81"


def main() -> None:
    """Consume the exact six-profile admission without mutating source or native receipts."""
    source = ROOT / "operator-native4"
    review_path = ROOT / "receipts/INDEPENDENT_R6_PROFILE_REVIEW.json"
    review = json.loads(review_path.read_text())
    if digest(review_path) != REVIEW_SHA or review["review_status"] != (
        "approved_for_exact_six_stage1_full_fit_profile_identities"
    ):
        raise ValueError("independent exact-fit approval changed")
    if source_identity(source, source / "scripts/train_sequence.py")["sha256"] != SOURCE_SHA:
        raise ValueError("immutable native source changed")
    allocation_path = ROOT / "receipts/b300-fit-allocation-r3.json"
    allocation = json.loads(allocation_path.read_text())
    deadline = json.loads((ROOT / "receipts/first-gpu-profile-r2.json").read_text())
    wall = int(min(7190, deadline["reserve_start_epoch"] - time.time() - 20))
    if wall < 1:
        raise RuntimeError("admission closed before collection reserve")
    owners = []
    for entry in review["profiles"]:
        key, family, gpu = entry["id"], entry["family"], entry["gpu_index"]
        device = allocation["devices"][gpu]
        if device.get("fit_allowed") is not True or device["uuid"] != entry["gpu_uuid"]:
            raise ValueError("candidate GPU differs from the owned allocation")
        profile_path, config = Path(entry["native_profile_path"]), Path(entry["remote_config_path"])
        if digest(profile_path) != entry["native_profile_sha256"]:
            raise ValueError("profile bytes changed")
        if digest(config) != entry["config_raw_sha256"]:
            raise ValueError("candidate config changed")
        profile = json.loads(profile_path.read_text())
        if profile["identity"]["config_sha256"] != entry["native_normalized_config_identity_sha256"]:
            raise ValueError("normalized native config identity changed")
        approved = profile | {"approved": True, "external_review": {
            "path": str(review_path), "sha256": REVIEW_SHA,
            "policy_sha256": digest(ROOT / "receipts/PROFILE_REVIEW_POLICY_R4.json"),
            "native_profile_sha256": digest(profile_path),
            "allocation_sha256": digest(allocation_path),
        }}
        approved_path = profile_path.with_name("approved_profile_receipt_r4.json")
        write_new(approved_path, approved)
        validate_profile_receipt(approved_path, profile["identity"])
        output = ROOT / "runs" / f"r6-{key}"
        owner_path = ROOT / "receipts" / f"fit-r6-{key}.owner.json"
        if output.exists() or owner_path.exists():
            raise FileExistsError("preserve the existing scientific attempt")
        argv = [
            "/usr/bin/timeout", "--signal=TERM", "--kill-after=5s", f"{wall}s", PYTHON,
            str(source / "scripts/train_sequence.py"), "--config", str(config),
            "--data-root", str(ROOT / "data"), "--split-manifest",
            str(ROOT / "data/cache/multistage/auxiliary-r3.json"),
            "--output", str(output), "--profile-receipt", str(approved_path),
        ]
        env = os.environ.copy()
        env.update(
            PYTHONPATH=f"{source / 'src'}:{ROOT / 'python-deps'}",
            CUDA_VISIBLE_DEVICES=entry["gpu_uuid"], OMP_NUM_THREADS="4",
            MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4",
            WNN_PORTABLE_DATA_MANIFEST=str(ROOT / "inputs/portable-data-r1.json"),
        )
        for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE"):
            env.pop(name, None)
        started = time.time()
        with (ROOT / "logs" / f"fit-r6-{key}.log").open("xb") as log:
            proc = subprocess.Popen(argv, cwd=source, env=env, stdin=subprocess.DEVNULL,
                                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        fields = Path(f"/proc/{proc.pid}/stat").read_text().rsplit(")", 1)[1].split()
        owner = {
            "schema": "wnn.multistage-external-worker.v1", "worker_id": f"root-r6-{key}",
            "attempt_id": f"preexisting-stage1-r6-{key}", "mode": "fit", "stage": 1,
            "gpu_key": str(gpu), "pid": proc.pid, "birth": fields[19], "pgid": int(fields[2]),
            "command": argv,
            "command_sha256": hashlib.sha256(json.dumps(argv, separators=(",", ":")).encode()).hexdigest(),
            "output_dir": str(output), "started_at_epoch": started,
            "deadline_epoch": started + wall + 5, "campaign_deadline_epoch": deadline["deadline_epoch"],
            "config_identity_sha256": profile["identity"]["config_sha256"],
            "profile_receipt_path": str(approved_path), "profile_receipt_sha256": digest(approved_path),
            "training_source_sha256": SOURCE_SHA, "worker_owner_path": str(owner_path),
            "review_sha256": REVIEW_SHA, "family": family,
        }
        write_new(owner_path, owner)
        owners.append(owner)
        print(f"FULL_FIT_DETACHED id={key} gpu={gpu} pid={proc.pid}", flush=True)
    write_new(ROOT / "receipts/r6-full-fits-r1.json", {
        "new_scientific_fit_attempts": 6, "total_scientific_fit_attempts": 8,
        "owners": owners, "status": "launched",
    })


if __name__ == "__main__":
    main()
