"""Launch the two exact independently reviewed lower-LR continuations."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

from launch_first_two_fits import PYTHON, ROOT, digest, write_new

from wnn.training.sequence_provenance import source_identity, validate_profile_receipt

REVIEW_SHA = "73924297b520ae234ad3c45aa1d8fc5faaf6a3bd2388e10709cdca5cb9a9bb63"
SOURCE_SHA = "2661a89285bd528e1bca690a340e71a21816a7b54e3ba1635d14925f3a9c1e4e"


def main() -> None:
    """Consume actual parent/profile bindings and detach two exit-recording guardians."""
    source = ROOT / "source5b-core"
    review_path = ROOT / "receipts/INDEPENDENT_STAGE2_PROFILE_REVIEW_R1.json"
    review = json.loads(review_path.read_text())
    if digest(review_path) != REVIEW_SHA or review["review_status"] != (
        "approved_for_exact_two_stage2_full_fit_profile_identities"
    ):
        raise ValueError("Stage2 approval changed")
    if source_identity(source, source / "scripts/train_sequence.py")["sha256"] != SOURCE_SHA:
        raise ValueError("Stage2 source changed")
    deadline = json.loads((ROOT / "receipts/first-gpu-profile-r2.json").read_text())
    wall = int(min(7190, deadline["reserve_start_epoch"] - time.time() - 20))
    if wall < 1:
        raise RuntimeError("campaign admission closed")
    owners = []
    for index, entry in enumerate(review["profiles"]):
        family, gpu = entry["family"], entry["gpu_index"]
        config = Path(entry["config_remote_path"])
        profile_path = Path(entry["native_profile_path"])
        parent = Path(entry["parent"]["checkpoint_path"])
        if digest(config) != entry["config_raw_sha256"]:
            raise ValueError("Stage2 config changed")
        if digest(profile_path) != entry["native_profile_sha256"]:
            raise ValueError("Stage2 profile changed")
        if digest(parent) != entry["initializer_checkpoint_sha256"]:
            raise ValueError("parent checkpoint changed")
        profile = json.loads(profile_path.read_text())
        approved = profile | {"approved": True, "external_review": {
            "path": str(review_path), "sha256": REVIEW_SHA,
            "policy_sha256": digest(ROOT / "receipts/PROFILE_REVIEW_POLICY_R5.json"),
            "native_profile_sha256": digest(profile_path),
        }}
        approved_path = profile_path.with_name("approved_profile_receipt_stage2_r1.json")
        write_new(approved_path, approved)
        validate_profile_receipt(approved_path, profile["identity"])
        output = ROOT / "runs" / f"stage2-{family}-polish-r1"
        owner_path = ROOT / "receipts" / f"fit-stage2-{family}-polish-r1.owner.json"
        if output.exists() or owner_path.exists():
            raise FileExistsError("preserve existing Stage2 fit")
        child_command = [
            "/usr/bin/timeout", "--signal=TERM", "--kill-after=5s", f"{wall}s", PYTHON,
            str(source / "scripts/train_sequence.py"), "--config", str(config),
            "--data-root", str(ROOT / "data"), "--split-manifest",
            str(ROOT / "data/cache/multistage/auxiliary-r3.json"),
            "--output", str(output), "--profile-receipt", str(approved_path),
            "--initializer", str(parent),
        ]
        request_path = ROOT / "requests" / f"stage2-{family}-fit-r1.json"
        write_new(request_path, {
            "command": child_command, "source_root": str(source), "stage": 2,
            "trial_number": -2001 - index, "gpu_key": str(gpu),
            "output_dir": str(output), "timeout_seconds": wall + 5,
            "log_path": str(ROOT / "logs" / f"fit-stage2-{family}-polish-r1.log"),
            "child_receipt_path": str(ROOT / "receipts" / f"fit-stage2-{family}.child.json"),
            "review_sha256": REVIEW_SHA, "initializer_sha256": digest(parent),
        })
        argv = [PYTHON, str(ROOT / "requests/fit_guardian.py"), "--request", str(request_path)]
        env = os.environ.copy()
        env.update(
            PYTHONPATH=f"{source / 'src'}:{ROOT / 'python-deps'}",
            CUDA_VISIBLE_DEVICES=entry["gpu_uuid"], OMP_NUM_THREADS="4",
            MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4",
            WNN_PORTABLE_DATA_MANIFEST=str(ROOT / "inputs/portable-data-r1.json"),
            WNN_PROJECT_ROOT=str(ROOT / "data"),
        )
        for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE"):
            env.pop(name, None)
        started = time.time()
        with (ROOT / "logs" / f"guardian-stage2-{family}-r1.log").open("xb") as log:
            proc = subprocess.Popen(argv, cwd=source, env=env, stdin=subprocess.DEVNULL,
                                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        fields = Path(f"/proc/{proc.pid}/stat").read_text().rsplit(")", 1)[1].split()
        owner = {
            "worker_id": f"root-stage2-{family}-r1", "stage": 2, "gpu_key": str(gpu),
            "pid": proc.pid, "birth": fields[19], "pgid": int(fields[2]), "command": argv,
            "command_sha256": hashlib.sha256(json.dumps(argv, separators=(",", ":")).encode()).hexdigest(),
            "output_dir": str(output), "started_at_epoch": started,
            "deadline_epoch": started + wall + 5, "campaign_deadline_epoch": deadline["deadline_epoch"],
            "profile_receipt_path": str(approved_path), "review_sha256": REVIEW_SHA,
        }
        write_new(owner_path, owner)
        owners.append(owner)
        print(f"STAGE2_FULL_FIT_DETACHED family={family} gpu={gpu} guardian_pid={proc.pid}", flush=True)
    write_new(ROOT / "receipts/stage2-full-fits-r1.json", {
        "new_scientific_fit_attempts": 2, "total_scientific_fit_attempts": 10,
        "owners": owners, "status": "launched",
    })


if __name__ == "__main__":
    main()
