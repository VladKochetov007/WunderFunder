"""Deliver the one remaining frozen W18 fit and checked CPU package."""

from __future__ import annotations

import hashlib
import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

F = Path("/data/epanov_s/tabdata/runtime/final-submit-all-labeled-20261031-r1")
R = Path("/data/epanov_s/tabdata/runtime/w18-submit-recovery-20261103-r1")
S = R / "source"
P = "/data/epanov_s/tabdata/.venv/bin/python"
GPU = "GPU-76951d88-add3-0d6b-5bee-c0390915ca47"
NODE = "w18final_distilled_s5"
STOP_REQUESTED = False
sys.path[:0] = [str(S), str(S / "src"), str(F / "control-code-r2/scripts")]

from wnn.final_submit_manifest import load_final_fit_membership  # noqa: E402
from wnn.multistage_search import atomic_write_json  # noqa: E402
from wnn.multistage_search_owned_command import run_owned_command  # noqa: E402
from wnn.multistage_search_runtime import (  # noqa: E402
    base_environment,
    hold_gpu_slot,
    process_identity,
)
from wnn.training.environment import TrainingEnvironment  # noqa: E402
from wnn.training.source_identity import training_source_identity  # noqa: E402


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def create(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as out:
        json.dump(value, out, indent=2, allow_nan=False)


def canonical(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def headroom() -> None:
    raw = subprocess.check_output(
        ["nvidia-smi", "-i", GPU, "--query-gpu=name,memory.free", "--format=csv,noheader,nounits"],
        text=True,
        timeout=15,
    )
    name, free = raw.strip().rsplit(",", 1)
    if "B300" not in name or int(free) < 32768:
        raise RuntimeError("Selected B300 no longer has the admitted 32 GiB free headroom")


def main() -> None:
    authority = read(R / "ROOT_AUTHORITY_R1.json")
    source = read(R / "inputs/SOURCE_BINDING_R1.json")
    execution = read(R / "inputs/EXECUTION_BINDING_R1.json")
    deadline = authority["hard_deadline_epoch"]
    if authority["max_scientific_fits"] != 1 or authority["fit_epochs"] != 2:
        raise RuntimeError("Recovery authority differs from the one frozen W18 fit")
    if time.time() + 10800 + 1800 > deadline:
        raise TimeoutError(
            "Full fit and collection reserve no longer fit the authorized wall bound"
        )
    if sha(R / "inputs/w18-source-recovery-r1.tar.gz") != source["source_capsule_sha256"]:
        raise RuntimeError("Recovery source capsule changed")
    for relative, expected in execution["files"].items():
        if sha(Path(relative)) != expected:
            raise RuntimeError("Pinned execution input changed: " + relative)
    if (
        read(R / "reports/CACHE_VALIDATION_R1.json")["metadata_sha256"]
        != source["cache_metadata_sha256"]
    ):
        raise RuntimeError("Actual existing-cache validation is missing or differs")
    cache_metadata = F / "data/artifacts/runs/final-fit-w18-mixed-targets/metadata.json"
    if sha(cache_metadata) != source["cache_metadata_sha256"]:
        raise RuntimeError("Actual teacher cache changed after its independent validation")
    plan = read(F / "inputs/final-plan-r2.json")
    if plan["plan_sha256"] != "29802499ab6ba62586b612cd10164d27c9737a96cf8de41c09fef3f6d0402cf4":
        raise RuntimeError("Frozen plan identity differs")
    nodes = [n for n in plan["nodes"] if n["id"] == NODE]
    if len(nodes) != 1:
        raise RuntimeError("Frozen W18 recipe is not unique")
    node = nodes[0]
    args = list(node["args"])
    output = F / "data/artifacts/runs" / node["run"]
    fit_receipt = F / "control/final-submit-chain/receipts/fits" / (NODE + ".json")
    if output.exists() or fit_receipt.exists():
        raise FileExistsError("A W18 scientific attempt already exists; no automatic retry")
    previous = read(R / "inputs/PREVIOUS_FITS_MANIFEST.json")
    if previous["completed_model_artifacts"] != 17:
        raise RuntimeError("Expected seventeen preserved final fits")
    for item in previous["items"]:
        if sha(Path(item["checkpoint_path"])) != item["checkpoint_sha256"]:
            raise RuntimeError("Preserved checkpoint changed: " + item["node_id"])
    membership = load_final_fit_membership(
        F / "inputs/final-fit-membership-r1.json",
        F / "data",
        "1e8a8a794a0b8d9677a76030e570681074503d38aa2d54b80e36067566b94ce0",
    )
    owner = process_identity(os.getpid())
    if owner is None:
        raise RuntimeError("Cannot establish recovery process ownership")
    owner = dict(owner, controller_root=str(R / "control"))
    os.environ["PYTHONPATH"] = str(S) + ":" + str(S / "src")
    env = base_environment(GPU, S)
    env.update(
        WNN_PROJECT_ROOT=str(F / "data"),
        PYTHONUNBUFFERED="1",
        NVIDIA_TF32_OVERRIDE="0",
        TORCH_ALLOW_TF32_CUBLAS_OVERRIDE="0",
    )
    create(
        R / "receipts/INTENT_R1.json",
        {
            "schema": "wnn.w18-recovery-intent.v1",
            "node_id": NODE,
            "plan_sha256": plan["plan_sha256"],
            "recipe_sha256": canonical(args),
            "source_capsule_sha256": source["source_capsule_sha256"],
            "gpu_uuid": GPU,
            "max_scientific_fits": 1,
            "epochs": 2,
            "expected_optimizer_updates": 15600,
            "fixed_endpoint": 1,
            "fit_members_sha256": membership.fit_members_sha256,
            "preserved_prior_fits": 17,
            "deadline_epoch": deadline,
            "created_epoch": time.time(),
            "independent_score_claim": "forbidden",
        },
    )
    last_telemetry = 0.0
    phase = "profile"

    def progress() -> None:
        nonlocal last_telemetry
        if STOP_REQUESTED:
            raise RuntimeError("Recovery stop requested; terminate only the owned child")
        now = time.time()
        if now - last_telemetry < 30:
            return
        last_telemetry = now
        gpu = subprocess.check_output(
            [
                "nvidia-smi",
                "-i",
                GPU,
                "--query-gpu=utilization.gpu,memory.used,memory.free",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=15,
        ).strip()
        row = {
            "epoch_time": now,
            "phase": phase,
            "gpu_uuid": GPU,
            "shared_gpu_util_used_free": gpu,
            "independent_score_claim": "forbidden",
        }
        row["host_memory"] = {
            line.split(":", 1)[0]: line.split(":", 1)[1].strip()
            for line in Path("/proc/meminfo").read_text().splitlines()
            if line.startswith(("MemAvailable:", "MemTotal:"))
        }
        with (R / "reports/resources.jsonl").open("a") as out:
            out.write(json.dumps(row) + "\n")
        atomic_write_json(R / "reports/progress.json", row)

    profile_args = args.copy()
    profile_args.remove("--save-training-state")
    profile_args[profile_args.index("--name") + 1] += "_engineering_r2"
    chunk = int(profile_args[profile_args.index("--chunk") + 1])
    batches = math.ceil(50 / (20000 / chunk))
    profile_args.extend(["--profile-steps", str(batches), "--profile-warmup", "1"])
    profile_log = R / "logs/profile-r1.log"
    profile_process = R / "receipts/profile-r1.json"
    command = [
        "/usr/bin/timeout",
        "--signal=TERM",
        "--kill-after=5s",
        "590s",
        P,
        str(R / "profile_wrapper_r2.py"),
        str(profile_log),
        *profile_args,
    ]
    identity = training_source_identity(TrainingEnvironment(source_root=S, root=F / "data"))
    request = {
        "schema": "wnn.final-submit-profile-request.v1",
        "mode": "legacy_tl3",
        "node_id": NODE,
        "plan_sha256": plan["plan_sha256"],
        "recipe_sha256": canonical(args),
        "recipe_argv": args,
        "argv": command,
        "argv_sha256": canonical(command),
        "source_capsule_sha256": source["source_capsule_sha256"],
        "full_source_code_sha256": identity["sha256"],
        "expected_source": identity,
        "final_fit_manifest_sha256": membership.manifest_sha256,
        "source_audit_sha256": membership.source_validation_sha256,
        "fit_members_sha256": membership.fit_members_sha256,
        "fit_member_count": 12480,
        "evaluation_member_count": 0,
        "no_eval_during_fit": True,
        "independent_score_claim": "forbidden",
        "gpu_uuid": GPU,
        "cpu_threads": 4,
        "profile_output": str(R / "profiles/w18-r1"),
        "log_path": str(profile_log),
        "process_receipt": str(profile_process),
        "profile_updates": None,
        "profile_batches": batches,
        "warmup_batches": 1,
        "config_file_sha256": None,
        "config_identity_sha256": None,
    }
    request_path = R / "inputs/profile-request-r1.json"
    create(request_path, request)
    if STOP_REQUESTED:
        raise RuntimeError("Recovery stopped before profiling")
    with hold_gpu_slot(F / "control/final-submit-chain", GPU):
        headroom()
        code, timed_out, _ = run_owned_command(
            command,
            cwd=S,
            env=env,
            log_path=profile_log,
            receipt_path=profile_process,
            owner=owner,
            stage=0,
            trial_number=17,
            mode="final_submit_engineering_r2",
            gpu_key=GPU,
            output_dir=R / "profiles/w18-r1",
            timeout_seconds=600,
            progress=progress,
        )
    if code != 0 or timed_out:
        raise RuntimeError(
            "W18 profile failed; preserve evidence and diagnose without automatic retry"
        )
    phase = "independent_profile_review"
    review_command = [
        P,
        str(R / "control/review_profile.py"),
        "--request",
        str(request_path),
        "--expected-request-sha256",
        sha(request_path),
        "--plan",
        str(F / "inputs/final-plan-r2.json"),
        "--membership",
        str(F / "inputs/final-fit-membership-r1.json"),
        "--source-audit",
        str(F / "inputs/SOURCE_VALIDATION_R1.json"),
        "--source-capsule",
        str(R / "inputs/w18-source-recovery-r1.tar.gz"),
        "--process",
        str(profile_process),
        "--legacy-log",
        str(profile_log),
        "--output-dir",
        str(R / "profile-reviews/w18-r1"),
    ]
    with (R / "logs/profile-review-r1.log").open("xb") as log:
        result = subprocess.run(
            review_command,
            cwd=S,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=180,
            check=False,
        )
    if (
        result.returncode != 0
        or read(R / "profile-reviews/w18-r1/profile_review.json")["status"] != "approved"
    ):
        raise RuntimeError("Independent profile review did not approve the exact W18 fit")
    phase = "fit"
    if STOP_REQUESTED:
        raise RuntimeError("Recovery stopped before fitting")
    if sha(cache_metadata) != source["cache_metadata_sha256"]:
        raise RuntimeError("Actual teacher cache changed between profile and fit")
    if output.exists() or fit_receipt.exists() or time.time() + 10800 + 1800 > deadline:
        raise RuntimeError("W18 output exists or remaining authorized fit budget is insufficient")
    command = [
        "/usr/bin/timeout",
        "--signal=TERM",
        "--kill-after=10s",
        "10790s",
        P,
        str(S / "scripts/train_rnn.py"),
        *args,
    ]
    with hold_gpu_slot(F / "control/final-submit-chain", GPU):
        headroom()
        code, timed_out, _ = run_owned_command(
            command,
            cwd=S,
            env=env,
            log_path=R / "logs/fit-r1.log",
            receipt_path=fit_receipt,
            owner=owner,
            stage=1,
            trial_number=18,
            mode="final_submit_finite_chain",
            gpu_key=GPU,
            output_dir=output,
            timeout_seconds=10800,
            progress=progress,
        )
    if code != 0 or timed_out or not (output / "model_ep1.pt").is_file():
        raise RuntimeError("W18 did not reach its successful fixed endpoint; no automatic retry")
    if sha(cache_metadata) != source["cache_metadata_sha256"]:
        raise RuntimeError("Actual teacher cache changed during the W18 fit")
    phase = "package_and_publish"
    if STOP_REQUESTED:
        raise RuntimeError("Recovery stopped before packaging")
    progress()
    cpu_env = dict(env, CUDA_VISIBLE_DEVICES="")
    result = subprocess.run(
        [
            P,
            str(R / "control/publish_w18.py"),
            "--runtime-root",
            str(F),
            "--recovery-root",
            str(R),
            "--deadline-epoch",
            str(deadline),
            "--poll-seconds",
            "15",
        ],
        cwd=R,
        env=cpu_env,
        check=False,
        timeout=max(1, int(deadline - time.time() - 10)),
    )
    if result.returncode != 0:
        raise RuntimeError("W18 package/publication did not pass; inspect preserved reports")
    create(
        R / "reports/COMPLETE_R1.json",
        {
            "status": "delivered",
            "finished_epoch": time.time(),
            "independent_score_claim": "forbidden",
        },
    )


if __name__ == "__main__":

    def terminate(signum: int, frame: object) -> None:
        global STOP_REQUESTED
        STOP_REQUESTED = True

    signal.signal(signal.SIGTERM, terminate)
    try:
        main()
    except BaseException as error:
        atomic_write_json(
            R / "reports/FAILURE_R1.json",
            {"error": type(error).__name__ + ": " + str(error), "epoch_time": time.time()},
        )
        raise
