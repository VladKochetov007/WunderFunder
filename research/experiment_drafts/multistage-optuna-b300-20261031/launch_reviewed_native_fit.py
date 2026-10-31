"""Launch one exact independently reviewed native profile through the proven trainer."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

from wnn.multistage_gpu_identity import normalize_gpu_uuid
from wnn.multistage_search_budget_fit import reserve_fit
from wnn.training.sequence_config import load_config
from wnn.training.sequence_provenance import source_identity, validate_profile_receipt

SOURCE_SHA = "c520c67ca8d1fb49fc0c452a4606f5ad130ed51e7fc689fdaf9daea2b429dd81"


def digest(path: Path) -> str:
    """Hash exact artifact bytes independently of JSON serialization."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_new(path: Path, value: object) -> None:
    """Publish a new record while refusing to overwrite a previous attempt."""
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def launch(root: Path, number: int, review_path: Path, review_sha: str) -> dict[str, object]:
    """Bind the corrected independent review, native profile, source and fit budget."""
    campaign = root / "campaign-r2"
    trial = campaign / "stage1/trials" / f"{number:06d}"
    request_path = trial / "trial_request.json"
    request = json.loads(request_path.read_bytes())
    review = json.loads(review_path.read_bytes())
    if digest(review_path) != review_sha or review.get("status") != "approved":
        raise ValueError("independent approved review bytes differ from the pinned launch input")
    if review.get("reviewer") not in {"multistage_independent_critic", "independent"}:
        raise ValueError("review has no recognized independent issuer")
    bindings = review.get("bindings", review)
    scope = review.get("scope", review)
    if scope.get("stage") != 1 or scope.get("trial_number") != number:
        raise ValueError("independent review covers a different scientific candidate")
    native_path = Path(request["profile_receipt_path"])
    config_path = Path(request["config_path"])
    native = json.loads(native_path.read_bytes())
    config = load_config(config_path)
    identity = native["identity"]
    checks = {
        "native_profile_sha256": digest(native_path),
        "trial_request_sha256": digest(request_path),
        "source_sha256": SOURCE_SHA,
    }
    if any(bindings.get(key) != value for key, value in checks.items()):
        raise ValueError("review does not bind the exact native/request/source artifacts")
    raw_sha = bindings.get("config_raw_sha256", bindings.get("config_sha256"))
    if raw_sha != digest(config_path) or request["config_sha256"] != raw_sha:
        raise ValueError("raw configuration differs from the independent review")
    normalized_sha = hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if identity["config_sha256"] != normalized_sha:
        raise ValueError("normalized configuration differs from the native producer")
    source = root / "operator-native4"
    if source_identity(source, source / "scripts/train_sequence.py")["sha256"] != SOURCE_SHA:
        raise ValueError("immutable scientific source changed")
    if native["status"] != "profile_complete" or native["updates"] != 50 or native["errors"]:
        raise ValueError("native profile is incomplete or contains numerical errors")
    if native["approved"] is not False or request["primary_valid_used"] is not False:
        raise ValueError("profile approval or validation scope is inconsistent")
    uuid = request["expected_device_uuid_set"][0]
    if normalize_gpu_uuid(uuid) != normalize_gpu_uuid(identity["runtime"]["devices"][0]["uuid"]):
        raise ValueError("profile GPU differs from the bound launch GPU")
    approved_path = trial / "profile/approved_manual_receipt.json"
    write_new(
        approved_path,
        native
        | {
            "approved": True,
            "external_review": {
                "path": str(review_path),
                "sha256": review_sha,
                "native_profile_sha256": digest(native_path),
                "method": "independent correction of preserved engineering consumer rejection",
            },
        },
    )
    validate_profile_receipt(approved_path, identity)
    attempt_id = f"manual-stage1-trial-{number:06d}"
    output = root / "runs" / f"tpe-stage1-{number:06d}"
    if output.exists():
        raise FileExistsError("scientific output already exists; no silent retry")
    _, wall = reserve_fit(campaign, 1, attempt_id, number, 7190)
    command = [
        "/usr/bin/timeout",
        "--signal=TERM",
        "--kill-after=5s",
        f"{wall}s",
        "/data/epanov_s/tabdata/.venv/bin/python",
        str(source / "scripts/train_sequence.py"),
        "--config",
        str(config_path),
        "--data-root",
        str(root / "data"),
        "--split-manifest",
        request["auxiliary_split_manifest"],
        "--output",
        str(output),
        "--profile-receipt",
        str(approved_path),
    ]
    child_environment = {
        "PYTHONPATH": f"{source / 'src'}:{root / 'python-deps'}",
        "CUDA_VISIBLE_DEVICES": uuid,
        "OMP_NUM_THREADS": "4",
        "MKL_NUM_THREADS": "4",
        "OPENBLAS_NUM_THREADS": "4",
        "WNN_PORTABLE_DATA_MANIFEST": str(root / "inputs/portable-data-r1.json"),
        "WNN_PROJECT_ROOT": str(root / "data"),
    }
    fit_request = root / "requests" / f"{attempt_id}.json"
    write_new(
        fit_request,
        {
            "command": command,
            "source_root": str(source),
            "stage": 1,
            "trial_number": number,
            "gpu_key": request["gpu_key"],
            "output_dir": str(output),
            "timeout_seconds": wall + 5,
            "log_path": str(root / "logs" / f"{attempt_id}.log"),
            "child_receipt_path": str(root / "receipts" / f"{attempt_id}.child.json"),
            "budget_run_dir": str(campaign),
            "fit_attempt_id": attempt_id,
            "child_environment": child_environment,
            "independent_review_sha256": review_sha,
            "original_checker_rejection_preserved": True,
        },
    )
    environment = os.environ.copy() | child_environment
    environment["PYTHONPATH"] = f"{root / 'operator-r2/src'}:{root / 'python-deps'}"
    guardian = root / "requests/fit_guardian_r2.py"
    argv = [command[4], str(guardian), "--request", str(fit_request)]
    with (root / "logs" / f"{attempt_id}.guardian.log").open("xb") as log:
        process = subprocess.Popen(
            argv,
            cwd=source,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    birth = Path(f"/proc/{process.pid}/stat").read_text().rsplit(") ", 1)[1].split()[19]
    owner = {
        "pid": process.pid,
        "pgid": os.getpgid(process.pid),
        "birth": birth,
        "argv": argv,
        "started_epoch": time.time(),
        "attempt_id": attempt_id,
        "stage": 1,
        "trial_number": number,
        "gpu_key": request["gpu_key"],
        "gpu_uuid": uuid,
        "output_dir": str(output),
        "review_sha256": review_sha,
        "request_sha256": digest(fit_request),
        "native_source_sha256": SOURCE_SHA,
    }
    write_new(root / "receipts" / f"{attempt_id}.owner.json", owner)
    return owner


def main() -> None:
    """Admit one new native fit from an exact independently approved profile."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--trial-number", type=int, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--review-sha256", required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            launch(args.runtime_root, args.trial_number, args.review, args.review_sha256), indent=2
        )
    )


if __name__ == "__main__":
    main()
