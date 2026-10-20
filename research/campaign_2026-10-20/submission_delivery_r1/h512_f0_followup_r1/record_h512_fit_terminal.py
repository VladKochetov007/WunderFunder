"""Write a hash-bound receipt after the admitted H512 service succeeds."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from h512_swa345_validation import SOURCE_ROOT, TRAINER_SHA, current_training_source

RUN = "r01_h512_f0_s1003_local_cp5_r1"
COMMIT = "68ff682977b822f38351aac0f621146f5f13b440"
PLAN_SHA = "51d797d1d568d98b0d9553c934721aab2826ba5d4d61b96a24e496ce514c7ec9"
SCREEN_SHA = "f6b3b954cd2253a9eac10032c9a053f7abcd55e359c41c2062ed3bdeff2920d8"
ADMISSION_SHA = "a17a468f5d0d247db92cbaac0e3d4f0800536c46d6682f2665abb4b781e776f1"
PACKET = Path("research/campaign_2026-10-20/submission_delivery_r1/h512_f0_followup_r1")
OUTPUT = PACKET / "FIT_TERMINAL_ACCEPTANCE_R1.json"


def sha256(path: Path) -> str:
    """Hash a file in bounded memory."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    """Read an object metadata file."""
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def service_properties(service: str) -> dict[str, str]:
    """Read the completed systemd unit's result and invocation identity."""
    result = subprocess.run(
        [
            "systemctl",
            "show",
            service,
            "--property=ActiveState",
            "--property=Result",
            "--property=ExecMainStatus",
            "--property=InvocationID",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)


def record(root: Path, service: str) -> dict[str, Any]:
    """Verify terminal fit evidence and create one immutable receipt."""
    root = root.resolve(strict=True)
    packet = root / PACKET
    admission = packet / "FIT_ADMISSION_R1.json"
    start = packet / "FIT_START_R1.json"
    log = packet / "fit_r1.log"
    provenance_path = root / "artifacts/runs" / RUN / "provenance.json"
    for path, expected in (
        (admission, ADMISSION_SHA),
        (packet / "H512_FIT_AND_PRODUCTION_PLAN_R1.json", PLAN_SHA),
        (packet / "H512_F0_SCORE_SCREEN_R1.json", SCREEN_SHA),
    ):
        if sha256(path) != expected:
            raise ValueError(f"frozen input differs: {path}")
    started = read_json(start)
    properties = service_properties(service)
    invocation = started.get("service", {}).get("InvocationID")
    if (
        properties.get("ActiveState") != "inactive"
        or properties.get("Result") != "success"
        or properties.get("ExecMainStatus") != "0"
        or properties.get("InvocationID") != invocation
    ):
        raise ValueError("service is incomplete, failed, or has a different invocation")

    fit_epochs: set[int] = set()
    selected_fixed = False
    for line in log.read_text().splitlines():
        if line.startswith("FIT "):
            row = json.loads(line[4:])
            if isinstance(row.get("epoch"), int):
                fit_epochs.add(row["epoch"])
        elif line.strip() == "SELECTED_FIXED 5":
            selected_fixed = True
    if fit_epochs != set(range(6)) or not selected_fixed:
        raise ValueError("fit log lacks six completed epochs or fixed epoch-5 selection")

    provenance = read_json(provenance_path)
    args = provenance.get("args", {})
    if (
        args.get("name") != RUN
        or args.get("epochs") != 6
        or args.get("checkpoint_epoch") != 5
        or args.get("checkpoint_policy") != "fixed"
        or args.get("seed") != 1003
        or args.get("hidden") != 512
        or args.get("loss") != "wmse"
        or args.get("precision") != "fp32"
        or args.get("train_source") != "train+valid"
        or args.get("valid_fold") != 0
        or args.get("include_heldout_valid")
        or not args.get("no_eval_during_fit")
        or args.get("split_manifest_sha256")
        != "09263ad4fe2b18e2b2f3b79f4562678dce801b067b868134ddc79279f9316a3d"
    ):
        raise ValueError("run provenance does not match the frozen H512 fit")
    trainer = provenance.get("trainer_implementation", {})
    source = current_training_source(SOURCE_ROOT.resolve(strict=True))
    if (
        not isinstance(trainer, dict)
        or trainer.get("files") != source["files"]
        or trainer.get("files", {}).get("scripts/train_rnn.py") != TRAINER_SHA
        or trainer.get("sha256") != source["sha256"]
    ):
        raise ValueError("checkpoint trainer source closure differs from the pinned source")
    checkpoints = provenance.get("checkpoint_sha256", {})
    hashes: dict[str, str] = {}
    for epoch in range(6):
        name = f"model_ep{epoch}.pt"
        path = provenance_path.parent / name
        actual = sha256(path)
        if checkpoints.get(name) != actual:
            raise ValueError(f"checkpoint hash is absent or differs: {name}")
        hashes[name] = actual

    if (root / OUTPUT).exists():
        raise FileExistsError(root / OUTPUT)
    receipt = {
        "schema": "wnn.h512-fit-terminal.v1",
        "status": "FIT_COMPLETE_ACCEPTED",
        "run_name": RUN,
        "source_execution_commit": COMMIT,
        "plan_sha256": PLAN_SHA,
        "screen_sha256": SCREEN_SHA,
        "admission_sha256": ADMISSION_SHA,
        "service": service,
        "service_properties": properties,
        "fit_log_sha256": sha256(log),
        "provenance_sha256": sha256(provenance_path),
        "receipt_producer_sha256": sha256(Path(__file__).resolve()),
        "checkpoint_sha256": hashes,
        "fit_epochs": sorted(fit_epochs),
        "selected_fixed_epoch": 5,
        "fit_time_evaluation": False,
    }
    output = root / OUTPUT
    with output.open("x") as stream:
        stream.write(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    return receipt


def main() -> None:
    """Verify the exact service and write the terminal receipt once."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--service", required=True)
    args = parser.parse_args()
    print(json.dumps(record(args.root, args.service), indent=2))


if __name__ == "__main__":
    main()
