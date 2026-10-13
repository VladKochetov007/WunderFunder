"""Deliver the frozen per-member GPU diagnostic bank in one admitted worker."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    admission = json.loads((HERE / "MEMBER_BANK_ADMISSION_R1.json").read_text())
    if admission["status"] != "admitted":
        raise ValueError("member bank is not admitted")
    plan_path = HERE / "MEMBER_BANK_PLAN_R1.json"
    if digest(plan_path) != admission["plan_sha256"]:
        raise ValueError("member plan changed")
    plan = json.loads(plan_path.read_text())
    bank = HERE / "member_bank"
    bank.mkdir(exist_ok=True)
    receipts = []
    for member in plan["members"]:
        for relative, expected in admission["source_files"].items():
            if digest(ROOT / relative) != expected:
                raise ValueError(f"frozen source changed: {relative}")
        out = bank / member["id"]
        if out.exists():
            raise FileExistsError(out)
        command = [
            sys.executable,
            str(ROOT / "scripts/export_team_member_fp32.py"),
            "--inventory",
            str(HERE / "INVENTORY.json"),
            "--candidate",
            member["candidate"],
            "--member",
            member["member"],
            "--data",
            str(ROOT / "datasets/valid.parquet"),
            "--batch-sequences",
            "16",
            "--chunk-rows",
            "500",
            "--out",
            str(out),
        ]
        receipt = {
            "id": member["id"],
            "started_utc": datetime.now(UTC).isoformat(),
            "command": command,
            "weight_sha256": member["sha256"],
        }
        print(json.dumps({"event": "started", **receipt}), flush=True)
        with (bank / f"{member['id']}.log").open("xb") as log:
            result = subprocess.run(
                command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=900, check=False
            )
        receipt.update(exit_code=result.returncode, finished_utc=datetime.now(UTC).isoformat())
        if result.returncode == 0:
            terminal = json.loads((out / "RESULT.json").read_text())
            if (
                terminal["status"] != "complete"
                or terminal["weight_sha256"] != member["sha256"]
                or terminal["prediction_shape"] != [1873, 20000, 2]
                or terminal["row_groups"] != list(range(1873))
            ):
                raise ValueError(f"invalid terminal member: {member['id']}")
            receipt["result_sha256"] = digest(out / "RESULT.json")
            receipt["prediction_sha256"] = terminal["prediction_sha256"]
            (out / "README.md").write_text(
                f"# {member['id']}\n\n[RESULT.json](RESULT.json) records the completed "
                "FP32 diagnostic export, source/weight/data identity and resources. "
                "`predictions.npy` is ignored physical storage; `progress.jsonl` is the "
                "operational trace. See the [bank index](../README.md) for exposure limits.\n"
            )
        (bank / f"{member['id']}_ATTEMPT.json").write_text(json.dumps(receipt, indent=2) + "\n")
        receipts.append(receipt)
        print(json.dumps({"event": "finished", **receipt}), flush=True)
        if result.returncode != 0:
            raise RuntimeError(f"member failed; diagnose before a new attempt: {member['id']}")
    (HERE / "MEMBER_BANK_TERMINAL_R1.json").write_text(json.dumps(receipts, indent=2) + "\n")


if __name__ == "__main__":
    main()
