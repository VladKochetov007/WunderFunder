"""Run the fixed CPU endpoint screen only after the admitted GPU handoff succeeds."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path("/home/vlad/development/wunder/wnn_connectome_starterpack")
PACKET = Path(__file__).resolve().parent
FIT_PACKET = PACKET.parent / "h512_f0_followup_r1"
sys.path.insert(0, str(FIT_PACKET))
from record_h512_fit_terminal import service_properties  # noqa: E402


def sha256(path: Path) -> str:
    """Hash the frozen execution files with bounded memory."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def stopped() -> bool:
    """Honor either handoff's stop marker before admitting more CPU work."""
    return (PACKET / "STOP_NATIVE_SCREEN").exists() or (FIT_PACKET / "STOP_POST_FIT").exists()


def run() -> None:
    """Wait on the exact existing watcher and execute five fixed, fail-fast commands."""
    admission_path = PACKET / "NATIVE_EXECUTION_ADMISSION_R1.json"
    admission = json.loads(admission_path.read_text())
    if admission.get("status") != "ADMITTED_CONDITIONAL_ON_POSTFIT_SUCCESS":
        raise RuntimeError("native endpoint execution is not admitted")
    while True:
        if stopped():
            return
        properties = service_properties(admission["postfit_watcher_unit"])
        if properties.get("InvocationID") != admission["postfit_watcher_invocation"]:
            raise RuntimeError("the admitted post-fit watcher changed or disappeared")
        if properties.get("MainPID") == "0":
            break
        time.sleep(15)
    if properties.get("Result") != "success" or properties.get("ExecMainStatus") != "0":
        raise RuntimeError("post-fit watcher did not succeed")
    postfit = service_properties("wnn-h512-postfit-r1-20261021.service")
    if (
        postfit.get("MainPID") != "0"
        or postfit.get("Result") != "success"
        or postfit.get("ExecMainStatus") != "0"
    ):
        raise RuntimeError("GPU post-fit service did not complete successfully")
    sequence = json.loads((FIT_PACKET / "POSTFIT_SEQUENCE_REPORT_R1.json").read_text())
    audit = json.loads((FIT_PACKET / "EXPORT_AUDIT_R1.json").read_text())
    fit = json.loads((FIT_PACKET / "FIT_TERMINAL_ACCEPTANCE_R1.json").read_text())
    if (
        sequence.get("status") != "complete"
        or len(sequence.get("steps", [])) != 7
        or any(step.get("returncode") != 0 for step in sequence["steps"])
        or audit.get("status") != "PASS"
        or fit.get("status") != "FIT_COMPLETE_ACCEPTED"
    ):
        raise RuntimeError("post-fit evidence chain is incomplete")
    for name, digest in admission["execution_pins"].items():
        if sha256(ROOT / name) != digest:
            raise RuntimeError(f"native screen source changed: {name}")
    for name in admission["must_not_exist"]:
        if os.path.lexists(name):
            raise FileExistsError(name)
    plan = json.loads((PACKET / "ENDPOINT_SCREEN_PLAN_R1.json").read_text())
    commands = plan["commands"]
    steps = [
        *zip(
            ("package_cp5", "package_swa345"),
            commands["after_root_accepts_fit_and_before_replay"],
            strict=True,
        ),
        *zip(("native_cp5", "native_swa345"), commands["replay"], strict=True),
        ("paired_native_grade", commands["grade"]),
    ]
    report = {
        "status": "running",
        "started_at_utc": datetime.now(UTC).isoformat(),
        "admission_sha256": sha256(admission_path),
        "accepted_fit_receipt_sha256": sha256(FIT_PACKET / "FIT_TERMINAL_ACCEPTANCE_R1.json"),
        "gpu_postfit_service": postfit,
        "steps": [],
        "new_fits": 0,
        "submission_release_admitted": False,
    }
    report_path = PACKET / "NATIVE_EXECUTION_REPORT_R1.json"
    with report_path.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    environment = dict(os.environ, **admission["environment"])
    for name, argv in steps:
        if stopped():
            report["status"] = "stopped_before_next_step"
            report_path.write_text(json.dumps(report, indent=2) + "\n")
            return
        started = time.monotonic()
        print(json.dumps({"step": name, "event": "starting"}), flush=True)
        result = subprocess.run(argv, cwd=ROOT, env=environment, check=False)
        report["steps"].append(
            {
                "name": name,
                "argv": argv,
                "returncode": result.returncode,
                "wall_seconds": time.monotonic() - started,
            }
        )
        report["status"] = "failed" if result.returncode else "running"
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        if result.returncode:
            raise RuntimeError(f"fixed native screen step failed: {name}")
    report["status"] = "complete_pending_independent_evidence_review"
    report_path.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    run()
