"""Controller receipt for one independently reviewed real state-drift comparison."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
CONT = HERE.parent
sys.path.insert(0, str(CONT / "round4"))
from approve_capture import command, sha, unit  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reviewer-completed-at", required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    completed = datetime.fromisoformat(options.reviewer_completed_at)
    if completed.tzinfo is None or not 0 <= time.time() - completed.timestamp() < 900:
        raise ValueError("Recent observed terminal independent review required")
    review = HERE / "state_drift_review_v2_binding"
    candidate = HERE / "state_drift_v2"
    options.output.resolve().relative_to(candidate.resolve())
    if options.output.exists() or options.receipt.exists():
        raise ValueError("Fresh output and receipt required")
    state = json.loads((review / "STATE.json").read_text())
    if not state["verdict"].startswith("PASS") or "frozen" not in state["phase"]:
        raise ValueError("Frozen independent PASS missing")
    manifest = json.loads((review / "HASHES.json").read_text())
    for name, expected in manifest.items():
        path = (review / name).resolve(strict=True)
        path.relative_to(review)
        if sha(path) != expected:
            raise ValueError(f"Review artifact changed: {name}")
    inventory = json.loads((review / "candidate_runtime_hashes.json").read_text())
    if inventory["verdict"] != state["verdict"]:
        raise ValueError("Inventory verdict mismatch")
    for name, expected in inventory["candidate_sha256"].items():
        path = (candidate / name).resolve(strict=True)
        path.relative_to(candidate)
        if sha(path) != expected:
            raise ValueError(f"Candidate changed: {name}")
    for section in ("runtime_sha256", "source"):
        for name, expected in inventory[section].items():
            path = (ROOT / name).resolve(strict=True)
            path.relative_to(ROOT)
            if sha(path) != expected:
                raise ValueError(f"Reviewed {section} changed: {name}")
    measurement = HERE / "capture_measurement_review/STATE.json"
    trajectory = HERE / "real_trajectory.json"
    if (sha(trajectory) != inventory["trajectory_sha256"]
        or sha(measurement) != inventory["independent_measurement_review_state_sha256"]
        or sha(HERE / "prepare_capture_metadata.py") != inventory["metadata_helper_sha256"]):
        raise ValueError("Reviewed trajectory, measurement review or provenance helper changed")
    if json.loads(measurement.read_text())["verdict"] != "PASS_MEASURED_PARITY_FOR_RECORDED_WARM_V4_DEFAULT_TRAJECTORIES_ONLY":
        raise ValueError("Independent default measurement missing")
    watch = json.loads((CONT / "watch_status.json").read_text())
    if (not (CONT / "READY").is_file() or (CONT / "RESOURCE_FAILURE").exists()
        or watch["barrier"] or watch["monitoring_degraded"] or watch["active_gpu_jobs"]
        or not 0 <= time.time() - datetime.fromisoformat(watch["time"]).timestamp() < 120
        or time.time() + 325 >= 1791057600):
        raise ValueError("Live admission or completion slack missing")
    units = {name: unit(name) for name in (
        "wnn-codex-research-queue-v2.service", "wnn-codex-fold23-scores.service")}
    queue = ROOT / "research/campaign_2026-10-01/jobs/queue6.txt"
    if any(line.strip() and not line.lstrip().startswith("#") for line in queue.read_text().splitlines()):
        raise ValueError("Training queue not drained")
    processes = command(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"]).strip()
    if processes:
        raise ValueError("GPU has a live compute process")
    for path in (ROOT / "research/campaign_2026-09-29/jobs").glob("*.json"):
        job = json.loads(path.read_text())
        if job.get("kind") == "gpu" and job.get("status") == "running":
            raise ValueError("GPU job remains running")
    spec = importlib.util.spec_from_file_location("drift_runtime", candidate / "runtime.py")
    if spec is None or spec.loader is None:
        raise ValueError("Cannot load reviewed constants")
    runtime = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runtime)
    capture = CONT / "round4/gpu_capture_v4_default_2120r/real_fit_panel/normal"
    output_manifest = capture.parents[1] / "output_hashes.json"
    if sha(output_manifest) != inventory["output_manifest_sha256"]:
        raise ValueError("Independently reviewed capture manifest changed")
    observed_capture = {"pre": sha(capture / "pre.pt"), "post": sha(capture / "post.pt")}
    capture_files = json.loads(output_manifest.read_text())["files"]
    if observed_capture != {kind: capture_files[f"real_fit_panel/normal/{kind}.pt"] for kind in observed_capture}:
        raise ValueError("Capture files differ from measured manifest")
    receipt = {
        "status": "approved", "default_v4_gate": "PASS", "independent_state_drift_review": "PASS",
        "reviewer": "Helmholtz/01a101e6-0368-7002-9b08-952d37d2129d",
        "reviewer_completion_observed_at": options.reviewer_completed_at,
        "controller_attested_at": datetime.now().astimezone().isoformat(),
        "candidate_sha256": inventory["candidate_sha256"],
        "runtime_sha256": inventory["runtime_sha256"], "runtime_limits": runtime.LIMITS,
        "capture_sha256": {**observed_capture, "trajectory": sha(trajectory)},
        "v4_review_receipt": str(CONT / "round4/controller_receipts/capture_v4_2120.json"),
        "review_bindings": {name: sha(review / name) for name in manifest},
        "independent_measurement_review_sha256": sha(measurement),
        "watch": watch, "live_queue_units": units,
        "scope": "One frozen warm4/no-skip binary incoming student-hidden intervention; no full fit or score claim",
        "launch": {
            "entrypoint": str((candidate / "run_real.py").resolve()),
            "pre": str((capture / "pre.pt").resolve()),
            "post": str((capture / "post.pt").resolve()),
            "trajectory": str(trajectory.resolve()),
            "output": str(options.output.resolve()),
            "receipt": str(options.receipt.resolve()),
        },
    }
    options.receipt.parent.mkdir(parents=True, exist_ok=True)
    with options.receipt.open("x") as output:
        json.dump(receipt, output, indent=2, allow_nan=False)
        output.write("\n")
    print(json.dumps({"receipt": str(options.receipt), "sha256": sha(options.receipt), "GPU_launched": False}))


if __name__ == "__main__":
    main()
