"""Authorize only a reviewed descriptive fixed-two-case gradient discrepancy audit."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONT = HERE.parent
sys.path.insert(0, str(CONT / "round4"))
from approve_capture import sha  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reviewer-completed-at", required=True)
    parser.add_argument("--approval", type=Path, required=True)
    options = parser.parse_args()
    completed = datetime.fromisoformat(options.reviewer_completed_at)
    if completed.tzinfo is None or not 0 <= time.time() - completed.timestamp() < 900:
        raise ValueError("Recent observed terminal review required")
    review = HERE / "amp_linearity_review_v2"
    candidate = HERE / "amp_linearity_audit_v2"
    state = json.loads((review / "STATE.json").read_text())
    if not state["verdict"].startswith("PASS") or "frozen" not in state["phase"]:
        raise ValueError("Frozen independent PASS missing")
    manifest = json.loads((review / "HASHES.json").read_text())
    for name, expected in manifest.items():
        path = (review / name).resolve(strict=True)
        path.relative_to(review)
        if sha(path) != expected:
            raise ValueError(f"Review changed: {name}")
    candidate_sha = json.loads((review / "candidate_sha256.json").read_text())
    integration = json.loads((review / "integration_bindings.json").read_text())
    if integration["verdict"] != state["verdict"] or integration["candidate_sha256"] != candidate_sha:
        raise ValueError("Reviewed integration and candidate bindings differ")
    if set(candidate_sha) != {"PREREG.md", "audit.py", "metadata.py", "tests.py"}:
        raise ValueError("Exact reviewed descriptive instrument file set required")
    for name, expected in candidate_sha.items():
        if sha(candidate / name) != expected:
            raise ValueError(f"Candidate changed: {name}")
    measurement = HERE / "capture_measurement_review/STATE.json"
    if json.loads(measurement.read_text())["verdict"] != "PASS_MEASURED_PARITY_FOR_RECORDED_WARM_V4_DEFAULT_TRAJECTORIES_ONLY":
        raise ValueError("Actual default measurement review missing")
    watch = json.loads((CONT / "watch_status.json").read_text())
    if (not (CONT / "READY").exists() or (CONT / "RESOURCE_FAILURE").exists()
        or watch["barrier"] or watch["monitoring_degraded"]
        or not 0 <= time.time() - datetime.fromisoformat(watch["time"]).timestamp() < 120
        or time.time() + 60 >= 1791057600):
        raise ValueError("Healthy live admission and completion slack required")
    metadata = HERE / "fixed_case_metadata_v2.json"
    if (sha(metadata) != integration["fixed_case_metadata_sha256"]
        or sha(measurement) != integration["independent_measurement_STATE_sha256"]
        or sha(HERE / "prepare_capture_metadata_v2.py") != integration["prepare_controller_helper_sha256"]
        or sha(HERE / "real_trajectory.json") != integration["existing_trajectory_sha256"]):
        raise ValueError("Reviewed metadata or measurement identity changed")
    binding = json.loads(metadata.read_text())
    if binding["measurement_review_sha256"] != sha(measurement) or binding["gradients_read"] is not False:
        raise ValueError("Metadata-only measurement binding changed")
    output_manifest = CONT / "round4/gpu_capture_v4_default_2120r/output_hashes.json"
    if sha(output_manifest) != integration["output_manifest_sha256"]:
        raise ValueError("Independently reviewed capture manifest changed")
    output_hashes = json.loads(output_manifest.read_text())["files"]
    for name, prefix in (("synthetic_actual_architecture_normal", "synthetic_actual_architecture/normal"),
                         ("real_panel_normal", "real_fit_panel/normal")):
        case = binding["cases"][name]
        expected_paths = {
            kind: str(CONT / "round4/gpu_capture_v4_default_2120r" / prefix / f"{kind}.pt")
            for kind in ("pre", "post", "components")
        }
        if case["paths"] != expected_paths:
            raise ValueError("Fixed reviewed capture paths differ")
        if case["sha256"] != {kind: output_hashes[f"{prefix}/{kind}.pt"] for kind in ("pre", "post", "components")}:
            raise ValueError("Fixed measured capture hashes differ")
    approval = {
        "schema_version": 4, "independent_measurement_review_passed": True,
        "controller_analysis_authorized": True,
        "independent_measurement_review_receipt_sha256": sha(measurement),
        "cases": binding["cases"], "candidate_sha256": candidate_sha,
        "independent_audit_review_bindings": {name: sha(review / name) for name in manifest},
        "fixed_metadata_sha256": sha(metadata),
        "controller_attested_at": datetime.now().astimezone().isoformat(),
        "reviewer_completion_observed_at": options.reviewer_completed_at,
        "scope": "Descriptive FP64 discrepancy only; no acceptance bound, conflict/update attribution or score claim",
    }
    with options.approval.open("x") as output:
        json.dump(approval, output, indent=2, allow_nan=False)
        output.write("\n")
    print(json.dumps({"approval": str(options.approval), "sha256": sha(options.approval), "gradients_read": False}))


if __name__ == "__main__":
    main()
