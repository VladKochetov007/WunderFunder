from __future__ import annotations

import json
from pathlib import Path

from verify import digest

OWN = Path(__file__).resolve().parent


def main() -> None:
    """Freeze narrow corrected-instrument verdict, never execute or authorize data analysis."""
    tests = json.loads((OWN / "tests.json").read_text())
    constructor = json.loads((OWN / "constructor_guard_tests.json").read_text())
    verification = json.loads((OWN / "protected_verification.json").read_text())
    integration = json.loads((OWN / "integration_bindings.json").read_text())
    candidate = json.loads((OWN / "candidate_sha256.json").read_text())
    assert set(candidate) == {"PREREG.md", "audit.py", "metadata.py", "tests.py"}
    assert candidate == integration["candidate_sha256"]
    assert (
        tests["passed"] and constructor["pass"] and verification["all_author_and_prior_files_match"]
    )
    # Final external small-file identity check, not a capture-content pass.
    expected = {
        OWN.parent / "approve_amp_audit.py": integration["approval_controller_helper_sha256"],
        OWN.parent / "prepare_capture_metadata_v2.py": integration[
            "prepare_controller_helper_sha256"
        ],
        OWN.parent / "fixed_case_metadata_v2.json": integration["fixed_case_metadata_sha256"],
        OWN.parent / "capture_measurement_review/STATE.json": integration[
            "independent_measurement_STATE_sha256"
        ],
        OWN.parent / "real_trajectory.json": integration["existing_trajectory_sha256"],
    }
    assert all(digest(path) == value for path, value in expected.items())
    state = {
        "phase": "frozen_independent_v2_amp_linearity_review_complete",
        "verdict": integration["verdict"],
        "blocking_findings": [],
        "metadata_JSON_schema_correction": "PASS",
        "controller_projection_integration": "PASS",
        "final_constructor_binding_guards": "PASS",
        "source_hashes": candidate,
        "metadata_projection_sha256": candidate["metadata.py"],
        "projection_version": "metadata-json-v1",
        "approval_controller_helper_sha256": integration["approval_controller_helper_sha256"],
        "independent_measurement_STATE_sha256": integration["independent_measurement_STATE_sha256"],
        "fixed_case_metadata_sha256": integration["fixed_case_metadata_sha256"],
        "fixed_cases": list(integration["fixed_case_identities"]),
        "prior_reviews_and_author_artifacts_preserved": True,
        "actual_capture_or_gradient_tensor_reads": 0,
        "actual_diagnostic_executed": False,
        "GPU_launched": False,
        "receipt_written": False,
        "scientific_acceptance_bound_inferred": False,
        "conflict_update_or_WP_claim": False,
        "converted_array_workspace_ceiling_bytes": 8388608,
        "helper_input_scanning_fallback_approved": False,
        "controller_explicit_analysis_approval_required": True,
        "pending": [],
        "last_failure": None,
    }
    (OWN / "STATE.json").write_text(json.dumps(state, indent=2) + "\n")
    manifest = {
        str(path.relative_to(OWN)): digest(path)
        for path in sorted(OWN.rglob("*"))
        if path.is_file() and path.name not in {"HASHES.json", "freeze.log"}
    }
    (OWN / "HASHES.json").write_text(json.dumps(manifest, indent=2) + "\n")
    assert all(digest(OWN / name) == value for name, value in manifest.items())
    print(
        json.dumps(
            {
                "verdict": state["verdict"],
                "final_manifest_verified": True,
                "report_sha256": manifest["REPORT.md"],
                "manifest_sha256": digest(OWN / "HASHES.json"),
                "candidate_inventory_sha256": manifest["candidate_sha256.json"],
                "approval_controller_helper_sha256": integration[
                    "approval_controller_helper_sha256"
                ],
            }
        )
    )


if __name__ == "__main__":
    main()
