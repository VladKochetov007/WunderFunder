from __future__ import annotations

import json
from pathlib import Path

from verify_inputs import digest

OWN = Path(__file__).resolve().parent


def main() -> None:
    """Freeze narrow independent verdict; never issue a GPU receipt."""
    result = json.loads((OWN / "results.json").read_text())
    verified = json.loads((OWN / "protected_verification.json").read_text())
    assert result["all_pass"] and verified["all_match"]
    smoke = result["CPU_reproduction"]
    assert all(
        smoke[key] == 0
        for key in (
            "normal_max_delta",
            "repeat_max_delta",
            "skip_max_delta",
            "uninstrumented_default_max_delta",
        )
    )
    assert smoke["independent_floor"]["normal_max_floor"] == 0
    assert smoke["independent_floor"]["skip_max_floor"] == 0
    state = {
        "phase": "frozen_v4_independent_review_complete",
        "verdict": "PASS_FOR_BOUNDED_UNCHANGED_DEFAULT_CUDA_MEASUREMENT_WITH_MANDATORY_GATES",
        "blocking_findings": [],
        "targeted_checks_pass": result["count"],
        "private_CPU_RNG_exact_roundtrip_fix": "PASS",
        "known_padding_offsets_rejected": [28, 20, 5048],
        "rejection_global_RNG_unchanged": True,
        "legitimate_advanced_cached_roundtrip": True,
        "CPU_reproduction": smoke,
        "protected_all_match": True,
        "all_prior_reports_and_amendments_preserved": True,
        "actual_v4_runtime_sha256": "4d54fc09d9c29646155d73d7293b7759b18671cfd48fc69c975e7db07384e929",
        "controller_current_queue_attestation_and_receipt_still_required": True,
        "CUDA_private_engine_and_parity_measured": False,
        "GPU_launched": False,
        "receipt_written": False,
        "candidate_edits": False,
        "runtime_edits": False,
        "scientist_arms_approved": False,
        "mature_full_fit_reconstruction": False,
        "wall_seconds": result["wall_seconds"],
        "max_rss_kib": result["max_rss_kib"],
        "pending": [],
        "last_failure": None,
    }
    (OWN / "STATE.json").write_text(json.dumps(state, indent=2) + "\n")
    hashes = {
        str(path.relative_to(OWN)): digest(path)
        for path in sorted(OWN.rglob("*"))
        if path.is_file() and path.name not in {"HASHES.json", "freeze.log"}
    }
    (OWN / "HASHES.json").write_text(json.dumps(hashes, indent=2) + "\n")
    assert all(digest(OWN / name) == value for name, value in hashes.items())
    print(
        json.dumps(
            {
                "verdict": state["verdict"],
                "final_manifest_verified": True,
                "report_sha256": hashes["REPORT.md"],
                "state_sha256": hashes["STATE.json"],
                "manifest_sha256": digest(OWN / "HASHES.json"),
                "file_count": len(hashes),
            }
        )
    )


if __name__ == "__main__":
    main()
