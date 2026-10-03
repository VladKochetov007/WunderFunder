from __future__ import annotations

import json
from pathlib import Path

from verify_inputs import digest

OWN = Path(__file__).resolve().parent


def main() -> None:
    """Bind final critic evidence; never produce an approval receipt."""
    tests = json.loads((OWN / "results.json").read_text())
    integrity = json.loads((OWN / "protected_verification.json").read_text())
    boundary = json.loads((OWN / "torch_engine_results.json").read_text())
    assert tests["all_pass"] and integrity["all_match"]
    assert boundary["actual_restore_witness"]["restored_RNG_compare_rejected"]
    state = {
        "phase": "frozen_v3_independent_review_complete",
        "verdict": "BLOCKED_TORCH_RNG_ACCEPTED_STATE_NORMALIZATION",
        "NumPy_and_moment_fixes": "PASS",
        "targeted_assertions_pass": tests["count"],
        "pristine_CPU_all_max_deltas": 0,
        "CPU_normal_and_skip_floors": 0,
        "legacy_E385_recovery": "PASS_EXACT_HASH",
        "legacy_STATE_reconstructed": False,
        "final_v2_amendment_preserved": True,
        "protected_all_match": True,
        "actual_runtime_payload_sha256": "d841cee9e6816384d49e5023a66972063d6aa9f51189c0343d40f11d22606917",
        "mandatory_next": "separate-version private RNG engine exact set/get check and independent retest",
        "CUDA_evidence": False,
        "GPU_launched": False,
        "receipt_written": False,
        "GPU_corruption_observed": False,
        "candidate_edits": False,
        "runtime_edits": False,
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
                "file_count": len(manifest),
                "report_sha256": manifest["REPORT.md"],
                "manifest_sha256": digest(OWN / "HASHES.json"),
            }
        )
    )


if __name__ == "__main__":
    main()
