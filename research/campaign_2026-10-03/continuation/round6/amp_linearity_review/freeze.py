from __future__ import annotations

import json
from pathlib import Path

from verify import digest

OWN = Path(__file__).resolve().parent


def main() -> None:
    """Freeze an exact schema blocker without author changes or data admission."""
    tests = json.loads((OWN / "tests.json").read_text())
    verified = json.loads((OWN / "protected_verification.json").read_text())
    assert tests["algebra_oracle_pass"] and verified["all_author_files_match"]
    assert tests["real_source_schema_tiny_witness"]["approved_JSON_case_rejected"]
    state = {
        "phase": "frozen_independent_amp_linearity_review_complete",
        "verdict": "BLOCKED_REAL_METADATA_JSON_IDENTITY_SCHEMA",
        "algebra_descriptive_FP64_residual": "PASS",
        "blockers": [
            "real pre.metadata contains cached_propensity/scored_mask Tensors; exact dictionary equality with approval JSON is incompatible"
        ],
        "tiny_json_metadata_positive_control": True,
        "actual_schema_witness": tests["real_source_schema_tiny_witness"],
        "author_frozen_files_preserved": True,
        "author_artifacts_verified": len(verified["author_checks"]),
        "converted_workspace_ceiling_bytes": verified["conservative_array_workspace_ceiling_bytes"],
        "actual_capture_tensor_reads": 0,
        "GPU_launched": False,
        "autograd_or_replay": False,
        "author_edits": False,
        "receipt_written": False,
        "scientific_acceptance_bound_inferred": False,
        "conflict_update_or_WP_claim": False,
        "required_next": "separate corrected metadata-only JSON projection and independent retest; no actual diagnostic admission as frozen",
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
            }
        )
    )


if __name__ == "__main__":
    main()
