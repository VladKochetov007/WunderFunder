from __future__ import annotations

import hashlib
import json
from pathlib import Path

OWN = Path(__file__).resolve().parent
ROOT = OWN.parents[4]


def digest(path: Path) -> str:
    """Stream small provenance/final-review files in1MiB chunks."""
    value = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def main() -> None:
    """Freeze the completed measured-parity audit without rewriting captures."""
    byte_audit = json.loads((OWN / "output_byte_verification.json").read_text())
    lineage = json.loads((OWN / "lineage_and_numeric_results.json").read_text())
    metadata = json.loads((OWN / "capture_metadata_results.json").read_text())
    guard = json.loads((OWN / "current_guard_results.json").read_text())
    assert (
        byte_audit["all_match"]
        and lineage["binding_checks_pass"]
        and metadata["all_pass"]
        and guard["all_pass"]
    )
    external = {
        name: digest(ROOT / name)
        for name in (
            "research/campaign_2026-09-29/jobs/c20261003_capture_v4_default_2120r.json",
            "research/campaign_2026-09-29/logs/c20261003_capture_v4_default_2120r.log",
            "research/campaign_2026-10-03/continuation/round4/controller_receipts/capture_v4_2120.json",
        )
    }
    (OWN / "external_provenance_hashes.json").write_text(json.dumps(external, indent=2) + "\n")
    job = json.loads(
        (
            ROOT / "research/campaign_2026-09-29/jobs/c20261003_capture_v4_default_2120r.json"
        ).read_text()
    )
    runtime = job["runtime_provenance"]
    assert runtime["slice"] == "wnn-campaign-gpu.slice"
    assert runtime["effective_slice_properties"]["CPUQuotaPerSecUSec"] == "8s"
    assert runtime["effective_slice_properties"]["MemoryMax"] == "25769803776"
    assert runtime["effective_slice_properties"]["MemorySwapMax"] == "0"
    assert runtime["environment"]["WNN_CAMPAIGN_VRAM_BYTES"] == "5500000000"
    state = {
        "phase": "frozen_independent_measurement_review_complete",
        "verdict": "PASS_MEASURED_PARITY_FOR_RECORDED_WARM_V4_DEFAULT_TRAJECTORIES_ONLY",
        "issues_blocking_narrow_measurement_claim": [],
        "job": lineage["job"],
        "output_files_verified": len(byte_audit["files"]),
        "output_bytes_streamed_once": byte_audit["total_output_bytes"],
        "full_output_byte_hash_passes": 1,
        "six_original_controls": 4,
        "total_original_samples_hash_verified": 24,
        "normal_default_repeat_skip_max_deltas": 0,
        "all_four_policy_floors_and_limits": 0,
        "metadata_only_CPU_mmap_captures": 4,
        "saved_stopped_targets_exact_supported_formula": True,
        "warm_attempts": [4, 5],
        "normal_Adam_steps": [4, 5],
        "skip_Adam_steps": [4, 4],
        "EMA_and_scheduler_advance_on_skip": True,
        "policy_order_evidence": "hash-bound source + consistent mtimes; no independent comparison-event trace",
        "parent_oom_count": 11,
        "gpu_family_oom_count": 0,
        "guard_watch_timestamp": guard["watch"]["time"],
        "unit_completion_limit": "transient units now not-found/inactive; terminal success supported by retained monitor job record/log and receipt",
        "GPU_launched_by_reviewer": False,
        "CUDA_private_engine_validation_by_reviewer": False,
        "GRU_or_backend_replay_by_reviewer": False,
        "capture_edits": False,
        "receipt_written": False,
        "scientific_component_bounds_inferred": False,
        "mature_full_fit_state_recovered": False,
        "external_full_prediction_cache_rehash": False,
        "initial_inspection_error": "KeyError evaluation.train; corrected schema inspection and resumed saved byte hashes, no second full pass",
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
