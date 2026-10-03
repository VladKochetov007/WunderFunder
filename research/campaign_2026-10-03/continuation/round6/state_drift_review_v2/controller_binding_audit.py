"""Source-only controller review plus authenticated small metadata hashes."""
from __future__ import annotations

import ast
import hashlib
import json
import resource
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
R6 = HERE.parent
ROOT = HERE.parents[4]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    started = time.perf_counter()
    constructor = R6 / "approve_state_drift.py"
    source = constructor.read_text()
    ast.parse(source)  # syntax only, no constructor execution
    guard_marker = source.index('raise ValueError("Reviewed trajectory, measurement review or provenance helper changed")')
    receipt_marker = source.index('receipt = {')
    fields = ["trajectory_sha256", "independent_measurement_review_state_sha256", "metadata_helper_sha256"]
    assert all(source.index('inventory["' + name + '"]') < guard_marker < receipt_marker for name in fields)
    assert source.index('raise ValueError("Capture files differ from measured manifest")') < receipt_marker
    review = R6 / "capture_measurement_review"
    review_manifest = json.loads((review / "HASHES.json").read_text())
    verified_path = review / "output_byte_verification.json"
    assert sha(verified_path) == review_manifest[verified_path.name]
    assert sha(review / "STATE.json") == review_manifest["STATE.json"]
    verified = json.loads(verified_path.read_text())
    output_manifest = R6.parents[0] / "round4/gpu_capture_v4_default_2120r/output_hashes.json"
    declared = json.loads(output_manifest.read_text())["files"]
    capture_hashes = {}
    for kind in ("pre", "post"):
        name = f"real_fit_panel/normal/{kind}.pt"
        assert declared[name] == verified["files"][name]["sha256"]
        capture_hashes[kind] = declared[name]
    result = {
        "passed": True,
        "constructor_metadata_guard_fields": fields,
        "constructor_guards_before_receipt": True,
        "constructor_capture_actual_hash_comparison_source_present": True,
        "capture_output_manifest_sha256": sha(output_manifest),
        "capture_output_manifest_path": str(output_manifest.relative_to(ROOT)),
        "declared_capture_sha256": capture_hashes,
        "capture_payloads_rehashed": False,
        "independent_byte_audit_metadata_sha256": sha(verified_path),
        "controller_source_sha256": {
            str(constructor.relative_to(ROOT)): sha(constructor),
            "research/campaign_2026-10-03/continuation/round4/approve_capture.py": sha(R6.parents[0] / "round4/approve_capture.py"),
        },
        "seconds": time.perf_counter() - started,
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "constructor_executed": False, "actual_tensor_reads": 0, "GPU_calls": 0,
    }
    (HERE / "controller_binding_results.json").write_text(json.dumps(result, indent=2) + "\n")
    print("PASS: corrected constructor metadata guards and declared capture hashes match authenticated review metadata; no payload reads.")


if __name__ == "__main__":
    main()
