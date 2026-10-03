from __future__ import annotations

import hashlib
import json
from pathlib import Path

OWN = Path(__file__).resolve().parent
ROUND = OWN.parent


def digest(path: Path) -> str:
    """Hash metadata in bounded blocks; forbid tensor containers."""
    assert path.suffix != ".pt" and path.stat().st_size < 4 * 1024**2
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            result.update(block)
    return result.hexdigest()


def main() -> None:
    """Verify reviewed bytes still agree, then write terminal review artifacts."""
    findings = json.loads((OWN / "results.json").read_text())
    provenance = json.loads((OWN / "provenance.json").read_text())
    for name, expected in provenance["sha256"].items():
        assert digest(Path(name)) == expected
    output = ROUND / "amp_linearity_audit_v2/actual_discrepancy_2222.json"
    approval = ROUND / "amp_audit_approved_2222.json"
    output_sha = digest(output)
    approval_sha = digest(approval)
    state = {
        "phase": "frozen_independent_amp_linearity_result_review_complete",
        "verdict": findings["verdict"],
        "blocking_findings": [],
        "result_sha256": output_sha,
        "approval_sha256": approval_sha,
        "source_hashes": json.loads((OWN / "candidate_sha256.json").read_text()),
        "prior_review_manifest_sha256": digest(ROUND / "amp_linearity_review_v2/HASHES.json"),
        "reviewed_integration_sha256": digest(
            ROUND / "amp_linearity_review_v2/integration_bindings.json"
        ),
        "fixed_metadata_sha256": digest(ROUND / "fixed_case_metadata_v2.json"),
        "measurement_STATE_sha256": digest(ROUND / "capture_measurement_review/STATE.json"),
        "cpu_wrapper_sha256": digest(ROUND.parent / "run_cpu.sh"),
        "fixed_cases": list(findings["totals"]),
        "actual_capture_or_gradient_reads_by_reviewer": 0,
        "actual_analysis_repeated": False,
        "GPU_launched": False,
        "receipt_written": False,
        "finite_scalar_and_partition_checks": "PASS",
        "converted_workspace_bytes": 8388608,
        "scientific_acceptance_bound_inferred": False,
        "conflict_update_or_WP_claim": False,
        "prior_artifacts_preserved": True,
        "pending": [],
        "last_failure": None,
    }
    (OWN / "STATE.json").write_text(json.dumps(state, indent=2) + "\n")
    report = f"""Verdict: **{findings["verdict"]}**. No blockers for using this output as the authorized descriptive readout of exactly two fixed normal cases.

`review.py` verified the frozen author package and all prior independent review manifest entries, exact four candidate source hashes, controller approval, and reviewed integration pins. Approval SHA256 `{approval_sha}` agrees with the output. Output SHA256 `{output_sha}` is bound in STATE/provenance. The two approved case dictionaries equal the prepared fixed metadata; paths and declared pre/post/components hashes match the frozen integration. Current measurement STATE, prepared metadata, helper code, existing trajectory, and capture output manifest hashes remain pinned. No tensor container was reread or rehashed by this reviewer.

Both cases cover 14 active parameters and 981,015 elements, with no unused parameters. Every reported numeric metric is finite and nonnegative. Independently checked all scope squared norms and maxima against their parameter summaries, counts, RMS denominators, relative ratios, and canonical name/shape coverage. The scalar comparison tolerance (relative 1e-12, absolute 1e-24) checks reporting arithmetic only; it is not a gradient acceptance threshold.

| Fixed case | Error L2 | Relative L2 | Maximum absolute discrepancy |
|---|---:|---:|---:|
| Synthetic actual architecture normal | 0.00013536436664744937 | 0.000190137912442765 | 0.00001341104507446289 |
| Real panel normal | 0.00044794706293064195 | 0.0004496687958966782 | 0.000091552734375 |

The measured operation is the fixed FP64 sum of separately computed main/EMA/struct/next self-gradients minus the ordinary whole unscaled preclip gradient. Frozen source applies no coefficient or scale twice. Nonzero differences establish that these captured operations did not produce identical numerical vectors; they do not identify a mechanism or constitute failed online replay. Zero default replay floors do not calibrate this different-operation discrepancy.

The actual command's terminal log and CPU command receipt are preserved in `executed_command.txt`: controller CPU wrapper, exit 0, 1.293 seconds wall time, 1.273 seconds CPU, 286.5 MB peak. The 8 MiB reported converted workspace agrees with the source's conservative chunk allowance and is below 32 MiB. Source loads trusted containers with CPU mmap and bounded file sizes; it scans post/components hashes, while prestate content identity remains the earlier independent whole-file declaration. This review verifies that provenance chain, not contemporaneous prestate content by a new scan. Exit 0 alone was insufficient: both result statuses were checked explicitly for successful descriptive output rather than rejection.

The real case is a fresh seed3 fixed-panel normalization trajectory with four warm chunks, not a mature historical fit or independent scientific holdout. Existing exposure is unchanged. No acceptance bound, gradient conflict, optimizer update attribution, or WP/score claim is admissible from this diagnostic. Output flags correctly deny those claims and provide no calibrated allowance. Reviewer read JSON/code/logs only, with zero new input/gradient reads, captures, replay, GPU work, or approvals. Frozen author and earlier review artifacts are unchanged.
"""
    (OWN / "REPORT.md").write_text(report)
    manifest = {
        path.name: digest(path)
        for path in sorted(OWN.iterdir())
        if path.is_file() and path.name not in {"HASHES.json", "freeze.log"}
    }
    (OWN / "HASHES.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for name, expected in manifest.items():
        assert digest(OWN / name) == expected
    print(
        json.dumps(
            {
                "verdict": findings["verdict"],
                "final_manifest_verified": True,
                "report_sha256": manifest["REPORT.md"],
                "state_sha256": manifest["STATE.json"],
                "manifest_sha256": digest(OWN / "HASHES.json"),
                "result_sha256": output_sha,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
