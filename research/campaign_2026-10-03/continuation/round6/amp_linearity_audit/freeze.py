from __future__ import annotations

import json
import re

from audit import ROOT, sha


def main() -> None:
    """Freeze an awaiting-approval audit, never a completed actual measurement."""
    for manifest in ["PREREG_SHA256.json", "FROZEN_SPEC.json"]:
        for name, expected in json.loads((ROOT / manifest).read_text()).items():
            assert sha(ROOT / name) == expected
    tests = json.loads((ROOT / "tests.json").read_text())
    assert tests["passed"] and tests["actual_gradient_reads"] == 0
    preserved = {}
    round5 = ROOT.parent.parent / "round5"
    for directory in ["mechanism_prereg", "mechanism_instrument_v2"]:
        prior = round5 / directory
        manifest = json.loads((prior / "HASHES.json").read_text())
        for name, item in manifest["artifacts"].items():
            assert sha(prior / name) == item["sha256"]
        preserved[directory] = dict(
            artifacts_unchanged=len(manifest["artifacts"]),
            manifest_sha256=sha(prior / "HASHES.json"),
        )
    (ROOT / "PRIOR_PRESERVATION.json").write_text(json.dumps(preserved, indent=2))
    body = (ROOT / "tests.log").read_text()
    elapsed = re.search(r"Elapsed \(wall clock\) time .*?: ([0-9:.]+)", body).group(1)
    seconds = 0.0
    for part in elapsed.split(":"):
        seconds = seconds * 60 + float(part)
    (ROOT / "COST.json").write_text(
        json.dumps(
            dict(
                test_process_wall_seconds=seconds,
                peak_test_rss_mib=int(
                    re.search(r"Maximum resident set size \(kbytes\): (\d+)", body).group(1)
                )
                / 1024,
                test_body_seconds=tests["seconds"],
                wrapper="round3-matched-usage1core2GiB900s",
                excludes="format/lint/hashing/editing/finalization",
                actual_inference_or_fit=0,
            ),
            indent=2,
        )
    )
    state = dict(
        phase="preregistered_audit_tested_frozen_awaiting_independent_review_and_controller",
        pass_complete=True,
        actual_gradient_reads=0,
        actual_analysis_executed=False,
        actual_analysis_authorized=False,
        independently_reviewed_actual_case_receipt=None,
        fixed_cases=["synthetic_actual_architecture_normal", "real_panel_normal"],
        synthetic_tests_passed=True,
        rejected_fixture_cases=len(tests["rejection_cases"]),
        measured_actual_discrepancy=None,
        decomposition_bound_inferred=False,
        calibrated_allowance=None,
        scientific_update_or_conflict_claim=False,
        gpu_or_actual_autograd_forward=0,
        preserved_original_artifacts=preserved,
        resource_failures=0,
        source_hashes={
            name: sha(ROOT / name) for name in ["PREREG.md", "audit.py", "tests.py", "REPORT.md"]
        },
    )
    (ROOT / "STATE.json").write_text(json.dumps(state, indent=2))
    artifacts = {
        p.name: dict(sha256=sha(p), bytes=p.stat().st_size)
        for p in sorted(ROOT.iterdir())
        if p.is_file() and p.name not in {"HASHES.json", "freeze.log"}
    }
    (ROOT / "HASHES.json").write_text(
        json.dumps(dict(artifacts=artifacts, exclusions="self and open freeze.log"), indent=2)
    )
    for name, item in artifacts.items():
        assert sha(ROOT / name) == item["sha256"]
    print(
        f"Frozen {len(artifacts)} hashes; original20+25 preserved; actual descriptive measurement UNEXECUTED"
    )


if __name__ == "__main__":
    main()
