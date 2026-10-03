from __future__ import annotations

import json
import re

from analyze import ROOT, sha


def main() -> None:
    """Freeze guard-only artifacts and verify preservation of the original study."""
    for name, expected in json.loads((ROOT / "FROZEN_SPEC.json").read_text()).items():
        assert sha(ROOT / name) == expected
    tests = json.loads((ROOT / "tests.json").read_text())
    assert tests["passed"] and not tests["actual_capture_analysis_executed"]
    original = ROOT.parent / "mechanism_prereg"
    manifest = json.loads((original / "HASHES.json").read_text())
    for name, item in manifest["artifacts"].items():
        assert sha(original / name) == item["sha256"]
    review = ROOT.parent / "mechanism_review"
    provenance = dict(
        original_manifest_sha256=sha(original / "HASHES.json"),
        original_artifacts_unchanged=len(manifest["artifacts"]),
        consumed_review_sha256={name: sha(review / name) for name in ["REPORT.txt", "STATE.json"]},
    )
    (ROOT / "INPUT_PROVENANCE.json").write_text(json.dumps(provenance, indent=2))
    costs = []
    for name in ["tests.log", "tests_final.log"]:
        body = (ROOT / name).read_text()
        elapsed = re.search(r"Elapsed \(wall clock\) time .*?: ([0-9:.]+)", body).group(1)
        seconds = 0.0
        for part in elapsed.split(":"):
            seconds = seconds * 60 + float(part)
        costs.append(
            dict(
                log=name,
                seconds=seconds,
                peak_rss_mib=int(
                    re.search(r"Maximum resident set size \(kbytes\): (\d+)", body).group(1)
                )
                / 1024,
            )
        )
    (ROOT / "COST.json").write_text(
        json.dumps(
            dict(
                timed_tests=costs,
                total_seconds=sum(c["seconds"] for c in costs),
                peak_rss_mib=max(c["peak_rss_mib"] for c in costs),
                excludes="format/lint/hash/report/import finalization",
                wrapper="round3-matched-usage1core2GiB900s",
            ),
            indent=2,
        )
    )
    state = dict(
        phase="guard_correction_frozen_awaiting_controller",
        pass_complete=True,
        verdict="BLOCK_ACTUAL_CAPTURE_SCIENCE_PENDING_APPROVED_V4_AND_INDEPENDENT_DECOMPOSITION_POLICY",
        actual_capture_reads=0,
        actual_capture_analysis_executed=False,
        measured_cuda_results_read=False,
        gpu_launched=False,
        fits=0,
        scores=0,
        guard_tests_passed=True,
        rejection_cases=len(tests["rejection_cases"]),
        independently_accepted_actual_decomposition_bound=None,
        default_floor_transfer_forbidden=True,
        descriptive_geometry="UNCERTIFIED componentselfgradient only",
        source_coefficients_and_denominators="UNAVAILABLE",
        tbptt_drift="UNIDENTIFIED",
        original_artifacts_preserved=provenance["original_artifacts_unchanged"],
        source_hashes={
            name: sha(ROOT / name) for name in ["PREREG.md", "analyze.py", "tests.py", "REPORT.md"]
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
        f"Frozen: {len(artifacts)} artifact hashes; original {provenance['original_artifacts_unchanged']} preserved; actual science BLOCKED"
    )


if __name__ == "__main__":
    main()
