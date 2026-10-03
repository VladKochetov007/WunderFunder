from __future__ import annotations

import json

from analyze import ROOT, sha


def main() -> None:
    """Freeze the completed preregistration pass, not an actual capture analysis."""
    spec = json.loads((ROOT / "FROZEN_SPEC.json").read_text())
    for name, expected in spec.items():
        assert sha(ROOT / name) == expected, name
    assert json.loads((ROOT / "controls.json").read_text())["passed"]
    state = dict(phase="preregistered_instrument_tested_awaiting_controller", pass_complete=True,
                 actual_capture_analysis_executed=False, future_measured_cuda_results_read=False,
                 actual_capture_paths=[], synthetic_controls_passed=True, actual_capture_review="pending",
                 gpu_launched=False, fits_executed=0, score_claims=False,
                 state_drift="UNIDENTIFIED without same-weight same-prefix reference",
                 structural_denominator="UNAVAILABLE until independent source binding",
                 resolved_fixture_failure="exact cosine assertions corrected to previously declared1e-12; old code/hash/log preserved",
                 resource_failures=0, prereg_sha256=sha(ROOT / "PREREG.md"),
                 analysis_sha256=sha(ROOT / "analyze.py"), report_sha256=sha(ROOT / "REPORT.md"))
    (ROOT / "STATE.json").write_text(json.dumps(state, indent=2))
    artifacts = {p.name: dict(sha256=sha(p), bytes=p.stat().st_size) for p in sorted(ROOT.iterdir())
                 if p.is_file() and p.name not in {"HASHES.json", "finalize.log"}}
    (ROOT / "HASHES.json").write_text(json.dumps(dict(artifacts=artifacts, exclusions="self and open finalize.log"), indent=2))
    for name, item in artifacts.items():
        assert sha(ROOT / name) == item["sha256"]
    print(f"Frozen preregistration pass: {len(artifacts)} hashes; actual scientific capture analysis UNEXECUTED")


if __name__ == "__main__":
    main()
