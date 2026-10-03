from __future__ import annotations

import json
import re

from audit import ROOT, sha


def main() -> None:
    """Freeze a tested metadata correction after critic exact terminal findings."""
    for name, expected in json.loads((ROOT / "FROZEN_SPEC.json").read_text()).items():
        assert sha(ROOT / name) == expected
    tests = json.loads((ROOT / "tests.json").read_text())
    assert tests["passed"] and tests["actual_gradient_reads"] == 0
    critic = ROOT.parent / "amp_linearity_review"
    critic_state = json.loads((critic / "STATE.json").read_text())
    assert (
        critic_state["verdict"] == "BLOCKED_REAL_METADATA_JSON_IDENTITY_SCHEMA"
        and critic_state["pending"] == []
    )
    preserved = {}
    locations = [
        ROOT.parent / "amp_linearity_audit",
        ROOT.parent.parent / "round5/mechanism_prereg",
        ROOT.parent.parent / "round5/mechanism_instrument_v2",
    ]
    for prior in locations:
        manifest = json.loads((prior / "HASHES.json").read_text())
        for name, item in manifest["artifacts"].items():
            assert sha(prior / name) == item["sha256"]
        preserved[prior.name] = dict(
            artifacts_unchanged=len(manifest["artifacts"]),
            manifest_sha256=sha(prior / "HASHES.json"),
        )
    provenance = dict(
        originals_preserved=preserved,
        critic_consumed_sha256={
            name: sha(critic / name)
            for name in ["REPORT.md", "STATE.json", "MINIMAL_FIX_SPEC.md", "tests.json"]
        },
    )
    (ROOT / "INPUT_PROVENANCE.json").write_text(json.dumps(provenance, indent=2))
    log = (ROOT / "tests.log").read_text()
    elapsed = re.search(r"Elapsed \(wall clock\) time .*?: ([0-9:.]+)", log).group(1)
    seconds = 0.0
    for part in elapsed.split(":"):
        seconds = seconds * 60 + float(part)
    (ROOT / "COST.json").write_text(
        json.dumps(
            dict(
                test_process_wall_seconds=seconds,
                peak_test_rss_mib=int(
                    re.search(r"Maximum resident set size \(kbytes\): (\d+)", log).group(1)
                )
                / 1024,
                test_body_seconds=tests["seconds"],
                wrapper="round3-matched-usage1core2GiB900s",
                actual_reads=0,
            ),
            indent=2,
        )
    )
    state = dict(
        phase="metadata_v2_cpu_tested_frozen_awaiting_independent_review",
        pass_complete=True,
        exact_critic_finding="BLOCKED_REAL_METADATA_JSON_IDENTITY_SCHEMA",
        metadata_projection="metadata-json-v1",
        public_helper="metadata.py:canonical_metadata",
        actual_capture_or_gradient_reads=0,
        actual_analysis_executed=False,
        corrected_instrument_independent_review="PENDING",
        controller_actual_analysis_authorized=False,
        tests_passed=True,
        rejected_fixture_cases=len(tests["rejection_cases"]),
        preserved_original_artifacts=preserved,
        fixed_cases=["synthetic_actual_architecture_normal", "real_panel_normal"],
        acceptance_threshold=None,
        decomposition_bound_inferred=False,
        scientific_conflict_update_score_claim=False,
        resource_failures=0,
        source_hashes={
            name: sha(ROOT / name)
            for name in ["PREREG.md", "audit.py", "metadata.py", "tests.py", "REPORT.md"]
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
        f"Frozen metadata v2: {len(artifacts)} hashes, original19+20+25 preserved, actual gradients UNREAD"
    )


if __name__ == "__main__":
    main()
