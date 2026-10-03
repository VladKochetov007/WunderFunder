from __future__ import annotations

import json
import re

from review import AUTHOR, ROOT, sha


def main() -> None:
    result = json.loads((ROOT / "results.json").read_text())
    check = json.loads((ROOT / "verification.json").read_text())
    assert result["passed"] and check["passed"] and result["fits_executed"] == 0
    for path, expected in json.loads((ROOT / "frozen_sources.json").read_text()).items():
        assert sha(ROOT / path) == expected
    manifest = json.loads((AUTHOR / "PROVENANCE.json").read_text())
    for path, expected in manifest["files_sha256"].items():
        assert sha(AUTHOR / path) == expected
    cost = {}
    for name in ["review", "verification"]:
        text = (ROOT / f"{name}.log").read_text()
        assert "Exit status: 0" in text
        cost[name] = dict(
            process_elapsed=re.search(r"Elapsed \(wall clock\) time .*?: (\S+)", text).group(1),
            peak_rss_kib=int(
                re.search(r"Maximum resident set size \(kbytes\): (\d+)", text).group(1)
            ),
        )
    cost.update(
        fits=0,
        raw_scans=0,
        GPU=False,
        resource="run_cpu.sh round3-matched-usage1core2GiB900s",
        failure=None,
    )
    (ROOT / "COST.json").write_text(json.dumps(cost, indent=2))
    state = dict(
        phase="complete_report_frozen",
        verdict="conditional synthetic predictability reproduced; crossinstrument/real candidate not justified",
        all_properties_pass=True,
        saved_inference_bootstrap_reproduced=True,
        author_artifacts_preserved=26,
        no_fits_or_raw_scans=True,
        matched_capacity_downgrade="equal width, different rank and penalty geometry",
        next_requested_test="ONE input-only peer innovation forecast beyond own history and frozenGRUstate, proposal only",
        next_work="await_controller",
        failure=None,
    )
    (ROOT / "STATE.json").write_text(json.dumps(state, indent=2))
    repo = ROOT.parents[4]
    own = [
        repo / "research/campaign_2026-10-03/eda/statphys/REPORT.txt",
        repo / "research/campaign_2026-10-03/continuation/round2/usage/REPORT.md",
        repo / "research/campaign_2026-10-03/continuation/round3/washout_usage/REPORT.md",
    ]
    provenance = dict(
        artifacts={
            p.name: dict(sha256=sha(p), bytes=p.stat().st_size)
            for p in sorted(ROOT.iterdir())
            if p.is_file()
            and p.name not in ["PROVENANCE.json", "finalize.log", "verification_final.log"]
        },
        author_manifest_sha256=sha(AUTHOR / "PROVENANCE.json"),
        author_source_hashes={
            n: sha(AUTHOR / n)
            for n in ["simulator.py", "representation.py", "experiment.py", "verify_freeze.py"]
        },
        own_prior_report_hashes={str(p.relative_to(repo)): sha(p) for p in own},
        source_freeze=json.loads((ROOT / "frozen_sources.json").read_text()),
        review_data="seeded synthetic only;1152fit/576test paths, no raw cohort/new capture reads",
        cost=cost,
        excluded_hashes="self plus current finalization/final-audit logs",
    )
    (ROOT / "PROVENANCE.json").write_text(json.dumps(provenance, indent=2))
    print(
        json.dumps(
            dict(
                phase=state["phase"],
                artifacts=len(provenance["artifacts"]),
                author_preserved=26,
                elapsed_since_prereg_seconds=(ROOT / "STATE.json").stat().st_mtime
                - (ROOT / "PREREG.json").stat().st_mtime,
            )
        )
    )


if __name__ == "__main__":
    main()
