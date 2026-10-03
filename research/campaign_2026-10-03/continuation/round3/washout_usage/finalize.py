from __future__ import annotations

import json
import re

from common import PRIOR, ROOT, dump, sha


def elapsed(text: str) -> float:
    value = re.search(r"Elapsed \(wall clock\) time .*?: (\S+)", text).group(1)
    return sum(float(v) * 60**i for i, v in enumerate(reversed(value.split(":"))))


def main() -> None:
    results = json.loads((ROOT / "results.json").read_text())
    verification = json.loads((ROOT / "verification.json").read_text())
    assert (
        verification["passed"]
        and results["gates"]["all_identity_pass"]
        and results["gates"]["all_future_pass"]
    )
    samples = json.loads((ROOT / "samples.json").read_text())
    times = []
    ids = [
        "component\tseq_ix\trow_group\tquery\taccepted\treason\tlag\tquery_pi\tinverse_probability_weight"
    ]
    for i, s in enumerate(samples["samples"]):
        text = (ROOT / f"seq_{i:02d}.log").read_text()
        assert "Exit status: 0" in text
        times.append(
            dict(
                sequence=i,
                wall_seconds=elapsed(text),
                peak_rss_kib=int(
                    re.search(r"Maximum resident set size \(kbytes\): (\d+)", text).group(1)
                ),
            )
        )
        records = json.loads((ROOT / f"gates_{i:02d}.json").read_text())["records"]
        for r in records:
            ids.append(
                "\t".join(
                    map(
                        str,
                        [
                            s["component"],
                            s["seq_ix"],
                            s["row_group"],
                            r["query"],
                            r["accepted"],
                            r["reason"],
                            r["lag"],
                            s["query_pi"],
                            1 / s["query_pi"],
                        ],
                    )
                )
            )
    (ROOT / "IDS.tsv").write_text("\n".join(ids) + "\n")
    cost = dict(
        per_sequence=times,
        sequence_process_wall_seconds=sum(t["wall_seconds"] for t in times),
        max_sequence_process_wall_seconds=max(t["wall_seconds"] for t in times),
        max_sequence_process_rss_mib=max(t["peak_rss_kib"] for t in times) / 1024,
        body=results["cost"],
        verification=verification,
        verification_process_wall_seconds=elapsed((ROOT / "verification.log").read_text()),
        array_artifact_bytes=sum(p.stat().st_size for p in ROOT.glob("*.npz")),
        resource="run_cpu.sh round3-matched-usage1core2GiB900s, no resource failure",
        scope="per-process costs include imports; wrapper admission excluded",
    )
    dump(ROOT / "COST.json", cost)
    state = json.loads((ROOT / "STATE.json").read_text())
    assert state["completed_sequences"] == list(range(24)) and state["failure"] is None
    state.update(
        phase="complete_report_frozen",
        verdict=results["verdict"],
        selected=288,
        accepted=207,
        rejected=81,
        unmatched=0,
        feature_atol=0.0,
        feature_rtol=0.0,
        all_accepted_suffixes_identical=True,
        identity_future_pass=True,
        no_targets_or_wp_used=True,
        prior_artifacts_preserved=128,
        source_frozen=True,
        resource_failure=False,
        visually_inspected=["washout.png"],
        next_work="await_controller_authorization",
        blockers=[
            "economic support of splice/donor",
            "predictive information and exploitable deficit untested",
        ],
        training_proposal=None,
    )
    dump(ROOT / "STATE.json", state)
    frozen = json.loads((ROOT / "frozen_sources.json").read_text())
    for path, expected in frozen.items():
        assert sha(ROOT / path) == expected
    artifacts = {
        p.name: dict(sha256=sha(p), bytes=p.stat().st_size)
        for p in sorted(ROOT.iterdir())
        if p.is_file()
        and p.name not in ["artifact_provenance.json", "finalize.log", "verification_final.log"]
    }
    control = samples["control"]
    dump(
        ROOT / "artifact_provenance.json",
        dict(
            artifacts=artifacts,
            checkpoint=control["checkpoint"],
            checkpoint_sha256=control["checkpoint_sha256"],
            split_sha256=control["split_sha256"],
            source_sha256=control["source_sha256"],
            prior_artifact_manifest_sha256=sha(PRIOR / "artifact_provenance.json"),
            prior_report_sha256=sha(PRIOR / "REPORT.md"),
            prior_array_preservation=verification["prior_artifacts_preserved"],
            sample_sha256=sha(ROOT / "samples.json"),
            prior_sample_sha256=sha(PRIOR / "samples.json"),
            sample_exposure="same previously exposed24components/288queries; full exposure metadata retained in samples.json",
            acceptance="all211 inputs on33rows exactly equal, input-only gate and donor; no origin replaced",
            source_freeze=frozen,
            resource_cost=cost,
            visually_inspected=["washout.png"],
            excluded_hashes="self manifest plus current finalization/final-audit output logs; transient cache directories omitted",
        ),
    )
    print(
        json.dumps(
            dict(
                phase=state["phase"],
                artifacts=len(artifacts),
                accepted=207,
                rejected=81,
                process_wall_seconds=cost["sequence_process_wall_seconds"],
                max_process_seconds=cost["max_sequence_process_wall_seconds"],
                peak_process_rss_mib=cost["max_sequence_process_rss_mib"],
            )
        )
    )


if __name__ == "__main__":
    main()
