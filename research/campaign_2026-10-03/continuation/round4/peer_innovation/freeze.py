from __future__ import annotations

import argparse
import json
import re

from common import ROOT, dump, sha


def audit() -> None:
    manifest = json.loads((ROOT / "artifact_provenance.json").read_text())
    for name, item in manifest["artifacts"].items():
        assert sha(ROOT / name) == item["sha256"], name
    assert json.loads((ROOT / "STATE.json").read_text())["phase"] == "complete_report_frozen"
    print(f"Frozen artifact audit passed: {len(manifest['artifacts'])} hashes")


def main() -> None:
    verification = json.loads((ROOT / "verification.json").read_text())
    results = json.loads((ROOT / "results.json").read_text())
    assert verification["passed"] and not results["promotion_allowed"]
    costs = []
    for name in [*[f"seq_{i:02d}.log" for i in range(24)], "controls.log", "real_fit.log", "verification.log", "verification_corrected.log"]:
        body = (ROOT / name).read_text()
        elapsed = re.search(r"Elapsed \(wall clock\) time .*?: ([0-9:.]+)", body).group(1)
        seconds = 0.0
        for part in elapsed.split(":"):
            seconds = seconds * 60 + float(part)
        costs.append(dict(log=name, process_wall_seconds=seconds,
                          peak_rss_mib=int(re.search(r"Maximum resident set size \(kbytes\): (\d+)", body).group(1))/1024,
                          exit_status=int(re.search(r"Exit status: (\d+)", body).group(1))))
    dump("COST.json", dict(processes=costs, timed_wall_sum_seconds=sum(c["process_wall_seconds"] for c in costs),
         max_process_wall_seconds=max(c["process_wall_seconds"] for c in costs),
         peak_process_rss_mib=max(c["peak_rss_mib"] for c in costs),
         all_admission="run_cpu.sh round3-matched-usage, CUDA unavailable,1core2GiB900s",
         excludes="shell admission/startup queue and editing/report/finalization time; finalization bounded hash-only",
         real_ridge_fits=3, toy_ridge_fits=6, gru_window_forwards=311, original_prefix_forwards=0,
         resource_failures=0, verification_failures_preserved=1))
    panel = json.loads((ROOT / "samples.json").read_text())
    origins = json.loads((ROOT / "origins.json").read_text())["origins"]
    missing = [r for r in origins if not r["eligible"]]
    dump("STATE.json", dict(phase="complete_report_frozen", completed_sequences=list(range(24)),
         verdict=results["verdict"], promotion_allowed=False, selected_origins=288, eligible_origins=287,
         missing_origins=missing, verification_passed=True,
         resolved_verification_failure="verification.log retained: float64 versus scalar float32 target recreation; no scored artifacts changed",
         resource_failure=None, all_prior_exposed=True, next_work="await controller; no representation/training proposal justified",
         checkpoint_sha256=panel["control"]["checkpoint_sha256"],
         source_sha256=panel["control"]["source_sha256"],
         prereg_sha256=sha(ROOT / "PREREG.json"), result_sha256=sha(ROOT / "results.json"),
         report_sha256=sha(ROOT / "REPORT.md")))
    excluded = {"artifact_provenance.json", "freeze.log", "final_audit.log"}
    artifacts = {str(p.relative_to(ROOT)): dict(sha256=sha(p), bytes=p.stat().st_size)
                 for p in sorted(ROOT.rglob("*")) if p.is_file() and p.name not in excluded and "__pycache__" not in p.parts}
    dump("artifact_provenance.json", dict(artifacts=artifacts, control=panel["control"],
         own_source_freeze=["frozen_sources.json", "frozen_fit_sources.json"],
         preserved_prior_artifacts=verification["prior_artifacts_unchanged"],
         exclusions="manifest cannot hash itself; open freeze/final_audit logs excluded to avoid mutable-file hashes"))
    print(json.dumps(dict(artifacts=len(artifacts), cost=json.loads((ROOT / "COST.json").read_text()), missing=missing), indent=2))
    audit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", action="store_true")
    args = parser.parse_args()
    audit() if args.audit else main()
