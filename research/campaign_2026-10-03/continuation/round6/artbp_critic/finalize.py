from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def sha(path: Path) -> str:
    """Hash small critic/source artifacts without any actual capture access."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    """Freeze the blind math report, measured fixtures and preserved first failure."""
    assert json.loads((ROOT / "results.json").read_text())["passed"]
    assert (
        sha(ROOT / "PREREG.md")
        == json.loads((ROOT / "PREREG_SHA256.json").read_text())["PREREG.md"]
    )
    initial = json.loads((ROOT / "HASHES_first_execution.json").read_text())
    assert sha(ROOT / "checks_executed.py") == initial["artifacts"]["checks.py"]["sha256"]
    assert sha(ROOT / "results.json") == initial["artifacts"]["results.json"]["sha256"]
    preserved = {}
    prior_paths = [
        ROOT.parent / "amp_linearity_audit",
        ROOT.parent / "amp_linearity_audit_v2",
        ROOT.parent.parent / "round5/mechanism_prereg",
        ROOT.parent.parent / "round5/mechanism_instrument_v2",
    ]
    for directory in prior_paths:
        manifest = json.loads((directory / "HASHES.json").read_text())
        for name, item in manifest["artifacts"].items():
            assert sha(directory / name) == item["sha256"]
        preserved[directory.name] = dict(
            artifacts_unchanged=len(manifest["artifacts"]),
            manifest_sha256=sha(directory / "HASHES.json"),
        )
    (ROOT / "PRESERVATION.json").write_text(json.dumps(preserved, indent=2))
    costs = []
    for name in ["checks.log", "checks_final.log"]:
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
                processes=costs,
                total_seconds=sum(c["seconds"] for c in costs),
                initial_admission_failure=75,
                retry="explicit controller authorization, unchanged conditions",
                intentional_scalar_fp16_overflow_fixture=True,
                body_resource_failures=0,
                wrapper="round3-matched-usage1core2GiB900s",
            ),
            indent=2,
        )
    )
    state = dict(
        phase="blind_math_premortem_frozen_ready_for_actor_crossreview",
        pass_complete=True,
        actor_reads=0,
        actual_capture_or_gradient_reads=0,
        gpu_or_model_training_runs=0,
        symbolic_derivations_complete=True,
        tiny_exhaustive_checks_passed=True,
        no_update_or_score_promotion=True,
        first_admission_failure_preserved=True,
        explicit_controller_retry_authorized=True,
        critic_runtime_or_barrier_edits=False,
        prereg_erratum="wrong marginal mean11.5 corrected symbolically to13 before numerical execution",
        originals_preserved=preserved,
        actor_crossreview="NOT_STARTED_WAIT_CONTROLLER",
        source_hashes={
            name: sha(ROOT / name)
            for name in ["PREREG.md", "ERRATUM.txt", "checks.py", "checks_executed.py", "REPORT.md"]
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
    print(f"Frozen blind critic {len(artifacts)} hashes; actor UNREAD, math controls PASS")


if __name__ == "__main__":
    main()
