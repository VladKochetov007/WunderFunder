"""Prepare v3 controller patch and read-only preservation/critic lineage."""

from __future__ import annotations

import difflib
import hashlib
import json

from live_capture import HERE, ROOT, sha


def main() -> None:
    preserved = {}
    for folder in ("live_capture", "live_capture_v2"):
        previous = HERE.parent / folder
        protected = {}
        for line in (previous / "protected_hashes.sha256").read_text().splitlines():
            expected, name = line.split(maxsplit=1)
            actual = sha(ROOT / name)
            if actual != expected:
                raise ValueError("previous protected hash changed: " + name)
            protected[name] = actual
        artifacts = json.loads((previous / "artifact_hashes.json").read_text())
        for name, expected in artifacts.items():
            if sha(previous / name) != expected:
                raise ValueError("previous artifact changed: " + name)
        preserved[folder] = {
            "protected": protected,
            "protected_count": len(protected),
            "artifact_count": len(artifacts),
            "all_file_sha256": {
                str(path.relative_to(previous)): sha(path)
                for path in previous.rglob("*")
                if path.is_file()
            },
        }
    payload = HERE.parent / "controller_gpu_payload.sh"
    original = payload.read_text()
    prior_scope = "scope=research/campaign_2026-10-03/continuation/round4/live_capture_v2"
    if prior_scope not in original or "timeout --signal=TERM --kill-after=5s 585" not in original:
        raise ValueError("controller current payload not reviewed v2/time-bound source")
    revised = original.replace(
        prior_scope, "scope=research/campaign_2026-10-03/continuation/round4/live_capture_v3"
    )
    (HERE / "controller_payload.patch").write_text(
        "".join(
            difflib.unified_diff(
                original.splitlines(True),
                revised.splitlines(True),
                fromfile=str(payload.relative_to(ROOT)),
                tofile=str(payload.relative_to(ROOT)),
            )
        )
    )
    runtime_files = (
        "scripts/campaign_job.py",
        "research/campaign_2026-10-03/continuation/run_gpu.sh",
        "scripts/limited",
        str(payload.relative_to(ROOT)),
    )
    runtime = {
        "current_runtime_sha256": {name: sha(ROOT / name) for name in runtime_files},
        "controller_payload_expected_after_patch": hashlib.sha256(revised.encode()).hexdigest(),
        "unapplied_patch_sha256": sha(HERE / "controller_payload.patch"),
        "child_TERM_seconds": 585,
        "child_KILL_grace_seconds": 5,
        "max_nominal_child_seconds": 590,
        "outer_execution_seconds": 600,
        "absolute_cutguard": "2026-10-03 20:00UTC =23:00 Europe/Kyiv,605s entry guard unchanged",
        "launch_chain": "campaign_job gpu threads8 timeout600 -> run_gpu.sh -> controller_gpu_payload.sh CONTROLLER_RECEIPT FRESH_V3_OUTPUT",
        "controller_patch_applied": False,
        "actor_shell_launch": "unsupported",
    }
    (HERE / "runtime_manifest.json").write_text(json.dumps(runtime, indent=2) + "\n")
    review = HERE.parent / "capture_review_v2"
    files = (
        "REPORT.md",
        "STATE.json",
        "results.json",
        "V3_MINIMAL_SPEC.md",
        "followup_results.json",
        "rng_prestate_results.json",
        "moment_cast_results.json",
    )
    if any(not (review / name).exists() for name in files):
        raise ValueError("critic final report/spec not yet frozen")
    critic_state = json.loads((review / "STATE.json").read_text())
    lineage = {
        "preserved": preserved,
        "review_v2_verdict": critic_state["verdict"],
        "original_three_fixes_independently_pass": critic_state[
            "prior_three_blocking_corrections_retested_pass"
        ],
        "review_files": {name: sha(review / name) for name in files},
        "panel_byte_identical_to_v2": sha(HERE / "fit_panel.json")
        == sha(HERE.parent / "live_capture_v2/fit_panel.json"),
        "actual_trainer_sha256": sha(ROOT / "scripts/train_rnn.py"),
        "v3_only_fix": "canonical RNG validation/exact NumPy-scalar bytes and final-critic moment storage grammar; policy unchanged",
    }
    (HERE / "LINEAGE.json").write_text(json.dumps(lineage, indent=2) + "\n")
    print(
        json.dumps(
            {
                "previous_versions_preserved": list(preserved),
                "critic_final_read": True,
                "critic_verdict": critic_state["verdict"],
                "controller_patch_applied": False,
            }
        )
    )


if __name__ == "__main__":
    main()
