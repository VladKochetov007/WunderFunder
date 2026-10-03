"""Build unapplied controller-only runtime patch; preserve v1 identity evidence."""

from __future__ import annotations

import difflib
import json

from live_capture import HERE, ROOT, sha

V1 = HERE.parent / "live_capture"
REVIEW = HERE.parent / "capture_review"


def main() -> None:
    protected = {}
    for line in (V1 / "protected_hashes.sha256").read_text().splitlines():
        expected, name = line.split(maxsplit=1)
        actual = sha(ROOT / name)
        if actual != expected:
            raise ValueError("v1 protected hash changed: " + name)
        protected[name] = actual
    old_artifacts = json.loads((V1 / "artifact_hashes.json").read_text())
    for name, expected in old_artifacts.items():
        if sha(V1 / name) != expected:
            raise ValueError("v1 artifact changed: " + name)
    payload_path = HERE.parent / "controller_gpu_payload.sh"
    original = payload_path.read_text()
    revised = original.replace("round4/live_capture", "round4/live_capture_v2").replace(
        "timeout --signal=TERM 590", "timeout --signal=TERM --kill-after=5s 585"
    )
    if original == revised:
        raise ValueError("controller patch base not expected v1 payload")
    (HERE / "controller_payload.patch").write_text(
        "".join(
            difflib.unified_diff(
                original.splitlines(True),
                revised.splitlines(True),
                fromfile=str(payload_path.relative_to(ROOT)),
                tofile=str(payload_path.relative_to(ROOT)),
            )
        )
    )
    import hashlib

    files = (
        "scripts/campaign_job.py",
        "research/campaign_2026-10-03/continuation/run_gpu.sh",
        "scripts/limited",
        str(payload_path.relative_to(ROOT)),
    )
    runtime = {
        "current_runtime_sha256": {name: sha(ROOT / name) for name in files},
        "controller_payload_expected_after_patch": hashlib.sha256(revised.encode()).hexdigest(),
        "unapplied_patch_sha256": sha(HERE / "controller_payload.patch"),
        "child_TERM_seconds": 585,
        "child_KILL_grace_seconds": 5,
        "max_nominal_child_seconds": 590,
        "outer_execution_seconds": 600,
        "absolute_cutguard": "2026-10-03 20:00 UTC =23:00 Europe/Kyiv; controller payload requires605s remaining",
        "launch_chain": "campaign_job launch --kind gpu --timeout600 --threads8 -> run_gpu.sh -> controller_gpu_payload.sh CONTROLLER_RECEIPT FRESH_V2_OUTPUT",
        "actor_shell_launch": "UNSUPPORTED; no standalone GPU launch script supplied",
        "controller_patch_applied": False,
    }
    (HERE / "runtime_manifest.json").write_text(json.dumps(runtime, indent=2) + "\n")
    lineage = {
        "v1_preserved": True,
        "v1_protected": protected,
        "v1_artifact_count": len(old_artifacts),
        "v1_protected_manifest_sha256": sha(V1 / "protected_hashes.sha256"),
        "v1_artifact_manifest_sha256": sha(V1 / "artifact_hashes.json"),
        "review_verdict": "FAIL_AS_FROZEN",
        "review_sha256": {
            path.name: sha(path)
            for path in REVIEW.iterdir()
            if path.is_file()
            and (
                path.name.startswith(("REPORT", "STATE", "results", "review"))
                or path.name == "adversaries.py"
            )
        },
        "copy_scope": str(HERE.relative_to(ROOT)),
        "panel_byte_identical_to_v1": sha(HERE / "fit_panel.json") == sha(V1 / "fit_panel.json"),
        "actual_trainer_sha256": sha(ROOT / "scripts/train_rnn.py"),
    }
    (HERE / "LINEAGE.json").write_text(json.dumps(lineage, indent=2) + "\n")
    print(
        json.dumps(
            {
                "v1_protected_preserved": len(protected),
                "v1_artifacts_preserved": len(old_artifacts),
                "panel_unchanged": lineage["panel_byte_identical_to_v1"],
                "controller_patch_applied": False,
            }
        )
    )


if __name__ == "__main__":
    main()
