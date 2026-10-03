from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

OWN = Path(__file__).resolve().parent
ROOT = OWN.parents[4]
BUILDER = OWN.parent / "live_capture_v4"


def digest(path: Path) -> str:
    """Stream frozen bytes without loading model arrays."""
    value = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def main() -> None:
    """Verify prior evidence and actual selected v4 runtime, without receipt issuance."""
    checks = {}

    def check(name: str, entries: dict[str, str], base: Path) -> None:
        checks[name] = {}
        for path, expected in entries.items():
            actual = digest(base / path)
            checks[name][path] = {
                "expected": expected,
                "actual": actual,
                "matches": actual == expected,
            }

    for version in ("live_capture", "live_capture_v2", "live_capture_v3", "live_capture_v4"):
        directory = OWN.parent / version
        inventory = {
            name: value
            for value, name in (
                line.split(maxsplit=1)
                for line in (directory / "protected_hashes.sha256").read_text().splitlines()
            )
        }
        check(version + "_protected", inventory, ROOT)
        check(
            version + "_artifacts",
            json.loads((directory / "artifact_hashes.json").read_text()),
            directory,
        )
    for name in (
        "capture_critic",
        "teacher_lineage",
        "capture_review",
        "capture_review_v2",
        "capture_review_v3",
    ):
        directory = OWN.parent / name
        inventory = json.loads((directory / "HASHES.json").read_text())
        check(name, inventory.get("own", inventory), directory)
    prior = OWN.parent / "capture_review_v3"
    check(
        "v3_additive_spec",
        json.loads((prior / "SUPPLEMENT_HASHES.json").read_text())["sha256"],
        prior,
    )
    recovery = json.loads((prior / "legacy_recovery.json").read_text())
    check(
        "E385_exact_originals",
        recovery["recovered_sha256"],
        OWN.parent / "capture_review_v2/versions/E385",
    )
    request = json.loads((BUILDER / "review_request.json").read_text())
    check("v4_candidate", request["candidate_sha256"], BUILDER)
    check("source", request["source"], ROOT)
    check("generated", json.loads((BUILDER / "generated_hashes.json").read_text()), BUILDER)
    manifest = json.loads((BUILDER / "runtime_manifest.json").read_text())
    runtime = dict(manifest["current_runtime_sha256"])
    wrapper = "research/campaign_2026-10-03/continuation/round4/controller_gpu_payload.sh"
    runtime[wrapper] = manifest["controller_payload_expected_after_patch"]
    check("actual_applied_runtime", runtime, ROOT)
    check(
        "request_manifest_bindings",
        {
            "runtime_manifest.json": request["runtime_manifest_sha256"],
            "controller_payload.patch": request["controller_patch_sha256"],
        },
        BUILDER,
    )
    unchanged = (
        "runner.py",
        "baselines.py",
        "fit_panel.py",
        "fit_panel.json",
        "source_binding.json",
        "trainer.patch",
        "instrumented_train_rnn.py",
        "chunk_source.py.txt",
        "generated_hashes.json",
    )
    checks["unchanged_math_and_runner"] = {
        name: {"matches": digest(BUILDER / name) == digest(OWN.parent / "live_capture_v3" / name)}
        for name in unchanged
    }
    syntax = subprocess.run(
        ["bash", "-n", str(ROOT / wrapper)], capture_output=True, text=True, check=False
    )
    assert syntax.returncode == 0, syntax.stderr
    previous = (ROOT / wrapper).read_bytes().replace(b"/live_capture_v4\n", b"/live_capture_v3\n")
    checks["scope_only_runtime_change"] = {
        "reverse_to_v3": {
            "matches": hashlib.sha256(previous).hexdigest()
            == manifest["current_runtime_sha256"][wrapper]
        }
    }
    result = {
        "all_match": all(row["matches"] for group in checks.values() for row in group.values()),
        "groups": checks,
        "counts": {name: len(group) for name, group in checks.items()},
        "bash_syntax_pass": True,
    }
    (OWN / "protected_verification.json").write_text(json.dumps(result, indent=2) + "\n")
    assert result["all_match"], "Frozen identity mismatch"
    bindings = {
        "kind": "independent_review_identity_inventory_NOT_controller_receipt",
        "verdict": "PASS_FOR_BOUNDED_UNCHANGED_DEFAULT_CUDA_MEASUREMENT_WITH_MANDATORY_GATES",
        "candidate_sha256": request["candidate_sha256"],
        "source": request["source"],
        "runtime_sha256": runtime,
        "actual_v4_wrapper_applied": True,
        "actor_runtime_manifest_is_historical_unapplied_record": True,
        "permitted_scope": "only bounded unchanged/default synthetic actual architecture then fixed-panel trajectory; no scientist arms",
        "mandatory_gates": [
            "controller current queue/exclusive admission attestation and independent hash-bound receipt",
            "six original/default baselines before online comparisons; reject excessive or unstable floors",
            "online/default/repeat full-state replay and deliberate AMP skip clocks",
            "fail closed on CUDA private RNG roundtrip or backend state mismatch",
            "TERM585/KILL5 child, timeout600 outer, deadline/exclusive-family/device memory guards",
        ],
        "CUDA_parity_measured": False,
        "mature_full_fit_state_recovered": False,
        "GPU_launched": False,
        "receipt_written": False,
    }
    (OWN / "candidate_runtime_hashes.json").write_text(json.dumps(bindings, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ("all_match", "counts", "bash_syntax_pass")}))


if __name__ == "__main__":
    main()
