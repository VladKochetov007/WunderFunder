from __future__ import annotations

import hashlib
import json
from pathlib import Path

OWN = Path(__file__).resolve().parent
ROOT = OWN.parents[4]
BUILDER = OWN.parent / "live_capture_v3"


def digest(path: Path) -> str:
    """Stream artifact bytes without model loading."""
    result = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            result.update(chunk)
    return result.hexdigest()


def main() -> None:
    """Verify all frozen identities, including recovered historical E385 bytes."""
    checks = {}

    def check(name: str, entries: dict[str, str], base: Path) -> None:
        checks[name] = {
            path: {
                "expected": expected,
                "actual": digest(base / path),
                "matches": digest(base / path) == expected,
            }
            for path, expected in entries.items()
        }

    for version in ("live_capture", "live_capture_v2", "live_capture_v3"):
        directory = OWN.parent / version
        protected = {
            name: value
            for value, name in (
                line.split(maxsplit=1)
                for line in (directory / "protected_hashes.sha256").read_text().splitlines()
            )
        }
        check(version + "_protected", protected, ROOT)
        check(
            version + "_artifacts",
            json.loads((directory / "artifact_hashes.json").read_text()),
            directory,
        )
    for name in ("capture_critic", "teacher_lineage", "capture_review", "capture_review_v2"):
        directory = OWN.parent / name
        inventory = json.loads((directory / "HASHES.json").read_text())
        check(name, inventory.get("own", inventory), directory)
    recovery = json.loads((OWN / "legacy_recovery.json").read_text())
    check(
        "E385_recovered",
        recovery["recovered_sha256"],
        OWN.parent / "capture_review_v2/versions/E385",
    )
    request = json.loads((BUILDER / "review_request.json").read_text())
    check("candidate", request["candidate_sha256"], BUILDER)
    check("source", request["source"], ROOT)
    check("generated", json.loads((BUILDER / "generated_hashes.json").read_text()), BUILDER)
    manifest = json.loads((BUILDER / "runtime_manifest.json").read_text())
    runtime = dict(manifest["current_runtime_sha256"])
    wrapper = "research/campaign_2026-10-03/continuation/round4/controller_gpu_payload.sh"
    runtime[wrapper] = manifest["controller_payload_expected_after_patch"]
    check("actual_applied_runtime", runtime, ROOT)
    check(
        "review_manifest",
        {
            "runtime_manifest.json": request["runtime_manifest_sha256"],
            "controller_payload.patch": request["controller_patch_sha256"],
        },
        BUILDER,
    )
    checks["unchanged_baseline_math"] = {
        name: {"matches": digest(BUILDER / name) == digest(OWN.parent / "live_capture_v2" / name)}
        for name in (
            "baselines.py",
            "fit_panel.py",
            "fit_panel.json",
            "trainer.patch",
            "instrumented_train_rnn.py",
            "chunk_source.py.txt",
        )
    }
    result = {
        "groups": checks,
        "counts": {name: len(entries) for name, entries in checks.items()},
        "all_match": all(row["matches"] for entries in checks.values() for row in entries.values()),
    }
    (OWN / "protected_verification.json").write_text(json.dumps(result, indent=2) + "\n")
    bindings = {
        "kind": "independent_review_identity_inventory_NOT_receipt",
        "verdict": "BLOCKED_TORCH_RNG_ACCEPTED_STATE_NORMALIZATION",
        "candidate_sha256": request["candidate_sha256"],
        "source": request["source"],
        "runtime_sha256": runtime,
        "runtime_manifest_is_historical_unapplied_record": True,
        "actual_wrapper_applied": True,
        "scope_if_corrected": "bounded unchanged/default CUDA parity only; no scientist arms, no full-fit reconstruction",
        "GPU_launched": False,
        "receipt_written": False,
        "controller_live_queue_attestation_required": True,
    }
    (OWN / "candidate_runtime_hashes.json").write_text(json.dumps(bindings, indent=2) + "\n")
    print(json.dumps({name: result[name] for name in ("all_match", "counts")}))
    if not result["all_match"]:
        raise AssertionError("Frozen integrity verification failed")


if __name__ == "__main__":
    main()
