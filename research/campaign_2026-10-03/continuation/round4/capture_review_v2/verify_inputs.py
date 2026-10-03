"""Verify frozen candidates/prior verdicts and bind final applied runtime bytes."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

OWN = Path(__file__).resolve().parent
ROOT = OWN.parents[4]
V2 = OWN.parent / "live_capture_v2"
LOGGER = logging.getLogger(__name__)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def check(entries: dict, directory: Path) -> dict:
    return {name: {"expected": expected, "actual": digest(directory / name), "matches": digest(directory / name) == expected}
            for name, expected in entries.items()}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    checks = {}
    for version in ("live_capture", "live_capture_v2"):
        directory = OWN.parent / version
        protected = {}
        for line in (directory / "protected_hashes.sha256").read_text().splitlines():
            value, name = line.split(maxsplit=1)
            protected[name] = value
        checks[version + "_protected"] = check(protected, ROOT)
        checks[version + "_artifacts"] = check(json.loads((directory / "artifact_hashes.json").read_text()), directory)
    for name in ("capture_critic", "teacher_lineage", "capture_review"):
        directory = OWN.parent / name
        manifest = json.loads((directory / "HASHES.json").read_text())
        checks[name + "_preserved"] = check(manifest.get("own", manifest), directory)
    request = json.loads((V2 / "review_request.json").read_text())
    checks["v2_candidate"] = check(request["candidate_sha256"], V2)
    checks["source"] = check(request["source"], ROOT)
    manifest = json.loads((V2 / "runtime_manifest.json").read_text())
    final_runtime = {name: digest(ROOT / name) for name in manifest["current_runtime_sha256"]}
    wrapper = "research/campaign_2026-10-03/continuation/round4/controller_gpu_payload.sh"
    runtime_checks = {name: actual == (manifest["controller_payload_expected_after_patch"] if name == wrapper else manifest["current_runtime_sha256"][name]) for name, actual in final_runtime.items()}
    authorization_inputs = {
        "kind": "BLOCKED_review_inputs_ONLY_NOT_approval_or_receipt",
        "verdict": "BLOCKED_V2_NUMPY_RNG_AND_MOMENT_TYPE_IDENTITY; require separate correction candidate",
        "scope": "bounded default CUDA parity measurement ONLY; no intervention or full-fit state claim",
        "candidate_sha256": request["candidate_sha256"], "source": request["source"],
        "runtime_sha256": final_runtime,
        "generated_files_bound_by_manifest_sha256": request["candidate_sha256"]["generated_hashes.json"],
        "command_contract": "campaign_job launch --kind gpu --timeout600 --threads8 -> run_gpu.sh -> controller_gpu_payload.sh CONTROLLER_RECEIPT FRESH_V2_OUTPUT",
        "child_TERM_seconds": 585, "child_KILL_grace_seconds": 5,
        "outer_seconds": 600, "absolute_deadline_UTC": "2026-10-03T20:00:00Z",
        "runtime_manifest_interpretation": "frozen unapplied record is historical; actual controller payload now matches expected patched hash",
        "controller_attestation_required": "actual units/job records/queue and independent receipt; reviewer did not issue receipt",
        "external_postcheck_is_not_sufficient": True,
        "CUDA_parity_established": False,
    }
    (OWN / "authorization_inputs.json").write_text(json.dumps(authorization_inputs, indent=2) + "\n")
    verification = {"groups": checks, "runtime_checks": runtime_checks,
                    "all_match": all(v["matches"] for group in checks.values() for v in group.values()) and all(runtime_checks.values()),
                    "counts": {name: len(group) for name, group in checks.items()}}
    (OWN / "protected_verification.json").write_text(json.dumps(verification, indent=2) + "\n")
    LOGGER.info("VERIFIED %s", json.dumps({"all_match": verification["all_match"], "counts": verification["counts"], "runtime": final_runtime, "runtime_checks": runtime_checks}))
    if not verification["all_match"]:
        raise RuntimeError("Protected/final-runtime mismatch")


if __name__ == "__main__":
    main()
