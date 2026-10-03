"""Freeze blocking independent review; never write a GPU authorization receipt."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

OWN = Path(__file__).resolve().parent
LOGGER = logging.getLogger(__name__)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    results = json.loads((OWN / "results.json").read_text())
    verified = json.loads((OWN / "protected_verification.json").read_text())
    state = json.loads((OWN / "STATE.json").read_text())
    state.update(phase="frozen_v2_review", verdict="BLOCKED_EXACT_RNG_AND_MOMENT_TYPE_IDENTITY",
                 pending=[], last_failure=None, protected_all_match=verified["all_match"],
                 prior_three_blocking_corrections_retested_pass=True,
                 remaining_blockers=["NumPy dtype/numeric aliases accepted by validate/compare and silently cast on restore", "warm Muon/Adam moment float64 dtype aliases accepted and silently cast to float32"],
                 external_postcheck_not_approval_workaround=True,
                 required_next="separate builder version per V3_MINIMAL_SPEC.md; no receipt for v2",
                 CPU_reproduction=results["CPU_reproduction"], cost=results["cost"],
                 runtime_applied_payload_sha256="335233c719cea03a9cef689a12ad89150682dc1caa0f0bc49038913b266af8ad")
    (OWN / "STATE.json").write_text(json.dumps(state, indent=2) + "\n")
    hashes = {str(p.relative_to(OWN)): digest(p) for p in sorted(OWN.rglob("*")) if p.is_file() and p.name not in {"HASHES.json", "freeze.log"}}
    (OWN / "HASHES.json").write_text(json.dumps(hashes, indent=2) + "\n")
    LOGGER.info("FROZEN %s", json.dumps({"verdict": state["verdict"], "protected_match": verified["all_match"], "GPU_launched": False, "receipt_written": False}))
    if not verified["all_match"]:
        raise RuntimeError("Protected verification failed")


if __name__ == "__main__":
    main()
