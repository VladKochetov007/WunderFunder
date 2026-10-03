"""Freeze independent review results without approving or launching a candidate."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

OWN = Path(__file__).resolve().parent
ROOT = OWN.parents[4]
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
    protected = json.loads((OWN / "protected_verification.json").read_text())
    runtime = {name: digest(ROOT / name) for name in (
        "research/campaign_2026-10-03/continuation/round4/controller_gpu_payload.sh",
        "research/campaign_2026-10-03/continuation/round4/RUNTIME_INTEGRATION.md",
        "research/campaign_2026-10-03/continuation/run_gpu.sh",
        "scripts/campaign_job.py", "scripts/limited")}
    (OWN / "runtime_hashes.json").write_text(json.dumps(runtime, indent=2) + "\n")
    state = json.loads((OWN / "STATE.json").read_text())
    state.update(phase="frozen_independent_review", verdict="FAIL_AS_FROZEN",
                 protected_hashes_all_match=protected["all_match"],
                 runtime_chain_design="PASS conditional on controller exclusive launch after independently verified queue drain",
                 blocking_findings=["optimizer serialized IDs can swap warm named moments", "integer tensor clocks inherit floating tolerance", "CUDA default tolerance not calibrated from empirical floor"],
                 pristine_cpu_byte_differences=results["pristine_byte_differences"],
                 cost=results["cost"], pending=[], last_failure=None,
                 required_next="builder/controller remediation and independent retest; no GPU approval issued")
    (OWN / "STATE.json").write_text(json.dumps(state, indent=2) + "\n")
    own_hashes = {str(p.relative_to(OWN)): digest(p) for p in sorted(OWN.rglob("*")) if p.is_file() and p.name not in {"HASHES.json", "freeze.log"}}
    (OWN / "HASHES.json").write_text(json.dumps(own_hashes, indent=2) + "\n")
    LOGGER.info("FROZEN %s", json.dumps({"verdict": state["verdict"], "protected_all_match": protected["all_match"], "GPU_launched": False, "receipt_written": False}))
    if not protected["all_match"]:
        raise RuntimeError("Protected hashes failed")


if __name__ == "__main__":
    main()
