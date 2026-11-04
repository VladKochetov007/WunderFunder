"""Profile an independently amended recipe without rewriting its frozen predecessor."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from wnn.multistage_search_profile import run_profile_trial


def main() -> None:
    """Verify a derived admission and run one bounded, weights-free native profile."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--admission", type=Path, required=True)
    parser.add_argument("--admission-sha256", required=True)
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--gpu-key", required=True)
    args = parser.parse_args()
    raw = args.admission.read_bytes()
    if hashlib.sha256(raw).hexdigest() != args.admission_sha256:
        raise ValueError("derived admission changed before profiling")
    admission = json.loads(raw)
    amendment_ref = admission["provenance"]["dropout_repair_amendment"]
    amendment_path = Path(amendment_ref["path"])
    amendment_raw = amendment_path.read_bytes()
    if hashlib.sha256(amendment_raw).hexdigest() != amendment_ref["sha256"]:
        raise ValueError("independent engineering amendment changed")
    amendment = json.loads(amendment_raw)
    if amendment.get("reviewer") != "multistage_independent_critic":
        raise ValueError("dropout repair has no independent reviewer")
    if amendment.get("status") not in {"approved", "approved_engineering_compatibility_only"}:
        raise ValueError("independent amendment does not approve the engineering repair")
    owner = args.run_dir / "workers" / f"{args.worker_id}.launch_receipt.json"
    for _ in range(60):
        if owner.is_file():
            break
        time.sleep(0.05)
    else:
        raise FileNotFoundError("immutable profile-worker launch receipt was not published")
    device = admission["profile_policy"]["devices"][args.gpu_key]
    result = run_profile_trial(args.run_dir, admission, args.worker_id, args.gpu_key, device)
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
