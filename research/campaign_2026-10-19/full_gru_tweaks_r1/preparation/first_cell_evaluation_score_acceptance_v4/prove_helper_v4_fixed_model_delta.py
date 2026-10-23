"""Prove helper V4 is an exact fixed-model contract correction from V3."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

ROOT = Path("/Users/nb/Documents/wunder/WunderFunder-maintenance")
PREP = ROOT / "research/campaign_2026-10-19/full_gru_tweaks_r1/preparation"
V3 = PREP / "first_cell_evaluation_score_acceptance_v3/score_result_acceptance_v3.py"
V4 = PREP / "first_cell_evaluation_score_acceptance_v4/score_result_acceptance_v4.py"
PROOF = PREP / "first_cell_evaluation_score_acceptance_v4/HELPER_V4_FIXED_MODEL_DELTA_PROOF.json"
V3_PIN = (46_448, "72f25104d5a338e55d3d7667df79882cd699d0798c7d8fc19638fea392ad0e8c")
V4_PIN = (46_350, "55b4814921e522d9fa24f6c9e4d830218989d42a834ebf9a30af056d617741dc")


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def main() -> None:
    old_raw, new_raw = V3.read_bytes(), V4.read_bytes()
    if (len(old_raw), sha(old_raw)) != V3_PIN:
        raise ValueError("frozen helper V3 pin differs")
    if (len(new_raw), sha(new_raw)) != V4_PIN:
        raise ValueError("helper V4 source pin differs")
    old = (
        b'                "checkpoint": {\n'
        b'                    **arm["model"],\n'
        b'                    "loaded_sha256": arm["model"]["sha256"],\n'
        b'                },'
    )
    new = b'                "checkpoint": dict(arm["model"]),'
    if new_raw.count(new) != 1:
        raise ValueError("helper V4 expected-model checkpoint delta is missing or duplicated")
    reconstructed = new_raw.replace(new, old, 1)
    if reconstructed != old_raw:
        raise ValueError("helper V4 differs from frozen V3 outside the declared expected-model fix")
    proof = {
        "schema": "fg19.first-cell-score-acceptance-helper-v4-fixed-model-delta.v1",
        "status": "PASS_EXACT_EXPECTED_FIXED_MODEL_CONTRACT_CORRECTION",
        "predecessor": {"path": str(V3.relative_to(ROOT)), "bytes": len(old_raw), "sha256": sha(old_raw)},
        "candidate": {"path": str(V4.relative_to(ROOT)), "bytes": len(new_raw), "sha256": sha(new_raw)},
        "declared_change": "Expected fixed_models checkpoint equals the exact scorer-emitted dict(arm['model']); do not invent loaded_sha256 there. Export acceptance/sidecar loaded model hashes remain validated by their existing checks.",
        "inverse_byte_reconstruction_matches_frozen_v3": True,
        "helper_executed": False,
        "metadata_fixture_run": False,
        "b_or_ssh_attempted": False,
        "quality_claimed": False,
    }
    raw = (json.dumps(proof, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()
    with PROOF.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    directory_fd = os.open(PROOF.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    print(f"{len(raw)} {sha(raw)} {PROOF}")


if __name__ == "__main__":
    main()
