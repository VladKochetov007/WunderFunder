"""Prove caller V9 changes only the helper V4 binding from reviewed V8."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

ROOT = Path("/Users/nb/Documents/wunder/WunderFunder-maintenance")
C = ROOT / "research/campaign_2026-10-19/full_gru_tweaks_r1"
PREP = C / "preparation/first_cell_evaluation_score_acceptance_v1"
V8 = PREP / "ROOT_ACCEPT_SCORE_RESULT_V8.py"
V9 = PREP / "ROOT_ACCEPT_SCORE_RESULT_V9.py"
V8_REVIEW = PREP / "ROOT_CALLER_SOURCE_REVIEW_V8.json"
HELPER_V4 = C / "preparation/first_cell_evaluation_score_acceptance_v4/score_result_acceptance_v4.py"
HELPER_REVIEW_V4 = C / "preparation/first_cell_evaluation_score_acceptance_v4/HELPER_SOURCE_REVIEW_V4.json"
PROOF = PREP / "ROOT_CALLER_V9_HELPER_V4_REBASE_PROOF.json"
V8_PIN = (32_463, "5a0c984140e227933db5c94ca7c6b02ad4278ade2542dc920173c940da31b8ec")
V9_PIN = (32_464, "1c3b5f91dbb0cd72938ea3d62f3a0f9b85c9badb1e3042e50eb2d3b004e1ec1b")
V8_REVIEW_PIN = (6_453, "54294f758e806cc0251244d9496edc8b4ffa28982e204e89b1ddeedad1a1be44")
HELPER_PIN = (46_350, "55b4814921e522d9fa24f6c9e4d830218989d42a834ebf9a30af056d617741dc")
HELPER_REVIEW_PIN = (15_718, "3597f9be9196a5ad0fa9aed7c208e26f29e4c10876d8ecfb0ef546e3e4e2370b")


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def pin(path: Path, expected: tuple[int, str]) -> bytes:
    raw = path.read_bytes()
    if (len(raw), sha(raw)) != expected:
        raise ValueError(f"fixed source or review pin differs: {path.name}")
    return raw


def main() -> None:
    old_raw = pin(V8, V8_PIN)
    new_raw = pin(V9, V9_PIN)
    text = new_raw.decode("utf-8", errors="strict")
    reverse = (
        (
            "preparation/first_cell_evaluation_score_acceptance_v4/score_result_acceptance_v4.py",
            "preparation/first_cell_evaluation_score_acceptance_v3/score_result_acceptance_v3.py",
        ),
        (
            '(46_350, "55b4814921e522d9fa24f6c9e4d830218989d42a834ebf9a30af056d617741dc")',
            '(46_448, "72f25104d5a338e55d3d7667df79882cd699d0798c7d8fc19638fea392ad0e8c")',
        ),
        (
            "preparation/first_cell_evaluation_score_acceptance_v4/HELPER_SOURCE_REVIEW_V4.json",
            "preparation/first_cell_evaluation_score_acceptance_v3/HELPER_SOURCE_REVIEW_V3.json",
        ),
        (
            '(15_718, "3597f9be9196a5ad0fa9aed7c208e26f29e4c10876d8ecfb0ef546e3e4e2370b")',
            '(8_451, "9ad6a92a06e1c481666f11446fbfc9eeb741e31769f0c016370f44442dcc9fed")',
        ),
        (
            "fg19.first-cell-score-acceptance-helper-source-review.v4",
            "fg19.first-cell-score-acceptance-helper-source-review.v3",
        ),
        (
            "PASS_SOURCE_ONLY_SCORE_ACCEPTANCE_HELPER_V4",
            "PASS_SOURCE_ONLY_SCORE_ACCEPTANCE_HELPER_V3",
        ),
        (
            'fg19_root_score_acceptance_v4',
            'fg19_root_score_acceptance_v1',
        ),
    )
    for candidate, predecessor in reverse:
        if text.count(candidate) != 1:
            raise ValueError("caller V9 helper binding delta is missing or duplicated")
        text = text.replace(candidate, predecessor, 1)
    if text.encode("utf-8") != old_raw:
        raise ValueError("caller V9 differs from reviewed V8 outside helper V4 bindings")

    v8_review = json.loads(pin(V8_REVIEW, V8_REVIEW_PIN))
    if v8_review.get("schema") != "fg19.first-cell-score-root-acceptance-caller-review.v8" or v8_review.get(
        "status"
    ) != "PASS_SOURCE_ONLY_ROOT_SCORE_ACCEPTANCE_CALLER_V8":
        raise ValueError("caller V8 predecessor review schema/status differs")
    if v8_review.get("candidate") != {
        "path": str(V8.relative_to(ROOT)),
        "bytes": V8_PIN[0],
        "sha256": V8_PIN[1],
    }:
        raise ValueError("caller V8 predecessor review does not bind exact source bytes")
    helper_review = json.loads(pin(HELPER_REVIEW_V4, HELPER_REVIEW_PIN))
    if helper_review.get("schema") != "fg19.first-cell-score-acceptance-helper-source-review.v4" or helper_review.get(
        "status"
    ) != "PASS_SOURCE_ONLY_SCORE_ACCEPTANCE_HELPER_V4":
        raise ValueError("helper V4 review schema/status differs")
    if helper_review.get("candidate") != {
        "path": str(HELPER_V4.relative_to(ROOT)),
        "bytes": HELPER_PIN[0],
        "sha256": HELPER_PIN[1],
    }:
        raise ValueError("helper V4 review does not bind exact helper source bytes")

    proof = {
        "schema": "fg19.first-cell-score-root-caller-v9-helper-v4-rebase-proof.v1",
        "status": "PASS_EXACT_HELPER_V4_REBIND_FROM_REVIEWED_V8",
        "predecessor": {"path": str(V8.relative_to(ROOT)), "bytes": len(old_raw), "sha256": sha(old_raw)},
        "candidate": {"path": str(V9.relative_to(ROOT)), "bytes": len(new_raw), "sha256": sha(new_raw)},
        "reviewed_predecessor": {
            "path": str(V8_REVIEW.relative_to(ROOT)),
            "bytes": V8_REVIEW_PIN[0],
            "sha256": V8_REVIEW_PIN[1],
            "schema": v8_review["schema"],
            "status": v8_review["status"],
        },
        "new_helper_review": {
            "path": str(HELPER_REVIEW_V4.relative_to(ROOT)),
            "bytes": HELPER_REVIEW_PIN[0],
            "sha256": HELPER_REVIEW_PIN[1],
            "schema": helper_review["schema"],
            "status": helper_review["status"],
        },
        "declared_change": "Rebind helper source and source-review path, byte/SHA pins, schema, status, and in-memory module origin name to the independently reviewed helper V4; all caller V8 AST outside these seven literals is byte-identical.",
        "inverse_byte_reconstruction_matches_reviewed_v8": True,
        "caller_executed": False,
        "b_or_ssh_attempted": False,
        "score_or_quality_claimed": False,
    }
    raw = (json.dumps(proof, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()
    with PROOF.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    directory_fd = os.open(PREP, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    print(f"{len(raw)} {sha(raw)} {PROOF}")


if __name__ == "__main__":
    main()
