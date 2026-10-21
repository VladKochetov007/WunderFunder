"""Prove caller V8 differs from frozen V7 only in empty-stderr handling."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

ROOT = Path("/Users/nb/Documents/wunder/WunderFunder-maintenance")
PREP = ROOT / "research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/first_cell_evaluation_score_acceptance_v1"
V7 = PREP / "ROOT_ACCEPT_SCORE_RESULT_V7.py"
V8 = PREP / "ROOT_ACCEPT_SCORE_RESULT_V8.py"
PROOF = PREP / "ROOT_CALLER_V8_EMPTY_STDERR_PROOF.json"
V7_PIN = (32_281, "811e44d27f1e6c798d11f26d450b7aed76c6c6a6cbd888de047e91cbb25a4c2e")
V8_PIN = (32_463, "5a0c984140e227933db5c94ca7c6b02ad4278ade2542dc920173c940da31b8ec")


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def main() -> None:
    old_raw = V7.read_bytes()
    new_raw = V8.read_bytes()
    if (len(old_raw), sha(old_raw)) != V7_PIN:
        raise ValueError("frozen V7 source pin differs")
    if (len(new_raw), sha(new_raw)) != V8_PIN:
        raise ValueError("V8 source pin differs")
    text = new_raw.decode("utf-8", errors="strict")
    reverse = [
        (
            "def read(\n    path: Path, cap: int = MAX_METADATA_BYTES, *, allow_empty: bool = False\n) -> bytes:\n",
            "def read(path: Path, cap: int = MAX_METADATA_BYTES) -> bytes:\n",
        ),
        (
            "if (\n        not stat.S_ISREG(before.st_mode)\n        or before.st_size > cap\n        or (before.st_size == 0 and not allow_empty)\n    ):\n",
            "if not stat.S_ISREG(before.st_mode) or before.st_size > cap:\n",
        ),
        (
            "if len(raw) != before.st_size or len(raw) > cap or (not raw and not allow_empty):\n",
            "if len(raw) != before.st_size or len(raw) > cap:\n",
        ),
        (
            'stderr_raw = read(args.capture / "STDERR.raw.txt", MAX_STDERR_BYTES, allow_empty=True)\n',
            'stderr_raw = read(args.capture / "STDERR.raw.txt", MAX_STDERR_BYTES)\n',
        ),
        (
            'launch_stderr_raw = read(RECORD / "STDERR.raw.txt", MAX_STDERR_BYTES, allow_empty=True)\n',
            'launch_stderr_raw = read(RECORD / "STDERR.raw.txt", MAX_STDERR_BYTES)\n',
        ),
    ]
    for candidate, predecessor in reverse:
        if text.count(candidate) != 1:
            raise ValueError("V8 exact empty-stderr delta is missing or duplicated")
        text = text.replace(candidate, predecessor, 1)
    reconstructed = text.encode("utf-8")
    if reconstructed != old_raw:
        raise ValueError("V8 differs from V7 outside the five declared source edits")
    proof = {
        "schema": "fg19.first-cell-score-root-caller-v8-empty-stderr-proof.v1",
        "status": "PASS_EXACT_V7_EMPTY_STDERR_HANDLING_DELTA",
        "predecessor": {"path": str(V7.relative_to(ROOT)), "bytes": len(old_raw), "sha256": sha(old_raw)},
        "candidate": {"path": str(V8.relative_to(ROOT)), "bytes": len(new_raw), "sha256": sha(new_raw)},
        "declared_changes": [
            "read rejects empty files by default while preserving existing regular-file, O_NOFOLLOW, size, and stable-stat checks",
            "read exposes allow_empty keyword for explicit empty stderr records",
            "exact status-capture stderr read opts into allow_empty",
            "exact score-launch stderr read opts into allow_empty",
        ],
        "inverse_byte_reconstruction_matches_frozen_v7": True,
        "caller_executed": False,
        "old_tests_replayed": False,
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
