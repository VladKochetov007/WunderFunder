"""Prove the score-v3 stop packet is a narrow score-v2 namespace fork."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

ROOT = Path("/Users/nb/Documents/wunder/WunderFunder-maintenance")
C = ROOT / "research/campaign_2026-10-19/full_gru_tweaks_r1"
OLD_DIR = C / "preparation/first_cell_evaluation_score_stop_v3"
NEW_DIR = C / "preparation/first_cell_evaluation_score_stop_v4"
OLD_DRIVER = OLD_DIR / "ROOT_CAPTURE_SCORE_STOP_V3.py"
NEW_DRIVER = NEW_DIR / "ROOT_CAPTURE_SCORE_STOP_V4.py"
OLD_TEMPLATE = OLD_DIR / "STOP_SCORE_TEMPLATE_V2.py"
NEW_TEMPLATE = NEW_DIR / "STOP_SCORE_TEMPLATE_V3.py"
PROOF = NEW_DIR / "SCORE_STOP_V4_NAMESPACE_PROOF.json"
OLD_DRIVER_PIN = (12_159, "ce9220dcf37ccf1fc96e184e90d9feaf47dffe2ff16c60d55a2832034fb0d055")
OLD_TEMPLATE_PIN = (11_984, "88166701e24daa43b8a06244d5b0712eaeaba01c078afb569c43d81db17b49f5")
OLD_UNIT = "computer-vision-homework-fg19-first-cell-score-v2.service"
NEW_UNIT = "computer-vision-homework-fg19-first-cell-score-v3.service"


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def transform_exact(raw: bytes, replacements: list[tuple[bytes, bytes]]) -> bytes:
    for old, new in replacements:
        if raw.count(old) != 1:
            raise ValueError(f"expected one exact predecessor byte string: {old!r}")
        raw = raw.replace(old, new, 1)
    return raw


def main() -> None:
    old_driver, new_driver = OLD_DRIVER.read_bytes(), NEW_DRIVER.read_bytes()
    old_template, new_template = OLD_TEMPLATE.read_bytes(), NEW_TEMPLATE.read_bytes()
    if (len(old_driver), sha(old_driver)) != OLD_DRIVER_PIN:
        raise ValueError("frozen score-v2 root stop driver pin differs")
    if (len(old_template), sha(old_template)) != OLD_TEMPLATE_PIN:
        raise ValueError("frozen score-v2 remote stop template pin differs")
    expected_driver = transform_exact(
        old_driver,
        [
            (b"records/first_cell_evaluation_v1/score_v3/exact_stop", b"records/first_cell_evaluation_v1/score_v4/exact_stop"),
            (f'SCORE_UNIT = "{OLD_UNIT}"'.encode(), f'SCORE_UNIT = "{NEW_UNIT}"'.encode()),
            (f'if SCORE_UNIT != "{OLD_UNIT}":'.encode(), f'if SCORE_UNIT != "{NEW_UNIT}":'.encode()),
            (b"score unit differs from the reviewed score-v2 scope", b"score unit differs from the reviewed score-v3 scope"),
        ],
    )
    expected_template = transform_exact(
        old_template,
        [(f'SCORE_UNIT = "{OLD_UNIT}"'.encode(), f'SCORE_UNIT = "{NEW_UNIT}"'.encode())],
    )
    if new_driver != expected_driver or new_template != expected_template:
        raise ValueError("stop packet changed beyond the declared namespace literals")
    import ast
    if ast.dump(ast.parse(old_driver.decode()), include_attributes=False) == ast.dump(ast.parse(new_driver.decode()), include_attributes=False):
        raise ValueError("driver fork unexpectedly has no AST changes")
    report = {
        "schema": "fg19.first-cell-score-stop-v4-namespace-proof.v1",
        "status": "PASS_NARROW_SCORE_V3_STOP_NAMESPACE_FORK",
        "predecessor_driver": {"path": str(OLD_DRIVER.relative_to(ROOT)), "bytes": len(old_driver), "sha256": OLD_DRIVER_PIN[1]},
        "candidate_driver": {"path": str(NEW_DRIVER.relative_to(ROOT)), "bytes": len(new_driver), "sha256": sha(new_driver)},
        "predecessor_template": {"path": str(OLD_TEMPLATE.relative_to(ROOT)), "bytes": len(old_template), "sha256": OLD_TEMPLATE_PIN[1]},
        "candidate_template": {"path": str(NEW_TEMPLATE.relative_to(ROOT)), "bytes": len(new_template), "sha256": sha(new_template)},
        "changed_driver_literals": ["D: score_v3/exact_stop -> score_v4/exact_stop", "SCORE_UNIT score-v2 -> score-v3", "unit guard score-v2 -> score-v3", "unit diagnostic text score-v2 -> score-v3"],
        "changed_template_literals": ["SCORE_UNIT score-v2 -> score-v3"],
        "acceptance_body_review_invocation_pins_remain_unbound": True,
        "only_unit_targeted": "computer-vision-homework-fg19-first-cell-score-v3.service",
        "no_stop_submitted_or_executed": True,
        "no_B_or_SSH_action": True,
        "no_tests_or_receipts_or_result_payloads_read": True,
    }
    raw = (json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()
    with PROOF.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    directory_fd = os.open(NEW_DIR, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    print(f"{len(raw)} {sha(raw)} {PROOF}")


if __name__ == "__main__":
    main()
