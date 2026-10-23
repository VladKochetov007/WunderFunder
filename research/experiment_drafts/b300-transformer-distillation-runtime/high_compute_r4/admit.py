"""Extend the frozen R3 receiver admission with R4 component checks."""

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

PACKET = Path(__file__).resolve().parent
R3 = PACKET.parent / "large_teacher_r3_muon30"
sys.path.insert(0, str(R3))
import admission as base  # noqa: E402 - reuse the frozen R3 admission implementation

NEW_TESTS = (
    "test_sequence_inner_splits",
    "test_sequence_cache_ensemble",
    "test_high_compute_campaign",
    "test_high_compute_runner",
)


def accept_checks(directory: Path) -> None:
    """Require execution of every R4 module before extending the R3 receipt."""
    cases = list(ET.parse(directory / "pytest.xml").getroot().iter("testcase"))
    for name in NEW_TESTS:
        if not any(name in case.get("classname", "").split(".") for case in cases):
            raise ValueError(f"R4 receiver checks did not execute {name}")
    base.accept_checks(directory)
    receipt = base.read(directory / "acceptance.json")
    paths = [base.ROOT / "tests" / f"{name}.py" for name in NEW_TESTS]
    paths += [
        base.ROOT / "scripts" / name
        for name in ("prepare_sequence_inner_split.py", "merge_sequence_teacher_caches.py")
    ]
    paths += [p for p in PACKET.rglob("*") if p.suffix in {".py", ".sh", ".json"}]
    paths += [p for p in R3.iterdir() if p.suffix in {".py", ".sh"}]
    receipt["files"].update({str(path.resolve()): base.file_hash(path) for path in paths})
    receipt["extension"] = "wnn.high-compute-checks.v1"
    receipt["limits"] = "Receiver correctness checks only; no fit, quality or resource approval."
    base.write_new(directory / "r4_acceptance.json", receipt)


def main() -> None:
    """Run explicit correctness or per-cell profile admission on the receiver."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    checks = sub.add_parser("checks")
    checks.add_argument("--directory", type=Path, required=True)
    profile = sub.add_parser("profile")
    for name in ("directory", "checks", "config"):
        profile.add_argument(f"--{name}", type=Path, required=True)
    profile.add_argument("--reviewer", required=True)
    profile.add_argument("--host-ram-gib", type=float, required=True)
    args = parser.parse_args()
    if args.action == "checks":
        accept_checks(args.directory)
    else:
        if base.read(args.checks).get("extension") != "wnn.high-compute-checks.v1":
            raise ValueError("R4 requires its complete receiver acceptance receipt")
        base.admit(args)


if __name__ == "__main__":
    main()
