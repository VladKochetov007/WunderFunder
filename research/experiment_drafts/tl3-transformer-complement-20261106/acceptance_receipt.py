"""Bind successful receiver checks before the explicit profile/fit stages."""

from __future__ import annotations

import argparse
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

XML_REPORTS = ("pytest.xml", "bf16.xml", "bf16x9.xml")
EVIDENCE = (*XML_REPORTS, "ddp.json", "complement-ddp.json", "muon-ddp.json")


def digest(path: Path) -> str:
    """Hash small check evidence without importing training code."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    """Write only after the shell's successful checks, then verify exact evidence bytes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--mode", choices=("write", "verify"), required=True)
    args = parser.parse_args()
    for name in XML_REPORTS:
        suite = ET.parse(args.root / name).getroot()
        cases = suite.findall(".//testcase")
        if suite.findall(".//failure") or suite.findall(".//error") or not cases:
            raise ValueError("receiver test suite failed or contains no test cases")
        if name != "pytest.xml" and suite.findall(".//skipped"):
            raise ValueError("required compiled GPU objective acceptance was skipped")
    for name in ("ddp.json", "complement-ddp.json", "muon-ddp.json"):
        result = json.loads((args.root / name).read_text())
        if result.get("passed") is not True or result.get("world_size") != 2:
            raise ValueError("two-rank objective or optimizer acceptance did not pass")
        if name == "complement-ddp.json" and result.get("blend_objective_alpha") != 0.1:
            raise ValueError("DDP acceptance did not exercise the frozen complement objective")
    expected = {
        "schema": "wnn.tl3-complement-acceptance.v1",
        "status": "checks_completed",
        "commit": args.commit,
        "split_sha256": digest(args.split),
        "evidence_sha256": {name: digest(args.root / name) for name in EVIDENCE},
    }
    path = args.root / "COMPLETE.json"
    if args.mode == "verify":
        if json.loads(path.read_text()) != expected:
            raise ValueError("receiver acceptance evidence, source or split changed")
    else:
        with path.open("x") as stream:
            stream.write(json.dumps(expected, indent=2) + "\n")


if __name__ == "__main__":
    main()
