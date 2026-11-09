"""Bind receiver checks and exact production/peak-profile evidence before fitting."""

from __future__ import annotations

import argparse
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

XML = ("pytest.xml", "bf16.xml", "bf16_bf16x9.xml")
EVIDENCE = (*XML, "stem-ddp.json", "muon-ddp.json")


def digest(path: Path) -> str:
    """Hash small receiver evidence, not model payloads."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checks(root: Path, commit: str, split: Path, *, write: bool) -> None:
    """Admit only complete, nonempty test suites and successful two-rank checks."""
    folder = root / "acceptance"
    for name in XML:
        suite = ET.parse(folder / name).getroot()
        cases = suite.findall(".//testcase")
        if not cases or suite.findall(".//failure") or suite.findall(".//error"):
            raise ValueError("receiver correctness suite failed or did not execute")
        if len(suite.findall(".//skipped")) == len(cases):
            raise ValueError("receiver correctness suite was entirely skipped")
        if name != "pytest.xml" and suite.findall(".//skipped"):
            raise ValueError("required BF16/BF16x9 acceptance was skipped")
    for name in ("stem-ddp.json", "muon-ddp.json"):
        value = json.loads((folder / name).read_text())
        if value.get("passed") is not True or value.get("world_size") != 2:
            raise ValueError("two-rank acceptance failed")
    expected = {
        "schema": "wnn.input-stem-acceptance.v1",
        "commit": commit,
        "split_sha256": digest(split),
        "data_check_sha256": digest(root / "data-check.json"),
        "evidence_sha256": {name: digest(folder / name) for name in EVIDENCE},
    }
    path = folder / "COMPLETE.json"
    if write:
        with path.open("x") as stream:
            stream.write(json.dumps(expected, indent=2) + "\n")
    elif json.loads(path.read_text()) != expected:
        raise ValueError("accepted source/split/check evidence changed")


def profile_pair(root: Path, cell: str, split: Path) -> None:
    """Reject approvals for another recipe or missing actual-state reload proof."""
    from wnn.training.sequence_config import load_config
    from wnn.training.sequence_provenance import canonical_sha256, source_identity

    packet = Path(__file__).resolve().parent
    if cell not in {
        f"{arm}_s{seed}" for arm in ("plain", "k1", "k16") for seed in (203201, 203202)
    }:
        raise ValueError("unknown frozen cell")
    repository = packet.parents[2]
    source = source_identity(repository, repository / "scripts/train_sequence.py")
    folder = root / "runs" / cell
    production = load_config(packet / "configs" / f"{cell}.json")
    receipts = []
    for stage, config in (
        ("profile", production),
        ("profile-peak", production | {"warmup_fraction": 0.0}),
    ):
        receipt = json.loads((folder / stage / "profile_receipt.json").read_text())
        identity = receipt.get("identity", {})
        if (
            receipt.get("status") != "profile_complete"
            or receipt.get("approved") is not False
            or receipt.get("updates") != 50
            or receipt.get("errors") != []
            or receipt.get("error_count") != 0
            or receipt.get("scientific_checkpoint_written") is not False
            or identity.get("source_sha256") != source["sha256"]
            or identity.get("source_files") != source["files"]
            or identity.get("config_sha256") != canonical_sha256(config)
            or identity.get("data", {}).get("split_sha256") != digest(split)
            or identity.get("world_size") != 2
        ):
            raise ValueError("profile identity/work/status differs from the frozen cell")
        reload = receipt.get("engineering_reload", {})
        ranks = reload.get("by_rank", [])
        if (
            reload.get("status") != "passed"
            or [r.get("rank") for r in ranks] != [0, 1]
            or any(r.get("model_optimizer_scheduler_rng_equal") is not True for r in ranks)
        ):
            raise ValueError("actual-shaped model/optimizer/scheduler/RNG reload is incomplete")
        receipts.append(receipt)
    if receipts[0]["identity"]["runtime"] != receipts[1]["identity"]["runtime"]:
        raise ValueError("peak and production profiles must use the same runtime/devices")
    approved = json.loads((folder / "approved_profile.json").read_text())
    mutable = {"approved", "reviewer", "reviewed_at", "approval_note"}
    if (
        approved.get("approved") is not True
        or not approved.get("reviewer")
        or not approved.get("reviewed_at")
        or any(
            approved.get(key) != value for key, value in receipts[0].items() if key not in mutable
        )
    ):
        raise ValueError("reviewer approval must preserve the exact production profile evidence")


def main() -> None:
    """Validate immutable evidence without approving any profile automatically."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--mode", choices=("write", "verify", "profiles"), required=True)
    parser.add_argument("--cell", default="")
    args = parser.parse_args()
    checks(args.root, args.commit, args.split, write=args.mode == "write")
    if args.mode == "profiles":
        profile_pair(args.root, args.cell, args.split)


if __name__ == "__main__":
    main()
