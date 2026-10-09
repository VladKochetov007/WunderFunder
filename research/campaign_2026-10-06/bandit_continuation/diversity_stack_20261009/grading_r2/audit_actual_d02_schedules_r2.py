#!/usr/bin/env python3
"""Independently compare the frozen D02 quartet's actual producer traces."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from schedule_semantics import audit_quartet_schedules

ROOT = Path(__file__).resolve().parents[5]
CAMPAIGN = Path("research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009")
SPEC_PATH = CAMPAIGN / "grading_r1/FROZEN_D02_QUARTET_R1.json"
SPEC_SHA256 = "b6e8ff174166ae66259b872b6bf3c6809a17fccb3754b472201aaf8fb1f56134"
R5_ROOT = (
    CAMPAIGN / "cloud_delivery/gpu_loss_pair_r5_export/terminal_collect_r1/attempt_001/outputs/out"
)
D02_ROOT = CAMPAIGN / "cloud_delivery/d02_r5/terminal_collect_r1/attempt_001/outputs/out"
OUTPUT = CAMPAIGN / "grading_r2/D02_SCHEDULE_SEMANTIC_AUDIT_R2.json"
ARMS = {
    "adamw_wmse": ("adamw", "wmse", R5_ROOT / "arms/wmse"),
    "adamw_huber_d1": ("adamw", "huber", R5_ROOT / "arms/huber_d1"),
    "muon_wmse": ("muon", "wmse", D02_ROOT / "arms/wmse"),
    "muon_huber_d1": ("muon", "huber", D02_ROOT / "arms/huber_d1"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUTPUT}")
    spec = json.loads((ROOT / SPEC_PATH).read_text())
    spec_sha = sha256(ROOT / SPEC_PATH)
    if SPEC_SHA256 and spec_sha != SPEC_SHA256:
        raise ValueError("frozen D02 quartet spec hash changed")
    if spec.get("status") != "frozen_before_muon_fit_results":
        raise ValueError("D02 quartet spec is not preregistered")
    expected_singles = {"adamw_wmse", "adamw_huber_d1", "muon_wmse", "muon_huber_d1"}
    if set(spec.get("singles", [])) != expected_singles:
        raise ValueError("D02 quartet cells differ from frozen preregistration")

    schedule_inputs: dict[str, dict[str, Any]] = {}
    artifact_pins: dict[str, dict[str, Any]] = {}
    for name, (optimizer, loss, arm_root_rel) in ARMS.items():
        arm_root = ROOT / arm_root_rel
        provenance_path = arm_root / "model/provenance.json"
        provenance = json.loads(provenance_path.read_text())
        args = provenance.get("args", {})
        if args.get("optim") != optimizer or args.get("loss") != loss:
            raise ValueError(f"{name}: actual producer optimizer/loss mismatch")
        if provenance.get("seed") != 20261009 or args.get("checkpoint_epoch") != 1:
            raise ValueError(f"{name}: actual producer seed/selected epoch mismatch")
        if provenance.get("split_manifest_sha256") != spec["split"]["sha256"]:
            raise ValueError(f"{name}: actual producer split differs from preregistration")
        schedule_path = arm_root / "schedule_trace.json"
        update_path = arm_root / "update_trace.jsonl"
        schedule_bytes = schedule_path.read_bytes()
        update_bytes = update_path.read_bytes()
        schedule = json.loads(schedule_bytes)
        update_rows = [json.loads(line) for line in update_bytes.splitlines() if line.strip()]
        schedule_inputs[name] = {
            "optimizer": optimizer,
            "schedule_trace": schedule,
            "update_rows": update_rows,
        }
        artifact_pins[name] = {
            "optimizer": optimizer,
            "loss": loss,
            "run": args.get("name"),
            "provenance": {
                "path": str(provenance_path.relative_to(ROOT)),
                "sha256": sha256(provenance_path),
            },
            "schedule_trace": {
                "path": str(schedule_path.relative_to(ROOT)),
                "sha256": hashlib.sha256(schedule_bytes).hexdigest(),
                "raw_group_layout": [
                    len(row.get("optimizer_groups", [])) for row in schedule.get("rows", [])
                ],
            },
            "update_trace": {
                "path": str(update_path.relative_to(ROOT)),
                "sha256": hashlib.sha256(update_bytes).hexdigest(),
                "rows": len(update_rows),
            },
        }
    semantics = audit_quartet_schedules(schedule_inputs)
    result = {
        "schema": "wnn.d02-quartet-schedule-semantic-audit.r2",
        "status": "pass",
        "created_utc": datetime.now(UTC).isoformat(),
        "frozen_comparison": {"path": str(SPEC_PATH), "sha256": spec_sha},
        "grader_source": {
            "path": str(Path(__file__).relative_to(ROOT)),
            "sha256": sha256(Path(__file__)),
        },
        "semantic_validator": {
            "path": str((Path(__file__).parent / "schedule_semantics.py").relative_to(ROOT)),
            "sha256": sha256(Path(__file__).parent / "schedule_semantics.py"),
        },
        "arms": artifact_pins,
        "comparison": semantics,
        "interpretation": (
            "Actual AdamW control and Muon candidate traces both align step-for-step. "
            "Each Muon optimizer group matches the single AdamW group's LR and beta1 at each update; "
            "raw traces retain their different one-group/two-group producer serialization."
        ),
        "scoring": "not performed by this schedule-only audit",
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, OUTPUT)
    print(
        json.dumps(
            {"path": str(OUTPUT), "sha256": sha256(ROOT / OUTPUT), "comparison": semantics},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
