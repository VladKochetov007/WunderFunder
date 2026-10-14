"""Build single-shape R5 plans with guarded file-cache handoffs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

SHAPES = ((256, 1), (192, 2), (512, 1), (384, 2), (640, 1), (512, 2), (448, 3))
CPU_RESERVE_BYTES = 400_000_000
HANDOFF_TIMEOUT = "120s"


def read_json(path: Path, maximum_bytes: int = 512_000) -> dict[str, Any]:
    """Read bounded plan metadata after checking the file length."""
    size = path.stat().st_size
    if size <= 0 or size > maximum_bytes:
        raise ValueError(f"{path} exceeds its {maximum_bytes}-byte read bound")
    raw = path.read_bytes()
    if len(raw) != size:
        raise ValueError(f"{path} changed while it was being read")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def sha256_file(path: Path, maximum_bytes: int = 512_000) -> str:
    """Hash one bounded plan file."""
    size = path.stat().st_size
    if size <= 0 or size > maximum_bytes:
        raise ValueError(f"{path} exceeds its {maximum_bytes}-byte hash bound")
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def argv_value(command: list[str], option: str) -> str:
    """Read exactly one CLI value from a frozen argv vector."""
    indexes = [index for index, token in enumerate(command) if token == option]
    if len(indexes) != 1 or indexes[0] + 1 >= len(command):
        raise ValueError(f"expected one value after {option}")
    return command[indexes[0] + 1]


def handoff_step(
    *,
    main: Path,
    ops_worktree: Path,
    source_plan_path: Path,
    source_plan_sha256: str,
    source_plan: dict[str, Any],
    hidden: int,
    layers: int,
    scope: str,
    attempt_id: str,
    receipt_path: Path,
) -> dict[str, Any]:
    """Construct one CPU-guarded handoff step bound to the R4 source plan."""
    run = str(source_plan["run"])
    runtime = str(source_plan["runtime"])
    python = main / ".venv/bin/python"
    controller = main / "scripts/bandit_resources.py"
    helper = ops_worktree / "scripts/reclaim_student_frontier_cache.py"
    worker = [
        "timeout",
        "--signal=TERM",
        "--kill-after=10s",
        HANDOFF_TIMEOUT,
        "/usr/bin/env",
        f"WNN_PROJECT_ROOT={main}",
        "WNN_THREADS=1",
        str(python),
        str(helper),
        "--root",
        str(main),
        "--runtime",
        runtime,
        "--base-plan",
        str(source_plan_path),
        "--base-plan-sha256",
        source_plan_sha256,
        "--scope",
        scope,
        "--run",
        run,
        "--hidden",
        str(hidden),
        "--layers",
        str(layers),
        "--self-attempt",
        attempt_id,
        "--out",
        str(receipt_path),
    ]
    next_phase = (
        "the matching B16 full-valid export" if scope == "after-profile" else "the next shape's fit"
    )
    profile = (
        f"student-frontier {scope} scoped file-cache handoff for {run}; require exact terminal evidence; "
        f"prepare {next_phase}; no source/artifact deletion or cap changes"
    )
    launch = [
        str(python),
        str(controller),
        "--root",
        str(ops_worktree),
        "--state",
        runtime,
        "launch",
        "--attempt",
        attempt_id,
        "--kind",
        "cpu",
        "--cpus",
        "1",
        "--reserve-bytes",
        str(CPU_RESERVE_BYTES),
        "--profile",
        profile,
        "--contract",
        str(source_plan_path),
        "--python",
        str(python),
        "--admission-timeout",
        "300",
        "--",
        *worker,
    ]
    base_plan_steps = source_plan["steps"]
    if scope == "after-profile":
        prerequisites = {
            "profile_attempt": base_plan_steps[3]["attempt"],
            "profile_validation_attempt": base_plan_steps[4]["attempt"],
            "profile_validation_receipt": str(
                Path(argv_value(base_plan_steps[4]["expected_command"], "--out"))
            ),
            "export_attempt": base_plan_steps[5]["attempt"],
        }
        phase = "file-cache-handoff-after-profile-before-export"
    else:
        prerequisites = {
            "profile_validation_receipt": str(
                Path(argv_value(base_plan_steps[4]["expected_command"], "--out"))
            ),
            "export_attempt": base_plan_steps[5]["attempt"],
            "grade_attempt": base_plan_steps[6]["attempt"],
            "grade_validation_attempt": base_plan_steps[7]["attempt"],
            "grade_report": str(Path(argv_value(base_plan_steps[6]["expected_command"], "--out"))),
            "grade_validation_receipt": str(
                Path(argv_value(base_plan_steps[7]["expected_command"], "--out"))
            ),
        }
        phase = "file-cache-handoff-after-grade-before-next-fit"
    return {
        "attempt": attempt_id,
        "launch_command": launch,
        "expected_command": worker,
        "must_not_exist": [str(receipt_path)],
        "output_bytes": 100_000,
        "run": run,
        "phase": phase,
        "shape": [hidden, layers],
        "handoff_scope": scope,
        "handoff_receipt_path": str(receipt_path),
        "handoff_base_plan_sha256": source_plan_sha256,
        "prerequisites": prerequisites,
    }


def build(
    main: Path,
    fit_worktree: Path,
    base_ops_worktree: Path,
    ops_worktree: Path,
) -> list[dict[str, Any]]:
    """Derive the seven R5 sequence plans without changing any model command."""
    main = main.resolve(strict=True)
    fit_worktree = fit_worktree.resolve(strict=True)
    base_ops_worktree = base_ops_worktree.resolve(strict=True)
    ops_worktree = ops_worktree.resolve(strict=True)
    recipes = ops_worktree / "research/campaign_2026-10-14/cpu_student_frontier_r1/recipes"
    results = main / "research/campaign_2026-10-14/cpu_student_frontier_r1/results"
    if fit_worktree.name != "student-frontier-sequence-20261014":
        raise ValueError("unexpected frozen fit worktree")
    if base_ops_worktree.name != "student-frontier-ops-r4-20261014":
        raise ValueError("expected the frozen R4 source worktree")
    if ops_worktree.name != "student-frontier-ops-r5-20261014":
        raise ValueError("unexpected R5 successor worktree")
    output_plans: list[dict[str, Any]] = []
    for hidden, layers in SHAPES:
        shape_id = f"H{hidden}L{layers}"
        stem = f"CPU_STUDENT_FRONTIER_F0_S3_{shape_id}_R3_SEQUENCE.json"
        source_plan_path = (
            base_ops_worktree
            / "research/campaign_2026-10-14/cpu_student_frontier_r1/recipes"
            / stem
        )
        source_plan_sha256 = sha256_file(source_plan_path)
        source_plan = read_json(source_plan_path)
        run = f"student_frontier_f0s3_h{hidden}_l{layers}_r2"
        if source_plan.get("shape") != [hidden, layers] or source_plan.get("run") != run:
            raise ValueError(f"frozen R4 source plan identity differs for {shape_id}")
        plan = json.loads(json.dumps(source_plan))
        plan["phase"] = f"f0_seed3_h{hidden}l{layers}_fixed_epoch_screen_r5_scoped_cache_handoffs"
        plan["sequence_successor_worktree"] = str(ops_worktree)
        plan["source_plan_path"] = str(source_plan_path)
        plan["source_plan_sha256"] = source_plan_sha256
        plan["step_count"] = 10
        plan["resource_policy"]["cache_handoff_reserve_bytes"] = CPU_RESERVE_BYTES
        plan["resource_policy"]["cache_handoff_target_parent_bytes"] = 8_500_000_000
        plan["resource_policy"]["execution_order"] = (
            "one shape only: fit, fit assertions, grader-input precheck, exact-shape profile, "
            "profile validation, source-bound idle-GPU file-cache handoff, full-valid export, "
            "official grade, grade validation, then source-bound idle-GPU file-cache handoff; "
            "no next architecture is auto-admitted"
        )
        after_profile_id = f"student-frontier-f0s3-cache-profile-h{hidden}l{layers}-r1"
        after_grade_id = f"student-frontier-f0s3-cache-grade-h{hidden}l{layers}-r1"
        profile_receipt_path = (
            results
            / f"student_frontier_f0s3_h{hidden}_l{layers}_r5_after_profile_cache_handoff.json"
        )
        grade_receipt_path = (
            results / f"student_frontier_f0s3_h{hidden}_l{layers}_r5_after_grade_cache_handoff.json"
        )
        profile_step = handoff_step(
            main=main,
            ops_worktree=ops_worktree,
            source_plan_path=source_plan_path,
            source_plan_sha256=source_plan_sha256,
            source_plan=source_plan,
            hidden=hidden,
            layers=layers,
            scope="after-profile",
            attempt_id=after_profile_id,
            receipt_path=profile_receipt_path,
        )
        grade_step = handoff_step(
            main=main,
            ops_worktree=ops_worktree,
            source_plan_path=source_plan_path,
            source_plan_sha256=source_plan_sha256,
            source_plan=source_plan,
            hidden=hidden,
            layers=layers,
            scope="after-grade",
            attempt_id=after_grade_id,
            receipt_path=grade_receipt_path,
        )
        plan["steps"] = [
            *source_plan["steps"][:5],
            profile_step,
            *source_plan["steps"][5:],
            grade_step,
        ]
        output = recipes / f"CPU_STUDENT_FRONTIER_F0_S3_{shape_id}_R5_SEQUENCE.json"
        if output.exists() or profile_receipt_path.exists() or grade_receipt_path.exists():
            raise FileExistsError(f"R5 sequence or receipt already exists for {shape_id}")
        output.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
        output_plans.append(
            {
                "shape": [hidden, layers],
                "source_plan_path": str(source_plan_path),
                "source_plan_sha256": source_plan_sha256,
                "plan_path": str(output),
                "after_profile_attempt": after_profile_id,
                "after_profile_receipt": str(profile_receipt_path),
                "after_grade_attempt": after_grade_id,
                "after_grade_receipt": str(grade_receipt_path),
            }
        )
    return output_plans


def main_cli() -> None:
    """Generate the exact seven R5 shape plans once."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main", type=Path, required=True)
    parser.add_argument("--fit-worktree", type=Path, required=True)
    parser.add_argument("--base-ops-worktree", type=Path, required=True)
    parser.add_argument("--ops-worktree", type=Path, required=True)
    args = parser.parse_args()
    plans = build(args.main, args.fit_worktree, args.base_ops_worktree, args.ops_worktree)
    print(json.dumps({"status": "built", "shape_plan_count": len(plans), "plans": plans}))


if __name__ == "__main__":
    main_cli()
