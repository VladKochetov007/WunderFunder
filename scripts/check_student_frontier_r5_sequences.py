"""Static audit for the seven single-shape R5 cache-handoff sequences."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

SHAPES = ((256, 1), (192, 2), (512, 1), (384, 2), (640, 1), (512, 2), (448, 3))
BASE_PHASES = (
    "full-fit",
    "post-fit-assertions",
    "frozen-grader-input-check",
    "shape-matched-inference-resource-profile",
    "validate-shape-matched-inference-resource-profile",
    "full-valid-prediction",
    "official-fold0-grade",
    "postgrade-assertions",
)
R5_PHASES = (
    *BASE_PHASES[:5],
    "file-cache-handoff-after-profile-before-export",
    *BASE_PHASES[5:],
    "file-cache-handoff-after-grade-before-next-fit",
)
BASE_TO_R5_INDEX = (0, 1, 2, 3, 4, 6, 7, 8)
SHARED_ATTEMPT = "student-frontier-f0s3-grader-inputs-check-r3"
HANDOFF_RESERVE_BYTES = 400_000_000
PROFILE_NEXT_RESERVE_BYTES = 6_000_000_000
FIT_NEXT_RESERVE_BYTES = 5_000_000_000


def sha256_file(path: Path, maximum_bytes: int = 512_000) -> str:
    """Hash a small frozen recipe or admission record."""
    size = path.stat().st_size
    if size <= 0 or size > maximum_bytes:
        raise ValueError(f"{path} exceeds its {maximum_bytes}-byte hash bound")
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path, maximum_bytes: int = 512_000) -> dict[str, Any]:
    """Read a small JSON object after checking the file size."""
    size = path.stat().st_size
    if size <= 0 or size > maximum_bytes:
        raise ValueError(f"{path} exceeds its {maximum_bytes}-byte read bound")
    raw = path.read_bytes()
    if len(raw) != size:
        raise ValueError(f"{path} changed while it was being read")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def argv_value(command: list[str], option: str) -> str:
    """Return a unique argv option value."""
    indexes = [index for index, token in enumerate(command) if token == option]
    if len(indexes) != 1 or indexes[0] + 1 >= len(command):
        raise ValueError(f"expected exactly one value for {option}")
    return command[indexes[0] + 1]


def require(actual: object, expected: object, label: str) -> None:
    """Fail closed on a static recipe mismatch."""
    if actual != expected:
        raise ValueError(f"{label} differs: expected {expected!r}, got {actual!r}")


def validate(root: Path, main: Path, base_ops: Path, output: Path) -> dict[str, Any]:
    """Check all seven plans preserve R4 commands and add only scoped gates."""
    recipes = root / "research/campaign_2026-10-14/cpu_student_frontier_r1/recipes"
    base_recipes = base_ops / "research/campaign_2026-10-14/cpu_student_frontier_r1/recipes"
    admission_path = (
        main
        / "research/campaign_2026-10-14/cpu_student_frontier_r1/recipes/ROOT_REMAINING_F0_SEED3_ADMISSION_R4.json"
    )
    admission = read_json(admission_path)
    require(
        admission.get("order"),
        [f"h{hidden}l{layers}" for hidden, layers in SHAPES],
        "root-admitted remaining-shape order",
    )
    first_base_plan = read_json(base_recipes / "CPU_STUDENT_FRONTIER_F0_S3_H256L1_R3_SEQUENCE.json")
    profile_contract_path = Path(first_base_plan["profile_contract"]).resolve(strict=True)
    profile_contract_sha256 = sha256_file(profile_contract_path)
    profile_contract = read_json(profile_contract_path)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite static preflight receipt: {output}")

    reports: list[dict[str, Any]] = []
    used_attempts: dict[str, str] = {}
    for hidden, layers in SHAPES:
        shape_id = f"H{hidden}L{layers}"
        stem = f"CPU_STUDENT_FRONTIER_F0_S3_{shape_id}_R3_SEQUENCE.json"
        base_path = base_recipes / stem
        base_plan = read_json(base_path)
        base_sha = sha256_file(base_path)
        r5_path = recipes / f"CPU_STUDENT_FRONTIER_F0_S3_{shape_id}_R5_SEQUENCE.json"
        plan = read_json(r5_path)
        run = f"student_frontier_f0s3_h{hidden}_l{layers}_r2"
        require(plan.get("format"), "wnn.bandit-sequence-plan", f"{shape_id} plan format")
        require(plan.get("shape"), [hidden, layers], f"{shape_id} shape")
        require(plan.get("run"), run, f"{shape_id} run")
        require(plan.get("source_plan_path"), str(base_path), f"{shape_id} source plan path")
        require(plan.get("source_plan_sha256"), base_sha, f"{shape_id} source plan SHA")
        require(
            plan.get("profile_contract_sha256"),
            profile_contract_sha256,
            f"{shape_id} profile contract SHA",
        )
        require(plan.get("step_count"), 10, f"{shape_id} step count")
        steps = plan.get("steps")
        if not isinstance(steps, list) or [item.get("phase") for item in steps] != list(R5_PHASES):
            raise ValueError(f"{shape_id} R5 phase order differs")
        require(base_plan.get("shape"), [hidden, layers], f"{shape_id} base shape")
        require(base_plan.get("run"), run, f"{shape_id} base run")
        for base_index, r5_index in enumerate(BASE_TO_R5_INDEX):
            require(
                steps[r5_index],
                base_plan["steps"][base_index],
                f"{shape_id} preserved R4 step {base_index}",
            )
        if steps[0]["attempt"] != f"student-frontier-f0s3-fit-h{hidden}l{layers}-r2":
            raise ValueError(f"{shape_id} fit attempt identity changed")

        model = next(
            item
            for item in profile_contract["models"]
            if item["hidden"] == hidden and item["layers"] == layers
        )
        gate_step = steps[5]
        grade_handoff = steps[9]
        for handoff, scope, reserve in (
            (gate_step, "after-profile", "6000000000"),
            (grade_handoff, "after-grade", "5000000000"),
        ):
            attempt_id = handoff.get("attempt")
            if attempt_id in used_attempts:
                raise ValueError(f"duplicate handoff attempt ID: {attempt_id}")
            used_attempts[str(attempt_id)] = shape_id
            expected_script = str(root / "scripts/reclaim_student_frontier_cache.py")
            command = handoff.get("expected_command")
            launch = handoff.get("launch_command")
            if not isinstance(command, list) or not isinstance(launch, list):
                raise ValueError(f"{shape_id} {scope} handoff command is missing")
            require(expected_script in command, True, f"{shape_id} {scope} helper path")
            require(argv_value(command, "--scope"), scope, f"{shape_id} {scope} scope")
            require(argv_value(command, "--run"), run, f"{shape_id} {scope} run")
            require(argv_value(command, "--hidden"), str(hidden), f"{shape_id} {scope} hidden")
            require(argv_value(command, "--layers"), str(layers), f"{shape_id} {scope} layers")
            require(
                argv_value(command, "--base-plan"), str(base_path), f"{shape_id} {scope} base plan"
            )
            require(
                argv_value(command, "--base-plan-sha256"), base_sha, f"{shape_id} {scope} base SHA"
            )
            require(
                argv_value(command, "--self-attempt"),
                str(attempt_id),
                f"{shape_id} {scope} self attempt",
            )
            require(
                argv_value(launch, "--contract"),
                str(base_path),
                f"{shape_id} {scope} controller contract",
            )
            require(
                argv_value(launch, "--reserve-bytes"),
                str(HANDOFF_RESERVE_BYTES),
                f"{shape_id} {scope} reservation",
            )
            require(argv_value(launch, "--kind"), "cpu", f"{shape_id} {scope} worker kind")
            require(argv_value(launch, "--cpus"), "1", f"{shape_id} {scope} CPU reservation")
            require(
                handoff.get("must_not_exist"),
                [handoff.get("handoff_receipt_path")],
                f"{shape_id} {scope} no-overwrite path",
            )
            require(
                argv_value(command, "--out"),
                handoff["handoff_receipt_path"],
                f"{shape_id} {scope} output path",
            )
            if Path(handoff["handoff_receipt_path"]).exists():
                raise FileExistsError(f"{shape_id} {scope} handoff receipt already exists")
            if not any(token == "--contract" for token in launch) or reserve not in (
                steps[6]["launch_command"]
                if scope == "after-profile"
                else steps[0]["launch_command"]
            ):
                raise ValueError(
                    f"{shape_id} {scope} does not preserve its next-phase resource gate"
                )
        export_step = steps[6]
        if model["profile_validation_receipt_path"] != argv_value(
            export_step["launch_command"], "--contract"
        ):
            raise ValueError(f"{shape_id} export does not depend on its matching profile receipt")
        if "predict_run.py" in export_step["expected_command"]:
            raise ValueError(f"{shape_id} export command regressed to the legacy predictor")
        predictor_copy = str(profile_contract["source"]["profiled_predictor_copy"])
        if predictor_copy not in export_step["expected_command"]:
            raise ValueError(f"{shape_id} export does not use the pinned student predictor")
        reports.append(
            {
                "shape": [hidden, layers],
                "base_plan_path": str(base_path),
                "base_plan_sha256": base_sha,
                "plan_path": str(r5_path),
                "plan_sha256": sha256_file(r5_path),
                "profile_handoff_attempt": gate_step["attempt"],
                "profile_handoff_receipt": gate_step["handoff_receipt_path"],
                "grade_handoff_attempt": grade_handoff["attempt"],
                "grade_handoff_receipt": grade_handoff["handoff_receipt_path"],
                "preserved_fit_attempt": steps[0]["attempt"],
            }
        )

    receipt = {
        "format": "wnn.student-frontier-r5-sequence-static-preflight",
        "version": 1,
        "root_admission_path": str(admission_path),
        "root_admission_sha256": sha256_file(admission_path),
        "root_admitted_order": [f"h{hidden}l{layers}" for hidden, layers in SHAPES],
        "profile_contract_path": str(profile_contract_path),
        "profile_contract_sha256": profile_contract_sha256,
        "plan_builder_path": str(root / "scripts/build_student_frontier_r5_sequences.py"),
        "plan_builder_source_sha256": sha256_file(
            root / "scripts/build_student_frontier_r5_sequences.py", 20_000_000
        ),
        "handoff_helper_path": str(root / "scripts/reclaim_student_frontier_cache.py"),
        "handoff_helper_source_sha256": sha256_file(
            root / "scripts/reclaim_student_frontier_cache.py", 20_000_000
        ),
        "static_checker_path": str(Path(__file__).resolve()),
        "static_checker_source_sha256": sha256_file(Path(__file__).resolve(), 20_000_000),
        "shape_plan_count": len(reports),
        "phase_order": list(R5_PHASES),
        "handoff_worker_reserve_bytes": HANDOFF_RESERVE_BYTES,
        "next_phase_reserves_bytes": {
            "after-profile": PROFILE_NEXT_RESERVE_BYTES,
            "after-grade": FIT_NEXT_RESERVE_BYTES,
        },
        "preserved_source_commands": "all eight original R4 phase steps are byte-identical in each R5 plan; only two guarded CPU handoff steps and sequence metadata were added",
        "resource_safety": "handoff helper requires terminal source-bound profile/grade evidence, acquires controller and GPU locks nonblocking, rejects active/unresolved GPU attempts, rechecks unchanged caps, and reclaims file cache only",
        "plans": reports,
        "static_checks": "passed; seven root-admitted shapes and order, exact R4 commands preserved, custom predictor/profile dependency pinned, matching profile and grade handoff phases, unique receipts and 400 MB one-CPU maintenance attempts",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite static preflight receipt: {output}")
    output.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt


def main_cli() -> None:
    """Run static R5 sequence checks without loading prediction payloads."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--main", type=Path, required=True)
    parser.add_argument("--base-ops-worktree", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    receipt = validate(
        args.root.resolve(strict=True),
        args.main.resolve(strict=True),
        args.base_ops_worktree.resolve(strict=True),
        args.out.resolve(),
    )
    print(
        json.dumps(
            {"status": receipt["static_checks"], "shape_plan_count": receipt["shape_plan_count"]}
        )
    )


if __name__ == "__main__":
    main_cli()
