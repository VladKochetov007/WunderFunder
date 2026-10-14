"""Static preflight for the one-shape-at-a-time F0 student R3 sequences."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

SHAPES = ((256, 2), (256, 1), (192, 2), (512, 1), (384, 2), (640, 1), (512, 2), (448, 3))
EXPECTED_PHASES = (
    "full-fit",
    "post-fit-assertions",
    "frozen-grader-input-check",
    "shape-matched-inference-resource-profile",
    "validate-shape-matched-inference-resource-profile",
    "full-valid-prediction",
    "official-fold0-grade",
    "postgrade-assertions",
)


def sha256_file(path: Path) -> str:
    """Stream an immutable recipe or source file through SHA256."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path, maximum_bytes: int = 512_000) -> dict[str, Any]:
    """Read a bounded recipe object."""
    size = path.stat().st_size
    if size <= 0 or size > maximum_bytes:
        raise ValueError(f"{path} exceeds its {maximum_bytes}-byte read bound")
    with path.open("rb") as source:
        raw = source.read(maximum_bytes + 1)
    if len(raw) != size:
        raise ValueError(f"{path} changed while being read")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def require(actual: object, expected: object, description: str) -> None:
    """Fail on any static recipe mismatch."""
    if actual != expected:
        raise ValueError(f"{description} differs: expected {expected!r}, got {actual!r}")


def argument(command: list[str], name: str) -> str:
    """Return one value from a frozen argv vector."""
    index = command.index(name)
    return command[index + 1]


def validate(root: Path, main: Path, fit_worktree: Path, output: Path) -> dict[str, Any]:
    """Check profile pins, each shape plan and the per-shape export gate."""
    recipes = root / "research/campaign_2026-10-14/cpu_student_frontier_r1/recipes"
    profile_path = recipes / "PREDICTION_PROFILE_F0_EIGHT_SHAPES_B16_R2.json"
    profile = read_json(profile_path)
    profile_sha = sha256_file(profile_path)
    require(profile.get("project_root"), str(main), "profile project root")
    require(profile.get("id"), "student-frontier-f0-seed3-b16-prediction-profiles-r2", "profile id")
    require(profile.get("status"), "prepared_for_root_admission", "profile admission state")
    models = profile.get("models")
    if not isinstance(models, list):
        raise ValueError("profile contract has no model list")
    model_shapes = [(item.get("hidden"), item.get("layers")) for item in models]
    require(model_shapes, list(SHAPES), "profile shape allowlist/order")
    if len({item.get("run") for item in models}) != len(SHAPES):
        raise ValueError("profile model run names are not unique")
    if len({item.get("output_path") for item in models}) != len(SHAPES):
        raise ValueError("profile output paths are not unique")
    plan_reports = []
    seen_attempts: set[str] = set()
    for hidden, layers in SHAPES:
        shape = f"h{hidden}l{layers}"
        plan_path = recipes / f"CPU_STUDENT_FRONTIER_F0_S3_{shape.upper()}_R3_SEQUENCE.json"
        plan = read_json(plan_path)
        require(plan.get("shape"), [hidden, layers], f"{shape} plan shape")
        run = f"student_frontier_f0s3_h{hidden}_l{layers}_r2"
        require(plan.get("run"), run, f"{shape} run")
        require(plan.get("step_count"), len(EXPECTED_PHASES), f"{shape} step count")
        steps = plan.get("steps")
        if not isinstance(steps, list) or [step.get("phase") for step in steps] != list(EXPECTED_PHASES):
            raise ValueError(f"{shape} sequence order is incomplete or differs")
        if any(step.get("attempt") in seen_attempts for step in steps):
            raise ValueError(f"{shape} plan reuses a nonshared attempt ID")
        # The identical grader-input preflight is deliberately shared by all
        # shape plans; every other attempt identifier is unique.
        for step in steps:
            if step["phase"] != "frozen-grader-input-check":
                seen_attempts.add(step["attempt"])
        model = next(item for item in models if item["hidden"] == hidden and item["layers"] == layers)
        fit_worker = steps[0]["expected_command"]
        for name, expected in {
            "--name": run,
            "--hidden": str(hidden),
            "--layers": str(layers),
            "--batch": "8",
            "--chunk": "2000",
            "--epochs": "1",
            "--checkpoint-epoch": "0",
            "--precision": "fp32",
            "--student-frontier-stage": "f0_seed3_fit",
        }.items():
            require(argument(fit_worker, name), expected, f"{shape} trainer argv {name}")
        require(steps[3].get("expected_command"), model["worker_command"], f"{shape} profile worker argv")
        require(steps[3].get("profile_contract_sha256"), profile_sha, f"{shape} profile contract pin")
        profile_launch = steps[3]["launch_command"]
        require(argument(profile_launch, "--contract"), str(profile_path), f"{shape} profile contract path")
        require(argument(profile_launch, "--reserve-bytes"), "5000000000", f"{shape} profile host reserve")
        gate_worker = steps[4]["expected_command"]
        require(argument(gate_worker, "--profile-contract-sha256"), profile_sha, f"{shape} profile gate hash")
        require(argument(gate_worker, "--attempt-id"), model["profile_attempt_id"], f"{shape} profile gate attempt")
        require(argument(gate_worker, "--out"), model["profile_validation_receipt_path"], f"{shape} profile receipt path")
        export_launch = steps[5]["launch_command"]
        require(argument(export_launch, "--contract"), model["profile_validation_receipt_path"], f"{shape} export dependency")
        require(argument(export_launch, "--reserve-bytes"), "6000000000", f"{shape} export host reserve")
        export_worker = steps[5]["expected_command"]
        if "--campaign-profile-first-batch" in export_worker:
            raise ValueError(f"{shape} quality export accidentally uses profile-only mode")
        if "--campaign-full-valid-feature-only" not in export_worker:
            raise ValueError(f"{shape} quality export is not full-valid feature-only")
        if model["output_name"] == run:
            raise ValueError(f"{shape} resource profile collides with its quality prediction name")
        grade_worker = steps[6]["expected_command"]
        scorer_index = grade_worker.index(str(main / "scripts/score_fold0_panel.py"))
        require(grade_worker[scorer_index + 1], f"{run}@ep0", f"{shape} grade candidate")
        profile_path_out = Path(model["output_path"])
        plan_reports.append({
            "shape": [hidden, layers],
            "plan_path": str(plan_path),
            "plan_sha256": sha256_file(plan_path),
            "run": run,
            "step_count": len(steps),
            "profile_attempt": model["profile_attempt_id"],
            "profile_output_path": str(profile_path_out),
            "export_contract_dependency": model["profile_validation_receipt_path"],
            "fit_worker_sha256": hashlib.sha256(json.dumps(fit_worker, separators=(",", ":")).encode()).hexdigest(),
        })

    fit_contract = fit_worktree / "research/campaign_2026-10-14/cpu_student_frontier_r1/recipes/CPU_STUDENT_FRONTIER_F0_SEED3_B8_R2_CONSUMER_CONTRACT.json"
    require(profile.get("consumer_contract"), str(fit_contract), "fit consumer path")
    require(profile.get("consumer_contract_sha256"), sha256_file(fit_contract), "fit consumer hash")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite preflight receipt: {output}")
    receipt = {
        "format": "wnn.student-frontier-r3-sequence-static-preflight",
        "version": 1,
        "profile_contract_path": str(profile_path),
        "profile_contract_sha256": profile_sha,
        "fit_consumer_contract_sha256": profile["consumer_contract_sha256"],
        "checker_source_sha256": sha256_file(Path(__file__).resolve()),
        "shape_order": [list(shape) for shape in SHAPES],
        "shape_plan_count": len(plan_reports),
        "attempt_count_including_shared_preflight": len(seen_attempts) + 1,
        "profiles_and_exports": plan_reports,
        "static_checks": "passed; exact shape allowlist; fixed B8/C2000/FP32 CP0 trainer args; exact per-shape B16 worker command; profile output and validator receipts are distinct; full exports depend on matching profile-validation receipt; profile/export host reservations are 5/6 GB; one plan is one shape and does not auto-advance",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt


def main() -> None:
    """Run static checks without loading model/data payloads."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--main", type=Path, required=True)
    parser.add_argument("--fit-worktree", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    receipt = validate(
        args.root.resolve(strict=True),
        args.main.resolve(strict=True),
        args.fit_worktree.resolve(strict=True),
        args.out.resolve(),
    )
    print(json.dumps({"status": receipt["static_checks"], "shape_plan_count": receipt["shape_plan_count"]}))


if __name__ == "__main__":
    main()
