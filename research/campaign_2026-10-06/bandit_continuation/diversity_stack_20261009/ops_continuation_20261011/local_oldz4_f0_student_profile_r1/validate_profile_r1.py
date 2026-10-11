#!/usr/bin/env python3
"""Static non-Torch validator for the frozen local H230 student profile packet."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sys

PACKET = Path(__file__).resolve().parent
MAIN = Path("/home/vlad/development/wunder/wnn_connectome_starterpack")
OPERATOR = Path("/home/vlad/development/wunder/wnn_old_pipeline_reconstruct_20261011_r1")
OPS = Path("research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/ops_continuation_20261011")
OLDZ4 = OPERATOR / OPS / "old_z4_cleanfold_successor_20261011_r3"
SOURCE = OPERATOR / "research/campaign_2026-10-06/bandit_continuation/continuous_execution_E529/h230_source_v3_all_data_distill"


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def derive_command(frozen: list[str], profile_name: str, trainer: str) -> list[str]:
    out = list(frozen)
    require(out[:4] == ["/usr/bin/timeout", "--signal=TERM", "--kill-after=20s", "10800s"],
            "frozen fit timeout prefix changed")
    out[3] = "600s"
    trainer_index = out.index(trainer)
    env_index = out.index(f"PYTHONPATH={SOURCE / 'src'}")
    out[env_index:env_index] = ["WNN_GPU_MEMORY_BYTES=4300000000", "WNN_LOADER_RAM_GB=6"]
    out[out.index("--name", trainer_index) + 1] = profile_name
    require("--save-training-state" in out, "frozen fit continuation flag disappeared")
    out.remove("--save-training-state")
    out.extend(["--grad-stats", "--profile-warmup", "1", "--profile-steps", "1"])
    return out


def main() -> int:
    contract_path = PACKET / "PROFILE_CONTRACT_R1.json"
    launch_path = PACKET / "PROFILE_LAUNCH_PACKET_R1.json"
    contract, launch = load(contract_path), load(launch_path)
    inputs = contract["inputs"]
    require(contract["launch_authorized"] is False and contract["fit_authorization"] is False,
            "packet must remain unadmitted")
    require(contract["scientific_result"] is False and contract["profile_only"] is True,
            "packet must be profile-only")
    require(digest(contract_path) == launch["profile_contract_sha256"], "contract SHA mismatch")

    plan_path = Path(contract["recipe_binding"]["sequence_plan"])
    fit_path = Path(contract["recipe_binding"]["fit_contract"])
    input_path = Path(inputs["input_bindings"])
    source_bindings_path = Path(inputs["source_bindings"])
    manifest_path = Path(inputs["source_manifest"])
    for path, expected, label in (
        (plan_path, contract["recipe_binding"]["sequence_plan_sha256"], "sequence"),
        (fit_path, contract["recipe_binding"]["fit_contract_sha256"], "fit contract"),
        (input_path, inputs["input_bindings_sha256"], "input bindings"),
        (source_bindings_path, inputs["source_bindings_sha256"], "source bindings"),
        (manifest_path, inputs["source_manifest_sha256"], "source manifest"),
    ):
        require(path.is_file() and digest(path) == expected, f"{label} missing or changed")

    plan = load(plan_path)
    fit = load(fit_path)
    step = next(item for item in plan["steps"] if item["attempt"] == contract["recipe_binding"]["fit_attempt"])
    derived = derive_command(step["expected_command"], contract["run_name"], inputs["executed_trainer"])
    require(derived == contract["profile"]["expected_command"] == launch["expected_command"],
            "profile argv diverges from exact fit transform")
    require(launch["launch_command"] == contract["launch_command"], "launch command copies disagree")
    delimiter = launch["launch_command"].index("--")
    require(launch["launch_command"][delimiter + 1:] == derived, "worker argv after delimiter differs")
    require(fit["input_bindings"]["sha256"] == inputs["input_bindings_sha256"],
            "fit contract input binding mismatch")

    fold = load(input_path)["folds"]["0"]
    require(fold["fold"] == 0 and fold["seed"] == 3, "wrong fold or seed")
    require(fold["split_manifest_sha256"] == inputs["split_manifest_sha256"], "split binding mismatch")
    require(fold["propensity_model_sha256"] == inputs["propensity_sha256"], "propensity binding mismatch")
    require(fold["teacher_cache_metadata_sha256"] == inputs["cache_metadata_sha256"],
            "cache metadata binding mismatch")
    require("10,606 train.parquet groups" in fold["source_qualified_train_membership"],
            "train membership changed")
    require(fold["eligible_inner_valid_group_count"] == 1405, "eligible valid membership changed")
    require(fold["evaluation_group_count"] == 468, "excluded evaluation membership changed")

    for label, path_key, hash_key in (
        ("split", "split_manifest", "split_manifest_sha256"),
        ("propensity", "propensity_model", "propensity_sha256"),
        ("cache metadata", "cache_metadata", "cache_metadata_sha256"),
        ("executed trainer", "executed_trainer", "executed_trainer_sha256"),
        ("manifest trainer", "manifest_trainer", "manifest_trainer_sha256"),
        ("optimizer diagnostics", "optim", "optim_sha256"),
    ):
        path = Path(inputs[path_key])
        require(path.is_file() and digest(path) == inputs[hash_key], f"{label} missing or changed")
    require(inputs["executed_trainer_sha256"] == inputs["manifest_trainer_sha256"],
            "executed trainer differs from manifest trainer")

    manifest = load(manifest_path)
    source_root = Path(inputs["source_root"])
    expected_root = MAIN / "research/campaign_2026-10-06/bandit_continuation/continuous_execution_E529/h230_source_v3_all_data_distill"
    require(manifest["source_root"] == str(expected_root), "source manifest root changed")
    require(len(manifest["files"]) == manifest["file_count"], "source manifest count mismatch")
    for entry in manifest["files"]:
        rel = Path(entry["path"])
        require(not rel.is_absolute() and ".." not in rel.parts, "unsafe source manifest path")
        source_file = source_root / rel
        require(source_file.is_file() and source_file.stat().st_size == entry["bytes"],
                f"source file missing/size changed: {entry['path']}")
        require(digest(source_file) == entry["sha256"], f"source file hash changed: {entry['path']}")

    trainer_text = Path(inputs["executed_trainer"]).read_text()
    optim_text = Path(inputs["optim"]).read_text()
    for needle in (
        'perf.add_argument("--profile-steps"',
        'perf.add_argument("--profile-warmup"',
        'research.add_argument("--grad-stats"',
        "args.max_batches = args.profile_warmup + args.profile_steps",
        'print("GRAD_STATS", json.dumps(epoch_grad_stats), flush=True)',
        'print("PROFILE", json.dumps({k: round(v, 2) for k, v in med.items()}), flush=True)',
    ):
        require(needle in trainer_text, f"trainer profile branch changed: {needle}")
    require("class GradStats:" in optim_text and "def observe(self)" in optim_text
            and '"nonfinite_steps"' in optim_text, "GradStats behavior changed")
    require(contract["profile"]["shape"]["expected_gradient_observations"] == 20,
            "gradient observation count changed")
    require(derived[-5:] == ["--grad-stats", "--profile-warmup", "1", "--profile-steps", "1"],
            "profile diagnostic flags mismatch")
    require("--save-training-state" not in derived, "profile can save continuation state")

    require(launch["attempt"] == contract["attempt_id"], "launch attempt mismatch")
    require(launch["profile_contract_path"] == str(contract_path), "launch contract path mismatch")
    require(launch["must_not_exist"] == contract["outputs"]["must_not_exist"], "collision lists differ")
    collisions = [path for path in launch["must_not_exist"] if Path(path).exists()]
    require(not collisions, f"profile output collision: {collisions}")

    parent = Path(inputs["initializer"]["checkpoint"])
    state = "present; root producer receipt and digest binding still required" if parent.exists() else "pending; ep3 checkpoint not yet present"
    print("PASS: exact fit-to-profile command transform and packet bindings")
    print(f"PASS: all {manifest['file_count']} source files match manifest")
    print(f"PENDING: {state}")
    print("PASS: no output collision; no Torch import or model/GPU launch")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, StopIteration, json.JSONDecodeError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1)
