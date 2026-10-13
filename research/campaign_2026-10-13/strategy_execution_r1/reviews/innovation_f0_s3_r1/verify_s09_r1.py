#!/usr/bin/env python3
"""Bounded read-only independent preflight for the frozen S09 R1 packet."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

MAIN = Path("/home/vlad/development/wunder/wnn_connectome_starterpack")
SOURCE = Path("/home/vlad/development/wunder/wnn-agent-worktrees/strategy-innovation-fit-20261013")
PACKET = SOURCE / "research/campaign_2026-10-13/strategy_execution_r1/innovation_f0_s3_r1"
PARENT = MAIN / "artifacts/runs/c20261003_combo_s3_f0/model_ep3.pt"
PARENT_SHA = "d4a1e50b1892b4d9edd7c255908b68073a760c059ca80223e98a63c7e023371e"
SOURCE_COMMIT = "c6e774edc8d5e0c63715a83287cf36fd3969a779"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command_flag(argv: list[str], flag: str) -> str | None:
    try:
        return argv[argv.index(flag) + 1]
    except (ValueError, IndexError):
        return None


def env_map(argv: list[str]) -> dict[str, str]:
    result = {}
    for value in argv:
        if "=" in value and not value.startswith("/"):
            key, val = value.split("=", 1)
            if key.replace("_", "").isalnum():
                result[key] = val
    return result


def main() -> None:
    commit = subprocess.check_output(["git", "-C", str(SOURCE), "rev-parse", "HEAD"], text=True).strip()
    status = subprocess.check_output(["git", "-C", str(SOURCE), "status", "--porcelain"], text=True)
    recipe = json.loads((PACKET / "RECIPE_R1.json").read_text())
    expected_sources = recipe["source_files"]
    source_hash_mismatches = []
    for relative, expected in expected_sources.items():
        path = SOURCE / relative
        actual = sha256(path) if path.is_file() else None
        if actual != expected:
            source_hash_mismatches.append({"path": relative, "expected": expected, "actual": actual})

    pair = {
        name: json.loads((PACKET / f"{name}_COMMANDS_R1.json").read_text())
        for name in ("MOMENTS_CONTROL", "INNOVATIONS")
    }
    common_mismatches = []
    arms = {}
    for key, packet in pair.items():
        stages = packet["stages"]
        arms[key] = []
        for stage in stages:
            argv = stage["worker_argv"]
            # Restrict parsing to the explicit trainer's option vector.
            args = argv[argv.index("--name") - 0:]
            feature_mode = command_flag(args, "--research-features")
            feature_control = command_flag(args, "--research-feature-control")
            arms[key].append({
                "part": stage["part"],
                "run_name": stage["run_name"],
                "attempt_id": stage["attempt_id"],
                "feature_mode": feature_mode,
                "feature_control": feature_control,
                "seed": command_flag(args, "--seed"),
                "fold": command_flag(args, "--valid-fold"),
                "parent": command_flag(args, "--init"),
                "split": command_flag(args, "--split-manifest"),
                "cache": command_flag(args, "--distill-cache-root"),
                "scale": command_flag(args, "--distill-cache-scale"),
                "alpha": command_flag(args, "--distill-alpha"),
                "batch": command_flag(args, "--batch"),
                "chunk": command_flag(args, "--chunk"),
                "epochs": command_flag(args, "--epochs"),
                "checkpoint_epoch": command_flag(args, "--checkpoint-epoch"),
                "resume_state": command_flag(args, "--resume-state"),
                "no_eval_during_fit": "--no-eval-during-fit" in args,
                "save_training_state": "--save-training-state" in args,
                "preprocessing_inherited": "--inherit-init-preprocessing" in args,
                "profile_flags": [value for value in args if value in ("--profile-warmup", "--profile-steps")],
                "authorized": stage["launch_authorized"],
            })
        if len(stages) != 3 or any(s["launch_authorized"] for s in stages):
            common_mismatches.append(f"{key}: unexpected stage count/launch authority")
    control_stages = arms["MOMENTS_CONTROL"]
    treatment_stages = arms["INNOVATIONS"]
    for control, treatment in zip(control_stages, treatment_stages, strict=True):
        for field in ("seed", "fold", "parent", "split", "cache", "scale", "alpha", "batch", "chunk", "epochs", "checkpoint_epoch", "no_eval_during_fit", "save_training_state", "preprocessing_inherited", "part"):
            if control[field] != treatment[field]:
                common_mismatches.append(f"{control['part']}/{field}: {control[field]!r} != {treatment[field]!r}")
        if control["feature_mode"] != "causal_innovations" or treatment["feature_mode"] != "causal_innovations":
            common_mismatches.append(f"{control['part']}: mode mismatch")
        if control["feature_control"] != "newest" or treatment["feature_control"] != "none":
            common_mismatches.append(f"{control['part']}: wrong control/treatment feature selector")
        if control["attempt_id"] == treatment["attempt_id"] or control["run_name"] == treatment["run_name"]:
            common_mismatches.append(f"{control['part']}: attempt or run name collides")
    profile = {key: packet["device_profile"] for key, packet in pair.items()}
    profile_summary = {}
    for key, p in profile.items():
        args = p["worker_argv"]
        profile_summary[key] = {
            "shape": p["shape"], "memory_bound_bytes": p["memory_bound_bytes"],
            "max_wall_seconds": p["max_wall_seconds"], "profile_only": p["profile_only"],
            "training_state_saved": p["training_state_saved"],
            "warmup": command_flag(args, "--profile-warmup"),
            "measured_steps": command_flag(args, "--profile-steps"),
            "controller_vector_present": bool(p["controller_argv"]),
        }

    # Bind actual checkpoint membership to the exact frozen source-qualified
    # manifest without opening the multi-gigabyte target-index archives that
    # the full split validator reads to rederive those identities.
    sys.path.insert(0, str(SOURCE / "src"))
    from wnn.run_identity import group_identity

    split_path = MAIN / "research/campaign_2026-09-29/artifacts/splits/fold0.json"
    split_manifest_bytes = split_path.read_bytes()
    split_manifest = json.loads(split_manifest_bytes)
    split_sha = hashlib.sha256(split_manifest_bytes).hexdigest()
    if sha256(PARENT) != PARENT_SHA:
        raise RuntimeError(f"parent checkpoint SHA mismatch: {sha256(PARENT)}")
    os.environ["WNN_PROJECT_ROOT"] = str(MAIN)
    import torch

    checkpoint = torch.load(PARENT, map_location="cpu", weights_only=False)
    args = checkpoint.get("args", {})
    provenance = checkpoint.get("provenance", {})
    fit_rows = provenance.get("fit_row_groups")
    if not isinstance(fit_rows, dict) or not isinstance(fit_rows.get("train"), list) or not isinstance(fit_rows.get("valid"), list):
        raise RuntimeError(f"parent lacks usable fit_row_groups: {type(fit_rows).__name__}")
    train_groups = tuple(map(int, fit_rows["train"]))
    valid_groups = tuple(map(int, fit_rows["valid"]))
    declared_train = set(map(int, split_manifest["fit"]["train"]))
    declared_valid = set(map(int, split_manifest["fit"]["valid"]))
    declared_eval = set(map(int, split_manifest["evaluation"]["valid"]))
    component_of = {
        (source_split, int(node["row_group"])): int(node["component"])
        for source_split, nodes in split_manifest["sequences"].items()
        for node in nodes
    }
    fit_components = {component_of[("train", group)] for group in train_groups}
    fit_components.update(component_of[("valid", group)] for group in valid_groups)
    eval_components = {component_of[("valid", group)] for group in declared_eval}
    if not set(train_groups).issubset(declared_train) or not set(valid_groups).issubset(declared_valid):
        raise RuntimeError("parent fitting groups are outside the exact frozen fit manifest")
    if fit_components & eval_components:
        raise RuntimeError("parent fitting components overlap the frozen evaluation components")
    group_ids = provenance.get("fit_membership_identity", {})
    identity_matches = (
        group_ids.get("train") == group_identity(train_groups)
        and group_ids.get("valid") == group_identity(valid_groups)
    )
    split_hash_matches = (
        args.get("split_manifest_sha256") == split_sha
        and provenance.get("split_manifest_sha256") == split_sha
        and Path(args.get("split_manifest", "")).resolve() == split_path.resolve()
    )
    if not split_hash_matches:
        raise RuntimeError("parent args/provenance do not bind the currently frozen split manifest")
    parent_metadata = {
        "checkpoint_sha256": PARENT_SHA,
        "checkpoint_epoch": checkpoint.get("epoch"),
        "args_fields": sorted(args.keys()) if isinstance(args, dict) else str(type(args)),
        "args": {key: args.get(key) for key in (
            "name", "seed", "valid_fold", "train_source", "train_seqs", "valid_seqs",
            "split_manifest_sha256", "no_eval_during_fit", "checkpoint_policy", "checkpoint_epoch",
            "research_features", "research_feature_control", "init", "distill", "distill_cache_root",
            "propensity", "prop_input", "device", "arch", "hidden", "layers",
        )} if isinstance(args, dict) else None,
        "provenance_fields": sorted(provenance.keys()),
        "provenance": {
            "split_manifest_sha256": provenance.get("split_manifest_sha256"),
            "fit_membership_identity": group_ids,
            "initial_checkpoint_sha256": provenance.get("initial_checkpoint_sha256"),
            "propensity_checkpoint_sha256": provenance.get("propensity_checkpoint_sha256"),
            "trainer_source_sha256": provenance.get("trainer_source_sha256"),
            "model_source_sha256": provenance.get("model_source_sha256"),
            "distillation_cache": provenance.get("distillation_cache"),
            "causal_feature_prior": provenance.get("causal_feature_prior"),
            "fit_row_groups_fields": sorted(fit_rows.keys()),
        },
        "fit_train_count": len(train_groups),
        "fit_valid_count": len(valid_groups),
        "campaign_split_manifest_sha256": split_sha,
        "args_split_manifest_sha256": args.get("split_manifest_sha256"),
        "args_split_manifest_path": args.get("split_manifest"),
        "provenance_split_manifest_sha256": provenance.get("split_manifest_sha256"),
        "campaign_train_count": len(declared_train),
        "campaign_fit_valid_count": len(declared_valid),
        "campaign_evaluation_count": len(declared_eval),
        "parent_fit_groups_are_complete_manifest_fit_membership": (
            set(train_groups) == declared_train and set(valid_groups) == declared_valid
        ),
        "membership_identity_matches_ordered_groups": identity_matches,
        "fit_eval_components_disjoint": True,
        "parent_component_count": len(fit_components),
        "evaluation_component_count": len(eval_components),
    }
    output = {
        "status": "pass" if not source_hash_mismatches and not common_mismatches and commit == SOURCE_COMMIT and not status.strip() and parent_metadata["parent_fit_groups_are_complete_manifest_fit_membership"] and identity_matches else "fail",
        "source_commit": commit,
        "source_worktree_clean": not status.strip(),
        "source_files_expected": len(expected_sources),
        "source_hash_mismatches": source_hash_mismatches,
        "arm_vectors": arms,
        "command_mismatches": common_mismatches,
        "profile_summary": profile_summary,
        "packet_fit_authorized": {key: packet["fit_authorized"] for key, packet in pair.items()},
        "packet_exports_authorized": {key: packet["export"]["launch_authorized"] for key, packet in pair.items()},
        "packet_grades_authorized": {key: packet["grade"]["launch_authorized"] for key, packet in pair.items()},
        "parent": parent_metadata,
    }
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
