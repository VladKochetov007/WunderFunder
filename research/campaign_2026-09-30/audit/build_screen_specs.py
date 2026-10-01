"""Freeze screen designs before fitting; bind immutable completed lineage later.

This builder never opens target arrays or computes a prediction metric. Root
supplies the exact job/source closure and independently recorded fit diagnostics.
The frozen generic campaign_score.py remains unchanged.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path

from build_reference_spec import CAMPAIGN, checkpoint_binding, identity, publish, reference_design
from campaign_score import ROOT, artifact, file_sha256, validate_member
from wnn.splits import load_campaign_split

AUDIT = Path(__file__).resolve().parent
FAMILIES = {
    "spread": ["c20260930_ft_spread_current_f0", "c20260930_ft_spread_proxy_f0"],
    "structured": ["c20260930_ssm_multi_f0", "c20260930_ssm_single_f0"],
    "event": ["c20260930_mem_content_f0", "c20260930_mem_position_f0", "c20260930_mem_raw_f0"],
}


def read(path: Path) -> dict:
    """Read a small JSON commitment, never a target or prediction array."""
    return json.loads(path.read_text())


def require_args(args: dict, expected: dict, name: str) -> None:
    """Reject any changed frozen training argument before binding artifacts."""
    for key, value in expected.items():
        if args.get(key) != value:
            raise ValueError(f"{name}: changed argument {key}: {args.get(key)!r} != {value!r}")


def preset(name: str, family: str) -> dict:
    """Use the clean recipe metadata plus only explicitly named interventions."""
    raw = copy.deepcopy(read(ROOT / "artifacts/runs/c20260929_since_s3_f0/provenance.json")["args"])
    for key in ["split_manifest", "split_manifest_sha256"]:
        raw.pop(key, None)
    raw["name"] = name
    raw["spread_aux_mode"] = "original"
    raw["spread_cache_root"] = ""
    if family == "structured":
        raw.update(
            arch="structured",
            hidden=96,
            layers=2,
            decay_init="multi" if "multi" in name else "single",
        )
    elif family == "event":
        raw.update(
            arch="event-content",
            hidden=96,
            layers=1,
            memory_arm={
                "content": "endpoint-content",
                "position": "endpoint-position",
                "raw": "raw-code-content",
            }[name.split("_")[-2]],
        )
    elif family == "continuation":
        raw.update(
            epochs=1,
            checkpoint_epoch=0,
            init="c20260929_since_s3_f0@ep3",
            inherit_init_preprocessing=True,
            spread_aux_mode="current_known" if "current" in name else "proxy_known",
            spread_cache_root=str(ROOT / "research/campaign_2026-09-30/spread_aux"),
        )
    elif family in ["head_ft", "full_ft"]:
        stage = "c20260929_ft_since_s3_f0h" if family == "head_ft" else "c20260929_ft_since_s3_f0"
        raw = copy.deepcopy(read(ROOT / "artifacts/runs" / stage / "provenance.json")["args"])
        for key in ["split_manifest", "split_manifest_sha256"]:
            raw.pop(key, None)
        final = name[:-1] if family == "head_ft" else name
        continuation = final.replace("c20260930_ft_spread_", "c20260930_spread_")
        raw.update(
            name=name,
            init=f"{continuation}@ep0" if family == "head_ft" else f"{final}h@ep0",
            spread_aux_mode="original",
            spread_cache_root="",
        )
    else:
        raise ValueError(f"unknown recipe {family}")
    return raw


def stage_names(family: str) -> dict[str, list[str]]:
    """Declare all ancestors covered by each new durable final-run job."""
    return {
        final: [final.replace("c20260930_ft_spread_", "c20260930_spread_"), final + "h", final]
        if family == "spread"
        else [final]
        for final in FAMILIES[family]
    }


def pipeline_arm(command: list[str], recipe_path: Path, recipe_hash: str) -> str:
    """Bind exact pipeline argv to the same immutable stage recipe."""
    if "scripts/campaign_screen_pipeline.py" not in command:
        raise ValueError("new screens must use durable serial pipeline")
    for option in ["--recipe", "--recipe-sha256", "--arm"]:
        if command.count(option) != 1 or command.index(option) + 1 >= len(command):
            raise ValueError(f"pipeline option missing/duplicated: {option}")
    supplied = Path(command[command.index("--recipe") + 1])
    supplied = supplied if supplied.is_absolute() else ROOT / supplied
    if (
        supplied.resolve() != recipe_path.resolve()
        or command[command.index("--recipe-sha256") + 1] != recipe_hash
    ):
        raise ValueError("pipeline command recipe differs from source closure")
    return command[command.index("--arm") + 1]


def freeze(family: str, closure_path: Path, plan_path: Path) -> dict:
    """Build one immutable, target-free preregistration before any new fit."""
    closure = read(closure_path)
    stages = stage_names(family)
    recipe_path = (
        artifact(closure["pipeline_recipe"]) if "pipeline_recipe" in closure else closure_path
    )
    recipe = read(recipe_path)
    if recipe.get("status") != "frozen-before-fit":
        raise ValueError("pipeline recipe not frozen before fit")
    source = closure["source_sha256"]
    for relative, digest in source.items():
        artifact({"path": relative, "sha256": digest})
    required = {
        "scripts/train_rnn.py",
        "scripts/predict_run.py",
        "src/wnn/nn.py",
        "src/wnn/features.py",
        "src/wnn/outcomes.py",
        "src/wnn/run_identity.py",
    }
    required.add(
        {
            "spread": "src/wnn/spread_aux.py",
            "structured": "src/wnn/structured_state.py",
            "event": "src/wnn/event_content.py",
        }[family]
    )
    required.add("scripts/campaign_screen_pipeline.py")
    if family == "spread":
        required.add("scripts/ft_gclp.sh")
    if not required.issubset(source):
        raise ValueError("source closure lacks screen implementation/recipe dependencies")
    expected = {}
    commands = {}
    pipeline_arms = {}
    for final, names in stages.items():
        supplied = closure.get("job_commands", {}).get(final)
        if supplied is None:
            matches = [
                key
                for key, arm in recipe["arms"].items()
                if any(final in stage["argv"] for stage in arm["stages"])
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"{final}: pipeline arm cannot be uniquely associated with final run"
                )
            supplied = [
                ".venv/bin/python",
                "scripts/campaign_screen_pipeline.py",
                "--recipe",
                str(recipe_path.relative_to(ROOT)),
                "--recipe-sha256",
                file_sha256(recipe_path),
                "--arm",
                matches[0],
            ]
        if not isinstance(supplied, list) or not supplied:
            raise ValueError("empty job command")
        commands[final] = supplied
        key = pipeline_arm(supplied, recipe_path, file_sha256(recipe_path))
        chosen = recipe["arms"][key]
        if not chosen["stages"] or len({stage["stage"] for stage in chosen["stages"]}) != len(
            chosen["stages"]
        ):
            raise ValueError("empty/duplicated pipeline stages")
        pipeline_arms[final] = {
            "arm": key,
            "log_directory": chosen["log_directory"],
            "stages": chosen["stages"],
        }
        if any(
            recipe["source_sha256"].get(relative) != digest for relative, digest in source.items()
        ):
            raise ValueError("pipeline and job closures disagree")
        if any(
            "scripts/train_rnn.py" not in stage["argv"]
            and "scripts/predict_run.py" not in stage["argv"]
            and "scripts/ft_gclp.sh" not in stage["argv"]
            for stage in chosen["stages"]
        ):
            raise ValueError("unrecognized pipeline stage executable")
        for i, name in enumerate(names):
            stage_recipe = ["continuation", "head_ft", "full_ft"][i] if family == "spread" else family
            declared = copy.deepcopy(closure["stage_expected_args"][name])
            require_args(declared, preset(name, stage_recipe), name)
            expected[name] = declared
        if any((ROOT / "artifacts/runs" / name / "provenance.json").exists() for name in names) or (
            CAMPAIGN / "jobs" / f"{final}.json"
        ).exists() or (ROOT / chosen["log_directory"]).exists():
            raise ValueError(f"{final}: screen started before design freeze")
    reference = reference_design()["folds"][0]
    roles = {"reference": reference["roles"]["reference"]}
    names = FAMILIES[family]
    epoch = 0 if family == "spread" else 3
    role_names = (
        ["candidate", "matched_control"]
        if family != "event"
        else ["candidate", "position_control", "raw_control"]
    )
    for role, name in zip(role_names, names):
        new = {
            "run": name,
            "epoch": epoch,
            "weight": 0.5 if family == "spread" else 1.0,
            "fit_source": "valid" if family == "spread" else "train+valid",
            "checkpoint": f"artifacts/runs/{name}/model_ep{epoch}.pt",
            "expected_args": expected[name],
            "source_sha256": source,
        }
        roles[role] = [copy.deepcopy(roles["reference"][1]), new] if family == "spread" else [new]
    derived = (
        {}
        if family == "spread"
        else {f"{role}_blend": {"reference": 0.8, role: 0.2} for role in role_names}
    )
    primary = "candidate" if family == "spread" else "candidate_blend"
    comparators = ["reference"] + [
        (role if family == "spread" else role + "_blend") for role in role_names[1:]
    ]
    design = {
        "metric_id": "global_wp_clip2_v1",
        "n_boot": 2000,
        "bootstrap_seed": 271828,
        "comparisons": [[primary, other] for other in comparators],
        "derived_predictions": derived,
        "folds": [{"fold": 0, "manifest": reference["manifest"], "roles": roles}],
    }
    manifest = read(ROOT / reference["manifest"]["path"])
    row_groups = sorted(row["row_group"] for row in manifest["sequences"]["valid"])
    if row_groups != list(range(1873)):
        raise ValueError("screen requires all1873 orderedVALID predictions")
    spread_cache = None
    if family == "spread":
        cache = ROOT / "research/campaign_2026-09-30/spread_aux"
        metadata = read(cache / "metadata.json")
        if metadata["status"] != "complete" or metadata["supervision_only"] is not True:
            raise ValueError("spread cache incomplete")
        spread_cache = {
            "metadata": identity(cache / "metadata.json"),
            "selection": identity(cache / "selection.json"),
            "map": identity(cache / "interior_code_table.json"),
            "arrays": {
                split: identity(cache / entry["file"]) for split, entry in metadata["files"].items()
            },
            "source_sha256": source["src/wnn/spread_aux.py"],
        }
    return {
        "purpose": f"independent fold0 {family} screen; inspected clean panel",
        "design": design,
        "root_screen_plan": identity(plan_path),
        "job_source_closure": identity(closure_path),
        "pipeline_recipe": identity(recipe_path),
        "job_commands": commands,
        "pipeline_arms": pipeline_arms,
        "stage_expected_args": expected,
        "stages_by_job": stages,
        "prediction_row_groups": row_groups,
        "spread_cache": spread_cache,
        "builder_source_sha256": file_sha256(Path(__file__)),
        "limits": "CPU proof/local screen only; no hidden untouched-panel/public claim",
    }


def check_stage(name: str, epoch: int, split: object, manifest: dict, expected: dict) -> dict:
    """Bind fixed checkpoint, exact clean fit rows and all frozen stage arguments."""
    stage = checkpoint_binding(name, epoch, split.sha256)
    require_args(stage["args"], expected, name)
    fit = stage["fit_row_groups"]
    split.assert_fit_groups(fit.get("train", []), fit.get("valid", []))
    train = manifest["fit"]["train"] if expected["train_source"] == "train+valid" else []
    if set(fit.get("train", [])) != set(train) or set(fit.get("valid", [])) != set(
        manifest["fit"]["valid"]
    ):
        raise ValueError(f"{name}: changed fitting membership")
    return stage


def check_parity(prior: dict, lineage: dict) -> dict:
    """Recompute parity from source-bound training instrumentation, no self-report."""
    stages = prior["stages_by_job"]
    history = []
    bindings = {}
    for job, names in stages.items():
        entries = []
        for name in names:
            stage = lineage[job]["stages"][name]
            provenance = read(artifact(stage["provenance"]))
            args = prior["stage_expected_args"][name]
            records = provenance["update_diagnostics"]
            if len(records) != args["epochs"] or [d["epoch_index"] for d in records] != list(
                range(args["epochs"])
            ):
                raise ValueError(f"{name}: missing prescribed epoch diagnostics")
            membership = provenance["fit_membership_identity"]
            normalization = provenance["normalization_identity"]
            expected_length = (
                len(provenance["fit_row_groups"]["train"])
                + len(provenance["fit_row_groups"]["valid"]) * args["valid_repeat"]
            )
            if (
                membership["dataset_length"] != expected_length
                or membership["valid_repeat"] != args["valid_repeat"]
            ):
                raise ValueError("instrumented dataset length differs")
            for source in ["train", "valid"]:
                groups = provenance["fit_row_groups"][source]
                # Explicit signed little-endian int64 grammar without NumPy arrays.
                encoded = b"".join(
                    int(group).to_bytes(8, "little", signed=True) for group in groups
                )
                if membership[source] != {
                    "count": len(groups),
                    "ordered_int64_le_sha256": hashlib.sha256(encoded).hexdigest(),
                }:
                    raise ValueError("instrumented ordered membership hash differs")
            batch_count = math.ceil(expected_length / args["batch"])
            attempt_count = batch_count * math.ceil(20000 / args["chunk"])
            cumulative = {}
            epoch_history = []
            for d in records:
                counts = d["epoch"]
                keys = [
                    "optimizer_attempts",
                    "amp_skipped_updates",
                    "optimizer_completed_updates",
                    "ema_updates",
                    "completed_batches",
                ]
                if set(counts) != set(keys) or any(
                    type(counts[k]) is not int or counts[k] < 0 for k in keys
                ):
                    raise ValueError("invalid optimizer counts")
                if (
                    counts["completed_batches"] != batch_count
                    or counts["optimizer_attempts"] != attempt_count
                    or counts["optimizer_completed_updates"] + counts["amp_skipped_updates"]
                    != attempt_count
                    or counts["optimizer_completed_updates"] <= 0
                ):
                    raise ValueError("incomplete/inconsistent successful optimizer updates")
                expected_ema = attempt_count if args["ema_teacher"] > 0 else 0
                if (
                    counts["ema_updates"] != expected_ema
                    or d["normalization_identity"] != normalization
                ):
                    raise ValueError("EMA calls/normalization history differs")
                for key in keys:
                    cumulative[key] = cumulative.get(key, 0) + counts[key]
                if d["cumulative"] != cumulative:
                    raise ValueError("cumulative diagnostic history differs")
                epoch_history.append(
                    {"counts": counts, "sample_order_identity": d["sample_order_identity"]}
                )
            if provenance["cumulative_updates"] != cumulative:
                raise ValueError("final cumulative updates differ")
            binding = {"checkpoint": stage["checkpoint"], "provenance": stage["provenance"]}
            bindings[name] = binding
            entries.append(
                {
                    "epochs": epoch_history,
                    "normalization": normalization,
                    "membership": membership,
                    "counter_implementation": provenance["update_counter_implementation"],
                    "counter_source_sha256": provenance["run_identity_source_sha256"],
                }
            )
        history.append(entries)
    if any(h != history[0] for h in history[1:]):
        raise ValueError("arms differ in successful updates/EMA/normalization history")
    return {
        "status": "pass",
        "source_bound_provenance": bindings,
        "stage_histories": history[0],
        "limits": "Per-epoch actual counts plus pre-iteration RNG/membership identities; no per-step skipped-index trace or actual sample-order materialization.",
    }


def bind(design_path: Path) -> dict:
    """Validate completed lineage and prediction headers without scoring targets."""
    prior = read(design_path)
    if prior["builder_source_sha256"] != file_sha256(Path(__file__)):
        raise ValueError("screen binder source changed after design freeze")
    design = prior["design"]
    closure = read(artifact(prior["job_source_closure"]))
    artifact(prior["pipeline_recipe"])
    artifact(prior["root_screen_plan"])
    manifest_path = artifact(design["folds"][0]["manifest"])
    split = load_campaign_split(manifest_path, ROOT)
    manifest = read(manifest_path)
    reference = reference_design()["folds"][0]["roles"]["reference"]
    artifacts = {}
    lineage = {}
    # Existing reference controls retain their own historical source identities.
    for member in reference:
        name = member["run"]
        pred = ROOT / "cache" / f"pred_{name}_valid.npy"
        bundle = {
            "prediction": identity(pred),
            "sidecar": identity(pred.with_suffix(".json")),
            "checkpoint": identity(ROOT / member["checkpoint"]),
            "provenance": identity(ROOT / "artifacts/runs" / name / "provenance.json"),
            "job": identity(CAMPAIGN / "jobs" / f"{name}.json"),
            "initial_checkpoint": identity(ROOT / "artifacts/runs" / f"{name}h" / "model_ep0.pt"),
        }
        validate_member(member, bundle, split, manifest, prior["prediction_row_groups"])
        middle = checkpoint_binding(name + "h", 0, split.sha256)
        parent_name = name.replace("c20260929_ft_since_", "c20260929_since_")
        parent = checkpoint_binding(parent_name, 3, split.sha256)
        propensity = checkpoint_binding("c20260929_prop_f0", 3, split.sha256)
        job_record = read(artifact(bundle["job"]))
        command = job_record["command"]
        location = command.index("scripts/ft_gclp.sh")
        if (
            middle["seed"] != 3
            or middle["args"]["init"] != parent_name + "@ep3"
            or middle["initial_checkpoint_sha256"] != parent["checkpoint"]["sha256"]
            or parent["propensity_checkpoint_sha256"] != file_sha256(ROOT / "artifacts/runs/c20260929_prop_f0/model.pt")
            or command[location + 1:location + 5] != [parent_name + "@ep3", name, "0", "3"]
        ):
            raise ValueError("reference FT parent differs")
        for stage in [middle, parent, propensity]:
            fit = stage["fit_row_groups"]
            split.assert_fit_groups(fit.get("train", []), fit.get("valid", []))
        artifacts[name] = bundle
        lineage[name] = {
            "stage1": middle, "parent": parent, "propensity": propensity,
            "parent_job": identity(CAMPAIGN / "jobs" / f"{parent_name}.json"),
            "propensity_job": identity(CAMPAIGN / "jobs/c20260929_prop_f0.json"),
            "both_ft_stages_owned_by_one_durable_job": bundle["job"],
        }
    for job, names in prior["stages_by_job"].items():
        job_path = CAMPAIGN / "jobs" / f"{job}.json"
        record = read(job_path)
        if (
            record.get("status") != "finished"
            or record.get("exit_code") != 0
            or record.get("kind") != "gpu"
            or record["command"] != prior["job_commands"][job]
        ):
            raise ValueError(f"{job}: unsuccessful/changed exact durable command")
        for relative, digest in closure["source_sha256"].items():
            if record["source_sha256"].get(relative) != digest:
                raise ValueError(f"{job}: source closure differs: {relative}")
        pipeline = prior["pipeline_arms"][job]
        log_root = ROOT / pipeline["log_directory"]
        reservation = read(log_root / "recipe_identity.json")
        if (
            reservation["sha256"] != prior["pipeline_recipe"]["sha256"]
            or reservation["arm"] != pipeline["arm"]
        ):
            raise ValueError("pipeline execution reservation differs")
        execution = []
        for index, stage in enumerate(pipeline["stages"]):
            path = log_root / f"{index:02d}_{stage['stage']}.json"
            executed = read(path)
            if (
                executed["stage"] != stage["stage"]
                or executed["command"] != stage["argv"]
                or executed["exit_code"] != 0
                or executed["duration_seconds"] < 0
            ):
                raise ValueError("pipeline stage failed/argv changed")
            execution.append(
                {
                    "record": identity(path),
                    "log": identity(log_root / f"{index:02d}_{stage['stage']}.log"),
                }
            )
        lineage[job] = {
            "job": identity(job_path),
            "stages": {},
            "pipeline_reservation": identity(log_root / "recipe_identity.json"),
            "pipeline_stage_execution": execution,
        }
        last = None
        for name in names:
            epoch = prior["stage_expected_args"][name]["checkpoint_epoch"]
            stage = check_stage(name, epoch, split, manifest, prior["stage_expected_args"][name])
            stage_provenance = read(artifact(stage["provenance"]))
            if (
                stage_provenance.get("run_identity_source_sha256")
                != closure["source_sha256"]["src/wnn/run_identity.py"]
            ):
                raise ValueError("update instrumentation source differs")
            if (
                last is not None
                and stage["initial_checkpoint_sha256"] != last["checkpoint"]["sha256"]
            ):
                raise ValueError(f"{name}: backward checkpoint binding differs")
            lineage[job]["stages"][name] = stage
            last = stage
        if prior["spread_cache"] is not None:
            parent = checkpoint_binding("c20260929_since_s3_f0", 3, split.sha256)
            continuation = lineage[job]["stages"][names[0]]
            if continuation["initial_checkpoint_sha256"] != parent["checkpoint"]["sha256"]:
                raise ValueError("spread initializer is not identical clean seed3parent ep3")
            cache = prior["spread_cache"]
            provenance = read(ROOT / "artifacts/runs" / names[0] / "provenance.json")
            aux = provenance["spread_aux"]
            for key in ["metadata", "selection", "map"]:
                artifact(cache[key])
            for entry in cache["arrays"].values():
                artifact(entry)
            if (
                aux["metadata_sha256"] != cache["metadata"]["sha256"]
                or aux["selection_sha256"] != cache["selection"]["sha256"]
                or aux["map_sha256"] != cache["map"]["sha256"]
                or aux["source_sha256"] != cache["source_sha256"]
                or aux["mode"] != prior["stage_expected_args"][names[0]]["spread_aux_mode"]
                or aux["array_sha256"] != {s: v["sha256"] for s, v in cache["arrays"].items()}
                or aux["supervision_only"] is not True
                or aux["unknown"] != 255
            ):
                raise ValueError("spread supervision/cache provenance differs")
            for name in names[1:]:
                prov = read(ROOT / "artifacts/runs" / name / "provenance.json")
                if (
                    prov.get("spread_aux") is not None
                    or prov["args"]["spread_aux_mode"] != "original"
                    or prov["args"]["spread_cache_root"]
                ):
                    raise ValueError("FT inherited spread labels/cache intervention")
            lineage[job]["clean_seed3_parent"] = parent
        else:
            field = "src/wnn/structured_state.py" if "ssm_" in job else "src/wnn/event_content.py"
            provenance = read(ROOT / "artifacts/runs" / job / "provenance.json")
            if provenance.get("model_source_sha256") != closure["source_sha256"][field]:
                raise ValueError("architecture source provenance differs")
        first = lineage[job]["stages"][names[0]]
        if prior["stage_expected_args"][names[0]]["propensity"] != "c20260929_prop_f0":
            raise ValueError("fold0propensity intervention changed")
        propensity = checkpoint_binding("c20260929_prop_f0", 3, split.sha256)
        prop_hash = file_sha256(ROOT / "artifacts/runs/c20260929_prop_f0/model.pt")
        if first["propensity_checkpoint_sha256"] != prop_hash:
            raise ValueError("propensity checkpoint binding differs")
        lineage[job]["propensity"] = propensity
        member = next(
            m for role in design["folds"][0]["roles"].values() for m in role if m["run"] == job
        )
        pred = ROOT / "cache" / f"pred_{job}_valid.npy"
        bundle = {
            "prediction": identity(pred),
            "sidecar": identity(pred.with_suffix(".json")),
            "checkpoint": identity(ROOT / member["checkpoint"]),
            "provenance": identity(ROOT / "artifacts/runs" / job / "provenance.json"),
            "job": identity(job_path),
        }
        init = member["expected_args"].get("init", "")
        if init:
            name, epoch = init.split("@ep")
            bundle["initial_checkpoint"] = identity(
                ROOT / "artifacts/runs" / name / f"model_ep{epoch}.pt"
            )
        validate_member(member, bundle, split, manifest, prior["prediction_row_groups"])
        artifacts[job] = bundle
    parity = check_parity(prior, lineage)
    return {
        "scorer_freeze": identity(AUDIT / "campaign_score_freeze.json"),
        "preregistration": identity(design_path),
        "targets": identity(ROOT / "cache/valid_targets.npz"),
        "design": design,
        "artifacts": {"0": artifacts},
        "backward_lineage": lineage,
        "fit_parity": parity,
        "builder_source_sha256": file_sha256(Path(__file__)),
        "no_target_array_access": True,
    }


def main() -> None:
    """Root freezes before fitting and binds after every required artifact exists."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["freeze", "bind"])
    parser.add_argument("--family", choices=FAMILIES)
    parser.add_argument("--closure", type=Path)
    parser.add_argument("--plan", type=Path, default=AUDIT.parent / "screen_plan.json")
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--spec", type=Path)
    args = parser.parse_args()
    if args.phase == "freeze":
        if not args.family or not args.closure:
            parser.error("freeze requires --family and --closure")
        publish(args.design, freeze(args.family, args.closure.resolve(), args.plan.resolve()))
    else:
        if not args.spec:
            parser.error("bind requires --spec")
        publish(args.spec, bind(args.design.resolve()))
    print(
        json.dumps(
            {
                "phase": args.phase,
                "design": str(args.design),
                "spec": str(args.spec) if args.spec else None,
            }
        )
    )


if __name__ == "__main__":
    main()
