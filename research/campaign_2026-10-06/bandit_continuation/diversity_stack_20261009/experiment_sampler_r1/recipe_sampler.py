#!/usr/bin/env python3
"""Sample a frozen, source-bound recipe pool for review; never launch fits."""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

MAX_RECIPES = 24
SCHEMA = "wnn.diversity.recipe.v1"
ALIASES = {
    "optimizer": {"adamw": "adamw", "muon": "muon", "sgd": "sgd"},
    "loss": {
        "wmse": "wmse",
        "weighted_mse": "wmse",
        "weighted-mse": "wmse",
        "huber": "huber",
        "huber_d1": "huber",
        "huber-d1": "huber",
    },
    "schedule": {"one_cycle": "onecycle", "onecycle": "onecycle", "wsd": "wsd", "sgdr": "sgdr"},
    "activation": {"gelu": "gelu", "fixed-mix": "fixed_mix", "fixed_mix": "fixed_mix"},
    "architecture": {"gru": "gru", "dual": "dual", "tcn": "tcn"},
    "precision": {"fp16": "fp16", "bf16": "bf16", "fp32": "fp32"},
}
DELTA_ALIASES = {"huber_d1": 1.0, "huber-d1": 1.0}
CLI_FLAGS = {
    "optimizer": "--optim",
    "muon_scope": "--muon-scope",
    "schedule": "--sched",
    "loss": "--loss",
    "activation": "--readout-activation",
    "architecture": "--arch",
    "precision": "--precision",
    "batch_order": "--batch-order",
    "train_source": "--train-source",
    "cell": "--cell",
    "encoder": "--enc",
    "recurrent_initialization": "--rnn-init",
}


class RecipeError(ValueError):
    """A recipe pool is stale, unsupported, or internally inconsistent."""


def canonical_json(value: object) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    )


def sha256_json(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_recipe(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize semantic aliases so spelling differences cannot create duplicate recipes."""
    recipe = copy.deepcopy(dict(raw))
    for key in ("model", "optimizer", "loss", "schedule", "sampling", "data", "ancestry", "fit"):
        if not isinstance(recipe.get(key), dict):
            raise RecipeError(f"recipe.{key} must be a JSON object")
    model, optimizer, loss = recipe["model"], recipe["optimizer"], recipe["loss"]
    sampling = recipe["sampling"]
    original_loss = str(loss.get("name", "")).strip().lower()
    for obj, key, alias_group in (
        (model, "arch", "architecture"),
        (model, "activation", "activation"),
        (optimizer, "name", "optimizer"),
        (loss, "name", "loss"),
        (recipe, "precision", "precision"),
        (recipe["schedule"], "name", "schedule"),
    ):
        value = obj.get(key)
        if isinstance(value, str):
            normalized = value.strip().lower()
            obj[key] = ALIASES.get(alias_group, {}).get(normalized, normalized)
    if original_loss in DELTA_ALIASES:
        value = loss.get("huber_delta")
        if value is not None and float(value) != DELTA_ALIASES[original_loss]:
            raise RecipeError(f"{original_loss} alias conflicts with huber_delta={value}")
        loss["huber_delta"] = DELTA_ALIASES[original_loss]
    if loss.get("name") != "huber":
        loss["huber_delta"] = None
    elif loss.get("huber_delta") is not None:
        loss["huber_delta"] = float(loss["huber_delta"])
    sampling["batch_order"] = str(sampling.get("batch_order", "")).strip().lower()
    optimizer.setdefault("muon_scope", None)
    optimizer.setdefault("momentum", None)
    return recipe


def recipe_hash(raw: Mapping[str, Any]) -> str:
    return sha256_json(canonical_recipe(raw))


def load_cli_choices(paths: Sequence[Path]) -> dict[str, set[str]]:
    """Extract literal argparse choices from the real trainer without importing it."""
    flag_to_field = {flag: field for field, flag in CLI_FLAGS.items()}
    choices: dict[str, set[str]] = {}
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not node.args:
                continue
            try:
                flag = ast.literal_eval(node.args[0])
            except (ValueError, TypeError):
                continue
            if flag not in flag_to_field:
                continue
            keyword = next((arg.value for arg in node.keywords if arg.arg == "choices"), None)
            try:
                values = ast.literal_eval(keyword)
            except (ValueError, TypeError):
                continue
            if isinstance(values, (list, tuple, set)) and all(
                isinstance(value, str) for value in values
            ):
                choices.setdefault(flag_to_field[flag], set()).update(values)
    missing = set(CLI_FLAGS) - choices.keys()
    if missing:
        missing_flags = ", ".join(CLI_FLAGS[field] for field in sorted(missing))
        raise RecipeError(f"cannot discover literal trainer CLI choices for {missing_flags}")
    return choices


def validate_recipe(raw: Mapping[str, Any], choices: Mapping[str, set[str]]) -> dict[str, Any]:
    recipe = canonical_recipe(raw)
    model, optimizer, loss = recipe["model"], recipe["optimizer"], recipe["loss"]
    schedule, sampling = recipe["schedule"], recipe["sampling"]
    requested = {
        "optimizer": optimizer.get("name"),
        "muon_scope": optimizer.get("muon_scope"),
        "schedule": schedule.get("name"),
        "loss": loss.get("name"),
        "activation": model.get("activation"),
        "architecture": model.get("arch"),
        "precision": recipe.get("precision"),
        "batch_order": sampling.get("batch_order"),
        "train_source": sampling.get("train_source"),
        "cell": model.get("cell"),
        "encoder": model.get("encoder"),
        "recurrent_initialization": model.get("recurrent_initialization"),
    }
    for field, value in requested.items():
        if field == "muon_scope" and value is None and optimizer["name"] != "muon":
            continue
        if not isinstance(value, str) or value not in choices[field]:
            raise RecipeError(f"unsupported {field} value: {value!r}")
    if model.get("activation_placement") != "readout_only":
        raise RecipeError("only the trainer's readout-only activation placement is supported")
    if not isinstance(model.get("hidden"), int) or not isinstance(model.get("layers"), int):
        raise RecipeError("model hidden width and layer count must be integers")
    if model["hidden"] <= 0 or model["layers"] <= 0:
        raise RecipeError("model hidden width and layer count must be positive")
    if model["activation"] == "fixed_mix" and not (
        model["arch"] == "gru" and model["cell"] == "gru" and model["encoder"] == "linear"
    ):
        raise RecipeError("fixed_mix requires a plain GRU with a linear encoder")
    if optimizer["name"] == "muon":
        policy = optimizer.get("muon_policy")
        if model["arch"] != "gru" or optimizer["muon_scope"] != "rnn":
            raise RecipeError("this pool only supports Muon on plain GRU recurrent gates")
        if not isinstance(policy, dict) or policy != {
            "scope": "all recurrent gates",
            "orthogonalization_dtype": "fp32",
            "ns_steps": 5,
            "rms": 0.2,
            "lr_multiplier": 1.0,
        }:
            raise RecipeError("Muon policy differs from the matched FP32 recurrent-gate contract")
    elif optimizer["muon_scope"] is not None:
        raise RecipeError("muon_scope must be null for non-Muon recipes")
    momentum = optimizer.get("momentum")
    if optimizer["name"] == "sgd":
        if (
            not isinstance(momentum, (int, float))
            or not math.isfinite(momentum)
            or not 0 <= momentum < 1
        ):
            raise RecipeError("SGD momentum must be finite and in [0, 1)")
    elif momentum not in (None, 0, 0.0):
        raise RecipeError("nonzero momentum is supported only with SGD")
    for key, allow_zero in (
        ("learning_rate", False),
        ("weight_decay", True),
        ("gradient_clip_norm", False),
    ):
        value = optimizer.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise RecipeError(f"optimizer.{key} must be a finite number")
        if value < 0 or (value == 0 and not allow_zero):
            raise RecipeError(f"optimizer.{key} is outside its supported range")
    if loss["name"] == "huber":
        delta = loss.get("huber_delta")
        if not isinstance(delta, (int, float)) or not math.isfinite(delta) or delta <= 0:
            raise RecipeError("Huber requires a finite positive delta")
    parameters = schedule.get("parameters")
    if not isinstance(parameters, dict) or parameters.get("peak_lr") != optimizer["learning_rate"]:
        raise RecipeError("schedule peak LR must match optimizer learning rate")
    implementations = {
        "onecycle": "torch.optim.lr_scheduler.OneCycleLR at trainer callsite defaults",
        "wsd": "wnn.training_dynamics.build_scheduler with wsd_lambda",
        "sgdr": "wnn.training_dynamics.build_scheduler with sgdr_lambda",
    }
    if schedule.get("implementation") != implementations[schedule["name"]]:
        raise RecipeError("schedule implementation differs from the selected trainer schedule")
    if schedule["name"] == "onecycle":
        required = {
            "total_steps",
            "pct_start",
            "anneal_strategy",
            "div_factor",
            "final_div_factor",
            "three_phase",
            "cycle_momentum",
            "base_momentum",
            "max_momentum",
        }
        if not required.issubset(parameters) or parameters.get("final_lr_fraction") is not None:
            raise RecipeError("OneCycle parameters are incomplete or contain a non-OneCycle field")
    elif schedule["name"] == "wsd":
        required = {"warmup_frac", "decay_share", "final_lr_fraction"}
        if (
            not required.issubset(parameters)
            or not 0 <= parameters["warmup_frac"] < 1
            or not 0 < parameters["decay_share"] <= 1
            or not 0 < parameters["final_lr_fraction"] <= 1
        ):
            raise RecipeError("WSD schedule parameters are incomplete or out of range")
    elif schedule["name"] == "sgdr":
        required = {"warmup_frac", "restarts", "final_lr_fraction"}
        if (
            not required.issubset(parameters)
            or not 0 <= parameters["warmup_frac"] < 1
            or not isinstance(parameters["restarts"], int)
            or parameters["restarts"] < 0
            or not 0 < parameters["final_lr_fraction"] <= 1
        ):
            raise RecipeError("SGDR schedule parameters are incomplete or out of range")
    for field in ("fold", "seed"):
        if not isinstance(sampling.get(field), int) or sampling[field] < 0:
            raise RecipeError(f"sampling.{field} must be a nonnegative integer")
    for key in ("source_sha256", "data_sha256", "ancestry_sha256"):
        digest = recipe.get("identity_hashes", {}).get(key)
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise RecipeError(f"identity_hashes.{key} must be a lowercase SHA-256 digest")
    if not re.fullmatch(r"[0-9a-f]{64}", str(sampling.get("membership_sha256", ""))):
        raise RecipeError("sampling.membership_sha256 must be a lowercase SHA-256 digest")
    if recipe["data"].get("sample_membership_sha256") != sampling["membership_sha256"]:
        raise RecipeError("data and sampling membership hashes disagree")
    return recipe


def _rank(seed: int, tag: str, digest: str) -> str:
    return hashlib.sha256(f"diversity-sampler-v1\0{seed}\0{tag}\0{digest}".encode()).hexdigest()


def _matched_signature(raw: Mapping[str, Any]) -> str:
    recipe = canonical_recipe(raw)
    recipe["optimizer"]["name"] = None
    recipe["optimizer"]["muon_scope"] = None
    recipe["optimizer"]["muon_policy"] = None
    recipe["loss"]["name"] = None
    recipe["loss"]["huber_delta"] = None
    return canonical_json(recipe)


def select_candidates(
    candidates: Sequence[Mapping[str, Any]],
    count: int,
    seed: int,
    mandatory_ids: Sequence[str],
    coverage_tags: Sequence[str],
) -> list[dict[str, Any]]:
    if not 4 <= count <= MAX_RECIPES or seed < 0:
        raise RecipeError(f"count must be 4–{MAX_RECIPES} and seed must be nonnegative")
    if len(mandatory_ids) != 4 or len(set(mandatory_ids)) != 4:
        raise RecipeError("exactly four unique mandatory IDs are required")
    unique: dict[str, dict[str, Any]] = {}
    ids_to_hash: dict[str, str] = {}
    for row in candidates:
        raw = row.get("recipe")
        if not isinstance(raw, Mapping):
            raise RecipeError("each candidate must contain a recipe object")
        digest = recipe_hash(raw)
        identifier = str(row.get("candidate_id", ""))
        if not identifier or identifier in ids_to_hash:
            raise RecipeError("candidate IDs must be nonempty and unique")
        ids_to_hash[identifier] = digest
        if row.get("recipe_hash") not in (None, digest):
            raise RecipeError(f"candidate {identifier} has a stale recipe hash")
        if digest in unique:
            unique[digest]["candidate_ids"].append(identifier)
            unique[digest]["mechanism_tags"] = sorted(
                set(unique[digest]["mechanism_tags"]) | set(row.get("mechanism_tags", []))
            )
        else:
            unique[digest] = {
                "candidate_ids": [identifier],
                "recipe_hash": digest,
                "recipe": canonical_recipe(raw),
                "mechanism_tags": sorted(set(row.get("mechanism_tags", []))),
                "description": str(row.get("description", "")),
            }
    mandatory = []
    for identifier in mandatory_ids:
        found = [item for item in unique.values() if identifier in item["candidate_ids"]]
        if len(found) != 1:
            raise RecipeError(f"mandatory candidate {identifier!r} must resolve exactly once")
        mandatory.append(found[0])
    expected = {
        ("adamw", "wmse", None),
        ("adamw", "huber", 1.0),
        ("muon", "wmse", None),
        ("muon", "huber", 1.0),
    }
    observed = {
        (
            item["recipe"]["optimizer"]["name"],
            item["recipe"]["loss"]["name"],
            item["recipe"]["loss"]["huber_delta"],
        )
        for item in mandatory
    }
    if observed != expected or len({_matched_signature(item["recipe"]) for item in mandatory}) != 1:
        raise RecipeError("mandatory quartet must vary only AdamW/Muon and WMSE/Huber(delta=1)")
    selected = list(dict.fromkeys(item["recipe_hash"] for item in mandatory))
    result = list(mandatory)
    remaining = {digest: item for digest, item in unique.items() if digest not in selected}
    covered = {tag for item in result for tag in item["mechanism_tags"]}
    for tag in coverage_tags:
        if tag in covered or len(result) >= count:
            continue
        eligible = [item for item in remaining.values() if tag in item["mechanism_tags"]]
        if eligible:
            chosen = min(eligible, key=lambda item: _rank(seed, tag, item["recipe_hash"]))
            result.append(chosen)
            remaining.pop(chosen["recipe_hash"])
            covered.update(chosen["mechanism_tags"])
    rest = sorted(
        remaining.values(),
        key=lambda item: (_rank(seed, "fill", item["recipe_hash"]), item["recipe_hash"]),
    )
    result.extend(rest[: count - len(result)])
    if len(result) != count:
        raise RecipeError(f"only {len(result)} unique candidates are available for count={count}")
    return result


def _experiment_name(recipe: Mapping[str, Any], digest: str) -> str:
    model, loss, sampling = recipe["model"], recipe["loss"], recipe["sampling"]
    loss_name = loss["name"] + (
        f"d{loss['huber_delta']:g}" if loss["huber_delta"] is not None else ""
    )
    return (
        f"ds1_{recipe['optimizer']['name']}-{loss_name}_{model['arch']}{model['hidden']}-"
        f"{model['activation']}_{recipe['schedule']['name']}_m{sampling['membership_sha256'][:8]}_"
        f"f{sampling['fold']}_s{sampling['seed']}_{recipe['precision']}_{digest[:10]}"
    )


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> bytes:
    data = ("".join(canonical_json(row) + "\n" for row in rows)).encode("utf-8")
    path.write_bytes(data)
    return data


def _markdown(rows: Sequence[Mapping[str, Any]]) -> bytes:
    headings = [
        "Name",
        "Mechanisms",
        "Optimizer",
        "Loss",
        "Architecture",
        "Activation",
        "Schedule",
        "Membership",
        "Fold",
        "Seed",
        "Precision",
        "Status",
    ]
    lines = ["| " + " | ".join(headings) + " |", "|" + "|".join("---" for _ in headings) + "|"]
    for row in rows:
        recipe = row["recipe"]
        model, loss, sampling = recipe["model"], recipe["loss"], recipe["sampling"]
        loss_label = loss["name"] + (
            f" δ={loss['huber_delta']:g}" if loss["huber_delta"] is not None else ""
        )
        cells = [
            row["experiment_name"],
            ", ".join(row["mechanism_tags"]),
            recipe["optimizer"]["name"],
            loss_label,
            f"{model['arch']} {model['hidden']}×{model['layers']}",
            model["activation"],
            recipe["schedule"]["name"],
            sampling["membership_sha256"][:8],
            str(sampling["fold"]),
            str(sampling["seed"]),
            recipe["precision"],
            "pending profile / not admitted",
        ]
        lines.append("| " + " | ".join(cell.replace("|", "\\|") for cell in cells) + " |")
    return ("\n".join(lines) + "\n").encode("utf-8")


def _verify_bindings(
    root: Path, spec: Mapping[str, Any], pool: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    current_sources: dict[str, str] = {}
    for relative in spec["source_code_paths"]:
        path = (root / relative).resolve()
        if root not in path.parents or not path.is_file():
            raise RecipeError(f"source is missing or escapes repository: {relative}")
        current_sources[relative] = file_sha256(path)
    cli_files = [(root / path).resolve() for path in spec["trainer_cli_choices"]]
    choices = load_cli_choices(cli_files)
    choice_snapshot = {key: sorted(values) for key, values in sorted(choices.items())}
    binding_hash = sha256_json(current_sources)
    checked_data: set[tuple[str, str]] = set()
    for candidate in pool:
        recipe = validate_recipe(candidate["recipe"], choices)
        source = recipe["source"]
        if (
            source.get("sha256_by_path") != current_sources
            or source.get("binding_hash") != binding_hash
        ):
            raise RecipeError(
                "candidate source hashes are stale; freeze a new candidate pool after source review"
            )
        if (
            source.get("files") != spec["source_code_paths"]
            or source.get("trainer_cli_choices") != spec["trainer_cli_choices"]
        ):
            raise RecipeError("candidate source paths disagree with the sampler spec")
        if source.get("trainer_cli_choice_snapshot") != choice_snapshot:
            raise RecipeError("trainer CLI choices changed since this recipe pool was frozen")
        expected_identities = {
            "source_sha256": sha256_json(source),
            "data_sha256": sha256_json({"data": recipe["data"], "sampling": recipe["sampling"]}),
            "ancestry_sha256": sha256_json(recipe["ancestry"]),
        }
        if recipe["identity_hashes"] != expected_identities:
            raise RecipeError("candidate source/data/ancestry identity hashes are inconsistent")
        for relative, digest in (
            (recipe["sampling"]["membership_path"], recipe["sampling"]["membership_file_sha256"]),
            (recipe["data"]["split_manifest_path"], recipe["data"]["split_manifest_sha256"]),
        ):
            path = (root / relative).resolve()
            if root not in path.parents or not path.is_file() or file_sha256(path) != digest:
                raise RecipeError(f"data membership source is missing or stale: {relative}")
        for source_row in recipe["data"].get("dataset_sources", []):
            if not re.fullmatch(r"[0-9a-f]{64}", str(source_row.get("sha256", ""))):
                raise RecipeError("dataset source is missing its declared SHA-256")
            checked_data.add((source_row["path"], source_row["sha256"]))
    return {
        "source_files": current_sources,
        "source_binding_sha256": binding_hash,
        "trainer_cli_choices": choice_snapshot,
        "dataset_sources": [
            dict(path=path, sha256=digest) for path, digest in sorted(checked_data)
        ],
    }


def generate(
    root: Path, spec_path: Path, output_dir: Path, count: int, seed: int
) -> dict[str, Any]:
    root, spec_path, output_dir = root.resolve(), spec_path.resolve(), output_dir.resolve()
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    if (
        spec.get("schema") != "wnn.diversity.recipe-sampler-spec.v1"
        or spec.get("max_recipes") != MAX_RECIPES
    ):
        raise RecipeError("unsupported sampler spec or recipe cap")
    pool_path = (spec_path.parent / spec["candidate_pool_path"]).resolve()
    pool = [
        json.loads(line)
        for line in pool_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    bindings = _verify_bindings(root, spec, pool)
    selected = select_candidates(
        pool, count, seed, spec["mandatory_candidate_ids"], spec["coverage_reserve_tags"]
    )
    rows = []
    for candidate in selected:
        recipe = candidate["recipe"]
        rows.append(
            {
                "schema": SCHEMA,
                "candidate_id": candidate["candidate_ids"][0],
                "candidate_aliases": candidate["candidate_ids"],
                "experiment_name": _experiment_name(recipe, candidate["recipe_hash"]),
                "recipe_hash": candidate["recipe_hash"],
                "mechanism_tags": candidate["mechanism_tags"],
                "status": {
                    "device_profile": "pending",
                    "admission": "not_admitted",
                    "queue_activation": "disabled",
                },
                "recipe": recipe,
            }
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    sampled_bytes = _write_jsonl(output_dir / "sampled_recipes_r1.jsonl", rows)
    pool_bytes = pool_path.read_bytes()
    (output_dir / "raw_candidate_pool_r1.jsonl").write_bytes(pool_bytes)
    table_bytes = _markdown(rows)
    (output_dir / "sampled_recipes_r1.md").write_bytes(table_bytes)
    receipt = {
        "schema": "wnn.diversity.recipe-sampler-receipt.v1",
        "sampler_version": 1,
        "sampler_source_sha256": file_sha256(Path(__file__).resolve()),
        "spec_sha256": file_sha256(spec_path),
        "candidate_pool_sha256": hashlib.sha256(pool_bytes).hexdigest(),
        "seed": seed,
        "selection_algorithm": "sha256-rank-v1",
        "maximum_recipes": MAX_RECIPES,
        "selected_count": len(rows),
        "selected_candidate_ids": [row["candidate_id"] for row in rows],
        "mandatory_candidate_ids": spec["mandatory_candidate_ids"],
        "coverage_reserve_tags": spec["coverage_reserve_tags"],
        "bindings": bindings,
        "sampled_recipes_sha256": hashlib.sha256(sampled_bytes).hexdigest(),
        "readable_table_sha256": hashlib.sha256(table_bytes).hexdigest(),
        "fit_allowed": False,
        "provider_mutations_allowed": False,
        "queue_activation_allowed": False,
    }
    (output_dir / "sampling_receipt_r1.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, required=True, help="read-only repository root for source/data hashes"
    )
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261009)
    args = parser.parse_args(argv)
    try:
        receipt = generate(args.root, args.spec, args.output_dir, args.count, args.seed)
    except (OSError, KeyError, TypeError, json.JSONDecodeError, RecipeError) as error:
        parser.error(str(error))
    print(
        json.dumps(
            {
                key: receipt[key]
                for key in (
                    "selected_count",
                    "selected_candidate_ids",
                    "sampled_recipes_sha256",
                    "fit_allowed",
                    "queue_activation_allowed",
                )
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
