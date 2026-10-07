"""Grade the preregistered equal-seed C1v2 pair against the fixed C0 pair.

This is a fixed-model fold-0 development comparison. It validates the sealed
commands and artifacts, then reuses the campaign evaluator's official metric,
mask, lineage audit, and paired-component bootstrap implementation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import evaluate_bandit_continuation as campaign_eval  # noqa: E402

from wnn.evaluation import paired_component_bootstrap, paired_component_moments  # noqa: E402
from wnn.gmetric import METRIC_ID  # noqa: E402
from wnn.splits import file_sha256, load_campaign_split  # noqa: E402

SPEC_PATH = Path(
    "research/campaign_2026-10-06/bandit_continuation/local/C1V2_C0_PAIRED_GRADER_SPEC.json"
)
DEFAULT_ARTIFACT_ROOT = ROOT
SEQUENCES_PER_BATCH = 4
EXCLUDED_CONTRACT_OPTIONS = {
    "name",
    "resume_state",
    "save_training_state",
    "stop_after_epoch",
    "split_manifest",
    "fit_groups_manifest",
}


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _assert_file_hash(path: Path, expected: str, label: str) -> None:
    actual = file_sha256(path)
    if actual != expected:
        raise ValueError(f"{label} SHA-256 changed: expected {expected}, observed {actual}")


def _hash_command(command: list[str], encoding: str) -> str:
    if encoding == "compact-json":
        payload = json.dumps(command, separators=(",", ":"))
    elif encoding == "json-default":
        payload = json.dumps(command)
    else:
        raise ValueError(f"unsupported frozen command encoding: {encoding}")
    return hashlib.sha256(payload.encode()).hexdigest()


def _commands_by_seed(commands: object, prefix: str, seeds: list[int]) -> dict[int, list[str]]:
    if not isinstance(commands, list):
        raise ValueError("frozen recipe lacks command arrays")
    result: dict[int, list[str]] = {}
    for command_value in commands:
        if not isinstance(command_value, list):
            raise ValueError("frozen command is malformed")
        command = list(map(str, command_value))
        try:
            run = command[command.index("--name") + 1]
            seed = int(command[command.index("--seed") + 1])
        except (ValueError, IndexError) as error:
            raise ValueError("frozen command lacks a run name or seed") from error
        if run != f"{prefix}_s{seed}_f0" or seed not in seeds or seed in result:
            raise ValueError(f"unexpected or duplicate fixed run: {run}")
        result[seed] = command
    if set(result) != set(seeds):
        raise ValueError("frozen commands do not cover exactly the two registered seeds")
    return result


def _validate_frozen_inputs(repository_root: Path, spec_path: Path) -> dict[str, object]:
    """Validate all pre-outcome inputs and return their decoded records."""
    spec = _read_json(spec_path)
    if (
        spec.get("format") != "fixed-pair-contrast-grader-v1"
        or spec.get("status") != "frozen_before_candidate_prediction_scoring"
        or spec.get("id") != "E512-C1V2-vs-C0-fixed-pair"
    ):
        raise ValueError("pair grader specification is not the frozen E512 record")

    for relative, digest in spec["grader_dependencies"].items():
        _assert_file_hash(repository_root / relative, str(digest), f"grader dependency {relative}")

    candidate_spec = spec["candidate"]
    recipe_path = repository_root / str(candidate_spec["recipe_path"])
    _assert_file_hash(recipe_path, str(candidate_spec["recipe_sha256"]), "candidate recipe")
    candidate = _read_json(recipe_path)
    seeds = list(map(int, candidate_spec["seeds"]))
    if (
        candidate.get("id") != candidate_spec["id"]
        or candidate.get("source_commit") != candidate_spec["source_commit"]
        or candidate.get("teacher_audit_receipt_sha256") != candidate_spec["teacher_audit_sha256"]
        or candidate.get("source_families_sha256") != spec["source_families"]["sha256"]
    ):
        raise ValueError("C1v2 recipe metadata differs from frozen grader inputs")
    candidate_commands = _commands_by_seed(candidate.get("commands"), str(candidate_spec["prefix"]), seeds)
    expected_candidate_hashes = list(map(str, candidate_spec["command_sha256"]))
    actual_candidate_hashes = [
        _hash_command(candidate_commands[seed], str(candidate_spec["command_hash_encoding"]))
        for seed in seeds
    ]
    if actual_candidate_hashes != expected_candidate_hashes or candidate.get("command_sha256") != expected_candidate_hashes:
        raise ValueError("C1v2 command arrays differ from the frozen compact-JSON hashes")
    training = candidate.get("training", {})
    if (
        training.get("checkpoint_epoch_zero_based") != candidate_spec["fixed_epoch_zero_based"]
        or training.get("epochs") != 2
        or training.get("no_eval_during_fit") is not True
    ):
        raise ValueError("C1v2 fixed epoch, training horizon, or no-eval setting differs")
    teacher_runs = list(map(str, candidate_spec["teacher_run_names"]))
    target_formula = candidate.get("target_formula", {})
    if "0.40625*T3 + 0.40625*T4" not in str(target_formula.get("mean_teacher", "")):
        raise ValueError("C1v2 teacher target formula differs from the frozen homogeneous pair")
    if float(candidate_spec["effective_teacher_coefficient_sum"]) != 0.8125:
        raise ValueError("frozen C1v2 teacher coefficient sum changed")
    source_families_path = repository_root / str(spec["source_families"]["path"])
    _assert_file_hash(source_families_path, str(spec["source_families"]["sha256"]), "source families")

    audit_path = repository_root / str(candidate_spec["teacher_audit_path"])
    _assert_file_hash(audit_path, str(candidate_spec["teacher_audit_sha256"]), "teacher audit")
    audit = _read_json(audit_path)
    if (
        audit.get("status") != "passed"
        or audit.get("split_manifest_sha256") != spec["split"]["sha256"]
        or audit.get("source_families_sha256") != candidate.get("source_families_sha256")
    ):
        raise ValueError("pinned C1v2 teacher audit is incomplete or belongs to another fold")
    target_recipe = audit.get("target_recipe", {})
    if (
        target_recipe.get("teacher_runs") != teacher_runs
        or target_recipe.get("effective_coefficients")
        != list(map(float, candidate_spec["effective_teacher_coefficients"]))
        or float(target_recipe.get("coefficient_sum", -1))
        != float(candidate_spec["effective_teacher_coefficient_sum"])
    ):
        raise ValueError("teacher audit coefficients or source runs differ from the frozen C1v2 pair")
    audit_teachers = {str(item.get("run")): item for item in audit.get("teachers", [])}
    recipe_teachers = {str(item.get("run")): item for item in candidate.get("sources", [])}
    if set(audit_teachers) != set(teacher_runs) or set(recipe_teachers) != set(teacher_runs):
        raise ValueError("teacher audit and recipe do not identify exactly the two strong-pair sources")
    teacher_prediction_hashes: dict[str, str] = {}
    for run in teacher_runs:
        audit_teacher = audit_teachers[run]
        recipe_teacher = recipe_teachers[run]
        if (
            audit_teacher.get("checkpoint_sha256") != recipe_teacher.get("checkpoint_sha256")
            or audit_teacher.get("provenance_sha256") != recipe_teacher.get("provenance_sha256")
        ):
            raise ValueError(f"teacher audit checkpoint provenance changed for {run}")
        for split_name in ("train", "valid"):
            prediction = audit_teacher.get("predictions", {}).get(split_name, {})
            expected_hash = recipe_teacher.get("prediction_sha256", {}).get(split_name)
            if (
                prediction.get("prediction_sha256") != expected_hash
                or prediction.get("checkpoint_sha256") != recipe_teacher.get("checkpoint_sha256")
                or prediction.get("dtype") != "float32"
                or prediction.get("row_groups_sha256")
                != recipe_teacher.get("prediction_row_groups_sha256", {}).get(split_name)
            ):
                raise ValueError(f"teacher prediction audit differs from recipe for {run}/{split_name}")
            teacher_prediction_hashes[f"pred_{run}_{split_name}.npy"] = str(expected_hash)

    reference_spec = spec["reference"]
    prereg_path = repository_root / str(reference_spec["preregistration_path"])
    report_path = repository_root / str(reference_spec["evaluation_report_path"])
    _assert_file_hash(prereg_path, str(reference_spec["preregistration_sha256"]), "C0 preregistration")
    _assert_file_hash(report_path, str(reference_spec["evaluation_report_sha256"]), "C0 frozen report")
    reference_prereg = _read_json(prereg_path)
    reference_report = _read_json(report_path)
    if (
        reference_prereg.get("id") != "E-502"
        or reference_prereg.get("status") != "preregistered"
        or reference_prereg.get("fold") != 0
        or reference_prereg.get("seeds") != seeds
        or reference_report.get("candidate_id") != "C0"
        or reference_report.get("candidate_prefix") != reference_spec["prefix"]
        or reference_report.get("metric_id") != spec["metric"]["id"]
        or reference_report.get("preregistration", {}).get("sha256")
        != reference_spec["preregistration_sha256"]
        or reference_report.get("preregistration", {}).get("commands")
        != reference_prereg.get("commands")
    ):
        raise ValueError("C0 preregistration and its frozen evaluation report disagree")
    reference_commands = _commands_by_seed(
        reference_prereg.get("commands"), str(reference_spec["prefix"]), seeds
    )
    reference_hashes = [
        _hash_command(reference_commands[seed], str(reference_spec["command_hash_encoding"]))
        for seed in seeds
    ]
    if (
        reference_hashes != list(map(str, reference_spec["command_sha256"]))
        or reference_prereg.get("command_sha256") != reference_hashes
    ):
        raise ValueError("C0 command arrays differ from their frozen default-JSON hashes")
    if (
        reference_report.get("split", {}).get("sha256") != spec["split"]["sha256"]
        or reference_report.get("targets", {}).get("sha256")
        != spec["split"]["valid_target_cache_sha256"]
        or reference_report.get("n_sequences") != spec["split"]["evaluation_group_count"]
        or reference_report.get("bootstrap", {}).get("replicates")
        != spec["metric"]["bootstrap"]["replicates"]
        or reference_report.get("bootstrap", {}).get("seed") != spec["metric"]["bootstrap"]["seed"]
    ):
        raise ValueError("C0 report does not use the frozen split, targets, groups, and bootstrap")

    expected_members = {int(item["seed"]): item for item in reference_spec["members"]}
    report_members = reference_report.get("members", {})
    if set(expected_members) != set(seeds):
        raise ValueError("frozen C0 comparator does not identify both reference seeds")
    for seed in seeds:
        expected = expected_members[seed]
        actual = report_members.get(str(expected["run"]), {})
        for field in ("checkpoint_sha256", "prediction_sha256", "provenance_sha256"):
            if actual.get(field) != expected[field]:
                raise ValueError(f"frozen C0 report member differs for seed {seed}: {field}")

    split_path = repository_root / str(spec["split"]["path"])
    _assert_file_hash(split_path, str(spec["split"]["sha256"]), "fold-0 split")
    return {
        "spec": spec,
        "candidate": candidate,
        "candidate_commands": candidate_commands,
        "candidate_teacher_prediction_hashes": teacher_prediction_hashes,
        "reference_prereg": reference_prereg,
        "reference_report": reference_report,
        "reference_commands": reference_commands,
        "reference_members": expected_members,
        "split_path": split_path,
    }


def _normalized(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _normalized(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_normalized(item) for item in value]
    return value


def _validate_candidate_training_state(
    artifact_root: Path,
    run: str,
    args: Mapping[str, object],
    expected_teacher_hashes: Mapping[str, str],
    expected_sources_sha256: str,
    expected_initializer_sha256: str,
) -> dict[str, object]:
    """Bind fit-time teacher bytes and trainer sources from the final epoch state."""
    state_path = artifact_root / "artifacts" / "runs" / run / "training_state.pt"
    state_sha256 = file_sha256(state_path)
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    receipt = _training_state_receipt(
        state,
        args,
        expected_teacher_hashes,
        expected_sources_sha256,
        expected_initializer_sha256,
        run,
    )
    del state
    return {"path": str(state_path), "sha256": state_sha256, **receipt}


def _training_state_receipt(
    state: object,
    args: Mapping[str, object],
    expected_teacher_hashes: Mapping[str, str],
    expected_sources_sha256: str,
    expected_initializer_sha256: str,
    run: str,
) -> dict[str, object]:
    """Validate the pinned fit contract without depending on a Torch file."""
    if (
        not isinstance(state, dict)
        or state.get("format") != "wnn.epoch-training-state"
        or state.get("version") != 1
        or state.get("boundary") != "epoch"
        or state.get("next_epoch") != 2
    ):
        raise ValueError(f"{run}: missing the fixed two-epoch final training-state receipt")
    contract = state.get("contract")
    if not isinstance(contract, dict) or contract.get("epochs") != 2:
        raise ValueError(f"{run}: final training-state contract has another schedule")
    actual_sources_sha256 = _canonical_sha256(contract.get("sources"))
    if actual_sources_sha256 != expected_sources_sha256:
        raise ValueError(f"{run}: trainer/package sources differ from E511's pinned source commit")
    options = contract.get("options")
    expected_options = {key: value for key, value in args.items() if key not in EXCLUDED_CONTRACT_OPTIONS}
    if _normalized(options) != _normalized(expected_options):
        raise ValueError(f"{run}: training-state command options differ from the fixed checkpoint")
    teacher_predictions = contract.get("teacher_predictions")
    if teacher_predictions != dict(expected_teacher_hashes):
        raise ValueError(f"{run}: fit-time teacher prediction hashes differ from the audited bytes")
    ancestors = contract.get("ancestors")
    if not isinstance(ancestors, dict) or ancestors.get("initial_checkpoint_sha256") != expected_initializer_sha256:
        raise ValueError(f"{run}: training-state initializer hash differs from the registered parent")
    return {
        "next_epoch": state["next_epoch"],
        "trainer_contract_sources_sha256": actual_sources_sha256,
        "teacher_predictions": dict(teacher_predictions),
        "initial_checkpoint_sha256": ancestors["initial_checkpoint_sha256"],
    }


def _grade_pair(
    artifact_root: Path,
    output_dir: Path,
    *,
    repository_root: Path = ROOT,
    spec_path: Path | None = None,
) -> dict[str, object]:
    """Validate fixed C1v2 and C0 members and grade their paired contrasts."""
    artifact_root = artifact_root.resolve(strict=True)
    repository_root = repository_root.resolve(strict=True)
    spec_path = spec_path or repository_root / SPEC_PATH
    frozen = _validate_frozen_inputs(repository_root, spec_path)
    spec = frozen["spec"]
    candidate_spec, reference_spec = spec["candidate"], spec["reference"]
    split = load_campaign_split(frozen["split_path"], artifact_root)
    if split.fold != 0 or len(split.evaluation_groups) != int(spec["split"]["evaluation_group_count"]):
        raise ValueError("expected all 468 frozen fold-0 evaluation groups")

    target_path = artifact_root / "cache" / "valid_targets.npz"
    _assert_file_hash(
        target_path,
        str(spec["split"]["valid_target_cache_sha256"]),
        "valid target cache",
    )
    with np.load(target_path, allow_pickle=False) as cache:
        row_groups = list(map(int, cache["row_group"]))
        targets = cache["y"]
        is_scored = cache["mask"]
        classes_all = cache["market_class"]
    if campaign_eval._list_sha256(row_groups) != spec["split"]["valid_target_row_groups_sha256"]:
        raise ValueError("valid target row order differs from the frozen comparison")
    expected_groups = campaign_eval._expected_target_groups(split.valid_groups, split.evaluation_groups)
    campaign_eval._validate_target_index(row_groups, targets, is_scored, classes_all, expected_groups)

    groups = list(map(int, split.evaluation_groups))
    group_positions = {group: index for index, group in enumerate(row_groups)}
    if len(groups) != len(set(groups)) or any(group not in group_positions for group in groups):
        raise ValueError("frozen evaluation groups are missing or duplicated in valid_targets")
    indices = np.asarray([group_positions[group] for group in groups], dtype=np.int64)
    group_to_component = {
        int(group): int(component)
        for source, group, component in split.components
        if source == "valid"
    }
    components = np.asarray([group_to_component[group] for group in groups], dtype=np.int64)
    classes = np.asarray(classes_all[indices], dtype=np.int64)
    selected_mask = campaign_eval._official_mask(np.asarray(is_scored[indices], dtype=bool))

    candidate_predictions: dict[int, np.memmap] = {}
    reference_predictions: dict[int, np.memmap] = {}
    bindings: dict[str, dict[str, object]] = {}
    training_states: dict[str, dict[str, object]] = {}
    seeds = list(map(int, candidate_spec["seeds"]))
    for seed in seeds:
        candidate_run = f"{candidate_spec['prefix']}_s{seed}_f0"
        reference_record = frozen["reference_members"][seed]
        reference_run = str(reference_record["run"])
        candidate_command = frozen["candidate_commands"][seed]
        reference_command = frozen["reference_commands"][seed]

        reference_prediction, reference_binding = campaign_eval._member(
            artifact_root,
            reference_run,
            seed,
            int(reference_spec["fixed_epoch_zero_based"]),
            split,
            row_groups,
            expected_init=None,
            source_record=reference_record,
            expected_command=reference_command,
        )
        if reference_binding["prediction_sha256"] != reference_record["prediction_sha256"]:
            raise ValueError(f"{reference_run}: C0 prediction bytes differ from the frozen report")

        candidate_prediction, candidate_binding = campaign_eval._member(
            artifact_root,
            candidate_run,
            seed,
            int(candidate_spec["fixed_epoch_zero_based"]),
            split,
            row_groups,
            expected_init=f"{candidate_spec['teacher_run_names'][seed - seeds[0]]}@ep3",
            expected_command=candidate_command,
        )
        candidate_provenance = _read_json(
            artifact_root / "artifacts" / "runs" / candidate_run / "provenance.json"
        )
        args = candidate_provenance.get("args")
        if not isinstance(args, dict):
            raise ValueError(f"{candidate_run}: provenance has no checkpoint arguments")
        recipe_source = next(
            item for item in frozen["candidate"]["sources"] if int(item["seed"]) == seed
        )
        training_state = _validate_candidate_training_state(
            artifact_root,
            candidate_run,
            args,
            frozen["candidate_teacher_prediction_hashes"],
            str(candidate_spec["training_contract_sources_sha256"]),
            str(recipe_source["checkpoint_sha256"]),
        )
        candidate_predictions[seed] = candidate_prediction
        reference_predictions[seed] = reference_prediction
        bindings[candidate_run] = candidate_binding
        bindings[reference_run] = reference_binding
        training_states[candidate_run] = training_state

    comparison_members = {
        "seed3_c1v2_vs_c0": ((candidate_predictions[3],), (reference_predictions[3],)),
        "seed4_c1v2_vs_c0": ((candidate_predictions[4],), (reference_predictions[4],)),
        "equal_pair_c1v2_vs_c0": (
            (candidate_predictions[3], candidate_predictions[4]),
            (reference_predictions[3], reference_predictions[4]),
        ),
    }
    moments = {
        name: (
            np.empty((len(groups), 2, 6), dtype=np.float64),
            np.empty((len(groups), 2, 6), dtype=np.float64),
        )
        for name in comparison_members
    }
    for start in range(0, len(groups), SEQUENCES_PER_BATCH):
        stop = min(start + SEQUENCES_PER_BATCH, len(groups))
        take = indices[start:stop]
        y = np.asarray(targets[take], dtype=np.float32)
        mask = selected_mask[start:stop]
        for name, (candidate_members, reference_members) in comparison_members.items():
            candidate_rows = (
                np.asarray(candidate_members[0][take], dtype=np.float32)
                if len(candidate_members) == 1
                else campaign_eval._blend_equal_pair(candidate_members[0][take], candidate_members[1][take])
            )
            reference_rows = (
                np.asarray(reference_members[0][take], dtype=np.float32)
                if len(reference_members) == 1
                else campaign_eval._blend_equal_pair(reference_members[0][take], reference_members[1][take])
            )
            candidate_moments, reference_moments = paired_component_moments(
                y, candidate_rows, reference_rows, mask
            )
            moments[name][0][start:stop] = candidate_moments
            moments[name][1][start:stop] = reference_moments

    comparisons: dict[str, dict[str, object]] = {}
    bootstrap = spec["metric"]["bootstrap"]
    for name, (candidate_moments, reference_moments) in moments.items():
        result = paired_component_bootstrap(
            candidate_moments,
            reference_moments,
            components,
            n_boot=int(bootstrap["replicates"]),
            seed=int(bootstrap["seed"]),
            centered=bool(bootstrap["centered"]),
        )
        comparisons[name] = dict(result) | {"selected_rows": int(selected_mask.sum())}

    component_provenance = campaign_eval._component_provenance(
        groups, components, classes, selected_mask
    )
    report = {
        "grader_id": spec["id"],
        "grader_spec": {"path": str(spec_path), "sha256": file_sha256(spec_path)},
        "grader_source_sha256": file_sha256(Path(__file__)),
        "candidate_recipe": {
            "path": str(repository_root / candidate_spec["recipe_path"]),
            "sha256": file_sha256(repository_root / candidate_spec["recipe_path"]),
            "id": candidate_spec["id"],
            "commands": frozen["candidate"]["commands"],
            "command_sha256": candidate_spec["command_sha256"],
            "teacher_audit_sha256": candidate_spec["teacher_audit_sha256"],
        },
        "reference": {
            "id": reference_spec["id"],
            "preregistration_sha256": file_sha256(
                repository_root / reference_spec["preregistration_path"]
            ),
            "evaluation_report_sha256": file_sha256(
                repository_root / reference_spec["evaluation_report_path"]
            ),
            "commands": frozen["reference_prereg"]["commands"],
            "members": {str(seed): frozen["reference_members"][seed] for seed in seeds},
        },
        "metric_id": METRIC_ID,
        "split": {
            "path": str(frozen["split_path"]),
            "fold": split.fold,
            "sha256": split.sha256,
            "evaluation_groups": groups,
        },
        "targets": {
            "path": str(target_path),
            "sha256": file_sha256(target_path),
            "row_groups_sha256": campaign_eval._list_sha256(row_groups),
        },
        "selection": spec["metric"]["selection"],
        "pair_blend": spec["metric"]["pair_blend"],
        "n_sequences": len(groups),
        "n_components": int(len(np.unique(components))),
        "selected_rows": int(selected_mask.sum()),
        "bootstrap": dict(bootstrap)
        | {"method": "paired frozen-component bootstrap over official per-sequence moments"},
        "training_state_receipts": training_states,
        "members": bindings,
        "comparisons": comparisons,
        "data_exposure": spec["data_exposure"],
        "public_forecast": None,
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (output_dir / "components.json").write_text(
        json.dumps(
            {
                "fold": 0,
                "split_manifest_sha256": split.sha256,
                "selection": spec["metric"]["selection"],
                "components": component_provenance,
            },
            indent=2,
        )
        + "\n"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=ROOT)
    parser.add_argument("--spec", type=Path)
    args = parser.parse_args()
    result = _grade_pair(
        args.artifact_root,
        args.output_dir,
        repository_root=args.repository_root,
        spec_path=args.spec,
    )
    print(json.dumps(result["comparisons"], indent=2))


if __name__ == "__main__":
    main()
