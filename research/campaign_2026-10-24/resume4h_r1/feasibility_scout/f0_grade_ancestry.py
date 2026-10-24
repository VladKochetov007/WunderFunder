"""Verify source-qualified parent and teacher ancestry for the saved F0 grade."""

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path
from typing import Any


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _row_refs_sha256(split: str, groups: list[int]) -> str:
    refs = [{"split": split, "row_group": int(group)} for group in groups]
    payload = json.dumps(refs, sort_keys=True, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _ordered_int64_sha256(groups: list[int]) -> str:
    digest = hashlib.sha256()
    for group in groups:
        digest.update(struct.pack("<q", int(group)))
    return digest.hexdigest()


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    resolved = (path if path.is_absolute() else root / path).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError(f"ancestry path escapes the repository: {value}")
    return resolved


def _validate_membership(
    run: str,
    provenance: dict[str, Any],
    evaluation_groups: set[int],
) -> dict[str, Any]:
    fit_groups = provenance.get("fit_row_groups")
    if not isinstance(fit_groups, dict) or "valid" not in fit_groups:
        raise ValueError(f"{run}: source-qualified valid fit membership is absent")
    membership: dict[str, list[int]] = {}
    for split, raw_groups in fit_groups.items():
        if split not in {"train", "valid"} or not isinstance(raw_groups, list):
            raise ValueError(f"{run}: malformed source-qualified fit groups")
        groups = [int(group) for group in raw_groups]
        if len(groups) != len(set(groups)):
            raise ValueError(f"{run}: duplicate {split} fit row groups")
        membership[split] = groups

    valid_groups = set(membership["valid"])
    overlap = valid_groups & evaluation_groups
    if overlap:
        raise ValueError(f"{run}: {len(overlap)} valid fit groups overlap the F0 evaluation set")

    raw_evaluation = provenance.get("evaluation_row_groups")
    if not isinstance(raw_evaluation, list):
        raise ValueError(f"{run}: evaluation membership is absent")
    run_evaluation = [int(group) for group in raw_evaluation]
    if len(run_evaluation) != len(set(run_evaluation)) or set(run_evaluation) != evaluation_groups:
        raise ValueError(f"{run}: evaluation groups differ from source-qualified F0 scope")

    identity = provenance.get("fit_membership_identity")
    membership_report: dict[str, Any] = {}
    for split, groups in membership.items():
        ordered_sha = _ordered_int64_sha256(groups)
        if isinstance(identity, dict) and isinstance(identity.get(split), dict):
            recorded = identity[split]
            if recorded.get("count") != len(groups):
                raise ValueError(f"{run}: {split} fit membership count differs")
            if recorded.get("ordered_int64_le_sha256") != ordered_sha:
                raise ValueError(f"{run}: {split} fit membership digest differs")
        membership_report[split] = {
            "count": len(groups),
            "ordered_int64_le_sha256": ordered_sha,
            "source_qualified_row_refs_sha256": _row_refs_sha256(split, groups),
        }

    return {
        "evaluation_groups": len(run_evaluation),
        "evaluation_membership_sha256": _row_refs_sha256("valid", run_evaluation),
        "valid_fit_evaluation_overlap": 0,
        "fit_membership": membership_report,
    }


def verify_parent_ancestry(
    root: Path,
    recipe: dict[str, Any],
    split_sha256: str,
    evaluation_groups: list[int],
) -> dict[str, Any]:
    """Walk every pinned parent, initializer, teacher, and propensity edge."""
    root = root.resolve()
    ancestry = recipe["ancestry"]
    run_pins = ancestry["runs"]
    expected_split = str(split_sha256)
    eval_set = {int(group) for group in evaluation_groups}
    if len(eval_set) != len(evaluation_groups):
        raise ValueError("F0 evaluation groups contain duplicates")
    if len(eval_set) != int(ancestry["evaluation_group_count"]):
        raise ValueError("F0 evaluation group count differs from the frozen ancestry contract")

    verified: dict[str, dict[str, Any]] = {}
    active: set[str] = set()
    edges: list[dict[str, Any]] = []
    cache_reports: list[dict[str, Any]] = []

    def verify_run(run: str, checkpoint_key: str, expected_checkpoint_sha: str) -> None:
        if run in active:
            raise ValueError(f"ancestry cycle detected at {run}")
        pin = run_pins.get(run)
        if not isinstance(pin, dict):
            raise ValueError(f"ancestry run {run} is not pinned by the exploratory recipe")
        if pin.get("checkpoint_key") != checkpoint_key:
            raise ValueError(f"{run}: checkpoint key differs from the frozen ancestry pin")
        if pin.get("checkpoint_sha256") != expected_checkpoint_sha:
            raise ValueError(f"{run}: checkpoint hash differs from the frozen ancestry pin")
        if run in verified:
            return

        active.add(run)
        run_dir = _resolve(root, f"artifacts/runs/{run}")
        provenance_path = run_dir / "provenance.json"
        provenance_sha = _sha256_file(provenance_path)
        if provenance_sha != pin.get("provenance_sha256"):
            raise ValueError(f"{run}: provenance differs from the frozen ancestry pin")
        provenance = _read_json(provenance_path)
        checkpoint_sha = provenance.get("checkpoint_sha256", {}).get(checkpoint_key)
        if checkpoint_sha != expected_checkpoint_sha:
            raise ValueError(f"{run}: checkpoint hash is absent from provenance")
        checkpoint_path = run_dir / checkpoint_key
        if _sha256_file(checkpoint_path) != expected_checkpoint_sha:
            raise ValueError(f"{run}: checkpoint payload hash differs")
        if provenance.get("split_manifest_sha256") != expected_split:
            raise ValueError(f"{run}: split manifest differs from the F0 source contract")
        membership = _validate_membership(run, provenance, eval_set)
        if len(provenance["fit_row_groups"]["valid"]) != int(ancestry["valid_fit_group_count"]):
            raise ValueError(f"{run}: valid fit group count differs from the frozen scope")

        record = {
            "run": run,
            "provenance_path": str(provenance_path),
            "provenance_sha256": provenance_sha,
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_key": checkpoint_key,
            "checkpoint_sha256": expected_checkpoint_sha,
            "split_manifest_sha256": expected_split,
            **membership,
        }
        verified[run] = record

        args = provenance.get("args") or {}
        init = args.get("init")
        if init:
            try:
                init_run, init_epoch_text = str(init).rsplit("@ep", maxsplit=1)
                init_epoch = int(init_epoch_text)
            except (ValueError, TypeError) as error:
                raise ValueError(f"{run}: malformed initializer reference") from error
            init_key = f"model_ep{init_epoch}.pt"
            init_sha = provenance.get("initial_checkpoint_sha256")
            if not isinstance(init_sha, str):
                raise ValueError(f"{run}: initializer checkpoint hash is absent")
            edges.append(
                {
                    "child": run,
                    "type": "initializer",
                    "parent": init_run,
                    "checkpoint_key": init_key,
                    "checkpoint_sha256": init_sha,
                }
            )
            verify_run(init_run, init_key, init_sha)
        elif provenance.get("initial_checkpoint_sha256"):
            raise ValueError(f"{run}: initializer hash exists without an initializer reference")

        resumed = provenance.get("resumed_from")
        if resumed:
            resumed_path = _resolve(root, str(resumed["path"]))
            expected_parent_dir = resumed_path.parent
            if (
                resumed_path.name != "training_state.pt"
                or expected_parent_dir.parent.name != "runs"
            ):
                raise ValueError(f"{run}: resume state path is not a run training state")
            previous_run = expected_parent_dir.name
            if _sha256_file(resumed_path) != resumed.get("sha256"):
                raise ValueError(f"{run}: resumed training-state payload hash differs")
            previous_epoch = int(resumed["next_epoch"]) - 1
            if previous_epoch < 0:
                raise ValueError(f"{run}: resumed epoch is invalid")
            previous_key = f"model_ep{previous_epoch}.pt"
            previous_provenance_path = expected_parent_dir / "provenance.json"
            previous_provenance = _read_json(previous_provenance_path)
            previous_sha = previous_provenance.get("checkpoint_sha256", {}).get(previous_key)
            if not isinstance(previous_sha, str):
                raise ValueError(f"{run}: preceding run lacks the resumed model checkpoint")
            edges.append(
                {
                    "child": run,
                    "type": "resume_state",
                    "parent": previous_run,
                    "state_path": str(resumed_path),
                    "state_sha256": resumed["sha256"],
                    "checkpoint_key": previous_key,
                    "checkpoint_sha256": previous_sha,
                }
            )
            verify_run(previous_run, previous_key, previous_sha)

        propensity_run = args.get("propensity")
        if propensity_run:
            propensity_sha = provenance.get("propensity_checkpoint_sha256")
            if not isinstance(propensity_sha, str):
                raise ValueError(f"{run}: propensity checkpoint hash is absent")
            edges.append(
                {
                    "child": run,
                    "type": "propensity",
                    "parent": str(propensity_run),
                    "checkpoint_key": "model.pt",
                    "checkpoint_sha256": propensity_sha,
                }
            )
            verify_run(str(propensity_run), "model.pt", propensity_sha)
        elif provenance.get("propensity_checkpoint_sha256"):
            raise ValueError(f"{run}: propensity hash exists without a source run")

        distill_edges = provenance.get("args", {}).get("distill_provenance") or []
        for edge in distill_edges:
            parent_run = str(edge["run"])
            epoch = int(edge["epoch"])
            parent_key = f"model_ep{epoch}.pt"
            parent_sha = str(edge["checkpoint_sha256"])
            edges.append(
                {
                    "child": run,
                    "type": "distillation_prediction",
                    "parent": parent_run,
                    "split": str(edge["split"]),
                    "checkpoint_key": parent_key,
                    "checkpoint_sha256": parent_sha,
                    "prediction_sha256": str(edge["prediction_sha256"]),
                }
            )
            verify_run(parent_run, parent_key, parent_sha)

        cache_record = provenance.get("distillation_cache")
        if cache_record:
            cache_path = _resolve(root, str(cache_record["root"])) / "metadata.json"
            cache_sha = _sha256_file(cache_path)
            if cache_sha != cache_record.get("metadata_sha256"):
                raise ValueError(f"{run}: distillation-cache metadata hash differs")
            cache_pin = ancestry.get("distillation_cache", {})
            if cache_pin.get("metadata_sha256") != cache_sha:
                raise ValueError(f"{run}: distillation-cache hash differs from the F0 recipe pin")
            if _resolve(root, str(cache_pin.get("metadata_path"))) != cache_path:
                raise ValueError(
                    f"{run}: distillation-cache metadata path differs from the F0 recipe"
                )
            cache = _read_json(cache_path)
            source_coefficients = cache.get("source_coefficients") or []
            expected_coefficients = cache_record.get("source_coefficients") or []

            def normalize_coefficients(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
                return [
                    {
                        "run": str(row["run"]),
                        "epoch": int(row["epoch"]),
                        "family": str(row["family"]),
                        "coefficient": float(row["coefficient"]),
                    }
                    for row in rows
                ]

            if normalize_coefficients(source_coefficients) != normalize_coefficients(
                expected_coefficients
            ):
                raise ValueError(f"{run}: distillation-cache source coefficients differ")
            if cache.get("split_manifest_sha256") != expected_split:
                raise ValueError(f"{run}: distillation-cache split hash differs")
            if set(map(int, cache.get("evaluation_valid_groups", []))) != eval_set:
                raise ValueError(f"{run}: distillation-cache evaluation group set differs")
            cache_fit_groups = cache.get("fit_groups", {})
            for split, run_groups in provenance["fit_row_groups"].items():
                if set(map(int, cache_fit_groups.get(split, []))) != set(map(int, run_groups)):
                    raise ValueError(f"{run}: distillation-cache {split} fit membership differs")
            if (
                int(
                    cache.get("source_qualification", {}).get(
                        "evaluation_valid_groups_in_cache", -1
                    )
                )
                != 0
            ):
                raise ValueError(f"{run}: distillation cache includes evaluation groups in its fit")
            if (
                cache.get("source_qualification", {}).get(
                    "prediction_arrays_joined_by_row_group_id"
                )
                is not True
            ):
                raise ValueError(f"{run}: distillation-cache source joins are not row-group keyed")
            source_records = cache.get("sources") or []
            if len(source_records) != len(source_coefficients):
                raise ValueError(f"{run}: distillation-cache source records are incomplete")
            for source, coefficient in zip(source_records, source_coefficients, strict=True):
                source_run = str(coefficient["run"])
                epoch = int(coefficient["epoch"])
                source_key = f"model_ep{epoch}.pt"
                source_sha = str(source["checkpoint_sha256"])
                if source.get("run") != source_run or int(source.get("epoch", -1)) != epoch:
                    raise ValueError(f"{run}: distillation-cache teacher identity differs")
                if source.get("coefficient") != coefficient.get("coefficient"):
                    raise ValueError(f"{run}: distillation-cache teacher coefficient differs")
                expected_checkpoint_path = _resolve(
                    root, f"artifacts/runs/{source_run}/{source_key}"
                )
                if _resolve(root, str(source["checkpoint_path"])) != expected_checkpoint_path:
                    raise ValueError(f"{run}: distillation-cache teacher checkpoint path differs")
                source_provenance_path = _resolve(root, str(source["provenance_path"]))
                expected_provenance_path = _resolve(
                    root, f"artifacts/runs/{source_run}/provenance.json"
                )
                if source_provenance_path != expected_provenance_path:
                    raise ValueError(f"{run}: distillation-cache teacher provenance path differs")
                if _sha256_file(source_provenance_path) != source.get("provenance_sha256"):
                    raise ValueError(f"{run}: distillation-cache teacher provenance hash differs")
                teacher_provenance = _read_json(source_provenance_path)
                teacher_fit = teacher_provenance.get("fit_row_groups", {})
                for split, group_record in source.get("fit_group_memberships", {}).items():
                    if split not in teacher_fit:
                        raise ValueError(
                            f"{run}: cache teacher {source_run} lacks {split} fit groups"
                        )
                    groups = [int(group) for group in teacher_fit[split]]
                    if int(group_record.get("count", -1)) != len(groups):
                        raise ValueError(f"{run}: cache teacher {source_run} {split} count differs")
                    if group_record.get("ordered_sha256") != _ordered_int64_sha256(groups):
                        raise ValueError(
                            f"{run}: cache teacher {source_run} {split} membership differs"
                        )
                prediction_records = source.get("predictions", {})
                train_prediction = prediction_records.get("train", {})
                if train_prediction.get("scope") != "train":
                    raise ValueError(f"{run}: cache teacher training targets lack source scope")
                valid_prediction = prediction_records.get("valid", {})
                if valid_prediction.get("scope") != "full-valid-feature-only":
                    raise ValueError(f"{run}: cache teacher valid predictions are not feature-only")
                edges.append(
                    {
                        "child": run,
                        "type": "distillation_cache_source",
                        "parent": source_run,
                        "checkpoint_key": source_key,
                        "checkpoint_sha256": source_sha,
                        "family": str(coefficient["family"]),
                        "coefficient": float(coefficient["coefficient"]),
                        "train_prediction_sha256": str(train_prediction["prediction_sha256"]),
                        "train_row_groups_sha256": str(
                            train_prediction["ordered_row_groups_sha256"]
                        ),
                        "valid_prediction_sha256": str(valid_prediction["prediction_sha256"]),
                        "valid_row_groups_sha256": str(
                            valid_prediction["ordered_row_groups_sha256"]
                        ),
                        "valid_prediction_scope": str(valid_prediction["scope"]),
                    }
                )
                verify_run(source_run, source_key, source_sha)
            cache_reports.append(
                {
                    "owner_run": run,
                    "metadata_path": str(cache_path),
                    "metadata_sha256": cache_sha,
                    "source_count": len(source_records),
                    "fit_membership_source_qualified": True,
                    "payload_arrays_rehashed": False,
                }
            )

        active.remove(run)

    parent = recipe["parent_recipe"]
    parent_recipe = _read_json(_resolve(root, parent["path"]))
    fold = next(item for item in parent_recipe["folds"] if int(item["fold"]) == 0)
    root_parent = fold["parent"]
    verify_run(
        str(root_parent["run"]),
        f"model_ep{int(root_parent['epoch'])}.pt",
        str(root_parent["checkpoint_sha256"]),
    )
    return {
        "root_parent": str(root_parent["run"]),
        "evaluation_group_count": len(eval_set),
        "valid_fit_group_count": int(ancestry["valid_fit_group_count"]),
        "runs": [verified[name] for name in sorted(verified)],
        "edges": edges,
        "distillation_caches": cache_reports,
        "large_prediction_payloads_rehashed": False,
        "source_qualified_overlap_check": "valid fit groups were checked against valid evaluation groups; train IDs remain a separate namespace",
    }
