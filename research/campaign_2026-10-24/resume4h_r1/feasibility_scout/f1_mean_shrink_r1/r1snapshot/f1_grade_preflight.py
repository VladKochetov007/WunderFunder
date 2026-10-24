"""Validate the saved F1 A-lag1 endpoint and clean source-qualified ancestry."""

from __future__ import annotations

import hashlib
import json
import sys
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[5]
HERE = Path(__file__).resolve().parent
ONLINE = ROOT / "research/experiment_drafts/online-relationships-20261022"
FEASIBILITY = HERE.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ONLINE))
sys.path.insert(0, str(FEASIBILITY))

from online_relationship_grade_contract import row_refs_sha256  # noqa: E402
from online_relationship_grade_core import npy_coverage, sha256_file  # noqa: E402

ROWS = 20_000
PREDICTION_SHAPE = [457, ROWS, 2]
PREDICTION_DTYPE = "<f4"


@dataclass(frozen=True)
class PreparedF1Grade:
    """Validated endpoint paths and split-qualified F1 bootstrap units."""

    recipe: dict[str, Any]
    recipe_sha256: str
    evaluation_groups: list[int]
    components: list[int]
    parent_prediction_path: Path
    candidate_prediction_path: Path
    target_cache_path: Path
    report: dict[str, Any]


def _read_json(path: Path) -> dict[str, Any]:
    """Read bounded JSON metadata and reject non-object roots."""
    if path.stat().st_size > 5_000_000:
        raise ValueError(f"metadata file exceeds the 5 MB bound: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _resolve(value: str) -> Path:
    """Resolve a repository-relative or absolute artifact path."""
    path = Path(value)
    return (path if path.is_absolute() else ROOT / path).resolve()


def _require(actual: Any, expected: Any, label: str) -> None:
    """Fail closed when a frozen field differs from observed metadata."""
    if actual != expected:
        raise ValueError(f"{label} differs from the frozen F1 contract")


def _source_manifest(recipe: dict[str, Any]) -> dict[str, Any]:
    """Verify the recipe-bound source manifest and each listed small source."""
    record = recipe["source_manifest"]
    path = _resolve(record["path"])
    _require(sha256_file(path), record["sha256"], "source manifest SHA-256")
    manifest = _read_json(path)
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("source manifest has no file pin map")
    for source_path, expected_sha in files.items():
        if not isinstance(expected_sha, str) or len(expected_sha) != 64:
            raise ValueError(f"malformed source hash for {source_path}")
        actual = sha256_file(_resolve(source_path))
        _require(actual, expected_sha, f"source SHA-256 for {source_path}")
    return manifest


def _verify_refs_and_components(
    recipe: dict[str, Any], subset: dict[str, Any], split: dict[str, Any]
) -> tuple[list[int], list[int], list[dict[str, int | str]]]:
    """Derive the exact eval ordering and component labels from frozen manifests."""
    groups = [int(group) for group in subset["evaluation_groups"]]
    expected = recipe["evaluation"]
    _require(subset.get("selected_members_sha256"), recipe["fit_membership"]["selected_members_sha256"], "selected fit members")
    _require(subset.get("selected_component_sha256"), recipe["fit_membership"]["selected_component_sha256"], "selected fit components")
    _require(subset.get("selected_component_count"), recipe["fit_membership"]["component_count"], "selected fit component count")
    _require(subset.get("selected_group_counts"), {"train": recipe["fit_membership"]["train_group_count"], "valid": recipe["fit_membership"]["valid_group_count"]}, "selected fit split counts")
    if len(groups) != 457 or len(groups) != len(set(groups)):
        raise ValueError("F1 evaluation scope must contain 457 unique valid groups")
    refs = [{"split": "valid", "row_group": group} for group in groups]
    _require(row_refs_sha256(refs), expected["row_refs_sha256"], "ordered eval membership")
    _require(subset.get("split_manifest_sha256"), recipe["split_manifest"]["sha256"], "subset split")
    _require(subset.get("evaluation_group_count"), len(groups), "subset eval group count")
    _require(subset.get("evaluation_membership_sha256"), expected["subset_membership_sha256"], "subset eval identity")
    split_rows = split.get("sequences", {}).get("valid")
    if not isinstance(split_rows, list):
        raise ValueError("split manifest lacks valid source-qualified sequence rows")
    component_by_group: dict[int, int] = {}
    for row in split_rows:
        if not isinstance(row, dict) or "row_group" not in row or "component" not in row:
            raise ValueError("malformed source-qualified valid sequence row")
        group = int(row["row_group"])
        if group in component_by_group:
            raise ValueError("duplicate valid row group in split component map")
        component_by_group[group] = int(row["component"])
    if any(group not in component_by_group for group in groups):
        raise ValueError("F1 subset contains a group absent from the valid split")
    components = [component_by_group[group] for group in groups]
    records = [
        {"split": "valid", "row_group": group, "component": component}
        for group, component in zip(groups, components, strict=True)
    ]
    digest = hashlib.sha256(
        json.dumps(records, sort_keys=True, separators=(",", ":")).encode("ascii")
    ).hexdigest()
    _require(len(set(components)), expected["component_count"], "bootstrap component count")
    _require(digest, expected["component_membership_sha256"], "component membership")
    selected_components = {int(component) for component in subset["selected_components"]}
    if len(selected_components) != recipe["fit_membership"]["component_count"]:
        raise ValueError("F1 readout fit component count differs from the subset recipe")
    if selected_components & set(components):
        raise ValueError("F1 readout fit components overlap the evaluation components")
    return groups, components, refs


def _ordered_int64_sha256(groups: list[int]) -> str:
    """Hash an ordered row-group list using the producer's little-endian encoding."""
    digest = hashlib.sha256()
    for group in groups:
        digest.update(struct.pack("<q", int(group)))
    return digest.hexdigest()


def _verify_f1_ancestry(
    recipe: dict[str, Any], split: dict[str, Any], groups: list[int]
) -> dict[str, Any]:
    """Verify the exact F1 parent/teacher/initializer/propensity graph and fit rows."""
    ancestry = recipe["ancestry"]
    pins = ancestry["runs"]
    _require(set(ancestry["expected_fit_membership"]), set(pins), "ancestry fit-membership pin keys")
    _require(ancestry["root_parent"], recipe["inputs"]["parent"]["run"], "F1 parent ancestry root")
    split_fit = {
        name: {int(group) for group in values}
        for name, values in split["fit"].items()
    }
    eval_set = set(groups)
    verified: dict[str, dict[str, Any]] = {}
    edges: list[dict[str, str]] = []
    reachable = {ancestry["root_parent"]}
    active: set[str] = set()

    def edge(child: str, edge_type: str, parent_run: str, checkpoint_key: str, sha: str) -> None:
        pin = pins.get(parent_run)
        if not isinstance(pin, dict):
            raise ValueError(f"F1 ancestry edge points to unpinned run {parent_run}")
        if pin.get("checkpoint_key") != checkpoint_key or pin.get("checkpoint_sha256") != sha:
            raise ValueError(f"F1 ancestry edge {child}->{parent_run} has a checkpoint mismatch")
        reachable.add(parent_run)
        edges.append(
            {
                "child": child,
                "type": edge_type,
                "parent": parent_run,
                "checkpoint_key": checkpoint_key,
                "checkpoint_sha256": sha,
            }
        )

    def verify_run(run: str) -> None:
        if run in active:
            raise ValueError(f"F1 ancestry cycle at {run}")
        if run in verified:
            return
        pin = pins.get(run)
        if not isinstance(pin, dict):
            raise ValueError(f"F1 ancestry run {run} is not frozen")
        active.add(run)
        run_dir = ROOT / "artifacts" / "runs" / run
        provenance_path = run_dir / "provenance.json"
        checkpoint_path = run_dir / str(pin["checkpoint_key"])
        provenance_sha = sha256_file(provenance_path)
        _require(provenance_sha, pin["provenance_sha256"], f"{run} provenance SHA-256")
        provenance = _read_json(provenance_path)
        actual_checkpoint_sha = provenance.get("checkpoint_sha256", {}).get(pin["checkpoint_key"])
        _require(actual_checkpoint_sha, pin["checkpoint_sha256"], f"{run} checkpoint provenance")
        _require(sha256_file(checkpoint_path), pin["checkpoint_sha256"], f"{run} checkpoint payload")
        _require(provenance.get("split_manifest_sha256"), recipe["split_manifest"]["sha256"], f"{run} fold1 split")

        expected_counts = ancestry["expected_fit_membership"][run]
        fit_record = provenance.get("fit_row_groups")
        if not isinstance(fit_record, dict):
            raise ValueError(f"{run} lacks source-qualified fit membership")
        membership_report: dict[str, Any] = {}
        for split_name in ("train", "valid"):
            raw = fit_record.get(split_name, [])
            if not isinstance(raw, list):
                raise ValueError(f"{run} has malformed {split_name} fit membership")
            values = [int(group) for group in raw]
            if len(values) != len(set(values)):
                raise ValueError(f"{run} has duplicate {split_name} fit groups")
            expected_count = int(expected_counts[split_name]["count"])
            _require(len(values), expected_count, f"{run} {split_name} fit count")
            if not set(values).issubset(split_fit[split_name]):
                raise ValueError(f"{run} {split_name} fit rows escape the fold1 fit namespace")
            ordered_sha = _ordered_int64_sha256(values)
            _require(ordered_sha, expected_counts[split_name]["ordered_sha256"], f"{run} {split_name} fit membership")
            identity = (provenance.get("fit_membership_identity") or {}).get(split_name)
            if identity is not None:
                _require(identity.get("count"), len(values), f"{run} {split_name} recorded fit count")
                _require(identity.get("ordered_int64_le_sha256"), ordered_sha, f"{run} {split_name} recorded fit digest")
            membership_report[split_name] = {
                "count": len(values),
                "ordered_int64_le_sha256": ordered_sha,
                "source_qualified_refs_sha256": row_refs_sha256(
                    [{"split": split_name, "row_group": group} for group in values]
                ),
                "outside_fold_fit_rows": 0,
            }
        if eval_set & set(map(int, fit_record.get("valid", []))):
            raise ValueError(f"{run} valid fit rows overlap the F1 evaluation groups")
        evaluation = [int(group) for group in provenance.get("evaluation_row_groups", [])]
        if len(evaluation) != len(set(evaluation)) or set(evaluation) != eval_set:
            raise ValueError(f"{run} evaluation membership differs from F1's 457 groups")

        record = {
            "run": run,
            "provenance_path": str(provenance_path),
            "provenance_sha256": provenance_sha,
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_key": pin["checkpoint_key"],
            "checkpoint_sha256": actual_checkpoint_sha,
            "evaluation_group_count": len(evaluation),
            "evaluation_row_refs_sha256": row_refs_sha256(
                [{"split": "valid", "row_group": group} for group in evaluation]
            ),
            "fit_membership": membership_report,
        }
        verified[run] = record
        args = provenance.get("args") or {}

        init = args.get("init")
        if init:
            try:
                parent_run, epoch_text = str(init).rsplit("@ep", maxsplit=1)
            except ValueError as error:
                raise ValueError(f"{run} has malformed initializer edge") from error
            parent_key = f"model_ep{int(epoch_text)}.pt"
            parent_sha = provenance.get("initial_checkpoint_sha256")
            if not isinstance(parent_sha, str):
                raise ValueError(f"{run} initializer lacks a checkpoint hash")
            edge(run, "initializer", parent_run, parent_key, parent_sha)
            verify_run(parent_run)
        elif provenance.get("initial_checkpoint_sha256"):
            raise ValueError(f"{run} has an initializer hash without a declared edge")

        propensity_run = args.get("propensity")
        if propensity_run:
            propensity_sha = provenance.get("propensity_checkpoint_sha256")
            if not isinstance(propensity_sha, str):
                raise ValueError(f"{run} propensity edge lacks a checkpoint hash")
            edge(run, "propensity", str(propensity_run), "model.pt", propensity_sha)
            verify_run(str(propensity_run))
        elif provenance.get("propensity_checkpoint_sha256"):
            raise ValueError(f"{run} has a propensity hash without a declared edge")

        for source in args.get("distill_provenance") or []:
            parent_run = str(source["run"])
            parent_key = f"model_ep{int(source['epoch'])}.pt"
            edge(run, f"distillation_{source['split']}", parent_run, parent_key, str(source["checkpoint_sha256"]))
            verify_run(parent_run)

        resumed = provenance.get("resumed_from")
        resume_arg = args.get("resume_state")
        if resumed:
            state_path = _resolve(str(resumed["path"]))
            expected_parent_dir = state_path.parent
            if state_path.name != "training_state.pt" or expected_parent_dir.parent.name != "runs":
                raise ValueError(f"{run} resume state does not point to a source run")
            parent_run = expected_parent_dir.name
            _require(resume_arg, str(state_path), f"{run} declared resume state")
            state_sha = sha256_file(state_path)
            _require(state_sha, resumed.get("sha256"), f"{run} resume state payload")
            previous_epoch = int(resumed["next_epoch"]) - 1
            if previous_epoch < 0:
                raise ValueError(f"{run} resume epoch is invalid")
            parent_key = f"model_ep{previous_epoch}.pt"
            parent_provenance = _read_json(expected_parent_dir / "provenance.json")
            parent_sha = parent_provenance.get("checkpoint_sha256", {}).get(parent_key)
            if not isinstance(parent_sha, str):
                raise ValueError(f"{run} previous run lacks its resume checkpoint")
            edge(run, "resume_state", parent_run, parent_key, parent_sha)
            edges[-1]["state_path"] = str(state_path)
            edges[-1]["state_sha256"] = state_sha
            verify_run(parent_run)
        elif resume_arg:
            raise ValueError(f"{run} declares a resume state without its provenance edge")

        active.remove(run)

    root_parent = str(ancestry["root_parent"])
    verify_run(root_parent)
    cache_record = ancestry["distillation_cache"]
    cache_path = _resolve(cache_record["path"])
    _require(sha256_file(cache_path), cache_record["sha256"], "parent distillation cache metadata")
    cache = _read_json(cache_path)
    owner = _read_json(ROOT / "artifacts" / "runs" / root_parent / "provenance.json")
    _require(cache.get("split_manifest_sha256"), recipe["split_manifest"]["sha256"], "teacher cache fold1 split")
    cached_eval = [int(group) for group in cache.get("evaluation_valid_groups", [])]
    if len(cached_eval) != len(set(cached_eval)):
        raise ValueError("teacher cache repeats an evaluation valid group")
    _require(set(cached_eval), eval_set, "teacher cache eval membership")
    cache_owner = owner.get("distillation_cache") or {}
    _require(cache_owner.get("metadata_sha256"), cache_record["sha256"], "parent cache provenance link")
    _require(_resolve(cache_owner.get("root", "")) / "metadata.json", cache_path, "parent cache metadata path")
    cache_fit_groups = cache.get("fit_groups") or {}
    owner_fit_groups = owner.get("fit_row_groups") or {}
    for split_name in ("train", "valid"):
        cache_fit = [int(group) for group in cache_fit_groups.get(split_name, [])]
        owner_fit = [int(group) for group in owner_fit_groups.get(split_name, [])]
        _require(set(cache_fit), set(owner_fit), f"parent cache/owner {split_name} fit refs")
        if len(cache_fit) != len(set(cache_fit)):
            raise ValueError(f"parent cache repeats {split_name} fit groups")
        if not set(cache_fit).issubset(split_fit[split_name]):
            raise ValueError(f"parent cache {split_name} fit refs escape fold1's source namespace")
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

    coefficients = cache.get("source_coefficients") or []
    _require(
        normalize_coefficients(coefficients),
        normalize_coefficients(cache_owner.get("source_coefficients") or []),
        "cache/owner source coefficients",
    )
    sources = cache.get("sources") or []
    _require(len(sources), len(coefficients), "teacher cache source count")
    _require(len(sources), cache_record["source_count"], "frozen teacher cache source count")
    for source, coefficient in zip(sources, coefficients, strict=True):
        run = str(coefficient["run"])
        epoch = int(coefficient["epoch"])
        pin = pins[run]
        reachable.add(run)
        verify_run(run)
        _require(pin["checkpoint_key"], f"model_ep{epoch}.pt", f"{run} cache checkpoint key")
        _require(source.get("run"), run, "teacher cache run")
        _require(int(source.get("epoch", -1)), epoch, f"{run} cache epoch")
        _require(source.get("checkpoint_sha256"), pin["checkpoint_sha256"], f"{run} cache checkpoint")
        _require(
            _resolve(source.get("checkpoint_path", "")),
            (ROOT / "artifacts" / "runs" / run / pin["checkpoint_key"]).resolve(),
            f"{run} cache checkpoint path",
        )
        _require(
            _resolve(source.get("provenance_path", "")),
            (ROOT / "artifacts" / "runs" / run / "provenance.json").resolve(),
            f"{run} cache provenance path",
        )
        source_provenance = _read_json(ROOT / "artifacts" / "runs" / run / "provenance.json")
        source_fit = source_provenance.get("fit_row_groups") or {}
        for split_name in ("train", "valid"):
            observed_groups = [int(group) for group in source_fit.get(split_name, [])]
            _require(
                source.get("fit_group_counts", {}).get(split_name),
                len(observed_groups),
                f"{run} cache {split_name} fit count",
            )
            _require(
                source.get("fit_row_group_hashes", {}).get(split_name),
                _ordered_int64_sha256(observed_groups),
                f"{run} cache {split_name} fit membership digest",
            )
        predictions = source.get("predictions") or {}
        for split_name in ("train", "valid"):
            prediction_record = predictions.get(split_name) or {}
            prediction_sidecar_path = Path(prediction_record.get("prediction_path", "")).with_suffix(".json")
            prediction_sidecar = _read_json(prediction_sidecar_path)
            _require(
                sha256_file(prediction_sidecar_path),
                prediction_record.get("sidecar_sha256"),
                f"{run} {split_name} teacher prediction sidecar hash",
            )
            for field, expected in (
                ("run", run),
                ("split", split_name),
                ("epoch", epoch),
                ("checkpoint_sha256", pin["checkpoint_sha256"]),
                ("prediction_sha256", prediction_record.get("prediction_sha256")),
                ("split_manifest_sha256", recipe["split_manifest"]["sha256"]),
            ):
                _require(prediction_sidecar.get(field), expected, f"{run} {split_name} teacher sidecar {field}")
            source_groups = [int(row["row_group"]) for row in split["sequences"][split_name]]
            sidecar_groups = [int(group) for group in prediction_sidecar.get("row_groups", [])]
            _require(sidecar_groups, source_groups, f"{run} {split_name} teacher source-qualified row order")
            _require(
                prediction_record.get("row_groups_sha256"),
                _ordered_int64_sha256(sidecar_groups),
                f"{run} {split_name} teacher row-group digest",
            )
            _require(prediction_record.get("epoch"), epoch, f"{run} {split_name} teacher prediction epoch")
            _require(prediction_record.get("checkpoint_sha256"), pin["checkpoint_sha256"], f"{run} {split_name} teacher prediction checkpoint")
        edges.append(
            {
                "child": root_parent,
                "type": "distillation_cache_source",
                "parent": run,
                "checkpoint_key": pin["checkpoint_key"],
                "checkpoint_sha256": pin["checkpoint_sha256"],
            }
        )
    if reachable != set(pins):
        raise ValueError(f"F1 ancestry contains unreachable or undeclared runs: {sorted(set(pins) ^ reachable)}")
    return {
        "root_parent": root_parent,
        "evaluation_group_count": len(eval_set),
        "runs": [verified[name] for name in sorted(verified)],
        "edges": edges,
        "distillation_cache": {"path": str(cache_path), "sha256": cache_record["sha256"], "source_count": len(sources)},
        "fit_row_namespace": "each train/valid membership was checked against fold1's matching source namespace",
        "teacher_prediction_scope_contract": "sidecars bind run, checkpoint, split and complete row order; producer metadata has no explicit no-target-read/scope field",
        "teacher_prediction_payload_arrays_rehashed": False,
    }


def _verify_f1_metadata(
    recipe: dict[str, Any],
    subset: dict[str, Any],
    groups: list[int],
    refs: list[dict[str, int | str]],
) -> dict[str, Any]:
    """Check the candidate, parent cache, normalizer, and source-qualified fit set."""
    inputs = recipe["inputs"]
    candidate = inputs["candidate"]
    parent = inputs["parent"]
    sidecar = _read_json(_resolve(candidate["sidecar_path"]))
    provenance = _read_json(_resolve(candidate["provenance_path"]))
    parent_sidecar = _read_json(_resolve(parent["sidecar_path"]))
    eval_cache = _read_json(_resolve(recipe["cache_metadata"]["evaluation_path"]))
    fit_cache = _read_json(_resolve(recipe["cache_metadata"]["fit_path"]))
    normalizer = _read_json(_resolve(recipe["cache_metadata"]["normalizer_path"]))
    _require(sha256_file(_resolve(candidate["sidecar_path"])), candidate["sidecar_sha256"], "candidate sidecar file")
    _require(sha256_file(_resolve(candidate["provenance_path"])), candidate["provenance_sha256"], "candidate provenance file")
    _require(sha256_file(_resolve(parent["sidecar_path"])), parent["sidecar_sha256"], "parent prediction sidecar file")
    for key in ("evaluation", "fit"):
        _require(
            sha256_file(_resolve(recipe["cache_metadata"][f"{key}_path"])),
            recipe["cache_metadata"][f"{key}_sha256"],
            f"parent {key} cache sidecar file",
        )
    _require(
        sha256_file(_resolve(recipe["cache_metadata"]["normalizer_path"])),
        recipe["cache_metadata"]["normalizer_sidecar_sha256"],
        "relationship normalizer sidecar file",
    )

    for field in ("fold", "seed", "lag", "arm", "run", "epoch", "shape", "dtype", "sha256"):
        _require(sidecar.get(field), candidate[field], f"candidate sidecar {field}")
    _require(sidecar.get("role"), "online_residual_evaluation", "candidate sidecar role")
    _require(sidecar.get("recipe_sha256"), recipe["base_recipe"]["sha256"], "candidate recipe")
    _require(sidecar.get("source_manifest_sha256"), recipe["split_manifest"]["sha256"], "candidate split source")
    _require(sidecar.get("groups"), refs, "candidate source-qualified group order")
    _require(sidecar.get("parent_cache_manifest_sha256"), recipe["cache_metadata"]["parent_prediction_sha256"], "candidate parent cache")
    _require(sidecar.get("provenance_sha256"), candidate["provenance_sha256"], "candidate provenance link")
    _require(sidecar.get("checkpoint"), {"path": candidate["checkpoint_path"], "sha256": candidate["checkpoint_sha256"]}, "candidate checkpoint link")

    _require(provenance.get("schema"), "wnn.online-relationships.readout-provenance.v1", "candidate provenance schema")
    for field, value in (("run", candidate["run"]), ("fold", 1), ("seed", 1002), ("lag", 1), ("arm", "A"), ("epoch", 2)):
        _require(provenance.get(field), value, f"candidate provenance {field}")
    _require(provenance.get("recipe"), {"path": recipe["base_recipe"]["absolute_path"], "sha256": recipe["base_recipe"]["sha256"]}, "candidate provenance recipe")
    _require(provenance.get("source_manifest"), {"path": recipe["split_manifest"]["absolute_path"], "sha256": recipe["split_manifest"]["sha256"]}, "candidate provenance split")
    _require(provenance.get("prediction"), {"path": candidate["prediction_path"], "sha256": candidate["prediction_sha256"], "sidecar": candidate["sidecar_path"]}, "candidate prediction provenance")
    _require(provenance.get("checkpoint"), {"epoch": 2, "path": candidate["checkpoint_path"], "sha256": candidate["checkpoint_sha256"]}, "candidate checkpoint provenance")
    _require(provenance.get("parent", {}).get("run"), parent["run"], "candidate parent run")
    _require(provenance.get("parent", {}).get("checkpoint_sha256"), parent["checkpoint_sha256"], "candidate parent checkpoint")
    _require(provenance.get("parent", {}).get("provenance_sha256"), parent["provenance_sha256"], "candidate parent provenance")
    _require(provenance.get("parent", {}).get("historical_precision"), "fp16_amp", "parent historical precision")
    _require(provenance.get("parent", {}).get("endpoint_precision"), "strict_fp32", "parent endpoint precision")
    _require(provenance.get("parent", {}).get("endpoint_backend"), "cpu", "parent endpoint backend")

    fit_groups = provenance.get("fit_membership", {}).get("groups")
    if not isinstance(fit_groups, list):
        raise ValueError("candidate source-qualified fit membership is absent")
    normalized_fit = [{"split": str(row["split"]), "row_group": int(row["row_group"])} for row in fit_groups]
    subset_fit_refs = [
        {"split": split_name, "row_group": int(group)}
        for split_name in ("train", "valid")
        for group in subset["selected_groups"][split_name]
    ]
    _require(normalized_fit, subset_fit_refs, "candidate/subset fit source order")
    cache_fit_refs = fit_cache.get("row_refs")
    _require(normalized_fit, cache_fit_refs, "candidate/cache fit source order")
    _require(provenance["fit_membership"].get("membership_sha256"), recipe["fit_membership"]["sha256"], "candidate fit membership hash")
    _require(fit_cache.get("membership_sha256"), recipe["fit_membership"]["sha256"], "fit cache membership hash")
    if len(normalized_fit) != recipe["fit_membership"]["count"]:
        raise ValueError("F1 fit subset is not the frozen 1,243 source-qualified groups")
    eval_ref_set = {(ref["split"], ref["row_group"]) for ref in refs}
    fit_ref_set = {(ref["split"], ref["row_group"]) for ref in normalized_fit}
    if eval_ref_set & fit_ref_set:
        raise ValueError("source-qualified readout fit membership overlaps evaluation")

    _require(parent_sidecar.get("row_refs"), refs, "parent prediction source-qualified order")
    _require(parent_sidecar.get("membership_sha256"), recipe["evaluation"]["row_refs_sha256"], "parent prediction membership")
    _require(parent_sidecar.get("prediction_sha256"), parent["prediction_sha256"], "parent prediction payload pin")
    _require(parent_sidecar.get("shape"), PREDICTION_SHAPE, "parent prediction shape")
    _require(parent_sidecar.get("dtype"), PREDICTION_DTYPE, "parent prediction dtype")
    _require(parent_sidecar.get("endpoint"), {"backend": "cpu", "precision": "strict_fp32"}, "parent endpoint execution")
    _require(parent_sidecar.get("split_manifest_sha256"), recipe["split_manifest"]["sha256"], "parent prediction split")
    _require(parent_sidecar.get("parent", {}).get("checkpoint_sha256"), parent["checkpoint_sha256"], "parent prediction checkpoint")
    _require(parent_sidecar.get("parent", {}).get("provenance_sha256"), parent["provenance_sha256"], "parent prediction provenance")
    _require(parent_sidecar.get("prediction_path"), parent["prediction_path"], "parent prediction path")
    _require(parent_sidecar.get("parent", {}).get("run"), parent["run"], "parent prediction run")
    _require(parent_sidecar.get("parent", {}).get("epoch"), parent["epoch"], "parent prediction epoch")
    _require(parent_sidecar.get("parent", {}).get("mean_identity"), recipe["parent_normalization"]["mean_sha256"], "parent mean identity")
    _require(parent_sidecar.get("parent", {}).get("std_identity"), recipe["parent_normalization"]["std_sha256"], "parent std identity")

    for cache, expected_scope, expected_shape, expected_read, expected_zero in (
        (eval_cache, "feature_only_eval", [457, ROWS, 117], {"targets": False, "scored_mask": False}, [112, 113, 114]),
        (fit_cache, "in_sample_fit", [1243, ROWS, 117], {"targets": True, "scored_mask": True}, []),
    ):
        _require(cache.get("schema"), "wnn.online-relationships.sequence-cache.v1", "parent cache schema")
        _require(cache.get("scope"), expected_scope, "parent cache scope")
        _require(cache.get("fold"), 1, "parent cache fold")
        _require(cache.get("seed"), 1002, "parent cache seed")
        _require(cache.get("recipe_sha256"), recipe["base_recipe"]["sha256"], "parent cache recipe")
        _require(cache.get("split_manifest_sha256"), recipe["split_manifest"]["sha256"], "parent cache split")
        _require(cache.get("subset_manifest_sha256"), recipe["subset_manifest"]["sha256"], "parent cache subset")
        _require(cache.get("membership_sha256"), recipe["evaluation"]["row_refs_sha256"] if expected_scope == "feature_only_eval" else recipe["fit_membership"]["sha256"], "parent cache membership")
        _require(cache.get("shape"), expected_shape, "parent cache shape")
        _require(cache.get("dtype"), "<f4", "parent cache dtype")
        _require(cache.get("read_contract"), expected_read, "parent cache read contract")
        _require(cache.get("zero_columns"), expected_zero, "parent cache zeroed columns")
        _require(cache.get("complete_membership"), True, "parent cache completeness")
        _require(cache.get("sampled_profile"), False, "parent cache sampling")
        cache_parent = cache.get("parent") or {}
        for field, expected in (
            ("run", parent["run"]),
            ("epoch", parent["epoch"]),
            ("checkpoint_sha256", parent["checkpoint_sha256"]),
            ("provenance_sha256", parent["provenance_sha256"]),
            ("endpoint_precision", "strict_fp32"),
            ("endpoint_backend", "cpu"),
            ("mean_sha256", recipe["parent_normalization"]["mean_sha256"]),
            ("std_sha256", recipe["parent_normalization"]["std_sha256"]),
            ("input_width", 211),
        ):
            _require(cache_parent.get(field), expected, f"{expected_scope} parent {field}")
    _require(eval_cache.get("row_refs"), refs, "parent evaluation cache group order")
    _require(fit_cache.get("row_refs"), normalized_fit, "parent fit cache source order")
    _require(normalizer.get("normalizer_sha256"), recipe["normalizer"]["sha256"], "relationship normalizer")
    _require(provenance.get("normalization", {}).get("relationship_normalizer_sha256"), recipe["normalizer"]["sha256"], "candidate normalizer link")
    _require(sidecar.get("normalizer_sha256"), recipe["normalizer"]["sha256"], "candidate sidecar normalizer")
    _require(provenance.get("propensity", {}).get("run"), recipe["propensity"]["run"], "candidate propensity source")
    _require(provenance.get("propensity", {}).get("checkpoint"), recipe["propensity"]["checkpoint_path"], "candidate propensity checkpoint path")
    _require(provenance.get("propensity", {}).get("provenance"), recipe["propensity"]["provenance_path"], "candidate propensity provenance path")
    _require(provenance.get("propensity", {}).get("checkpoint_sha256"), recipe["propensity"]["checkpoint_sha256"], "candidate propensity checkpoint")
    _require(provenance.get("propensity", {}).get("provenance_sha256"), recipe["propensity"]["provenance_sha256"], "candidate propensity provenance")
    candidate_sources = provenance.get("source_sha256")
    _require(candidate_sources, recipe["candidate_source_hashes"], "candidate source hash map")
    for source_path, expected_sha in candidate_sources.items():
        _require(sha256_file(_resolve(source_path)), expected_sha, f"candidate implementation source {source_path}")

    return {
        "candidate_sidecar_sha256": sha256_file(_resolve(candidate["sidecar_path"])),
        "candidate_provenance_sha256": sha256_file(_resolve(candidate["provenance_path"])),
        "parent_sidecar_sha256": sha256_file(_resolve(parent["sidecar_path"])),
        "eval_cache_sidecar_sha256": sha256_file(_resolve(recipe["cache_metadata"]["evaluation_path"])),
        "fit_cache_sidecar_sha256": sha256_file(_resolve(recipe["cache_metadata"]["fit_path"])),
        "normalizer_sidecar_sha256": sha256_file(_resolve(recipe["cache_metadata"]["normalizer_path"])),
        "fit_group_count": len(normalized_fit),
        "fit_source_qualified_overlap": 0,
    }


def prepare_f1_grade(recipe_path: Path, expected_recipe_sha256: str) -> PreparedF1Grade:
    """Validate every small source pin, ancestry edge, prediction header, and digest."""
    recipe_path = recipe_path.resolve()
    recipe_sha256 = sha256_file(recipe_path)
    _require(recipe_sha256, expected_recipe_sha256, "recipe SHA-256")
    recipe = _read_json(recipe_path)
    _require(recipe.get("schema"), "wnn.correction-shrink-f1-grade-recipe.v1", "recipe schema")
    _require(recipe.get("new_fit_count"), 0, "new fit count")
    _source_manifest(recipe)

    base = recipe["base_recipe"]
    subset_record = recipe["subset_manifest"]
    split_record = recipe["split_manifest"]
    base_path, subset_path, split_path = map(_resolve, (base["path"], subset_record["path"], split_record["path"]))
    _require(sha256_file(base_path), base["sha256"], "historical execution recipe")
    _require(sha256_file(subset_path), subset_record["sha256"], "F1 subset manifest")
    _require(sha256_file(split_path), split_record["sha256"], "F1 fold split")
    base_recipe = _read_json(base_path)
    _require(base_recipe.get("recipe_id"), "E-ONLINE-RELATIONSHIPS-20261022-R1", "historical recipe identity")
    _require(base_recipe.get("readout", {}).get("precision"), "strict FP32, autocast and TF32 off", "readout precision contract")
    _require(base_recipe.get("execution_actual_cpu_backend", {}).get("precision"), "strict_fp32", "actual candidate execution precision")
    _require(base_recipe.get("target_cache", {}).get("sha256"), recipe["target_cache"]["sha256"], "base recipe target identity")
    _require(_resolve(base_recipe.get("target_cache", {}).get("path", "")), _resolve(recipe["target_cache"]["path"]), "base recipe target path")
    subset, split = _read_json(subset_path), _read_json(split_path)
    groups, components, refs = _verify_refs_and_components(recipe, subset, split)
    metadata_report = _verify_f1_metadata(recipe, subset, groups, refs)
    ancestry = _verify_f1_ancestry(recipe, split, groups)

    inputs = recipe["inputs"]
    candidate, parent = inputs["candidate"], inputs["parent"]
    candidate_path, parent_path = _resolve(candidate["prediction_path"]), _resolve(parent["prediction_path"])
    _require(sha256_file(_resolve(candidate["checkpoint_path"])), candidate["checkpoint_sha256"], "candidate checkpoint payload")
    candidate_coverage = npy_coverage(candidate_path)
    parent_coverage = npy_coverage(parent_path)
    for coverage, expected_sha, expected_bytes, label in (
        (candidate_coverage, candidate["prediction_sha256"], candidate["prediction_bytes"], "candidate"),
        (parent_coverage, parent["prediction_sha256"], parent["prediction_bytes"], "parent"),
    ):
        _require(coverage["shape"], PREDICTION_SHAPE, f"{label} NPY shape")
        _require(coverage["dtype"], PREDICTION_DTYPE, f"{label} NPY dtype")
        _require(coverage["file_bytes"], expected_bytes, f"{label} NPY size")
        _require(sha256_file(candidate_path if label == "candidate" else parent_path), expected_sha, f"{label} prediction payload SHA-256")

    target_path = _resolve(recipe["target_cache"]["path"])
    _require(target_path.stat().st_size, recipe["target_cache"]["file_bytes"], "target archive size")
    target_sha = sha256_file(target_path)
    _require(target_sha, recipe["target_cache"]["sha256"], "target archive identity")
    report = {
        "schema": "wnn.correction-shrink-f1-preflight.v1",
        "status": "label_free_preflight_passed",
        "recipe_sha256": recipe_sha256,
        "source_manifest_sha256": recipe["source_manifest"]["sha256"],
        "base_recipe_sha256": base["sha256"],
        "subset_manifest_sha256": subset_record["sha256"],
        "split_manifest_sha256": split_record["sha256"],
        "evaluation": {
            "fold": 1,
            "seed": 1002,
            "group_count": len(groups),
            "row_refs_sha256": recipe["evaluation"]["row_refs_sha256"],
            "component_count": len(set(components)),
            "component_membership_sha256": recipe["evaluation"]["component_membership_sha256"],
        },
        "fit_membership": metadata_report,
        "ancestry": ancestry,
        "candidate_prediction": {**candidate_coverage, "sha256": candidate["prediction_sha256"]},
        "parent_prediction": {**parent_coverage, "sha256": parent["prediction_sha256"]},
        "target_cache": {
            "path": str(target_path),
            "file_bytes": target_path.stat().st_size,
            "sha256": target_sha,
            "target_values_read": False,
        },
        "prediction_values_read": False,
        "metadata": metadata_report,
    }
    return PreparedF1Grade(
        recipe=recipe,
        recipe_sha256=recipe_sha256,
        evaluation_groups=groups,
        components=components,
        parent_prediction_path=parent_path,
        candidate_prediction_path=candidate_path,
        target_cache_path=target_path,
        report=report,
    )

