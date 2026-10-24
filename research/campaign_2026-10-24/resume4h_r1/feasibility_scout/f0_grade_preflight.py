"""Validate the F0 exploratory grade inputs without opening targets."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any  # Recipe and provenance JSON use runtime-defined fields.

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
ONLINE_DIR = ROOT / "research/experiment_drafts/online-relationships-20261022"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ONLINE_DIR))

import numpy as np  # noqa: E402
from f0_grade_ancestry import verify_parent_ancestry  # noqa: E402
from online_relationship_grade_contract import (  # noqa: E402
    ROW_WEIGHT_CONTRACT,
    ROWS,
    ParentCacheArtifacts,
    _require_sha,
    _resolve,
    read_json,
    row_refs_sha256,
)
from online_relationship_grade_core import npy_coverage, sha256_file  # noqa: E402
from online_relationship_grade_parent import verify_parent_prediction_metadata  # noqa: E402
from online_relationship_grade_provenance import verify_candidate_provenance  # noqa: E402

from wnn.splits import load_campaign_split  # noqa: E402
from wnn.training.membership import load_component_fit_subset  # noqa: E402


@dataclass(frozen=True)
class PreparedF0Grade:
    """Validated paths and source-qualified F0 membership for later grading."""

    recipe: dict[str, Any]
    parent_recipe: dict[str, Any]
    fold: dict[str, Any]
    cells: dict[str, dict[str, Any]]
    parent_prediction_path: Path
    candidate_prediction_paths: dict[str, Path]
    evaluation_groups: list[int]
    components: np.ndarray
    target_cache_path: Path
    report: dict[str, Any]
    ancestry: dict[str, Any]


def _verify_source_manifest(recipe: dict[str, Any]) -> dict[str, Any]:
    source_record = recipe["source_manifest"]
    path = _resolve(ROOT, source_record["path"])
    _require_sha(path, source_record["sha256"], "F0 grader source manifest")
    manifest = read_json(path)
    for source_path, expected_sha in manifest["files"].items():
        _require_sha(_resolve(ROOT, source_path), expected_sha, f"grader source {source_path}")
    for record in manifest["evidence"].values():
        _require_sha(_resolve(ROOT, record["path"]), record["sha256"], "source receipt")
    return manifest


def _verify_historical_source_pins(
    terminal: dict[str, Any],
    collection: dict[str, Any],
    *,
    admission_sha256: str,
    recipe_sha256: str,
) -> dict[str, Any]:
    summary = terminal["source_validation"]
    pins = collection["source_pins"]
    checked: dict[str, Any] = {}
    for role in ("runtime", "grader"):
        record = pins.get(role)
        if not isinstance(record, dict):
            raise ValueError(f"historical {role} source-pin map is absent")
        expected = record.get("expected_sha256")
        observed = record.get("observed_sha256")
        if not isinstance(expected, dict) or not isinstance(observed, dict):
            raise ValueError(f"historical {role} source-pin map is malformed")
        if record.get("mismatches") or summary.get("mismatches", {}).get(role):
            raise ValueError(f"historical {role} source pins contain mismatches")
        if expected != observed:
            raise ValueError(f"historical {role} source receipt is internally inconsistent")
        expected_count = int(summary.get(f"{role}_pin_count", -1))
        if expected_count != len(expected):
            raise ValueError(f"historical {role} source-pin count differs from terminal receipt")
        for relative_path, expected_sha in expected.items():
            _require_sha(ROOT / relative_path, expected_sha, f"historical {role} source")
        checked[role] = {"count": len(expected), "sha256": dict(sorted(expected.items()))}
    if summary.get("admission_and_execution_recipe_hashes_agree") is not True:
        raise ValueError("historical terminal receipt reports an admission/recipe mismatch")
    if collection["source_pins"].get("admission_sha256") != admission_sha256:
        raise ValueError("historical source-pin inventory admission hash differs")
    if collection.get("admission_receipt", {}).get("sha256") != admission_sha256:
        raise ValueError("historical terminal inventory admission receipt hash differs")
    if collection.get("recipe", {}).get("sha256") != recipe_sha256:
        raise ValueError("historical source-pin inventory recipe hash differs")
    return checked


def _verify_cache_metadata(
    metadata: dict[str, Any],
    *,
    scope: str,
    refs: list[dict[str, Any]],
    recipe: dict[str, Any],
    recipe_sha: str,
    fold: dict[str, Any],
    subset: dict[str, Any],
    path: Path,
) -> None:
    expected = {
        "schema": "wnn.online-relationships.sequence-cache.v1",
        "scope": scope,
        "fold": 0,
        "seed": 1001,
        "recipe_id": recipe["recipe_id"],
        "recipe_sha256": recipe_sha,
        "split_manifest_sha256": fold["split_manifest_sha256"],
        "subset_manifest_sha256": fold["subset"]["sha256"],
        "subset_members_sha256": subset["selected_members_sha256"],
        "row_refs": refs,
        "membership_sha256": row_refs_sha256(refs),
        "complete_membership": True,
        "scope_label": scope,
        "data_file": path.name,
        "shape": [len(refs), ROWS, 117],
        "dtype": "<f4",
        "sampled_profile": False,
        "split_manifest": str(_resolve(ROOT, fold["split_manifest"])),
        "subset_manifest": fold["subset"]["path"],
    }
    for key, expected_value in expected.items():
        if metadata.get(key) != expected_value:
            raise ValueError(f"F0 {scope} cache sidecar field {key} differs from the recipe")
    if not isinstance(metadata.get("data_sha256"), str):
        raise ValueError(f"F0 {scope} cache sidecar lacks its payload digest")
    parent = fold["parent"]
    expected_parent = {
        "run": parent["run"],
        "epoch": parent["epoch"],
        "checkpoint": str(_resolve(ROOT, parent["checkpoint"])),
        "checkpoint_sha256": parent["checkpoint_sha256"],
        "provenance": str(_resolve(ROOT, parent["provenance"])),
        "provenance_sha256": parent["provenance_sha256"],
        "mean_sha256": parent["mean_identity"],
        "std_sha256": parent["std_identity"],
        "input_width": parent["input_width"],
    }
    if any(metadata.get("parent", {}).get(key) != value for key, value in expected_parent.items()):
        raise ValueError(f"F0 {scope} cache parent identity differs from the frozen recipe")
    if scope == "in_sample_fit":
        if metadata.get("read_contract") != {"targets": True, "scored_mask": True}:
            raise ValueError("F0 fit cache target-read contract is missing")
        if metadata.get("row_weight_contract") != ROW_WEIGHT_CONTRACT:
            raise ValueError("F0 fit cache row-weight contract differs from the frozen recipe")
        if metadata.get("row_weight_batch_semantics") != (
            "one complete sequence per main_row_weights call; true-mask mass scale is sequence-local"
        ):
            raise ValueError("F0 fit cache row-weight semantics differ")
    else:
        if metadata.get("read_contract") != {"targets": False, "scored_mask": False}:
            raise ValueError("F0 evaluation cache may contain target or mask reads")
        if metadata.get("zero_columns") != [112, 113, 114]:
            raise ValueError("F0 evaluation cache does not zero labels and weights")
    if not metadata.get("source_sha256"):
        raise ValueError(f"F0 {scope} cache sidecar lacks source hashes")


def _parent_identity(
    root_recipe: dict[str, Any],
    recipe_sha: str,
    fold: dict[str, Any],
    subset: dict[str, Any],
    eval_refs: list[dict[str, Any]],
    parent_meta: dict[str, Any],
    parent_meta_path: Path,
    parent_prediction_path: Path,
    fit_meta: dict[str, Any],
    fit_meta_path: Path,
    eval_meta: dict[str, Any],
    eval_meta_path: Path,
    parent_prediction_sha: str,
) -> dict[str, Any]:
    cache_dir = Path(fold["parent_cache"]).resolve()
    return {
        "parent_run": fold["parent"]["run"],
        "parent_epoch": fold["parent"]["epoch"],
        "parent_checkpoint_sha256": fold["parent"]["checkpoint_sha256"],
        "parent_provenance_sha256": fold["parent"]["provenance_sha256"],
        "fit_cache_sha256": fit_meta["data_sha256"],
        "fit_cache_sidecar_sha256": sha256_file(fit_meta_path),
        "evaluation_cache_sha256": eval_meta["data_sha256"],
        "evaluation_cache_sidecar_sha256": sha256_file(eval_meta_path),
        "evaluation_prediction_sha256": parent_prediction_sha,
        "evaluation_prediction_sidecar_sha256": sha256_file(parent_meta_path),
        "evaluation_prediction_path": str(parent_prediction_path),
        "evaluation_prediction_sidecar_path": str(parent_meta_path),
        "parent_endpoint_precision": eval_meta["parent"]["endpoint_precision"],
        "parent_endpoint_backend": eval_meta["parent"]["endpoint_backend"],
        "evaluation_refs_sha256": row_refs_sha256(eval_refs),
        "evaluation_groups": [int(ref["row_group"]) for ref in eval_refs],
        "cache_source_sha256": dict(sorted(eval_meta["source_sha256"].items())),
        "cache_directory": str(cache_dir),
        "recipe_sha256": recipe_sha,
    }


def _verify_candidate(
    cell_record: dict[str, Any],
    base_cell: dict[str, Any],
    recipe: dict[str, Any],
    recipe_path: Path,
    recipe_sha: str,
    fold: dict[str, Any],
    subset: dict[str, Any],
    parent_cache: ParentCacheArtifacts,
    parent_cache_manifest: Path,
    split_path: Path,
    groups: list[int],
) -> tuple[Path, dict[str, Any]]:
    run = str(base_cell["run"])
    if str(cell_record.get("run")) != run:
        raise ValueError(f"{run}: F0 recipe run identity differs from the historical cell")
    if _resolve(ROOT, str(cell_record.get("output"))) != _resolve(ROOT, str(base_cell["output"])):
        raise ValueError(f"{run}: F0 recipe output directory differs from the historical cell")
    output_dir = _resolve(ROOT, base_cell["output"])
    prediction_path = output_dir / "prediction.npy"
    sidecar_path = output_dir / "prediction.json"
    provenance_path = output_dir / "provenance.json"
    checkpoint_path = output_dir / "model_ep2.pt"
    sidecar = read_json(sidecar_path)
    provenance = read_json(provenance_path)
    checkpoint_sha = sha256_file(checkpoint_path)
    provenance_sha = sha256_file(provenance_path)
    sidecar_sha = sha256_file(sidecar_path)
    expected_hashes = {
        "prediction_sha256": sidecar.get("sha256"),
        "sidecar_sha256": sidecar_sha,
        "provenance_sha256": provenance_sha,
        "checkpoint_sha256": checkpoint_sha,
    }
    for key, actual in expected_hashes.items():
        if actual != cell_record[key]:
            raise ValueError(f"{run}: {key} differs from the F0 exploratory recipe")
    refs = [{"split": "valid", "row_group": int(group)} for group in groups]
    expected_sidecar = {
        "schema_version": 1,
        "role": "online_residual_evaluation",
        "run": run,
        "fold": 0,
        "seed": 1001,
        "lag": int(base_cell["lag"]),
        "arm": str(base_cell["arm"]),
        "epoch": 2,
        "recipe_sha256": recipe_sha,
        "source_manifest_sha256": fold["split_manifest_sha256"],
        "checkpoint": {"path": str(checkpoint_path), "sha256": checkpoint_sha},
        "parent_cache_manifest_sha256": sha256_file(parent_cache_manifest),
        "groups": refs,
        "shape": [len(groups), ROWS, 2],
        "dtype": "<f4",
        "provenance_sha256": provenance_sha,
    }
    for key, expected in expected_sidecar.items():
        if sidecar.get(key) != expected:
            raise ValueError(f"{run}: prediction sidecar field {key} differs")
    normalizer_sha = sidecar.get("normalizer_sha256")
    if not isinstance(normalizer_sha, str):
        raise ValueError(f"{run}: prediction sidecar lacks its normalizer hash")
    coverage = npy_coverage(prediction_path)
    if coverage["shape"] != [len(groups), ROWS, 2] or coverage["dtype"] != "<f4":
        raise ValueError(f"{run}: prediction header differs from the full F0 endpoint")
    if coverage["file_bytes"] != cell_record["prediction_bytes"]:
        raise ValueError(f"{run}: prediction length differs from the frozen recipe")
    prediction_sha = sha256_file(prediction_path)
    if prediction_sha != cell_record["prediction_sha256"]:
        raise ValueError(f"{run}: prediction payload digest differs from the frozen recipe")
    verify_candidate_provenance(
        ROOT,
        recipe,
        recipe_path,
        recipe_sha,
        fold,
        base_cell,
        subset,
        parent_cache,
        parent_cache_manifest,
        provenance,
        provenance_path,
        provenance_sha,
        checkpoint_path,
        checkpoint_sha,
        prediction_path,
        sidecar_path,
        prediction_sha,
        normalizer_sha,
        refs,
        split_path,
    )
    return prediction_path, {
        "run": run,
        "lag": int(base_cell["lag"]),
        "arm": str(base_cell["arm"]),
        "prediction_path": str(prediction_path),
        "prediction_sha256": prediction_sha,
        "prediction_sidecar_sha256": sidecar_sha,
        "provenance_sha256": provenance_sha,
        "checkpoint_sha256": checkpoint_sha,
        "shape": coverage["shape"],
        "dtype": coverage["dtype"],
        "file_bytes": coverage["file_bytes"],
        "source_qualified_groups": len(refs),
        "evaluation_membership_sha256": row_refs_sha256(refs),
    }


def prepare_f0_grade(recipe_path: Path, expected_sha256: str) -> PreparedF0Grade:
    """Verify all seven fixed endpoints and ancestry without reading labels."""
    recipe_path = recipe_path.resolve()
    recipe = read_json(recipe_path)
    recipe_sha = sha256_file(recipe_path)
    if recipe_sha != expected_sha256:
        raise ValueError("F0 exploratory recipe differs from the supplied SHA256")
    if recipe.get("schema") != "wnn.online-relationships-f0-exploratory.v1":
        raise ValueError("unsupported F0 exploratory grade recipe")
    if recipe.get("status") != "proposed_unadmitted" or recipe.get("new_fit_count") != 0:
        raise ValueError("F0 preflight only accepts the frozen unadmitted zero-fit recipe")
    source_manifest = _verify_source_manifest(recipe)
    parent_record = recipe["parent_recipe"]
    parent_recipe_path = _resolve(ROOT, parent_record["path"])
    _require_sha(parent_recipe_path, parent_record["sha256"], "historical online recipe")
    parent_recipe = read_json(parent_recipe_path)
    terminal_record = recipe["terminal_receipt"]
    terminal_path = _resolve(ROOT, terminal_record["path"])
    _require_sha(terminal_path, terminal_record["sha256"], "historical online terminal receipt")
    terminal = read_json(terminal_path)
    inventory_record = source_manifest["evidence"]["terminal_inventory"]
    inventory_path = _resolve(ROOT, inventory_record["path"])
    _require_sha(inventory_path, inventory_record["sha256"], "historical terminal inventory")
    if terminal.get("inventory", {}).get("path") != inventory_record["path"]:
        raise ValueError("historical terminal receipt points at a different source-pin inventory")
    if terminal.get("inventory", {}).get("sha256") != inventory_record["sha256"]:
        raise ValueError("historical terminal receipt inventory hash differs")
    collection = read_json(inventory_path)
    historical_source_pins = _verify_historical_source_pins(
        terminal,
        collection,
        admission_sha256=source_manifest["evidence"]["historical_admission"]["sha256"],
        recipe_sha256=source_manifest["evidence"]["historical_recipe"]["sha256"],
    )

    fold = next((item for item in parent_recipe["folds"] if int(item["fold"]) == 0), None)
    if fold is None or int(fold["seed"]) != 1001:
        raise ValueError("historical recipe has no F0/seed1001 cell")
    base_cells = {
        (int(cell["lag"]), str(cell["arm"])): cell
        for cell in parent_recipe["cells"]
        if int(cell["fold"]) == 0
    }
    recipe_cells = recipe.get("cells", [])
    expected_grid = {(lag, arm) for lag in (1, 16) for arm in ("A", "B", "C")}
    actual_grid = {(int(cell["lag"]), str(cell["arm"])) for cell in recipe_cells}
    if len(recipe_cells) != 6 or actual_grid != expected_grid or len(actual_grid) != 6:
        raise ValueError("F0 recipe must freeze exactly lag1/16 × A/B/C saved heads")
    if any(cell.get("fold") != 0 or cell.get("seed") != 1001 for cell in recipe_cells):
        raise ValueError("F0 recipe cells must retain the original fold and seed")

    split_path = _resolve(ROOT, fold["split_manifest"])
    _require_sha(split_path, fold["split_manifest_sha256"], "F0 campaign split")
    subset_path = _resolve(ROOT, fold["subset"]["path"])
    _require_sha(subset_path, fold["subset"]["sha256"], "F0 fit subset")
    campaign = load_campaign_split(split_path, ROOT)
    subset_result = load_component_fit_subset(
        subset_path, campaign, expected_experiment_id=parent_recipe["experiment_id"]
    )
    subset = subset_result["manifest"]
    groups = [int(group) for group in fold["evaluation_groups"]]
    if len(groups) != 468 or list(campaign.evaluation_groups) != groups:
        raise ValueError(
            "F0 evaluation membership or order differs from the source-qualified split"
        )
    eval_refs = [{"split": "valid", "row_group": group} for group in groups]
    fit_refs = [
        {"split": split_name, "row_group": int(group)}
        for split_name in ("train", "valid")
        for group in subset["selected_groups"][split_name]
    ]
    split = read_json(split_path)
    component_by_group = {
        int(record["row_group"]): int(record["component"]) for record in split["sequences"]["valid"]
    }
    if any(group not in component_by_group for group in groups):
        raise ValueError("F0 source-qualified component map omits an evaluation group")
    components = np.asarray([component_by_group[group] for group in groups], dtype=np.int64)
    if len(np.unique(components)) != 141:
        raise ValueError("F0 component count differs from the frozen 141-component scope")
    ancestry = verify_parent_ancestry(
        ROOT,
        recipe,
        str(fold["split_manifest_sha256"]),
        groups,
    )

    cache_dir = Path(fold["parent_cache"]).resolve()
    scratch = Path(parent_recipe["resource_contract"]["scratch"]).resolve()
    if not cache_dir.is_relative_to(scratch):
        raise ValueError("F0 parent cache escapes physical scratch")
    fit_meta_path = cache_dir / "fit_sequences.json"
    eval_meta_path = cache_dir / "evaluation_sequences.json"
    fit_meta = read_json(fit_meta_path)
    eval_meta = read_json(eval_meta_path)
    _verify_cache_metadata(
        fit_meta,
        scope="in_sample_fit",
        refs=fit_refs,
        recipe=parent_recipe,
        recipe_sha=parent_record["sha256"],
        fold=fold,
        subset=subset,
        path=cache_dir / "fit_sequences.npy",
    )
    _verify_cache_metadata(
        eval_meta,
        scope="feature_only_eval",
        refs=eval_refs,
        recipe=parent_recipe,
        recipe_sha=parent_record["sha256"],
        fold=fold,
        subset=subset,
        path=cache_dir / "evaluation_sequences.npy",
    )
    if fit_meta["source_sha256"] != eval_meta["source_sha256"]:
        raise ValueError("F0 fit and evaluation cache source pins differ")
    parent_prediction_path = cache_dir / "evaluation_predictions.npy"
    parent_meta_path = cache_dir / "evaluation_predictions.json"
    parent_meta = read_json(parent_meta_path)
    parent_sha = verify_parent_prediction_metadata(
        ROOT,
        parent_recipe,
        parent_record["sha256"],
        fold,
        parent_recipe_path,
        eval_refs,
        eval_meta_path,
        eval_meta,
        parent_prediction_path,
        parent_meta_path,
        parent_meta,
    )
    parent_coverage = npy_coverage(parent_prediction_path)
    if parent_coverage["shape"] != [468, ROWS, 2] or parent_coverage["dtype"] != "<f4":
        raise ValueError("F0 parent prediction header differs from the frozen endpoint")
    if parent_coverage["file_bytes"] != recipe["parent_prediction"]["file_bytes"]:
        raise ValueError("F0 parent prediction file length differs from the recipe")
    if parent_sha != recipe["parent_prediction"]["sha256"]:
        raise ValueError("F0 parent prediction digest differs from the recipe")
    if sha256_file(parent_prediction_path) != parent_sha:
        raise ValueError("F0 parent prediction payload digest is invalid")
    _require_sha(
        _resolve(ROOT, fold["parent"]["checkpoint"]),
        fold["parent"]["checkpoint_sha256"],
        "F0 parent checkpoint",
    )
    _require_sha(
        _resolve(ROOT, fold["parent"]["provenance"]),
        fold["parent"]["provenance_sha256"],
        "F0 parent provenance",
    )
    parent_cache = ParentCacheArtifacts(
        prediction=None,
        identity=_parent_identity(
            parent_recipe,
            parent_record["sha256"],
            fold,
            subset,
            eval_refs,
            parent_meta,
            parent_meta_path,
            parent_prediction_path,
            fit_meta,
            fit_meta_path,
            eval_meta,
            eval_meta_path,
            parent_sha,
        ),
    )

    cell_records: dict[str, dict[str, Any]] = {}
    prediction_paths: dict[str, Path] = {}
    artifact_records: list[dict[str, Any]] = [
        {
            "endpoint": "parent",
            "path": str(parent_prediction_path),
            "sha256": parent_sha,
            "shape": parent_coverage["shape"],
            "dtype": parent_coverage["dtype"],
            "file_bytes": parent_coverage["file_bytes"],
            "source_qualified_groups": len(eval_refs),
            "evaluation_membership_sha256": row_refs_sha256(eval_refs),
        }
    ]
    for cell_record in recipe_cells:
        key = f"{cell_record['arm']}_lag{cell_record['lag']}"
        base_cell = base_cells[(int(cell_record["lag"]), str(cell_record["arm"]))]
        path, artifact_record = _verify_candidate(
            cell_record,
            base_cell,
            parent_recipe,
            parent_recipe_path,
            parent_record["sha256"],
            fold,
            subset,
            parent_cache,
            parent_meta_path,
            split_path,
            groups,
        )
        cell_records[key] = base_cell
        prediction_paths[key] = path
        artifact_records.append(artifact_record)

    target_record = parent_recipe["target_cache"]
    target_path = _resolve(ROOT, target_record["path"])
    target_stat = target_path.stat()
    if target_stat.st_size != recipe["target_cache"]["file_bytes"]:
        raise ValueError("official target archive file length differs from the F0 recipe")
    report = {
        "schema": "wnn.online-relationships-f0-preflight.v1",
        "status": "passed_label_free_preflight",
        "recipe_sha256": recipe_sha,
        "parent_recipe_sha256": parent_record["sha256"],
        "source_manifest_sha256": recipe["source_manifest"]["sha256"],
        "metric": "global_wp_clip2_v1",
        "new_fit_count": 0,
        "fold": 0,
        "seed": 1001,
        "evaluation_groups": len(groups),
        "evaluation_components": int(len(np.unique(components))),
        "evaluation_membership_sha256": row_refs_sha256(eval_refs),
        "primary_contrasts": recipe["primary_contrasts"],
        "secondary_contrasts": recipe["secondary_contrasts"],
        "artifacts": artifact_records,
        "target_cache": {
            "path": str(target_path),
            "expected_sha256": target_record["sha256"],
            "file_bytes": target_stat.st_size,
            "content_hash_checked": False,
            "labels_read": False,
        },
        "historical_source_pin_counts": {
            role: record["count"] for role, record in historical_source_pins.items()
        },
        "parent_ancestry": ancestry,
        "large_sequence_cache_payloads_hashed": False,
        "exposure": "F0 exposed development evidence; no fresh holdout claim",
        "uncertainty": "component bootstrap is conditional on fixed endpoints",
    }
    return PreparedF0Grade(
        recipe=recipe,
        parent_recipe=parent_recipe,
        fold=fold,
        cells=cell_records,
        parent_prediction_path=parent_prediction_path,
        candidate_prediction_paths=prediction_paths,
        evaluation_groups=groups,
        components=components,
        target_cache_path=target_path,
        report=report,
        ancestry=ancestry,
    )


def write_json_exclusive(path: Path, payload: dict[str, Any]) -> None:
    """Write a UTF-8 JSON receipt without replacing previous evidence."""
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
