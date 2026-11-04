"""Grade the frozen W18 L4 cohort with the canonical scorer and fixed blends."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.heterogeneous_ensemble_contract import (  # noqa: E402
    RESERVED_ROLES,
    SCHEMA,
    SPLIT_SHA256,
    TL3_WEIGHTS,
    read_object,
)
from scripts.heterogeneous_ensemble_score import grade as canonical_grade  # noqa: E402
from wnn.chronological_source_index import OFFICIAL_SOURCE_PINS  # noqa: E402
from wnn.splits import file_sha256  # noqa: E402

JsonObject = dict[str, Any]
PANEL_SHA256 = "f3e0327fee7de17e445f4ddb8ed5983af8348490ac3ab7a2887cf19deb5661b2"
CONFIGURATIONS = ("control", "innovations", "alpha50", "family01")
SEEDS = (1001, 1002)
REFERENCE_ALIASES = {
    "tl3c_w18final_distilled_s5": "w18",
    "tl3c_fttall_tlead_s5": "tiedlead_s5",
    "tl3c_fttall_tlead_s6": "tiedlead_s6",
}


def sha_pin(path: Path, digest: str | None = None) -> JsonObject:
    """Return an absolute path and SHA256, hashing small metadata when needed."""
    if not path.is_file():
        raise FileNotFoundError(path)
    value = file_sha256(path) if digest is None else digest
    if re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"invalid SHA256 for {path}")
    return {"path": str(path.resolve()), "sha256": value}


def pending(reason: str) -> JsonObject:
    """Represent an unavailable artifact set explicitly."""
    return {"state": "pending", "reason": reason}


def expected_row_order(data_root: Path) -> list[int]:
    """Read actual evaluation row-group IDs in prediction-bank order."""
    with np.load(data_root / "cache/valid_index.npz", allow_pickle=False) as index:
        groups = index["row_group"].tolist()
    if (
        len(groups) != 1873
        or len(set(groups)) != 1873
        or any(type(group) is not int for group in groups)
    ):
        raise ValueError("valid_index must list 1873 unique row groups")
    return groups


def reference_checkpoint_filename(metadata_path: Path, epoch: int) -> str:
    """Use the exact pinned reference sidecar filename, allowing known bank layouts."""
    metadata = read_object(metadata_path)
    checkpoint_file = metadata.get("checkpoint_file")
    allowed = {"model.pt", f"model_ep{epoch}.pt"}
    if checkpoint_file not in allowed:
        raise ValueError("reference sidecar has an unsupported checkpoint filename")
    return checkpoint_file


def collect_member(
    *,
    run: str,
    epoch: int,
    prediction: Path,
    metadata_path: Path,
    checkpoint: Path,
    provenance: Path,
    row_order: list[int],
    checkpoint_filename: str | None = None,
    expected_checkpoint_sha256: str | None = None,
    expected_prediction_sha256: str | None = None,
) -> JsonObject:
    """Bind one immutable legacy bank to its sidecar, fixed checkpoint and provenance."""
    paths = {
        "predictions": prediction,
        "metadata": metadata_path,
        "checkpoint": checkpoint,
        "provenance": provenance,
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        return pending("producer artifacts incomplete: " + ", ".join(missing))

    metadata = read_object(metadata_path)
    checkpoint_filename = checkpoint_filename or f"model_ep{epoch}.pt"
    expected_metadata = {
        "run": run,
        "split": "valid",
        "epoch": epoch,
        "checkpoint_file": checkpoint_filename,
        "split_manifest_sha256": SPLIT_SHA256,
        "inference_precision": "fp32",
        "row_groups": row_order,
    }
    if any(metadata.get(key) != value for key, value in expected_metadata.items()):
        raise ValueError(f"prediction sidecar identity or row order differs: {run}")

    checkpoint_sha = metadata.get("checkpoint_sha256")
    prediction_sha = metadata.get("prediction_sha256")
    for label, value in (("checkpoint", checkpoint_sha), ("prediction", prediction_sha)):
        if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise ValueError(f"sidecar has an invalid {label} SHA256: {run}")
    if expected_checkpoint_sha256 is not None and checkpoint_sha != expected_checkpoint_sha256:
        raise ValueError(f"checkpoint SHA differs from frozen panel: {run}")
    if expected_prediction_sha256 is not None and prediction_sha != expected_prediction_sha256:
        raise ValueError(f"prediction SHA differs from frozen panel: {run}")

    return {
        "kind": "legacy_bank",
        "run": run,
        "epoch": epoch,
        "predictions": sha_pin(prediction, prediction_sha),
        "metadata": sha_pin(metadata_path),
        "checkpoint": sha_pin(checkpoint, checkpoint_sha),
        "provenance": sha_pin(provenance),
    }


def fixed_ensembles(member_names: list[str]) -> JsonObject:
    """Create TL3 and exactly eight prespecified 90/10 candidate mixtures."""
    required = {f"{configuration}_s{seed}" for configuration in CONFIGURATIONS for seed in SEEDS}
    if set(member_names) != {"w18", "tiedlead_s5", "tiedlead_s6", *required}:
        raise ValueError("ensemble members do not match the frozen three-reference/eight-cell panel")
    ensembles: JsonObject = {"TL3": TL3_WEIGHTS}
    for name in sorted(required, key=lambda value: (value.split("_s")[0], int(value.rsplit("s", 1)[1]))):
        ensembles[f"TL3_plus_{name}"] = {"TL3": 0.9, name: 0.1}
    return ensembles


def validate_panel(panel: JsonObject, panel_hash: str) -> tuple[Path, Path, list[JsonObject]]:
    """Validate the frozen R3 identity and return project, data root and cells."""
    if panel_hash != PANEL_SHA256:
        raise ValueError("panel bytes differ from frozen FROZEN_PANEL_R3.json")
    if panel.get("schema") != "wnn.w18-l4-cohort.v1" or panel.get("status") != "recipes_frozen":
        raise ValueError("frozen R3 cohort panel required")
    if panel.get("identity", {}).get("split_sha256") != SPLIT_SHA256:
        raise ValueError("panel split SHA differs from the admitted chronological primary split")

    paths = panel.get("paths", {})
    project_root = Path(str(paths.get("project_root", "")))
    split_path = Path(str(paths.get("split_manifest", "")))
    data_root = project_root
    if not data_root.is_absolute() or not split_path.is_absolute():
        raise ValueError("panel project and split paths must be absolute")
    endpoint = panel.get("endpoint", {})
    if endpoint.get("checkpoint_epoch") != 1 or endpoint.get("epochs") != 2:
        raise ValueError("grading is bound to the frozen epoch-1 endpoint")

    raw_cells = panel.get("cells")
    if not isinstance(raw_cells, list) or len(raw_cells) != 8:
        raise ValueError("the frozen cohort must contain eight fit cells")
    cells = [cell for cell in raw_cells if isinstance(cell, dict)]
    pairs = [(cell.get("configuration_id"), cell.get("seed")) for cell in cells]
    expected_pairs = {(configuration, seed) for configuration in CONFIGURATIONS for seed in SEEDS}
    if len(cells) != 8 or set(pairs) != expected_pairs or len(set(pairs)) != 8:
        raise ValueError("panel cells must be four frozen configurations paired on seeds 1001/1002")
    if len({cell.get("run_name") for cell in cells}) != 8:
        raise ValueError("panel run names must be unique")

    return project_root, data_root, cells


def make_manifest(panel: JsonObject, panel_path: Path, panel_hash: str) -> JsonObject:
    """Bind available panel members and explicit pending members to the official sources."""
    project_root, data_root, cells = validate_panel(panel, panel_hash)
    row_order = expected_row_order(data_root)
    split_path = Path(panel["paths"]["split_manifest"])
    split_pin = sha_pin(split_path)
    if split_pin["sha256"] != SPLIT_SHA256:
        raise ValueError("split manifest bytes differ from the frozen chronological primary split")
    validation_path = data_root / "datasets/valid.parquet"
    validation_pin = sha_pin(validation_path, OFFICIAL_SOURCE_PINS["valid"].sha256)

    members: JsonObject = {}
    references = panel.get("reference_predictions")
    if not isinstance(references, list) or len(references) != 3:
        raise ValueError("panel must bind the three canonical TL3 reference predictions")
    seen_references: set[str] = set()
    for reference in references:
        run = str(reference["run"])
        alias = REFERENCE_ALIASES.get(run)
        if alias is None or alias in seen_references:
            raise ValueError(f"unexpected or duplicate frozen reference: {run}")
        seen_references.add(alias)
        if float(reference.get("weight", -1.0)) != TL3_WEIGHTS[alias]:
            raise ValueError(f"canonical reference weight changed: {alias}")
        role_run, epoch = RESERVED_ROLES[alias]
        if role_run != run:
            raise ValueError(f"canonical reference role changed: {alias}")
        pred_path = Path(reference["prediction_path"])
        meta_path = Path(reference["sidecar_path"])
        source_root = pred_path.parent.parent
        if not meta_path.is_file():
            members[alias] = pending("producer artifacts incomplete: metadata")
            continue
        checkpoint_filename = reference_checkpoint_filename(meta_path, epoch)
        members[alias] = collect_member(
            run=run,
            epoch=epoch,
            prediction=pred_path,
            metadata_path=meta_path,
            checkpoint=source_root / "artifacts/runs" / run / checkpoint_filename,
            provenance=source_root / "artifacts/runs" / run / "provenance.json",
            row_order=row_order,
            checkpoint_filename=checkpoint_filename,
            expected_checkpoint_sha256=str(reference["checkpoint_sha256"]),
            expected_prediction_sha256=str(reference["prediction_sha256"]),
        )
    if seen_references != set(REFERENCE_ALIASES.values()):
        raise ValueError("frozen panel omits a canonical TL3 reference")

    for cell in cells:
        configuration = str(cell["configuration_id"])
        seed = int(cell["seed"])
        run = str(cell["run_name"])
        if re.fullmatch(r"[A-Za-z0-9_.-]+", run) is None:
            raise ValueError(f"unsafe run name in frozen panel: {run}")
        member_name = f"{configuration}_s{seed}"
        run_root = project_root / "artifacts/runs" / run
        pred_path = project_root / "cache" / f"pred_{run}_valid.npy"
        members[member_name] = collect_member(
            run=run,
            epoch=1,
            prediction=pred_path,
            metadata_path=pred_path.with_suffix(".json"),
            checkpoint=run_root / "model_ep1.pt",
            provenance=run_root / "provenance.json",
            row_order=row_order,
        )

    member_names = list(members)
    return {
        "schema": SCHEMA,
        "baseline": "TL3",
        "data_root": str(data_root.resolve()),
        "split": split_pin,
        "validation": validation_pin,
        "members": members,
        "ensembles": fixed_ensembles(member_names),
        "assembly": {
            "schema": "wnn.w18-l4-quality-grade-assembly.v1",
            "panel": str(panel_path.resolve()),
            "panel_sha256": panel_hash,
            "fixed_blend_weight": 0.1,
            "selection": "all members and weights were frozen before outcomes",
        },
    }


def matched_comparisons(report: JsonObject) -> list[JsonObject]:
    """Compute paired candidate/control and fixed-blend differences from canonical scores."""
    member_scores = report.get("member_scores", {})
    ensemble_rows = {row["name"]: row for row in report.get("rows", [])}
    comparisons: list[JsonObject] = []
    for configuration in CONFIGURATIONS:
        for seed in SEEDS:
            candidate = f"{configuration}_s{seed}"
            control = f"control_s{seed}"
            candidate_score = member_scores.get(candidate)
            control_score = member_scores.get(control)
            candidate_wp = None if candidate_score is None else candidate_score.get("wp")
            control_wp = None if control_score is None else control_score.get("wp")
            blend_name = f"TL3_plus_{candidate}"
            control_blend_name = f"TL3_plus_{control}"
            blend_row = ensemble_rows.get(blend_name, {})
            control_blend_row = ensemble_rows.get(control_blend_name, {})
            blend_score = blend_row.get("score")
            control_blend_score = control_blend_row.get("score")
            comparisons.append(
                {
                    "configuration": configuration,
                    "seed": seed,
                    "candidate_member": candidate,
                    "control_member": control,
                    "candidate_wp": candidate_wp,
                    "control_wp": control_wp,
                    "delta_wp_vs_control": (
                        None if candidate_wp is None or control_wp is None else candidate_wp - control_wp
                    ),
                    "fixed_blend": blend_name,
                    "fixed_blend_wp": None if blend_score is None else blend_score.get("wp"),
                    "fixed_blend_delta_wp_vs_tl3": blend_row.get("delta_to_tl3"),
                    "fixed_blend_delta_wp_vs_control_blend": (
                        None
                        if blend_score is None or control_blend_score is None
                        else blend_score["wp"] - control_blend_score["wp"]
                    ),
                    "status": "complete"
                    if candidate_wp is not None and control_wp is not None
                    else "incomplete",
                }
            )
    return comparisons


def write_matched_report(path: Path, rows: list[JsonObject]) -> None:
    """Write a compact matched comparison table with missing scores left blank."""
    columns = (
        "configuration",
        "seed",
        "candidate_wp",
        "control_wp",
        "delta_wp_vs_control",
        "fixed_blend_wp",
        "fixed_blend_delta_wp_vs_tl3",
        "fixed_blend_delta_wp_vs_control_blend",
        "status",
    )
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        values: list[str] = []
        for column in columns:
            value = row[column]
            if value is None:
                values.append("—")
            elif isinstance(value, float):
                values.append(f"{value:+.9f}" if "delta" in column else f"{value:.9f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(panel_path: Path, output: Path) -> JsonObject:
    """Create a fresh fixed-weight manifest and invoke the canonical numerical grader."""
    if output.exists():
        raise FileExistsError("choose a fresh output directory to preserve prior grading artifacts")
    panel_path = panel_path.resolve(strict=True)
    panel_hash = file_sha256(panel_path)
    panel = read_object(panel_path)
    manifest = make_manifest(panel, panel_path, panel_hash)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=False, exist_ok=False)
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    available = [record for record in manifest["members"].values() if record.get("state") != "pending"]
    if not available:
        result: JsonObject = {
            "status": "incomplete",
            "reason": "no pinned prediction artifacts are available; canonical scoring was not run",
            "pending_members": {
                name: record["reason"]
                for name, record in manifest["members"].items()
                if record.get("state") == "pending"
            },
        }
        comparisons = matched_comparisons({"member_scores": {}, "rows": []})
        (output / "ASSEMBLY_STATUS.json").write_text(
            json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
    else:
        result = canonical_grade(manifest_path, output / "canonical")
        comparisons = matched_comparisons(result)
    (output / "matched_per_seed.json").write_text(
        json.dumps(comparisons, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    write_matched_report(output / "MATCHED_PER_SEED.md", comparisons)
    return result


def main() -> None:
    """Grade only the frozen panel artifacts with canonical full-valid scoring."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.panel, args.output)
    print(json.dumps({key: report[key] for key in report if key in {"status", "reason"}}))


if __name__ == "__main__":
    main()
