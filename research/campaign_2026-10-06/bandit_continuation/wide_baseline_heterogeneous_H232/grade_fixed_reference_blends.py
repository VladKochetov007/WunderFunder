"""Grade delivered H232 predictions against the frozen exposed-fold reference."""

from __future__ import annotations

import argparse
import importlib.util
import json
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
PREREG = HERE / "H232_FIXED_EP3_REFERENCE_BLEND_PREREG_20261008_R2.json"
PREREG_SHA = "cc670d0302ec8c321f70923b0de908eb904b6922f81495061469b5a040d072a6"
SHARED = HERE.parent / "attention_training_H218/grade_fold0_reference_blends.py"


def load_module(name: str, path: Path) -> ModuleType:
    """Load our existing independently checked metric and comparison helpers."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    """Require a separate root receipt binding actual delivered artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve the original result; use a new identity")
    grade = load_module("h232_shared_grade", SHARED)
    approval = json.loads(args.authorization.read_text())
    if (
        approval.get("schema") != "h232-fixed-reference-grade-admission-v1"
        or approval.get("approved_by") != "root"
        or approval.get("status") != "approved"
        or approval.get("preregistration_sha256") != PREREG_SHA
        or grade.sha256_file(PREREG) != PREREG_SHA
        or approval.get("grader_sha256") != grade.sha256_file(Path(__file__))
    ):
        raise ValueError("Missing actual-artifact root grading admission")
    paths: dict[str, Path] = {}
    for name, entry in approval["artifacts"].items():
        path = (ROOT / entry["path"]).resolve(strict=True)
        if not path.is_relative_to(ROOT) or grade.sha256_file(path) != entry["sha256"]:
            raise ValueError(f"Delivered artifact identity mismatch: {name}")
        paths[name] = path
    required = {
        "prediction",
        "sidecar",
        "checkpoint",
        "provenance",
        "fit_terminal",
        "export_terminal",
        "source_archive",
        "source_manifest",
        "shared_grader",
        "scorer",
        "reference_helpers",
        "gmetric",
    }
    if set(paths) != required:
        raise ValueError("Incomplete or unexpected delivered artifact bindings")
    if paths["shared_grader"] != SHARED.resolve():
        raise ValueError("Shared grader path differs from this code's import")
    if (
        paths["scorer"] != grade.SCORER_PATH.resolve()
        or paths["reference_helpers"] != grade.H217_REFERENCE_HELPERS.resolve()
        or paths["gmetric"] != (ROOT / "src/wnn/gmetric.py").resolve()
    ):
        raise ValueError("Frozen scorer/helper paths differ")
    if approval.get("fit_completed") is not True or approval.get("export_completed") is not True:
        raise ValueError("Actual fit and export terminal evidence must be reviewed")
    prereg = json.loads(PREREG.read_text())
    candidate = prereg["candidate"]
    for name, field in (
        ("checkpoint", "checkpoint_sha256"),
        ("source_archive", "source_archive_sha256"),
        ("source_manifest", "source_manifest_sha256"),
    ):
        if grade.sha256_file(paths[name]) != candidate[field]:
            raise ValueError(f"Candidate changed from preregistration: {name}")
    design = prereg["comparison_design"]
    for record in (design["split_manifest"], design["target_cache"]):
        if grade.sha256_file(ROOT / record["path"]) != record["sha256"]:
            raise ValueError("Frozen split or target cache changed")
    reference = prereg["reference"]
    for stem in ("checkpoint", "provenance", "prediction", "sidecar"):
        if grade.sha256_file(ROOT / reference[f"{stem}_path"]) != reference[f"{stem}_sha256"]:
            raise ValueError(f"Frozen reference changed: {stem}")
    sidecar = json.loads(paths["sidecar"].read_text())
    for field, expected in {
        "run": candidate["run"],
        "split": "valid",
        "epoch": 3,
        "checkpoint_sha256": candidate["checkpoint_sha256"],
        "split_manifest_sha256": reference["split_manifest_sha256"],
        "inference_precision": "fp32",
        "prediction_sha256": approval["artifacts"]["prediction"]["sha256"],
    }.items():
        if sidecar.get(field) != expected:
            raise ValueError(f"Delivered sidecar differs: {field}")
    split = json.loads((ROOT / design["split_manifest"]["path"]).read_text())
    evaluation = split["evaluation"]["valid"]
    provenance = json.loads(paths["provenance"].read_text())
    membership = grade.validate_candidate_fit_membership(split, provenance, evaluation)
    helpers = load_module("h232_reference_helpers", paths["reference_helpers"])
    scorer = load_module("h232_official_score", paths["scorer"])
    reference_sidecar = json.loads((ROOT / reference["sidecar_path"]).read_text())
    helpers.verify_reference_lineage(split, reference_sidecar)
    targets = scorer.mmap_stored_npz(ROOT / design["target_cache"]["path"])
    groups = sidecar["row_groups"]
    if groups != targets["row_group"].tolist() or len(groups) != 1873:
        raise ValueError("Full-valid candidate row order differs from target cache")
    alignment = grade.align_evaluation_rows(
        evaluation, groups, reference_sidecar["row_groups"], targets["row_group"].tolist()
    )
    component = {row["row_group"]: row["component"] for row in split["sequences"]["valid"]}
    components = np.asarray([component[row[0][1]] for row in alignment], dtype=np.int64)
    prediction = np.load(paths["prediction"], mmap_mode="r", allow_pickle=False)
    reference_prediction = np.load(
        ROOT / reference["prediction_path"], mmap_mode="r", allow_pickle=False
    )
    grade.validate_prediction_array(prediction, len(groups))
    grade.validate_prediction_array(reference_prediction, len(reference_sidecar["row_groups"]))
    if list(grade.ALPHAS) != prereg["fixed_weights_on_candidate"]:
        raise ValueError("Weights differ from preregistration")
    result = grade.score_predeclared_blends(
        targets["y"],
        targets["mask"],
        prediction,
        reference_prediction,
        alignment,
        components,
        scorer,
        helpers,
    )
    for row in result["predeclared_blends_all_reported"]:
        row["alpha_on_h232"] = row.pop("alpha_on_h218")
    report = {
        "record_id": args.output.stem,
        "created_utc": datetime.now(UTC).isoformat(),
        "preregistration_sha256": PREREG_SHA,
        "authorization_sha256": grade.sha256_file(args.authorization),
        "grader_sha256": grade.sha256_file(Path(__file__)),
        "hypothesis": "H-232",
        "metric": "global_wp_clip2_v1",
        "status": "evaluated_exposed_discovery",
        "membership": membership,
        "comparability_limits": prereg["comparability_limits"],
        "public_forecast": None,
        "weight_selection": False,
        **result,
    }
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(
        [
            (row["alpha_on_h232"], row["score"]["wp"])
            for row in result["predeclared_blends_all_reported"]
        ]
    )


if __name__ == "__main__":
    main()
