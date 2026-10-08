"""Score the completed E540 pair at its three preregistered mixture weights."""

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
PREREG = HERE / "E540_FIXED_EP0_MATCHED_EMBEDDING_COMPARISON_PREREG_20261008_R1.json"
PREREG_SHA = "24ec8e827a344bddc8fede204e384c190701b5b59fac1b6c3cfc828fb88b6903"
TARGET_SHA = "dbb7a1a48dd0af9fa524a80c2694008a56cb91d5dae80e8d0d4984c0c5622955"
SHARED = HERE.parent / "attention_training_H218/grade_fold0_reference_blends.py"
SCORER = ROOT / "research/campaign_2026-09-30/audit/campaign_score.py"


def load_module(name: str, path: Path) -> ModuleType:
    """Import fixed project helpers without importing training code."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def score_pair(
    targets: dict[str, np.ndarray],
    candidate: np.ndarray,
    control: np.ndarray,
    alignment: list[tuple[tuple[str, int], int, int, int]],
    components: np.ndarray,
    scorer: ModuleType,
    weights: list[float],
    bootstrap: dict[str, int],
) -> list[dict[str, object]]:
    """Pool official per-sequence moments after raw blending, then resample components."""
    moments: dict[float, list[np.ndarray]] = {weight: [] for weight in weights}
    for _, candidate_row, control_row, target_row in alignment:
        labels = targets["y"][target_row]
        selected = np.asarray(targets["mask"][target_row], dtype=bool)
        selected = selected & (np.arange(labels.shape[0]) >= 99)
        for weight in weights:
            mixed = (1.0 - weight) * control[control_row] + weight * candidate[candidate_row]
            moments[weight].append(
                np.stack([scorer.moments(labels[:, t], mixed[:, t], selected) for t in range(2)])
            )
    pooled = {weight: np.stack(rows) for weight, rows in moments.items()}
    baseline = scorer.pooled(pooled[0.0])
    return [
        {
            "weight_on_ple": weight,
            "score": scorer.pooled(values),
            "delta_wp_vs_linear": scorer.pooled(values)["wp"] - baseline["wp"],
            "paired_component_bootstrap": scorer.paired_component_bootstrap(
                values,
                pooled[0.0],
                components,
                n_boot=bootstrap["replicates"],
                seed=bootstrap["seed"],
                centered=True,
            ),
        }
        for weight, values in pooled.items()
    ]


def main() -> None:
    """Require actual delivered-artifact root admission before any scoring."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve prior results; choose a new output identity")
    grade = load_module("e540_shared_grade", SHARED)
    admission = json.loads(args.authorization.read_text())
    if (
        admission.get("schema") != "e540-fixed-pair-grade-admission-v1"
        or admission.get("approved_by") != "root"
        or admission.get("status") != "approved"
        or admission.get("preregistration_sha256") != PREREG_SHA
        or grade.sha256_file(PREREG) != PREREG_SHA
        or admission.get("grader_sha256") != grade.sha256_file(Path(__file__))
        or admission.get("fits_and_export_terminal_reviewed") is not True
    ):
        raise ValueError("Missing actual-artifact root grading admission")
    required = {
        "linear_prediction",
        "linear_sidecar",
        "linear_checkpoint",
        "linear_provenance",
        "ple_prediction",
        "ple_sidecar",
        "ple_checkpoint",
        "ple_provenance",
        "export_terminal",
        "source_archive",
        "source_manifest",
        "artifact_binding",
        "subset_manifest",
        "split_manifest",
        "target_cache",
        "shared_grader",
        "scorer",
        "gmetric",
    }
    if set(admission["artifacts"]) != required:
        raise ValueError("Incomplete or unexpected artifact set")
    paths: dict[str, Path] = {}
    for name, entry in admission["artifacts"].items():
        path = (ROOT / entry["path"]).resolve(strict=True)
        if not path.is_relative_to(ROOT) or grade.sha256_file(path) != entry["sha256"]:
            raise ValueError(f"Artifact identity differs: {name}")
        paths[name] = path
    if paths["shared_grader"] != SHARED.resolve() or paths["scorer"] != SCORER.resolve():
        raise ValueError("Unexpected scoring helper path")
    if paths["gmetric"] != (ROOT / "src/wnn/gmetric.py").resolve():
        raise ValueError("Unexpected official metric path")
    if admission["artifacts"]["target_cache"]["sha256"] != TARGET_SHA:
        raise ValueError("Frozen target cache changed")
    prereg = json.loads(PREREG.read_text())
    for name, key in (
        ("artifact_binding", "artifact_binding_sha256"),
        ("subset_manifest", "subset_manifest_sha256"),
        ("split_manifest", "split_manifest_sha256"),
        ("linear_checkpoint", "control_checkpoint_sha256"),
        ("ple_checkpoint", "candidate_checkpoint_sha256"),
    ):
        if admission["artifacts"][name]["sha256"] != prereg[key]:
            raise ValueError(f"Artifact changed from preregistration: {name}")
    binding = json.loads(paths["artifact_binding"].read_text())
    subset = json.loads(paths["subset_manifest"].read_text())
    split = json.loads(paths["split_manifest"].read_text())
    evaluation = subset["evaluation_groups"]
    if evaluation != split["evaluation"]["valid"] or len(evaluation) != 468:
        raise ValueError("Excluded evaluation groups differ")
    selected_split = {**split, "fit": subset["selected_groups"]}
    scorer = load_module("e540_official_scorer", SCORER)
    if scorer.METRIC_ID != "global_wp_clip2_v1":
        raise ValueError("Official metric changed")
    targets = scorer.mmap_stored_npz(paths["target_cache"])
    arrays: dict[str, np.ndarray] = {}
    groups: dict[str, list[int]] = {}
    membership: dict[str, object] = {}
    for arm in ("linear", "ple"):
        provenance = paths[f"{arm}_provenance"]
        if grade.sha256_file(provenance) != binding["assets"][f"{arm}_provenance"]["sha256"]:
            raise ValueError(f"Original fitting provenance changed: {arm}")
        membership[arm] = grade.validate_candidate_fit_membership(
            selected_split, json.loads(provenance.read_text()), evaluation
        )
        sidecar = json.loads(paths[f"{arm}_sidecar"].read_text())
        expected = {
            "epoch": 0,
            "split": "valid",
            "inference_precision": "fp32",
            "checkpoint_sha256": admission["artifacts"][f"{arm}_checkpoint"]["sha256"],
            "prediction_sha256": admission["artifacts"][f"{arm}_prediction"]["sha256"],
            "run": binding["source_fits"][arm]["run_name"],
        }
        if any(sidecar.get(key) != value for key, value in expected.items()):
            raise ValueError(f"Export sidecar identity differs: {arm}")
        groups[arm] = sidecar["row_groups"]
        arrays[arm] = np.load(paths[f"{arm}_prediction"], mmap_mode="r", allow_pickle=False)
        grade.validate_prediction_array(arrays[arm], len(groups[arm]))
        if set(groups[arm]) != set(evaluation) or len(groups[arm]) != len(evaluation):
            raise ValueError(f"Export must contain exactly the excluded groups: {arm}")
    alignment = grade.align_evaluation_rows(
        evaluation, groups["ple"], groups["linear"], targets["row_group"].tolist()
    )
    component_of = {node["row_group"]: node["component"] for node in split["sequences"]["valid"]}
    components = np.asarray([component_of[key[1]] for key, _, _, _ in alignment])
    result = score_pair(
        targets,
        arrays["ple"],
        arrays["linear"],
        alignment,
        components,
        scorer,
        prereg["mixture_weights_on_ple"],
        prereg["bootstrap"],
    )
    report = {
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "evaluated_exposed_discovery",
        "metric": scorer.METRIC_ID,
        "preregistration_sha256": grade.sha256_file(PREREG),
        "authorization_sha256": grade.sha256_file(args.authorization),
        "grader_sha256": grade.sha256_file(Path(__file__)),
        "membership": membership,
        "group_count": len(alignment),
        "predeclared_weights_all_reported": result,
        "scientific_limits": prereg["scientific_limits"],
        "public_forecast": None,
        "weight_selection": False,
    }
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print([(row["weight_on_ple"], row["score"]["wp"]) for row in result])


if __name__ == "__main__":
    main()
