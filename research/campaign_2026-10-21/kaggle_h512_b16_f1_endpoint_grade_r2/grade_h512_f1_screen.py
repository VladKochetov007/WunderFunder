"""Grade the frozen H512 F1 panel after independent root fit acceptance."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import resource
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from f1_grade_contract import (
    DERIVED_SHA,
    FIT_NAME,
    ORIGINAL_SHA,
    SWA_NAME,
    check_acceptance,
    check_manifests,
    pinned,
    read_json,
    sha256,
)
from f1_grade_outputs import check_worker_outputs

ROOT = Path(__file__).resolve().parents[3]
PACKET = Path("research/campaign_2026-10-21/kaggle_h512_b16_f1_endpoint_grade_r2")
PLAN = PACKET / "F1_FIXED_SCREEN_R2.json"
PLAN_SHA256 = "2ab85601f8bd7d99c52df036d7f7dbed32310e594dab7bca580564a81249c29e"


def score_panel(
    root: Path,
    plan: dict[str, Any],
    derived: dict,
    predictions: dict[str, np.memmap],
    tracked: dict[str, Path],
) -> dict[str, Any]:
    """Score the five frozen endpoints with the repository's official scorer."""
    audit_dir = root / "research/campaign_2026-09-30/audit"
    sys.path.insert(0, str(audit_dir))
    spec = importlib.util.spec_from_file_location(
        "frozen_campaign_score", audit_dir / "campaign_score.py"
    )
    if spec is None or spec.loader is None:
        raise ImportError("Could not load the frozen campaign scorer")
    scorer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scorer)
    target_cache = scorer.mmap_stored_npz(tracked["targets"])
    if target_cache["y"].shape != tuple(plan["targets"]["sequence_shape"]):
        raise ValueError("Official target sequence cache shape changed")
    index_digest = hashlib.sha256()
    for field in ("row_group", "seq_ix"):
        index_digest.update(np.asarray(target_cache[field], dtype="<i8").tobytes())
    if index_digest.hexdigest() != plan["targets"]["row_index_sha256"]:
        raise ValueError("Official target sequence index identity changed")
    groups = np.asarray(target_cache["row_group"], dtype=np.int64)
    if len(groups) != 1873 or len(np.unique(groups)) != len(groups):
        raise ValueError("Target cache row-group index is incomplete")
    target_positions = {int(group): i for i, group in enumerate(groups)}

    evaluation_groups = [int(group) for group in derived["evaluation"]["valid"]]
    if len(evaluation_groups) != 457 or len(set(evaluation_groups)) != 457:
        raise ValueError("Derived manifest evaluation membership differs from the frozen F1 panel")
    for key, run in (("cp5", FIT_NAME), ("swa345", SWA_NAME)):
        sidecar = read_json(tracked[f"{key}_sidecar"])
        if sidecar.get("row_groups") != evaluation_groups:
            raise ValueError(f"Candidate row order differs from the derived manifest: {run}")

    oldz4_sidecar = read_json(tracked["oldz4_sidecar"])
    oldz4 = np.load(tracked["oldz4_prediction"], mmap_mode="r", allow_pickle=False)
    oldz4_provenance = read_json(tracked["oldz4_provenance"])
    baseline = plan["baseline"]
    if (
        oldz4_sidecar.get("run") != baseline["run"]
        or oldz4_sidecar.get("split") != "valid"
        or oldz4_sidecar.get("epoch") != baseline["epoch"]
        or oldz4_sidecar.get("inference_precision") != baseline["inference_precision"]
        or oldz4_sidecar.get("split_manifest_sha256") != ORIGINAL_SHA
        or oldz4_sidecar.get("row_groups") != list(range(1873))
        or oldz4.shape != tuple(baseline["prediction"]["shape"])
        or oldz4.dtype != np.float32
        or oldz4_sidecar.get("checkpoint_sha256") != baseline["checkpoint"]["sha256"]
        or oldz4_sidecar.get("prediction_sha256") != baseline["prediction"]["sha256"]
        or oldz4_provenance.get("split_manifest_sha256") != ORIGINAL_SHA
    ):
        raise ValueError("Legacy oldZ4 F1 reference metadata differs from its pinned identity")

    component_map = {
        int(node["row_group"]): int(node["component"]) for node in derived["sequences"]["valid"]
    }
    if not set(evaluation_groups).issubset(component_map):
        raise ValueError("Derived manifest lacks evaluation component assignments")
    names = [
        "oldz4",
        "cp5",
        "swa345",
        "oldz4_075_cp5_025",
        "oldz4_075_swa345_025",
    ]
    moments: dict[str, list[np.ndarray]] = {name: [] for name in names}
    softened: dict[str, dict[str, list[np.ndarray]]] = {
        str(lam): {name: [] for name in names} for lam in scorer.SOFTENED
    }
    blocks: dict[str, list[float | None]] = {name: [] for name in names}
    classes: list[int] = []
    components: list[int] = []
    selected_rows = 0
    for row, group in enumerate(evaluation_groups):
        baseline_prediction = np.asarray(oldz4[target_positions[group]], dtype=np.float32)
        cp5_prediction = np.asarray(predictions["cp5"][row], dtype=np.float32)
        swa_prediction = np.asarray(predictions["swa345"][row], dtype=np.float32)
        panel = {
            "oldz4": baseline_prediction,
            "cp5": cp5_prediction,
            "swa345": swa_prediction,
            "oldz4_075_cp5_025": (
                0.75 * baseline_prediction.astype(np.float64) + 0.25 * cp5_prediction
            ).astype(np.float32),
            "oldz4_075_swa345_025": (
                0.75 * baseline_prediction.astype(np.float64) + 0.25 * swa_prediction
            ).astype(np.float32),
        }
        position = target_positions[group]
        sequence = scorer.score_sequence(
            target_cache["y"][position], target_cache["mask"][position], panel
        )
        for name in names:
            moments[name].append(sequence["moments"][name])
            blocks[name].append(sequence["blocks"][name])
            for lam in scorer.SOFTENED:
                softened[str(lam)][name].append(sequence["soft"][str(lam)][name])
        selected_rows += sequence["selected_rows"]
        classes.append(int(target_cache["market_class"][position]))
        components.append(component_map[group])
    if selected_rows <= 0 or len(set(components)) != 149 or not np.isin(classes, range(4)).all():
        raise ValueError("Official mask, component, or class coverage is invalid")
    m = {name: np.stack(values) for name, values in moments.items()}
    soft = {
        lam: {name: np.stack(values) for name, values in arms.items()}
        for lam, arms in softened.items()
    }
    design = {
        "comparisons": [tuple(pair) for pair in plan["comparisons"]],
        "n_boot": plan["bootstrap"]["n_boot"],
        "bootstrap_seed": plan["bootstrap"]["seed"],
    }
    scored = scorer.summary(m, soft, blocks, np.asarray(classes), np.asarray(components), design)
    return {
        "metric_id": "global_wp_clip2_v1",
        "evaluation_groups": len(evaluation_groups),
        "evaluation_components": len(set(components)),
        "actual_official_scored_rows": selected_rows,
        "group_order_sha256": hashlib.sha256(
            np.asarray(evaluation_groups, dtype="<i8").tobytes()
        ).hexdigest(),
        **scored,
    }


def grade(root: Path, output: Path) -> dict[str, Any]:
    """Run the immutable F1 screen and write one create-only grade receipt."""
    started = time.monotonic()
    if output.exists():
        raise FileExistsError(output)
    plan_path = pinned(root, {"path": PLAN.as_posix(), "sha256": PLAN_SHA256})
    plan = read_json(plan_path)
    if (
        plan.get("schema") != "wnn.kaggle-h512-f1-fixed-screen.v1"
        or plan.get("status") != "FROZEN_SCREEN_NOT_GRADED"
    ):
        raise ValueError("F1 score screen is not frozen")
    acceptance, worker, base = check_acceptance(root, plan)
    _, derived = check_manifests(root, plan, worker)
    predictions, tracked = check_worker_outputs(plan, derived, acceptance, worker, base)
    tracked.update(
        {
            "plan": plan_path,
            "acceptance": (root / plan["acceptance"]["path"]).resolve(strict=True),
            "worker_receipt": (root / acceptance["worker_receipt_path"]).resolve(strict=True),
            "canonical_manifest": pinned(root, plan["identity"]["canonical_manifest"]),
            "derived_manifest": pinned(root, plan["identity"]["derived_execution_manifest"]),
            "portable_registry": pinned(root, plan["identity"]["portable_registry"]),
            "proposal": pinned(root, plan["identity"]["proposal"]),
            "targets": pinned(root, plan["targets"]),
            "oldz4_prediction": pinned(root, plan["baseline"]["prediction"]),
            "oldz4_sidecar": pinned(root, plan["baseline"]["sidecar"]),
            "oldz4_checkpoint": pinned(root, plan["baseline"]["checkpoint"]),
            "oldz4_provenance": pinned(root, plan["baseline"]["provenance"]),
            "grader_source": Path(__file__).resolve(),
            "grader_contract_source": (root / PACKET / "f1_grade_contract.py").resolve(strict=True),
            "grader_outputs_source": (root / PACKET / "f1_grade_outputs.py").resolve(strict=True),
        }
    )
    for source, digest in plan["scorer_sources"].items():
        tracked[source] = pinned(root, {"path": source, "sha256": digest})
    before = {name: {"path": str(path), "sha256": sha256(path)} for name, path in tracked.items()}
    comparison = score_panel(root, plan, derived, predictions, tracked)
    after = {name: {"path": str(path), "sha256": sha256(path)} for name, path in tracked.items()}
    if before != after:
        raise ValueError("An input or source changed during grading")
    result = {
        "schema": "wnn.kaggle-h512-f1-fixed-screen-grade.v1",
        "status": "GRADED_NOT_PROMOTED",
        "plan_sha256": sha256(plan_path),
        "script_sha256": sha256(Path(__file__).resolve()),
        "root_acceptance_sha256": sha256(tracked["acceptance"]),
        "worker_receipt_sha256": sha256(tracked["worker_receipt"]),
        "manifest_identities": {
            "canonical_original_sha256": ORIGINAL_SHA,
            "derived_execution_view_sha256": DERIVED_SHA,
            "only_source_field_changed": True,
        },
        "fit": {
            "run": FIT_NAME,
            "seed": 1004,
            "initialization": "fresh H512x1, no teacher or initializer",
            "membership": "fold1 fit train+valid; 457 separate evaluation groups",
            "training_updates": 44760,
            "cp5_sha256": acceptance["checkpoint_sha256"]["model_ep5.pt"],
            "swa_derivation_sha256": acceptance["swa_derivation_receipt_sha256"],
            "worker_quality_fit_claim": None,
        },
        "comparison": comparison,
        "baseline_interpretation": plan["baseline"]["interpretation"],
        "exposure_and_boundaries": plan["claim_boundaries"],
        "input_hashes_before_and_after": {"before": before, "after": after},
        "resources": {
            "wall_seconds": round(time.monotonic() - started, 3),
            "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def main() -> None:
    """Run the one frozen grade after root has written its terminal acceptance."""
    output = (
        ROOT
        / "research/campaign_2026-10-21/kaggle_h512_b16_f1_endpoint_grade_r2/F1_FIXED_SCREEN_GRADE_R2.json"
    )
    result = grade(ROOT, output)
    print(
        json.dumps(
            {
                "output": str(output),
                "status": result["status"],
                "selected_rows": result["comparison"]["actual_official_scored_rows"],
            }
        )
    )


if __name__ == "__main__":
    main()
