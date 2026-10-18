"""Grade the fixed geometry R3 endpoints and R2 correction-scale variants.

This is a small wrapper around the repository's official prediction alignment,
scoring, and paired component-bootstrap implementations. All variants use the
same exposed F0 evaluation groups. Alpha aliases are inference-only derivatives
of R2 and are not independent fitted models.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Candidate:
    """One scored cache entry and its dependence class."""

    name: str
    run: str
    role: str
    identity_group: str
    alpha: float | None = None
    source_checkpoint_sha256: str | None = None


def sha256(path: Path) -> str:
    """Hash a file in bounded blocks."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_grade_helpers(repo_root: Path) -> Any:
    """Load the repository's canonical alignment and official score helpers."""
    scripts = repo_root / "scripts"
    source = scripts / "grade_prediction_bank.py"
    if not source.is_file():
        raise FileNotFoundError(source)
    sys.path.insert(0, str(repo_root / "src"))
    spec = importlib.util.spec_from_file_location("geometry_r3_prediction_bank", source)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load grading helpers from {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ROOT = repo_root
    return module


def candidates(repo_root: Path, source_checkpoint_sha: str) -> list[Candidate]:
    """Declare fitted candidates, fixed references, and R2-derived alpha variants."""
    result = [
        Candidate(
            "parent",
            "warm_mechanism_f0_s3_c_r3",
            "reference",
            "parent_checkpoint",
        ),
        Candidate(
            "r2_full_geometry",
            "geometry_adapter_f0_s3_r2",
            "reference",
            "r2_checkpoint",
        ),
        Candidate(
            "r3_dictionary_only",
            "geometry_variation_f0_s3_r3_dictionary_only",
            "variant_fit",
            "r3_dictionary_fit",
        ),
        Candidate(
            "r3_moments_only",
            "geometry_variation_f0_s3_r3_moments_only",
            "variant_fit",
            "r3_moments_fit",
        ),
    ]
    alphas = (
        ("m0p50", -0.50),
        ("m0p25", -0.25),
        ("p0p25", 0.25),
        ("p0p50", 0.50),
    )
    result.extend(
        Candidate(
            f"r2_scale_{suffix}",
            f"geometry_adapter_f0_s3_r2_scale_a_{suffix}",
            "variant_inference_only",
            "r2_checkpoint_alpha_path",
            alpha,
            source_checkpoint_sha,
        )
        for suffix, alpha in alphas
    )
    return result


def validate_matched_fit_metadata(repo_root: Path) -> dict[str, Any]:
    """Verify that R3 siblings match R2's initializer, membership and update path."""
    run_names = (
        "geometry_adapter_f0_s3_r2",
        "geometry_variation_f0_s3_r3_dictionary_only",
        "geometry_variation_f0_s3_r3_moments_only",
    )
    provenance = {
        run: json.loads((repo_root / "artifacts/runs" / run / "provenance.json").read_text())
        for run in run_names
    }
    matched_fields = (
        "initial_parameter_sha256",
        "fit_membership_identity",
        "normalization_identity",
        "causal_feature_prior",
        "split_manifest_sha256",
    )
    reference = provenance[run_names[0]]
    for run in run_names[1:]:
        for field in matched_fields:
            if provenance[run].get(field) != reference.get(field):
                raise ValueError(f"{run}: R2/R3 matched provenance mismatch in {field}")

    histories = {
        run: json.loads((repo_root / "artifacts/runs" / run / "history.json").read_text())
        for run in run_names
    }
    if any(len(rows) != 1 or rows[0].get("epoch") != 0 for rows in histories.values()):
        raise ValueError("R2/R3 comparison expects one fixed epoch-zero history row")
    diagnostics = {run: histories[run][0].get("update_diagnostics", {}) for run in run_names}
    cumulative = {run: diagnostics[run].get("cumulative", {}) for run in run_names}
    sample_order = {run: diagnostics[run].get("sample_order_identity") for run in run_names}
    baseline_counts = cumulative[run_names[0]]
    baseline_order = sample_order[run_names[0]]
    for run in run_names[1:]:
        if cumulative[run] != baseline_counts:
            raise ValueError(f"{run}: R2/R3 cumulative update counts differ")
        if sample_order[run] != baseline_order:
            raise ValueError(f"{run}: R2/R3 sample_order_identity differs")

    return {
        "pass": True,
        "runs": list(run_names),
        "matched_provenance_fields": list(matched_fields),
        "cumulative_update_counts": baseline_counts,
        "sample_order_identity": baseline_order,
        "intentional_differences": ["causal_feature_control", "gradient_statistics"],
        "provenance_sha256": {
            run: sha256(repo_root / "artifacts/runs" / run / "provenance.json") for run in run_names
        },
        "history_sha256": {
            run: sha256(repo_root / "artifacts/runs" / run / "history.json") for run in run_names
        },
    }


def main() -> int:
    """Score all declared endpoints and bootstrap the top new variant."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", type=Path)
    parser.add_argument("--targets", type=Path)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261018)
    args = parser.parse_args()

    root = args.repo_root.resolve(strict=True)
    packet = root / "research/campaign_2026-10-18/geometry_variations_r3"
    split_path = (
        args.split or root / "research/campaign_2026-09-29/artifacts/splits/fold0.json"
    ).resolve()
    targets_path = (args.targets or root / "cache/valid_targets.npz").resolve()
    scale_plan_path = packet / "postfit_scaling_plan.json"
    scale_plan = json.loads(scale_plan_path.read_text())
    if scale_plan.get("source_run") != "geometry_adapter_f0_s3_r2@ep0":
        raise ValueError("alpha plan no longer identifies the frozen R2 source checkpoint")
    source_checkpoint_sha = str(scale_plan["source_checkpoint_sha256"])
    r2_receipt_path = (
        root / "research/campaign_2026-10-18/geometry_adapter_probe_r2/frozen_parent_check.json"
    )
    r3_recipe = json.loads((packet / "recipe.json").read_text())
    parent_checkpoint_sha = str(r3_recipe["resolved_parent_checkpoint_sha256"])
    frozen_r2 = json.loads(r2_receipt_path.read_text())
    if (
        frozen_r2.get("parent_sha256") != parent_checkpoint_sha
        or frozen_r2.get("child_sha256") != source_checkpoint_sha
    ):
        raise ValueError(
            "R2 frozen-parent receipt, R3 parent, and alpha source identities disagree"
        )
    matched_fit_receipt = validate_matched_fit_metadata(root)
    grade = load_grade_helpers(root)

    split = json.loads(split_path.read_text())
    if int(split["fold"]) != 0:
        raise ValueError("R3 grader expects the frozen F0 split")
    groups = [int(group) for group in split["evaluation"]["valid"]]
    if len(groups) != 468 or len(groups) != len(set(groups)):
        raise ValueError(f"unexpected or duplicate F0 evaluation groups: {len(groups)}")
    component_by_group = {
        int(row["row_group"]): int(row["component"]) for row in split["sequences"]["valid"]
    }
    if any(group not in component_by_group for group in groups):
        raise ValueError("F0 evaluation groups lack source-qualified component identities")
    components = np.asarray([component_by_group[group] for group in groups], dtype=np.int64)
    split_sha = sha256(split_path)

    with np.load(targets_path, allow_pickle=False) as cache:
        target_groups = np.asarray(cache["row_group"])
        if len(np.unique(target_groups)) != len(target_groups):
            raise ValueError("duplicate row groups in target cache")
        target_position = {int(group): index for index, group in enumerate(target_groups)}
        missing_targets = [group for group in groups if group not in target_position]
        if missing_targets:
            raise ValueError(f"target cache misses {len(missing_targets)} F0 groups")
        indices = [target_position[group] for group in groups]
        targets = np.asarray(cache["y"][indices], dtype=np.float32)
        mask = np.asarray(cache["mask"][indices], dtype=bool)
    if targets.ndim != 3 or targets.shape[0] != len(groups) or targets.shape[-1] != 2:
        raise ValueError(f"unexpected target shape: {targets.shape}")
    if mask.shape != targets.shape[:2]:
        raise ValueError(f"unexpected mask shape: {mask.shape}")
    mask &= np.arange(targets.shape[1])[None, :] >= 99
    if not mask.any():
        raise ValueError("official F0 mask has no scored rows after warmup")

    views: dict[str, Any] = {}
    identities: dict[str, dict[str, Any]] = {}
    for candidate in candidates(root, source_checkpoint_sha):
        prediction_path = root / "cache" / f"pred_{candidate.run}_valid.npy"
        sidecar_path = prediction_path.with_suffix(".json")
        if not prediction_path.is_file() or not sidecar_path.is_file():
            raise FileNotFoundError(
                f"missing prediction or sidecar for {candidate.name}: {prediction_path}"
            )
        sidecar = json.loads(sidecar_path.read_text())
        if sidecar.get("run") != candidate.run:
            raise ValueError(f"{candidate.name}: sidecar run identity mismatch")
        if int(sidecar.get("epoch", -1)) != (1 if candidate.name == "parent" else 0):
            raise ValueError(f"{candidate.name}: sidecar epoch identity mismatch")
        if sidecar.get("split_manifest_sha256") != split_sha:
            raise ValueError(f"{candidate.name}: split manifest SHA mismatch or missing")
        declared_groups = [int(group) for group in sidecar.get("row_groups", [])]
        if not declared_groups or len(declared_groups) != len(set(declared_groups)):
            raise ValueError(f"{candidate.name}: missing or duplicate sidecar row groups")
        if not sidecar.get("checkpoint_sha256"):
            raise ValueError(f"{candidate.name}: sidecar does not bind a producer checkpoint")
        if sidecar.get("inference_precision") != "fp32":
            raise ValueError(f"{candidate.name}: expected FP32 prediction precision")
        if candidate.name == "parent" and sidecar["checkpoint_sha256"] != parent_checkpoint_sha:
            raise ValueError("parent prediction does not match the frozen R3 initializer SHA")
        if (
            candidate.name == "r2_full_geometry"
            and sidecar["checkpoint_sha256"] != source_checkpoint_sha
        ):
            raise ValueError("R2 prediction does not match the frozen alpha-source SHA")
        if candidate.source_checkpoint_sha256 is not None:
            declared_source = sidecar.get("source_checkpoint_sha256")
            if (
                declared_source is not None
                and declared_source != candidate.source_checkpoint_sha256
            ):
                raise ValueError(f"{candidate.name}: alpha source checkpoint SHA mismatch")
        provenance_path = root / "artifacts/runs" / candidate.run / "provenance.json"
        checkpoint_path = root / "artifacts/runs" / candidate.run / f"model_ep{sidecar['epoch']}.pt"
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"{candidate.name}: missing producer checkpoint {checkpoint_path}"
            )
        checkpoint_sha = sha256(checkpoint_path)
        if checkpoint_sha != sidecar["checkpoint_sha256"]:
            raise ValueError(
                f"{candidate.name}: checkpoint file SHA differs from prediction sidecar"
            )
        if candidate.role == "variant_fit" and provenance_path.is_file():
            provenance = json.loads(provenance_path.read_text())
            if provenance.get("initial_checkpoint_sha256") != parent_checkpoint_sha:
                raise ValueError(
                    f"{candidate.name}: fitted endpoint did not inherit the frozen parent SHA"
                )
        elif candidate.role == "variant_fit":
            raise FileNotFoundError(f"{candidate.name}: fitted endpoint lacks provenance.json")
        if candidate.role == "variant_inference_only":
            if not provenance_path.is_file():
                raise FileNotFoundError(f"{candidate.name}: alpha alias lacks provenance.json")
            provenance = json.loads(provenance_path.read_text())
            source_record = provenance.get("source_checkpoint", {})
            transform = provenance.get("inference_transform", {})
            if source_record.get("sha256") != candidate.source_checkpoint_sha256:
                raise ValueError(f"{candidate.name}: alpha alias source checkpoint SHA mismatch")
            if float(transform.get("alpha", float("nan"))) != candidate.alpha:
                raise ValueError(f"{candidate.name}: alpha alias transform mismatch")
            if provenance.get("alias_checkpoint_sha256") != checkpoint_sha:
                raise ValueError(f"{candidate.name}: alias checkpoint SHA mismatch")
        prediction_sha = sha256(prediction_path)
        declared_prediction_sha = sidecar.get("prediction_sha256")
        if declared_prediction_sha and prediction_sha != declared_prediction_sha:
            raise ValueError(f"{candidate.name}: prediction sidecar SHA mismatch")
        aligned = grade.aligned(
            {"prediction": prediction_path.relative_to(root).as_posix()}, groups, split_sha
        )
        if isinstance(aligned, str):
            raise ValueError(f"{candidate.name}: {aligned}")
        array, order = aligned
        views[candidate.name] = grade.Lazy(array, order)
        identities[candidate.name] = {
            "run": candidate.run,
            "role": candidate.role,
            "identity_group": candidate.identity_group,
            "is_fitted_endpoint": candidate.role != "variant_inference_only",
            "is_new_r3_fit": candidate.role == "variant_fit",
            "is_independent_seed_replication": False,
            "alpha": candidate.alpha,
            "source_checkpoint_sha256": candidate.source_checkpoint_sha256,
            "producer_checkpoint_sha256": sidecar["checkpoint_sha256"],
            "producer_checkpoint_file_sha256": checkpoint_sha,
            "run_provenance_sha256": (
                sha256(provenance_path) if provenance_path.is_file() else None
            ),
            "prediction_sha256": prediction_sha,
            "sidecar_sha256": sha256(sidecar_path),
            "split_manifest_sha256": sidecar["split_manifest_sha256"],
            "aligned_evaluation_groups": len(order),
        }

    scores = {name: grade.score(targets, mask, view) for name, view in views.items()}
    ranking = sorted(scores, key=lambda name: (-scores[name]["wp"], name))
    variant_names = [name for name in scores if name not in {"parent", "r2_full_geometry"}]
    top_variant = max(variant_names, key=lambda name: scores[name]["wp"])
    loaded = {
        name: views[name].load().astype(np.float32, copy=False)
        for name in {top_variant, "parent", "r2_full_geometry"}
    }
    comparisons: dict[str, Any] = {}
    for reference in ("parent", "r2_full_geometry"):
        candidate_moments, reference_moments = grade.paired_component_moments(
            targets, loaded[top_variant], loaded[reference], mask
        )
        comparisons[f"{top_variant}_minus_{reference}"] = grade.paired_component_bootstrap(
            candidate_moments,
            reference_moments,
            components,
            n_boot=args.bootstrap,
            seed=args.bootstrap_seed,
            centered=True,
        )

    report = {
        "schema": "wnn.geometry-r3-independent-grade.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "grader_sha256": sha256(Path(__file__).resolve()),
        "repo_root": str(root),
        "split_path": str(split_path),
        "split_manifest_sha256": split_sha,
        "r3_recipe_sha256": sha256(packet / "recipe.json"),
        "alpha_plan_sha256": sha256(scale_plan_path),
        "r2_frozen_parent_receipt_sha256": sha256(r2_receipt_path),
        "matched_fit_metadata": matched_fit_receipt,
        "target_cache_path": str(targets_path),
        "fold": 0,
        "evaluation_groups": len(groups),
        "components": int(len(np.unique(components))),
        "selected_rows_after_row99": int(mask.sum()),
        "metric_id": "global_wp_clip2_v1",
        "exposure": "previously exposed F0; adaptive R1/R2/R3 and alpha-screen selection; no clean-holdout claim",
        "candidates": identities,
        "ranking": [
            {"rank": rank, "name": name, **scores[name]}
            for rank, name in enumerate(ranking, start=1)
        ],
        "top_new_variant": top_variant,
        "top_variant_vs_fixed_references": comparisons,
        "alpha_controls": {
            "grid": [-0.5, -0.25, 0.0, 0.25, 0.5, 1.0],
            "alpha0_prediction_alias": "parent",
            "alpha1_prediction_alias": "r2_full_geometry",
            "alpha0_kernel_rounding_caveat": "The cached parent endpoint is the mathematical alpha=0 control. A 271-wide expanded input can use a different GEMM reduction path than its native 211-wide parent, so bitwise prediction equality is not assumed.",
            "alpha_aliases_are_independent_fits": False,
            "source_checkpoint_sha256": source_checkpoint_sha,
        },
        "uncertainty_limitations": "Component bootstrap intervals condition on these fixed predictions and are reported after adaptive selection on exposed F0. They do not adjust for candidate/alpha selection, fitting-seed variation, cross-fold variation, or public transfer. Alpha aliases are correlated inference-only derivatives of R2, not independent fits.",
        "public_score_forecast": None,
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "top_new_variant": top_variant,
                "wp": scores[top_variant]["wp"],
                "output": str(output),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
