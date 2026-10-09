"""Evaluate the preregistered optimizer/loss quartet after independent review."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "src"))

_grade_helpers = import_module("grade_recovered_loss_pair")
_schedule_semantics = import_module("schedule_semantics")
_evaluation = import_module("wnn.evaluation")
_gmetric = import_module("wnn.gmetric")
ROOT = _grade_helpers.ROOT
read_pinned = _grade_helpers.read_pinned
sha256 = _grade_helpers.sha256
validate_sidecar = _grade_helpers.validate_sidecar
audit_quartet_schedules = _schedule_semantics.audit_quartet_schedules
paired_component_bootstrap = _evaluation.paired_component_bootstrap
paired_component_moments = _evaluation.paired_component_moments
GlobalAccumulator = _gmetric.GlobalAccumulator
reference_global_wp = _gmetric.reference_global_wp

# Producer provenance and root review receipts have heterogeneous JSON schemas.
Record = dict[str, Any]


def average_raw(predictions: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    """Average raw predictions before the official scorer clips the mixture."""
    if not weights or any(weight <= 0 for weight in weights.values()):
        raise ValueError("blend weights must be positive")
    if abs(sum(weights.values()) - 1.0) > 1e-12:
        raise ValueError("blend weights must sum to one")
    first = predictions[next(iter(weights))]
    result = np.zeros(first.shape, dtype=np.float64)
    for name, weight in weights.items():
        if predictions[name].shape != first.shape:
            raise ValueError("prediction shapes differ")
        result += np.asarray(predictions[name], dtype=np.float64) * weight
    return result.astype(np.float32)


def load_arms(
    root: Path,
    spec: Record,
    binding: Record,
    groups: list[int],
    component_map: dict[tuple[str, int], int],
) -> tuple[Record, Record, Record]:
    """Check exact reviewed artifacts, excluded components and shared controls."""
    if binding.get("status") != "independently_reviewed_complete_quartet":
        raise ValueError("quartet terminal review is pending")
    if set(binding["arms"]) != set(spec["singles"]):
        raise ValueError("binding does not contain the four declared cells")
    evaluation_components = {component_map["valid", group] for group in groups}
    predictions, identities = {}, {}
    common_controls = None
    common_fit = None
    schedule_inputs: Record = {}
    for name, arm in binding["arms"].items():
        review = read_pinned(root, arm["terminal_review"])
        reviewed_r5 = (
            review.get("schema") == "wnn.root-r5-terminal-review.r1"
            and review.get("state") == "verified_for_independent_grade_and_native_control_transport"
        )
        if review.get("status") != "pass" and not reviewed_r5:
            raise ValueError(f"terminal artifact review did not pass: {name}")
        provenance = read_pinned(root, arm["provenance"])
        args = provenance["args"]
        if args.get("init") or args.get("distill"):
            raise ValueError(f"undeclared teacher/initializer: {name}")
        expected_optimizer = "muon" if name.startswith("muon") else "adamw"
        expected_loss = "huber" if name.endswith("huber_d1") else "wmse"
        if args.get("optim") != expected_optimizer or args.get("loss") != expected_loss:
            raise ValueError(f"optimizer/loss cell differs: {name}")
        if expected_loss == "huber" and args.get("huber_delta") != 1.0:
            raise ValueError(f"Huber delta differs: {name}")
        if provenance.get("split_manifest_sha256") != spec["split"]["sha256"]:
            raise ValueError(f"fitting split differs: {name}")
        fit = provenance["fit_row_groups"]
        fit_components = {
            component_map[source, int(group)] for source, rows in fit.items() for group in rows
        }
        if fit_components & evaluation_components:
            raise ValueError(f"fit/evaluation leakage: {name}")
        if len(fit.get("train", [])) != 350 or fit.get("valid"):
            raise ValueError(f"fitting membership differs: {name}")
        if common_fit is None:
            common_fit = fit
        if fit != common_fit:
            raise ValueError(f"paired fitting membership/order differs: {name}")
        controls = {
            key: sha256(root / value["path"]) for key, value in arm["shared_controls"].items()
        }
        if set(controls) != {"initial_state", "normalization", "sample_orders"}:
            raise ValueError(f"shared control artifacts missing: {name}")
        if any(controls[key] != value["sha256"] for key, value in arm["shared_controls"].items()):
            raise ValueError(f"shared control bytes changed: {name}")
        if common_controls is None:
            common_controls = controls
        if controls != common_controls:
            raise ValueError(f"paired controls differ: {name}")
        schedule_ref = arm.get("schedule_trace")
        update_ref = arm.get("update_trace")
        if not isinstance(schedule_ref, dict) or not isinstance(update_ref, dict):
            raise ValueError(f"actual per-arm schedule/update trace pins are missing: {name}")
        schedule = read_pinned(root, schedule_ref)
        update_path = root / update_ref["path"]
        update_digest = sha256(update_path)
        if update_digest != update_ref["sha256"]:
            raise ValueError(f"update trace bytes changed: {name}")
        update_rows = [
            json.loads(line) for line in update_path.read_text().splitlines() if line.strip()
        ]
        schedule_inputs[name] = {
            "optimizer": expected_optimizer,
            "schedule_trace": schedule,
            "update_rows": update_rows,
        }
        path = root / arm["prediction"]["path"]
        digest = sha256(path)
        if digest != arm["prediction"]["sha256"]:
            raise ValueError(f"prediction bytes changed: {name}")
        sidecar = read_pinned(root, arm["sidecar"])
        validate_sidecar(sidecar, arm, spec, groups, digest)
        prediction = np.load(path, mmap_mode="r", allow_pickle=False)
        if prediction.shape != (468, 20_000, 2) or prediction.dtype != np.float32:
            raise ValueError(f"wrong prediction shape/dtype: {name}")
        if not np.isfinite(prediction).all():
            raise ValueError(f"nonfinite prediction: {name}")
        predictions[name] = prediction
        identities[name] = {
            "prediction_sha256": digest,
            "sidecar_sha256": arm["sidecar"]["sha256"],
            "checkpoint_sha256": arm["checkpoint_sha256"],
            "terminal_review_sha256": arm["terminal_review"]["sha256"],
            "controls": controls,
            "schedule_trace_sha256": schedule_ref["sha256"],
            "schedule_trace_path": schedule_ref["path"],
            "update_trace_sha256": update_digest,
            "update_trace_path": update_ref["path"],
        }
    schedule_semantics = audit_quartet_schedules(schedule_inputs)
    return predictions, identities, schedule_semantics


def grade(
    root: Path, spec_path: Path, spec_sha: str, bindings_path: Path, bindings_sha: str, output: Path
) -> None:
    """Grade all eleven endpoints together without selecting blend weights."""
    if output.exists() or output.with_suffix(".moments.npz").exists():
        raise ValueError("preserve prior result paths")
    spec = read_pinned(root, {"path": str(spec_path), "sha256": spec_sha})
    binding = read_pinned(root, {"path": str(bindings_path), "sha256": bindings_sha})
    if spec.get("status") != "frozen_before_muon_fit_results":
        raise ValueError("comparison is not preregistered")
    if binding.get("spec_sha256") != spec_sha:
        raise ValueError("binding refers to another comparison")
    split = read_pinned(root, spec["split"])
    groups = list(map(int, split["evaluation"]["valid"]))
    component_map = {
        (source, int(row["row_group"])): int(row["component"])
        for source, rows in split["sequences"].items()
        for row in rows
    }
    components = np.asarray([component_map["valid", group] for group in groups])
    if len(groups) != 468 or len(set(groups)) != 468 or len(np.unique(components)) != 141:
        raise ValueError("evaluation membership differs")
    predictions, identities, schedule_semantics = load_arms(
        root, spec, binding, groups, component_map
    )
    target_path = root / spec["targets"]["path"]
    if sha256(target_path) != spec["targets"]["sha256"]:
        raise ValueError("target cache changed")
    with np.load(target_path, allow_pickle=False) as cache:
        positions = {int(group): i for i, group in enumerate(cache["row_group"])}
        if len(positions) != len(cache["row_group"]) or not set(groups).issubset(positions):
            raise ValueError("duplicate or missing target groups")
        indices = [positions[group] for group in groups]
        targets = np.asarray(cache["y"][indices], dtype=np.float32)
        mask = np.asarray(cache["mask"][indices], dtype=bool)
    mask &= (np.arange(20_000) >= 99)[None, :]
    endpoints = {name: {name: 1.0} for name in spec["singles"]} | spec["blends"]
    if len(endpoints) != 11:
        raise ValueError("expected eleven declared endpoints")
    scores, moments = {}, {}
    for name, weights in endpoints.items():
        prediction = average_raw(predictions, weights)
        accumulator = GlobalAccumulator()
        for start in range(0, len(groups), 4):
            selection = slice(start, start + 4)
            accumulator.add(targets[selection], prediction[selection], mask[selection])
        scores[name] = accumulator.result()
        independent = reference_global_wp(targets, prediction, mask)
        if abs(scores[name]["wp"] - independent) > 1e-12:
            raise ValueError(f"independent metric disagreement: {name}")
        scores[name]["independent_centered_formula_wp"] = independent
        moments[name], _ = paired_component_moments(targets, prediction, prediction, mask)
    contrasts = {(name, "adamw_wmse") for name in endpoints if name != "adamw_wmse"}
    contrasts.add(("muon_huber_d1", "adamw_huber_d1"))
    contrasts.update(
        (name, member) for name, weights in spec["blends"].items() for member in weights
    )
    comparisons = {
        f"{candidate}_minus_{reference}": paired_component_bootstrap(
            moments[candidate],
            moments[reference],
            components,
            n_boot=spec["bootstrap"]["replicates"],
            seed=spec["bootstrap"]["seed"],
            centered=True,
        )
        for candidate, reference in sorted(contrasts)
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output.with_suffix(".moments.npz"),
        row_groups=np.asarray(groups),
        components=components,
        **moments,
    )
    result = {
        "created_utc": datetime.now(UTC).isoformat(),
        "spec_sha256": spec_sha,
        "bindings_sha256": bindings_sha,
        "grader_sha256": sha256(Path(__file__)),
        "artifact_identities": identities,
        "schedule_semantics": schedule_semantics,
        "scores": scores,
        "comparisons": comparisons,
        "primary_comparison": "adamw_wmse__muon_wmse_minus_adamw_wmse",
        "metric_id": "global_wp_clip2_v1",
        "evaluation_groups": 468,
        "evaluation_components": 141,
        "exposure": spec["exposure"],
        "uncertainty": spec["bootstrap"]["limitations"],
        "expected_public_score": None,
        "moments_sha256": sha256(output.with_suffix(".moments.npz")),
    }
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(output), "scores": scores}))


def main() -> None:
    """Require both frozen comparison and reviewed terminal artifact bindings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--spec-sha256", required=True)
    parser.add_argument("--bindings", type=Path, required=True)
    parser.add_argument("--bindings-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    grade(args.root, args.spec, args.spec_sha256, args.bindings, args.bindings_sha256, args.output)


if __name__ == "__main__":
    main()
