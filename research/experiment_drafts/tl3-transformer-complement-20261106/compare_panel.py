"""Compare the four fixed-final composed endpoints and their common frozen TL3."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from wnn.evaluation import paired_component_bootstrap
from wnn.splits import file_sha256
from wnn.training.sequence_blend_prediction import validate_blend_producer
from wnn.training.sequence_config import load_config
from wnn.training.sequence_provenance import canonical_sha256, source_identity

# Score/provenance JSON is heterogeneous and validated at this boundary.
SEEDS = (101001, 101002)
ARMS = ("labels", "complement")


def matched_identity(candidate: dict[str, Any], expected_source: dict[str, object]) -> str:
    """Bind actual source, config, data, runtime and normalization before pairing."""
    provenance = candidate["provenance"]
    identity = provenance["identity"]
    if (
        identity["source_sha256"] != expected_source["sha256"]
        or identity["source_files"] != expected_source["files"]
    ):
        raise ValueError("panel source differs from the reviewed implementation")
    if identity["config_sha256"] != canonical_sha256(candidate["config"]):
        raise ValueError("panel config is inconsistent with checkpoint provenance")
    data = identity["data"]
    if identity["data_sha256"] != canonical_sha256(data):
        raise ValueError("panel fitting data provenance digest mismatch")
    if data.get("reference") != candidate.get("reference"):
        raise ValueError("panel fitting reference is inconsistent with provenance")
    comparable_data = {key: value for key, value in data.items() if key != "reference"}
    return canonical_sha256(
        {
            "data": comparable_data,
            "normalization": provenance["normalization"],
            "runtime": identity["runtime"],
            "world_size": identity["world_size"],
            "optimizer": identity["optimizer"],
            "model_parameters": identity["model_parameters"],
            "completed_epochs": provenance["completed_epochs"],
            "global_updates": provenance["global_updates"],
            "checkpoint_policy": provenance["fixed_checkpoint_policy"],
        }
    )


def read_score(path: Path) -> dict[str, Any]:
    """Read bounded official-score evidence and reject incomplete component moments."""
    if not 0 < path.stat().st_size <= 32_000_000:
        raise ValueError("invalid score receipt size")
    value = json.loads(path.read_text())
    if (
        not isinstance(value, dict)
        or value.get("score", {}).get("metric_id") != "global_wp_clip2_v1"
    ):
        raise ValueError("expected an official score receipt")
    components = value.get("component_centered_moments")
    if not isinstance(components, dict) or not components:
        raise ValueError("missing evaluation component moments")
    return value


def contrast(candidate: dict[str, Any], control: dict[str, Any]) -> dict[str, Any]:
    """Resample paired evaluation components conditional on these fixed predictors."""
    if candidate["split_sha256"] != control["split_sha256"]:
        raise ValueError("score split mismatch")
    left, right = candidate["component_centered_moments"], control["component_centered_moments"]
    if set(left) != set(right):
        raise ValueError("evaluation component coverage mismatch")
    keys = sorted(left, key=int)
    arrays = [np.asarray([entry[k] for k in keys], dtype=np.float64) for entry in (left, right)]
    if any(a.shape != (len(keys), 2, 6) or not np.isfinite(a).all() for a in arrays):
        raise ValueError("component moment shape/dtype mismatch")
    result = paired_component_bootstrap(
        arrays[0],
        arrays[1],
        np.asarray([int(k) for k in keys]),
        n_boot=2000,
        seed=271828,
        centered=True,
    )
    for side, score in (("candidate", candidate), ("reference", control)):
        if not np.isclose(result[f"{side}_wp"], score["score"]["wp"], atol=1e-10, rtol=0):
            raise ValueError("component moments do not reproduce the official WP")
    return result


def main() -> None:
    """Require all four frozen cells; report every contrast without selecting a seed."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    packet = Path(__file__).resolve().parent
    repository = packet.parents[2]
    source = source_identity(repository, repository / "scripts/train_sequence.py")
    baseline_path = args.run_root / "baseline-score.json"
    baseline = read_score(baseline_path)
    if baseline.get("schema") != "wnn.tl3-reference-score.v1":
        raise ValueError("missing frozen baseline score")
    ref_hash = baseline["producer"]["reference_sha256"]
    records = {}
    source_shas = set()
    panel_identities = set()
    receipt_hashes = {"baseline": file_sha256(baseline_path)}
    for seed in SEEDS:
        for arm in ARMS:
            cell = f"{arm}_s{seed}"
            path = args.run_root / cell / "fit/score.json"
            score = read_score(path)
            producer = score["producer"]
            fitting = [tuple(member) for member in producer["fit_members"]]
            validate_blend_producer(producer, baseline["split_sha256"], fitting)
            if producer["reference"]["reference_sha256"] != ref_hash:
                raise ValueError("panel reference identity mismatch")
            if producer["reference"]["manifest_sha256"] != baseline["predictions_manifest_sha256"]:
                raise ValueError("panel reference prediction bytes differ from baseline replay")
            if producer["composition"]["alpha"] != 0.1:
                raise ValueError("panel coefficient changed")
            candidate = producer["candidate"]
            if candidate["config"] != load_config(packet / "configs" / f"{cell}.json"):
                raise ValueError("panel cell recipe differs from committed config")
            source_shas.add(candidate["provenance"]["identity"]["source_sha256"])
            panel_identities.add(matched_identity(candidate, source))
            records[cell] = score
            receipt_hashes[cell] = file_sha256(path)
    if len(source_shas) != 1:
        raise ValueError("panel mixes implementation revisions")
    if len(panel_identities) != 1:
        raise ValueError(
            "panel data, normalization, runtime, optimizer or completed budgets differ"
        )
    rows = []
    for seed in SEEDS:
        control, treatment = (records[f"{arm}_s{seed}"] for arm in ARMS)
        rows.append(
            {
                "seed": seed,
                "labels_composite_vs_tl3": contrast(control, baseline),
                "complement_vs_tl3": contrast(treatment, baseline),
                "complement_vs_labels_composite": contrast(treatment, control),
                "scores": {"labels": control["score"], "complement": treatment["score"]},
            }
        )
    improvements = [
        records[f"complement_s{s}"]["score"]["wp"] - baseline["score"]["wp"] for s in SEEDS
    ]
    treatment_deltas = [
        records[f"complement_s{s}"]["score"]["wp"] - records[f"labels_s{s}"]["score"]["wp"]
        for s in SEEDS
    ]
    result = {
        "schema": "wnn.tl3-transformer-complement-panel.v1",
        "status": "complete_exposed_source_screen",
        "primary_reference": baseline["score"],
        "split_sha256": baseline["split_sha256"],
        "receipt_sha256": receipt_hashes,
        "source_sha256": next(iter(source_shas)),
        "paired_results": rows,
        "mean_complement_minus_tl3": float(np.mean(improvements)),
        "mean_complement_minus_control": float(np.mean(treatment_deltas)),
        "followup_budget_trigger_only": bool(
            min(improvements) >= 0
            and np.mean(improvements) >= 0.0005
            and min(treatment_deltas) >= 0
        ),
        "uncertainty": "Two paired fitting seeds, one exposed source split; component intervals condition on fitted models. No public gain or significance established.",
        "next_fit_is_automatic": False,
    }
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
