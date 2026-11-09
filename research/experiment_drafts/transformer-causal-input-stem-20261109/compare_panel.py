"""Grade six fixed endpoints with actual checkpoint bytes and paired component contrasts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from wnn.evaluation import paired_component_bootstrap
from wnn.splits import file_sha256, load_campaign_split
from wnn.training.sequence_blend_prediction import validate_blend_producer
from wnn.training.sequence_cache_ensemble_lineage import validate_producer
from wnn.training.sequence_config import load_config
from wnn.training.sequence_provenance import canonical_sha256, source_identity

# JSON boundaries mix receipts, model identity and component arrays; validated before comparison.
SEEDS = (203201, 203202)
ARMS = ("plain", "k1", "k16")
SPLIT_SHA = "0c4de3a253d23b3a5d5ca73b7aeb7fcae2b974673e4f9ec51e94e593c59d114c"


def read(path: Path) -> dict[str, Any]:
    """Read bounded result metadata, never deserialize model tensors."""
    if not 0 < path.stat().st_size <= 32_000_000:
        raise ValueError("result metadata size is invalid")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("result metadata must be a JSON object")
    return value


def grade(path: Path) -> dict[str, Any]:
    """Require full canonical evaluation and component sufficient moments."""
    value = read(path)
    metric = value.get("score", {})
    if (
        value.get("split_sha256") != SPLIT_SHA
        or value.get("sequences") != 1873
        or metric.get("metric_id") != "global_wp_clip2_v1"
        or metric.get("selected_rows") != 3528273
        or not isinstance(value.get("component_centered_moments"), dict)
    ):
        raise ValueError("not a complete fixed official-valid grade")
    return value


def comparable_identity(candidate: dict[str, Any], kernel: int) -> dict[str, Any]:
    """Remove only the declared extra stem parameters from the common optimizer footprint."""
    provenance = candidate["provenance"]
    identity = provenance["identity"]
    if identity["config_sha256"] != canonical_sha256(candidate["config"]) or identity[
        "data_sha256"
    ] != canonical_sha256(identity["data"]):
        raise ValueError("config/data differs from its provenance hash")
    optimizer = json.loads(json.dumps(identity["optimizer"]))
    stem_groups = [
        group for group in optimizer["groups"] if "input_stem.weight" in group["param_names"]
    ]
    if len(stem_groups) != int(kernel > 0):
        raise ValueError("optimizer does not own exactly the expected stem parameter")
    if kernel:
        group = stem_groups[0]
        if group["muon"] or group["role"] != "adamw_matrix":
            raise ValueError("depthwise input filter must use the canonical AdamW matrix group")
        group["param_names"].remove("input_stem.weight")
        group["parameter_count"] -= 112 * kernel
    parameters = {key: count - 112 * kernel for key, count in identity["model_parameters"].items()}
    return {
        "data": identity["data"],
        "normalization": provenance["normalization"],
        "optimizer_without_declared_stem": optimizer,
        "backbone_parameters": parameters,
        "software": {key: value for key, value in identity["runtime"].items() if key != "devices"},
        "world_size": identity["world_size"],
        "completed_epochs": provenance["completed_epochs"],
        "global_updates": provenance["global_updates"],
    }


def contrast(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    """Resample entire paired components, conditional on these fixed fitted predictors."""
    first, second = left["component_centered_moments"], right["component_centered_moments"]
    if not first or set(first) != set(second):
        raise ValueError("component coverage differs")
    keys = sorted(first, key=int)
    arrays = [
        np.asarray([source[key] for key in keys], dtype=np.float64) for source in (first, second)
    ]
    if any(array.shape != (len(keys), 2, 6) or not np.isfinite(array).all() for array in arrays):
        raise ValueError("invalid centered component moments")
    result = paired_component_bootstrap(
        arrays[0],
        arrays[1],
        np.asarray([int(k) for k in keys]),
        n_boot=2000,
        seed=271828,
        centered=True,
    )
    for name, score in (("candidate_wp", left), ("reference_wp", right)):
        if not np.isclose(result[name], score["score"]["wp"], atol=1e-10, rtol=0):
            raise ValueError("component moments disagree with the canonical score")
    return result


def collect(
    root: Path, packet: Path, fitting: list[tuple[str, int]], with_blend: bool
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Bind all six complete endpoints to the reviewed source and frozen recipes."""
    repository = packet.parents[2]
    source = source_identity(repository, repository / "scripts/train_sequence.py")
    rows, hashes, common = {}, {}, set()
    recipe_templates = set()
    runtimes: dict[int, set[str]] = {seed: set() for seed in SEEDS}
    reference_manifests = set()
    for seed in SEEDS:
        for arm in ARMS:
            cell = f"{arm}_s{seed}"
            fit = root / "runs" / cell / "fit"
            terminal, score = read(fit / "terminal_status.json"), grade(fit / "score.json")
            producer = score["producer"]
            validate_producer(producer, SPLIT_SHA, fitting)
            checkpoint_sha = file_sha256(fit / "model.pt")
            config = load_config(packet / "configs" / f"{cell}.json")
            template = json.loads(json.dumps(config))
            if (
                template.pop("seed") != seed
                or template["model"].pop("input_stem_kernel", 0)
                != {"plain": 0, "k1": 1, "k16": 16}[arm]
            ):
                raise ValueError("frozen cell seed/stem does not match its label")
            recipe_templates.add(canonical_sha256(template))
            approved_sha = file_sha256(fit.parent / "approved_profile.json")
            identity = terminal.get("identity", {})
            if (
                terminal.get("status") != "completed"
                or terminal.get("completed_epochs") != 4
                or terminal.get("global_updates") != 3280
                or identity.get("world_size") != 2
                or checkpoint_sha != terminal.get("model_sha256")
                or checkpoint_sha != producer.get("checkpoint_sha256")
                or producer.get("config") != config
                or producer.get("teacher") is not None
                or "reference" in producer
                or producer.get("provenance", {}).get("identity") != identity
                or producer.get("provenance", {}).get("profile_receipt", {}).get("sha256")
                != approved_sha
                or identity.get("source_sha256") != source["sha256"]
                or identity.get("source_files") != source["files"]
            ):
                raise ValueError(f"incomplete/foreign fixed-final endpoint: {cell}")
            kernel = int(config["model"].get("input_stem_kernel", 0))
            common.add(canonical_sha256(comparable_identity(producer, kernel)))
            runtimes[seed].add(canonical_sha256(identity["runtime"]))
            hashes[cell] = {
                "checkpoint_sha256": checkpoint_sha,
                "score_sha256": file_sha256(fit / "score.json"),
                "approved_profile_sha256": approved_sha,
            }
            rows[cell] = {"primary": score}
            if with_blend:
                mixed = grade(fit / "score-blend.json")
                composite = mixed["producer"]
                validate_blend_producer(composite, SPLIT_SHA, fitting)
                if composite["candidate"] != producer or composite["composition"]["alpha"] != 0.1:
                    raise ValueError("secondary blend uses another candidate or coefficient")
                reference_manifests.add(canonical_sha256(composite["reference"]))
                rows[cell]["blend"] = mixed
                hashes[cell]["blend_score_sha256"] = file_sha256(fit / "score-blend.json")
    if (
        len(recipe_templates) != 1
        or len(common) != 1
        or any(len(values) != 1 for values in runtimes.values())
    ):
        raise ValueError(
            "panel changes data, normalization, backbone, optimizer, budget or paired GPU/runtime identity"
        )
    if with_blend and len(reference_manifests) != 1:
        raise ValueError("secondary mixtures must use identical frozen reference predictions")
    return rows, hashes


def main() -> None:
    """Publish every paired contrast without best-seed, epoch or mixture selection."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--with-blend", action="store_true")
    args = parser.parse_args()
    split = load_campaign_split(args.split_manifest, args.data_root)
    if split.sha256 != SPLIT_SHA or split.valid_groups or len(split.train_groups) != 10325:
        raise ValueError("wrong source split")
    rows, hashes = collect(
        args.run_root,
        Path(__file__).resolve().parent,
        [("train", group) for group in split.train_groups],
        args.with_blend,
    )
    contrasts, primary_deltas, total_deltas = [], [], []
    for seed in SEEDS:
        for outcome in ("primary", "blend") if args.with_blend else ("primary",):
            for candidate, control in (("k16", "k1"), ("k16", "plain"), ("k1", "plain")):
                left, right = (
                    rows[f"{candidate}_s{seed}"][outcome],
                    rows[f"{control}_s{seed}"][outcome],
                )
                result = contrast(left, right)
                contrasts.append(
                    {
                        "seed": seed,
                        "outcome": outcome,
                        "candidate": candidate,
                        "control": control,
                        **result,
                    }
                )
                delta = left["score"]["wp"] - right["score"]["wp"]
                if outcome == "primary" and (candidate, control) == ("k16", "k1"):
                    primary_deltas.append(delta)
                if outcome == "primary" and (candidate, control) == ("k16", "plain"):
                    total_deltas.append(delta)
    report = {
        "schema": "wnn.input-stem-comparison.v1",
        "status": "complete_exposed_fixed4E_screen",
        "split_sha256": SPLIT_SHA,
        "receipt_sha256": hashes,
        "scores": {
            cell: {name: value["score"] for name, value in outcomes.items()}
            for cell, outcomes in rows.items()
        },
        "contrasts": contrasts,
        "mean_k16_minus_k1": float(np.mean(primary_deltas)),
        "followup_budget_trigger_only": bool(
            min(primary_deltas) > 0 and np.mean(primary_deltas) >= 0.0005 and min(total_deltas) >= 0
        ),
        "automatic_next_fit": False,
        "uncertainty": "Two fitting seeds on one exposed split; component bootstrap conditions on fitted predictors. No significance, public gain or CPU-serving result.",
    }
    output = args.run_root / (
        "comparison-with-blend.json" if args.with_blend else "comparison.json"
    )
    with output.open("x") as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
