from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from wnn.data import SequenceFile
from wnn.evaluation import paired_component_bootstrap, paired_component_moments
from wnn.gmetric import METRIC_ID, GlobalAccumulator
from wnn.splits import load_campaign_split


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def _independent_gate(
    tl3: np.ndarray,
    r3: np.ndarray,
    proxy: np.ndarray,
    common: dict[str, Any],
    half_life: int,
) -> np.ndarray:
    """Rebuild the online mixture with scalar, targetwise state updates."""
    decay = math.exp(math.log(0.5) / half_life)
    complement = 1.0 - decay
    prior = float(common["initial_r3_weight"])
    log_prior = math.log(prior / (1.0 - prior))
    eta = float(common["eta"])
    warmup = int(common["warmup_observations"])
    low = float(common["prior_min_r3_weight"])
    high = float(common["prior_max_r3_weight"])
    output = np.empty((19_901, 2), dtype=np.float32)
    output[0] = 0.75 * tl3[99] + 0.25 * r3[99]

    for target in range(2):
        loss_a = 0.0
        loss_b = 0.0
        normalizer = 0.0
        for position in range(19_900):
            pseudo = min(2.0, max(-2.0, float(proxy[position, target])))
            feedback_weight = abs(pseudo)
            past_row = 99 + position
            now_row = 100 + position
            prior_a = min(2.0, max(-2.0, float(tl3[past_row, target])))
            prior_b = min(2.0, max(-2.0, float(r3[past_row, target])))
            loss_a = decay * loss_a + complement * feedback_weight * (pseudo - prior_a) ** 2
            loss_b = decay * loss_b + complement * feedback_weight * (pseudo - prior_b) ** 2
            normalizer = decay * normalizer + complement * feedback_weight * max(
                pseudo * pseudo, 0.01
            )

            weight_b = prior
            if position + 1 >= warmup:
                normalized_a = loss_a / max(normalizer, 1e-12)
                normalized_b = loss_b / max(normalizer, 1e-12)
                logit = log_prior - eta * (normalized_b - normalized_a)
                weight_b = 1.0 / (1.0 + math.exp(-min(40.0, max(-40.0, logit))))
                weight_b = min(high, max(low, weight_b))
            output[position + 1, target] = (1.0 - weight_b) * float(
                tl3[now_row, target]
            ) + weight_b * float(r3[now_row, target])
    return output


def grade(
    root: Path,
    recipe_path: Path,
    prediction_path: Path,
    prediction_receipt_path: Path,
    proxy_predictions_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    recipe_bytes = recipe_path.read_bytes()
    recipe = json.loads(recipe_bytes)
    group_specs = recipe["selection"]["groups"]
    if recipe["metric_id"] != METRIC_ID:
        raise ValueError(f"unexpected metric: {recipe['metric_id']}")
    if sha256_file(Path(__file__).resolve()) != recipe["code"]["independent_grader_sha256"]:
        raise ValueError("independent grader differs from its frozen source hash")
    if (
        sha256_file(Path(__file__).with_name("build_predictions.py"))
        != recipe["code"]["prediction_builder_sha256"]
    ):
        raise ValueError("label-blind prediction builder differs from the frozen recipe")
    build_receipt = json.loads(prediction_receipt_path.read_text())
    recipe_hash = hashlib.sha256(recipe_bytes).hexdigest()
    if build_receipt["recipe_sha256"] != recipe_hash:
        raise ValueError("prediction receipt is not bound to this frozen recipe")
    if sha256_file(prediction_path) != build_receipt["prediction_sha256"]:
        raise ValueError("prediction cache hash differs from the label-blind builder receipt")
    if sha256_file(proxy_predictions_path) != recipe["proxy"]["predictions_sha256"]:
        raise ValueError("proxy predictions differ from the frozen recipe")

    proxy_recipe_path = root / recipe["proxy"]["recipe_path"]
    proxy_grade_path = root / recipe["proxy"]["grade_path"]
    proxy_execution_path = root / recipe["proxy"]["execution_receipt_path"]
    proxy_fit_path = root / recipe["proxy"]["fit_receipt_path"]
    if sha256_file(proxy_recipe_path) != recipe["proxy"]["recipe_sha256"]:
        raise ValueError("lagged-proxy recipe differs from the frozen gate lineage")
    if sha256_file(proxy_grade_path) != recipe["proxy"]["grade_sha256"]:
        raise ValueError("lagged-proxy grade differs from the frozen gate lineage")
    if sha256_file(proxy_execution_path) != recipe["proxy"]["execution_receipt_sha256"]:
        raise ValueError("lagged-proxy execution receipt differs from the frozen gate lineage")
    if sha256_file(proxy_fit_path) != recipe["proxy"]["fit_receipt_sha256"]:
        raise ValueError("lagged-proxy fit receipt differs from the frozen gate lineage")
    proxy_recipe = json.loads(proxy_recipe_path.read_text())
    if proxy_recipe["selection"]["evaluation_groups"] != [
        {key: int(group[key]) for key in ("component", "row_group", "seq_ix")}
        for group in group_specs
    ]:
        raise ValueError("gate evaluation groups differ from the proxy's frozen evaluation order")
    fit_components = {int(item["component"]) for item in proxy_recipe["selection"]["fit_groups"]}
    if fit_components.intersection(int(group["component"]) for group in group_specs):
        raise ValueError("proxy fitting and gate evaluation components overlap")

    split = load_campaign_split(root / recipe["selection"]["split_manifest_path"], root)
    if split.sha256 != recipe["selection"]["split_manifest_sha256"]:
        raise ValueError("source-qualified split hash differs from the frozen gate recipe")
    component_by_group = {
        group: component for side, group, component in split.components if side == "valid"
    }
    sidecar_members: dict[str, dict[int, tuple[int, int]]] = {}
    for name, identity in recipe["expert_sources"].items():
        sidecar = json.loads((root / identity["sidecar_path"]).read_text())
        row_groups = [int(value) for value in sidecar["row_groups"]]
        sequence_ids = [int(value) for value in sidecar["sequence_ids"]]
        if sidecar["split_manifest_sha256"] != split.sha256:
            raise ValueError(f"{name} sidecar split differs from the frozen source split")
        sidecar_members[name] = {
            group: (index, sequence_ids[index]) for index, group in enumerate(row_groups)
        }
        if len(sidecar_members[name]) != len(row_groups):
            raise ValueError(f"{name} cache sidecar contains duplicate row groups")
    for group in group_specs:
        row_group = int(group["row_group"])
        component = int(group["component"])
        seq_ix = int(group["seq_ix"])
        if component_by_group.get(row_group) != component:
            raise ValueError(f"row group {row_group} no longer maps to its recipe component")
        for name in ("tl3", "r3"):
            cache_index, cache_seq_ix = sidecar_members[name][row_group]
            if cache_seq_ix != seq_ix or cache_index != int(group[f"{name}_cache_index"]):
                raise ValueError(f"{name} sidecar identity differs for row group {row_group}")

    predictions = np.load(prediction_path, mmap_mode="r", allow_pickle=False)
    names = list(recipe["endpoints"])
    rows_per_group = 19_901
    expected_shape = (len(names), len(group_specs) * rows_per_group, 2)
    if predictions.shape != expected_shape or predictions.dtype != np.float32:
        raise ValueError(
            f"unexpected gate prediction array: {predictions.shape}/{predictions.dtype}"
        )
    if not np.isfinite(predictions).all():
        raise ValueError("gate prediction cache contains nonfinite values")

    proxy_predictions = np.load(proxy_predictions_path, mmap_mode="r", allow_pickle=False)
    sources: dict[str, np.ndarray] = {}
    for name, source in recipe["expert_sources"].items():
        source_path = root / source["prediction_path"]
        if sha256_file(source_path) != source["prediction_sha256"]:
            raise ValueError(f"{name} expert cache differs from the frozen recipe")
        if sha256_file(root / source["sidecar_path"]) != source["sidecar_sha256"]:
            raise ValueError(f"{name} expert sidecar differs from the frozen recipe")
        sources[name] = np.load(source_path, mmap_mode="r", allow_pickle=False)

    # Reconstruct and verify the complete label-blind gate output before this
    # grader opens a target or scoring-mask payload.
    for group_index, group in enumerate(group_specs):
        start = group_index * rows_per_group
        end = start + rows_per_group
        row_group = int(group["row_group"])
        tl3 = np.asarray(sources["tl3"][int(group["tl3_cache_index"])], dtype=np.float32)
        r3 = np.asarray(sources["r3"][int(group["r3_cache_index"])], dtype=np.float32)
        proxy_start = int(group["proxy_offset"])
        proxy_end = proxy_start + int(group["proxy_rows"])
        proxy_group = np.asarray(proxy_predictions[proxy_start:proxy_end], dtype=np.float64)
        actual = {
            name: np.asarray(predictions[index, start:end], dtype=np.float32)
            for index, name in enumerate(names)
        }
        if not np.array_equal(actual["tl3"], tl3[99:]):
            raise ValueError(f"TL3 baseline output differs from source cache in group {row_group}")
        if not np.array_equal(actual["r3"], r3[99:]):
            raise ValueError(f"R3 baseline output differs from source cache in group {row_group}")
        if not np.allclose(
            actual["fixed_75_tl3_25_r3"], 0.75 * tl3[99:] + 0.25 * r3[99:], rtol=0.0, atol=1e-7
        ):
            raise ValueError(f"fixed blend differs from source caches in group {row_group}")
        for gate in recipe["gate"]["gate_configs"]:
            name = str(gate["id"])
            expected = _independent_gate(
                tl3, r3, proxy_group, recipe["gate"], int(gate["half_life_steps"])
            )
            if not np.allclose(actual[name], expected, rtol=0.0, atol=2e-7):
                max_error = float(np.max(np.abs(actual[name] - expected)))
                raise ValueError(
                    f"independent gate parity failed for {name}, max error {max_error}"
                )

    scorers = {name: GlobalAccumulator() for name in names}
    per_group_moments = {name: [] for name in names}
    tl3_moments: list[np.ndarray] = []
    components: list[int] = []
    source = SequenceFile(root / "datasets/valid.parquet")

    for group_index, group in enumerate(group_specs):
        row_group = int(group["row_group"])
        sequence = source.read(row_group, targets=True, mask=True)
        if sequence.seq_ix != int(group["seq_ix"]):
            raise ValueError(f"sequence identity mismatch at row group {row_group}")
        if sequence.targets is None or sequence.scored is None:
            raise ValueError(f"target or scoring mask missing at row group {row_group}")
        if sequence.features.shape[0] != 20_000:
            raise ValueError(f"unexpected sequence length in row group {row_group}")

        start = group_index * rows_per_group
        target = np.asarray(sequence.targets[99:], dtype=np.float32)
        mask = np.asarray(sequence.scored[99:], dtype=bool)
        component = int(group["component"])
        components.append(component)
        for name in names:
            output = np.asarray(
                predictions[names.index(name), start : start + rows_per_group], dtype=np.float32
            )
            if not np.isfinite(output).all():
                raise ValueError(f"nonfinite endpoint predictions in {name}")
            scorers[name].add(target, output, mask, group=component)
            candidate_moments, _ = paired_component_moments(
                target[None, ...], output[None, ...], output[None, ...], mask[None, ...]
            )
            per_group_moments[name].append(candidate_moments[0])
        tl3_moments.append(np.asarray(per_group_moments["tl3"][-1], dtype=np.float64))

    scores = {name: scorer.result() for name, scorer in scorers.items()}
    if len(set(components)) != len(components):
        raise ValueError("bootstrap groups must correspond to unique source-qualified components")
    component_array = np.asarray(components, dtype=np.int64)
    baseline = np.asarray(tl3_moments, dtype=np.float64)
    fixed_control = np.asarray(per_group_moments["fixed_75_tl3_25_r3"], dtype=np.float64)
    comparisons: dict[str, Any] = {}
    for name in names:
        if name == "tl3":
            continue
        candidate = np.asarray(per_group_moments[name], dtype=np.float64)
        comparisons[f"{name}_vs_tl3"] = paired_component_bootstrap(
            candidate, baseline, component_array, n_boot=2_000, seed=20261020, centered=True
        )
        if name not in {"r3", "fixed_75_tl3_25_r3"}:
            comparisons[f"{name}_vs_fixed_75_25"] = paired_component_bootstrap(
                candidate,
                fixed_control,
                component_array,
                n_boot=2_000,
                seed=20261021,
                centered=True,
            )

    selected_rows = next(iter(scorers.values())).selected_rows
    if any(scorer.selected_rows != selected_rows for scorer in scorers.values()):
        raise ValueError("endpoint mask row counts differ")
    ranking = sorted(names, key=lambda name: -float(scores[name]["wp"]))
    result = {
        "schema": "wnn.causal_proxy_gate_grade.v1",
        "id": "OPS-ONLINE-CAUSAL-PROXY-GATE-GRADE-20261018-R1",
        "status": "independently_reconstructed_and_scored",
        "metric_id": METRIC_ID,
        "recipe_sha256": recipe_hash,
        "prediction_sha256": sha256_file(prediction_path),
        "prediction_receipt_sha256": sha256_file(prediction_receipt_path),
        "independent_grader_sha256": sha256_file(Path(__file__).resolve()),
        "proxy_prediction_sha256": sha256_file(proxy_predictions_path),
        "evaluation_group_count": len(group_specs),
        "evaluation_components": components,
        "selected_rows": selected_rows,
        "mask": "official sequence scoring mask AND row index >= 99",
        "gate_reconstruction": "independent scalar targetwise recurrence matched label-blind prediction cache within 2e-7; no targets or masks enter this reconstruction",
        "exposure": recipe["selection"]["exposure"],
        "tl3_ancestry": recipe["expert_sources"]["tl3"]["ancestry"],
        "scores": scores,
        "point_ranking_descriptive_only": ranking,
        "paired_component_bootstrap_vs_tl3_and_fixed_control": comparisons,
        "bootstrap_scope": "conditional on one frozen pseudo-label proxy and cached expert predictions; resamples 32 exposed components and excludes seed, fold, ancestry, model-selection and public-transfer uncertainty",
        "interpretation_limits": [
            "The pseudo-label is y[t-1] predicted from x[t]; label maturity is undocumented and actual feedback is not established.",
            "The 32 F0 components are repeatedly exposed development data, and TL3 exact training/teacher/initializer ancestry is unknown.",
            "The experiment tests one fitted linear proxy, two frozen experts and three predeclared half-lives; it does not reject broader online learning if this screen is weak.",
            "Any positive result requires source-qualified cross-fold/fitting-seed replication and a deterministic callback implementation before a competition claim.",
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(output_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Independently reconstruct and grade the causal proxy gate."
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--prediction-receipt", type=Path, required=True)
    parser.add_argument("--proxy-predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    grade(
        args.root.resolve(strict=True),
        args.recipe.resolve(strict=True),
        args.predictions.resolve(strict=True),
        args.prediction_receipt.resolve(strict=True),
        args.proxy_predictions.resolve(strict=True),
        args.output.resolve(),
    )


if __name__ == "__main__":
    main()
