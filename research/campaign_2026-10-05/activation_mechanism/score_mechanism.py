"""Verify ten mechanism fits and fourteen fixed conditional comparisons."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
ACTIVATION = HERE.parent / "activation_optimizer"
sys.path.insert(0, str(ACTIVATION))
import score_activation as common  # noqa: E402

from wnn.evaluation import paired_component_bootstrap  # noqa: E402
from wnn.gmetric import global_wp  # noqa: E402

ACTIVATIONS = (
    "gelu",
    "gelu_plus_sine_01",
    "gelu_plus_linear_01",
    "scaled_gelu_11",
    "jax_tanh_gelu_plus_sine_01",
)
ARMS = tuple(
    f"{activation}__{optimizer}" for activation in ACTIVATIONS for optimizer in common.OPTIMIZERS
)
CONTRASTS = [
    (f"{activation}__{optimizer}", f"gelu__{optimizer}")
    for activation in ACTIVATIONS[1:]
    for optimizer in common.OPTIMIZERS
]
CONTRASTS += [
    (f"{activation}__{optimizer}", f"gelu_plus_sine_01__{optimizer}")
    for activation in ACTIVATIONS[2:]
    for optimizer in common.OPTIMIZERS
]


def main() -> None:
    """Refuse incomplete runs, verify provenance, then grade every fixed arm."""
    output = HERE / "RESULT.json"
    if output.exists():
        raise FileExistsError("refusing to overwrite grading result")
    run = HERE / "runs/ten_v1"
    report = json.loads((run / "result.json").read_text())
    attempt = json.loads(
        (
            ROOT / "research/campaign_2026-10-04/runner/attempts/e473-mechanism-ten-fit-v1.json"
        ).read_text()
    )
    if not (
        attempt["status"] == "finished"
        and attempt["exit_code"] == 0
        and attempt["teardown_verified"]
    ):
        raise ValueError("full fit is not complete with successful teardown")
    spec = json.loads((HERE / "e473_ten_fit_v1.sealed.json").read_text())
    sealed = {item["path"]: item["sha256"] for item in spec["code"]}
    for name, digest in report["source_sha256"].items():
        relative = str((HERE / name).relative_to(ROOT))
        if digest != sealed[relative] or digest != common.sha256_file(HERE / name):
            raise ValueError("mechanism fit source differs from seal")
    if (
        report["recipe_source_sha256"]
        != sealed[str((ACTIVATION / "fit_activation_optimizer.py").relative_to(ROOT))]
    ):
        raise ValueError("base recipe identity differs")
    if report["seed"] != 271828 or report["metric"] != "global_wp_clip2_v1":
        raise ValueError("seed or metric differs")
    nodes, components = common.load_nodes(run, report)
    records = {item["arm"]: item for item in report["arms"]}
    if len(report["arms"]) != 10 or set(records) != set(ARMS):
        raise ValueError("duplicate, missing or unregistered mechanism arm")
    scores, moments, hashes = {}, {}, {}
    targets, segmentation = None, None
    for arm in ARMS:
        item = records[arm]
        if item["steps"] != 474 or item["parameters"] != 8235:
            raise ValueError("fixed update/parameter budget differs")
        if [s["row_group"] for s in item["sequences"]] != [n["row_group"] for n in nodes]:
            raise ValueError("export sequence order differs")
        counts = [s["scored_rows"] for s in item["sequences"]]
        path = run / f"{arm}_selected_predictions.npz"
        with np.load(path, allow_pickle=False) as data:
            y = common.finite_float32(data["targets"], "raw targets")
            p = common.finite_float32(data["predictions"], "raw predictions")
        if any(c <= 0 for c in counts) or y.shape != (sum(counts), 2) or p.shape != y.shape:
            raise ValueError("invalid export shape or sequence segmentation")
        if targets is None:
            targets, segmentation = y.copy(), counts
            common.verify_target_export(y, nodes, counts)
        elif counts != segmentation or not np.array_equal(y, targets):
            raise ValueError("target alignment differs between arms")
        score = global_wp(y[None], p[None], np.ones((1, len(y)), dtype=bool))
        if not np.allclose(
            [score["wp"], score["t0"], score["t1"]],
            [item["wp"], *item["target_wp"]],
            atol=1e-7,
            rtol=0,
        ):
            raise ValueError("official score disagrees with fit receipt")
        scores[arm] = score
        moments[arm] = common.sequence_moments(y, p, counts)
        hashes[path.name] = common.sha256_file(path)
    original_run = ACTIVATION / "runs/sixteen_v2"
    original_report = json.loads((original_run / "result.json").read_text())
    original_nodes, original_components = common.load_nodes(original_run, original_report)
    original_loaded, original_targets, _ = common.load_exports(
        original_run, original_report, original_nodes
    )
    if not np.array_equal(original_components, components) or not np.array_equal(
        original_targets, targets
    ):
        raise ValueError("cuDNN backend control target/component alignment differs")
    reproduction = {}
    for activation, historical in (("gelu", "gelu"), ("gelu_plus_sine_01", "gelu_sine_01")):
        for optimizer in common.OPTIMIZERS:
            key = f"{activation}__{optimizer}"
            original_score = original_loaded[f"{historical}__{optimizer}"]["official"]
            reproduction[key] = {
                "historical": original_score,
                "new": scores[key],
                "delta": scores[key]["wp"] - original_score["wp"],
            }
            if not np.isclose(scores[key]["wp"], original_score["wp"], atol=1e-7, rtol=0):
                raise ValueError("repeated unchanged GELU/sine recipe differs from E470")
    contrasts = {
        f"{candidate}_minus_{reference}": paired_component_bootstrap(
            moments[candidate],
            moments[reference],
            components,
            n_boot=2000,
            seed=271828,
            centered=True,
        )
        for candidate, reference in CONTRASTS
    }
    result = {
        "scope": "Ten small-model fits on sixteen exposed components; fourteen conditional contrasts. No public-score, fresh-validation or pooled-training-seed claim.",
        "arm_scores": scores,
        "reproduction_controls": reproduction,
        "contrasts": contrasts,
        "attempt": attempt,
        "selected_prediction_sha256": hashes,
        "fit_result_sha256": common.sha256_file(run / "result.json"),
        "scorer_sha256": common.sha256_file(Path(__file__)),
        "shared_scorer_sha256": common.sha256_file(Path(common.__file__)),
    }
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"fits": 10, "contrasts": 14, "output": str(output)}))


if __name__ == "__main__":
    main()
