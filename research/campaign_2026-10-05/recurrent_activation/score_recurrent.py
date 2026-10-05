"""Verify all six sealed recurrent fits and nine fixed conditional comparisons."""

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

ARMS = tuple(
    f"{alpha}__{optimizer}" for alpha in ("m10", "zero", "p10") for optimizer in common.OPTIMIZERS
)
CONTRASTS = [
    (f"{alpha}__{optimizer}", f"zero__{optimizer}")
    for alpha in ("m10", "p10")
    for optimizer in common.OPTIMIZERS
] + [(f"{alpha}__muon_adamw", f"{alpha}__adamw") for alpha in ("m10", "zero", "p10")]
CONTRASTS += [(f"zero__{optimizer}", f"cudnn_tanh__{optimizer}") for optimizer in common.OPTIMIZERS]


def main() -> None:
    """Refuse incomplete runs, verify provenance, then grade every fixed arm."""
    output = HERE / "RESULT.json"
    if output.exists():
        raise FileExistsError("refusing to overwrite grading result")
    run = HERE / "runs/six_v1"
    report = json.loads((run / "result.json").read_text())
    attempt = json.loads(
        (
            ROOT / "research/campaign_2026-10-04/runner/attempts/e471-recurrent-six-fit-v1.json"
        ).read_text()
    )
    if not (
        attempt["status"] == "finished"
        and attempt["exit_code"] == 0
        and attempt["teardown_verified"]
    ):
        raise ValueError("full fit is not complete with successful teardown")
    spec = json.loads((HERE / "e471_six_fit_v1.sealed.json").read_text())
    sealed = {item["path"]: item["sha256"] for item in spec["code"]}
    wrapper = str((HERE / "fp32_policy_run.py").relative_to(ROOT))
    if report["precision_wrapper_sha256"] != sealed[wrapper]:
        raise ValueError("shared precision wrapper identity differs")
    policy = report["explicit_precision_policy"]
    if (
        policy["mode"] != "fit"
        or policy["cuda_matmul_allow_tf32"]
        or policy["cudnn_allow_tf32"]
        or not policy["cudnn_enabled"]
    ):
        raise ValueError("shared full-FP32 fitting policy differs")
    for name, digest in report["source_sha256"].items():
        relative = str((HERE / name).relative_to(ROOT))
        if digest != sealed[relative] or digest != common.sha256_file(HERE / name):
            raise ValueError("recurrent fit source differs from seal")
    if (
        report["recipe_source_sha256"]
        != sealed[str((ACTIVATION / "fit_activation_optimizer.py").relative_to(ROOT))]
    ):
        raise ValueError("base recipe identity differs")
    if report["seed"] != 271828 or report["metric"] != "global_wp_clip2_v1":
        raise ValueError("seed or metric differs")
    nodes, components = common.load_nodes(run, report)
    records = {item["arm"]: item for item in report["arms"]}
    if len(report["arms"]) != 6 or set(records) != set(ARMS):
        raise ValueError("duplicate, missing or unregistered recurrent arm")
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
    for optimizer in common.OPTIMIZERS:
        key = f"cudnn_tanh__{optimizer}"
        reference = original_loaded[f"tanh__{optimizer}"]
        scores[key] = reference["official"]
        moments[key] = reference["moments"]
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
        "scope": "Six small-model fits on sixteen exposed components; nine conditional contrasts. No public-score, fresh-validation or pooled-training-seed claim.",
        "arm_scores": scores,
        "contrasts": contrasts,
        "primary_contrast_keys": list(contrasts)[:7],
        "historical_backend_diagnostic_keys": list(contrasts)[7:],
        "explicit_precision_policy": policy,
        "attempt": attempt,
        "selected_prediction_sha256": hashes,
        "fit_result_sha256": common.sha256_file(run / "result.json"),
        "scorer_sha256": common.sha256_file(Path(__file__)),
        "shared_scorer_sha256": common.sha256_file(Path(common.__file__)),
    }
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"fits": 6, "contrasts": 9, "output": str(output)}))


if __name__ == "__main__":
    main()
