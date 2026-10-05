"""Recompute all three fixed activation seeds without pooling conditional CIs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import score_activation as original

from wnn.gmetric import global_wp

HERE = Path(__file__).parent
ROOT = HERE.resolve().parents[2]
SEEDS = (271828, 271829, 271830)


def read_complete_fit(seed: int) -> tuple[Path, dict, dict | None]:
    """Require the original run or a successful, torn-down replication worker."""
    run = HERE / "runs" / ("sixteen_v2" if seed == 271828 else f"seed{seed}")
    report = json.loads((run / "result.json").read_text())
    if report.get("seed") != seed or len(report.get("arms", [])) != 16:
        raise ValueError("wrong seed or partial activation result")
    expected_source = {
        name: hashlib.sha256((HERE / name).read_bytes()).hexdigest()
        for name in ("fit_activation_optimizer.py", "tests.py")
    }
    if report.get("source_sha256") != expected_source:
        raise ValueError("numerical source differs from E470")
    attempt = None
    if seed != 271828:
        attempt_path = ROOT / "research/campaign_2026-10-04/runner/attempts" / (
            f"e472-activation-seed{seed}-v1.json"
        )
        attempt = json.loads(attempt_path.read_text())
        if not (
            attempt.get("status") == "finished"
            and attempt.get("exit_code") == 0
            and attempt.get("teardown_verified")
        ):
            raise ValueError("replication worker is not successfully complete")
        wrapper_hash = hashlib.sha256((HERE / "fit_seed_replication.py").read_bytes()).hexdigest()
        if report.get("replication_wrapper_sha256") != wrapper_hash:
            raise ValueError("replication wrapper identity changed")
    return run, report, attempt


def load_recomputed(run: Path, report: dict) -> tuple[dict, np.ndarray, dict]:
    """Verify selected-row alignment, finite exports and every official score."""
    nodes, components = original.load_nodes(run, report)
    by_name = {item["arm"]: item for item in report["arms"]}
    if len(by_name) != 16 or set(by_name) != set(original.ARMS):
        raise ValueError("duplicated or missing fixed arm")
    expected_groups = [node["row_group"] for node in nodes]
    loaded, common_targets, common_counts, export_hashes = {}, None, None, {}
    for name in original.ARMS:
        item = by_name[name]
        if item["steps"] != 474 or item["parameters"] != 8235:
            raise ValueError("fixed fit budget or model shape changed")
        if [seq["row_group"] for seq in item["sequences"]] != expected_groups:
            raise ValueError("component sequence order changed")
        counts = [seq["scored_rows"] for seq in item["sequences"]]
        if any(count <= 0 for count in counts) or sum(counts) != item["scored_rows"]:
            raise ValueError("invalid selected-row segmentation")
        path = run / f"{name}_selected_predictions.npz"
        with np.load(path, allow_pickle=False) as data:
            y = original.finite_float32(data["targets"], "raw targets")
            p = original.finite_float32(data["predictions"], "raw predictions")
        if y.shape != (sum(counts), 2) or p.shape != y.shape:
            raise ValueError("export shape does not match selected rows")
        if common_targets is None:
            common_targets, common_counts = y.copy(), counts
            original.verify_target_export(y, nodes, counts)
        elif counts != common_counts or not np.array_equal(y, common_targets):
            raise ValueError("selected targets differ between arms")
        official = global_wp(y[None], p[None], np.ones((1, len(y)), dtype=bool))
        if not np.isclose(official["wp"], item["wp"], atol=1e-7, rtol=0):
            raise ValueError("recorded score differs from official recomputation")
        if not np.allclose(
            [official["t0"], official["t1"]], item["target_wp"], atol=1e-7, rtol=0
        ):
            raise ValueError("per-target scores differ from official recomputation")
        loaded[name] = {
            "summary": item,
            "official": official,
            "moments": original.sequence_moments(y, p, counts),
        }
        export_hashes[path.name] = original.sha256_file(path)
    return loaded, components, export_hashes


def main() -> None:
    """Retain all 48 scores and 90 conditional contrasts; no model selection."""
    seeds, references = {}, None
    for seed in SEEDS:
        run, report, attempt = read_complete_fit(seed)
        loaded, components, hashes = load_recomputed(run, report)
        if references is None:
            references = components
        elif not np.array_equal(references, components):
            raise ValueError("test components changed between seeds")
        seeds[str(seed)] = {
            "scores": {name: item["official"] for name, item in loaded.items()},
            "contrasts": original.build_contrasts(loaded, components),
            "fit_result_sha256": original.sha256_file(run / "result.json"),
            "prediction_export_sha256": hashes,
            "attempt": attempt,
        }
    summaries = {}
    for name in original.ARMS:
        values = [seeds[str(seed)]["scores"][name]["wp"] for seed in SEEDS]
        summaries[name] = {
            "seed_values": values,
            "mean": float(np.mean(values)),
            "sd": float(np.std(values, ddof=1)),
            "range": [min(values), max(values)],
        }
    result = {
        "seeds": seeds,
        "descriptive_cross_seed": summaries,
        "scope": "48 small-model fits on the same16exposedcomponents. No pooled confidence interval, fresh-generalization or public forecast.",
        "scorer_sha256": original.sha256_file(Path(__file__)),
        "original_scorer_sha256": original.sha256_file(HERE / "score_activation.py"),
    }
    (HERE / "THREE_SEED_ACTIVATION_RESULT.json").write_text(json.dumps(result, indent=2) + "\n")
    eda = HERE / "eda/seed_replication"
    eda.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, optimizer in zip(axes, original.OPTIMIZERS, strict=True):
        for seed in SEEDS:
            values = [
                seeds[str(seed)]["scores"][f"{act}__{optimizer}"]["wp"]
                for act in original.ACTIVATIONS
            ]
            ax.plot(range(8), values, marker="o", label=f"seed {seed}")
        ax.set_xticks(range(8), original.ACTIVATIONS, rotation=65, ha="right")
        ax.set_title(optimizer)
        ax.set_ylabel("Official pooled WP; exposed diagnostic panel")
        ax.legend()
    fig.suptitle("All sixteen recipes across three fixed seeds; no public-score forecast")
    fig.tight_layout()
    fig.savefig(eda / "all_activation_seeds.png", dpi=160)
    plt.close(fig)
    print(json.dumps({"fits": 48, "conditional_contrasts": 90, "output": str(eda)}))


if __name__ == "__main__":
    main()
