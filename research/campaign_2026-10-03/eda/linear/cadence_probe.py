"""Bounded, frozen comparison of causal low-pass and same-phase linear filters.

Run each subcommand through continuation/run_cpu.sh eda-linear .venv/bin/python -B.
Preparation freezes samples without reading targets; run fits only the declared models.
Render reads saved results, so headless notebook rendering does not repeat fitting.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import traceback
from typing import Any

import numpy as np
import pyarrow.parquet as pq
from scipy.signal import lfilter

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
FEATURES = (
    [f"i0_p{i}" for i in range(22)] + [f"i0_v{i}" for i in range(22)]
    + [f"i0_dp{i}" for i in range(4)] + [f"i0_dv{i}" for i in range(4)]
    + [f"i1_p{i}" for i in range(22)] + [f"i1_v{i}" for i in range(22)]
    + [f"i1_dp{i}" for i in range(4)] + [f"i1_dv{i}" for i in range(4)]
    + [f"a{i}" for i in range(8)]
)
NAMES = ["levels", "ema", "phase8", "phase7", "cycle8"]
LAMBDA = 0.001
STRIDE = 31


def save_json(name: str, data: dict[str, Any]) -> None:
    (OUT / name).write_text(json.dumps(data, indent=2) + "\n")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def update_state(phase: str, **changes: Any) -> None:
    state = json.loads((OUT / "STATE.json").read_text())
    state.update(phase=phase, **changes)
    save_json("STATE.json", state)


def sequence_id(pf: pq.ParquetFile, rg: int) -> int:
    column = pf.metadata.row_group(rg).column(pf.schema_arrow.get_field_index("seq_ix"))
    stat = column.statistics
    if stat is not None and stat.has_min_max and stat.min == stat.max:
        return int(stat.min)
    values = pf.read_row_group(rg, columns=["seq_ix"], use_threads=False).column(0)
    return int(values[0].as_py())


def prepare() -> None:
    if (OUT / "cadence_results.json").exists():
        raise RuntimeError("Results already exist; preserve them and choose a new experiment name.")
    train = pq.ParquetFile(ROOT / "datasets/train.parquet")
    valid = pq.ParquetFile(ROOT / "datasets/valid.parquet")
    with np.load(OUT / "gram_v1.npz") as gram:
        prior_train = gram["rgs"].astype(int)
    with np.load(OUT / "univariate.npz") as uni:
        prior_valid = uni["rgs"].astype(int)
    sources = {"univariate_and_eval_seed42": prior_valid.tolist(),
               "fixed_probes_and_figures": [0, 1, 2, 3, 500, 1000]}
    for name, seed, count in [("acf_seed0", 0, 60), ("mask_seed3", 3, 120),
                              ("phase_seed1", 1, 40), ("difference_fit_seed7", 7, 24),
                              ("possibly_unfinished_decomposition_seed5", 5, 40)]:
        sources[name] = np.random.default_rng(seed).choice(
            valid.metadata.num_row_groups, count, replace=False).astype(int).tolist()
    exposed = set(rg for values in sources.values() for rg in values)
    candidates = np.array(sorted(set(range(valid.metadata.num_row_groups)) - exposed))
    tr = np.random.default_rng(1732).choice(prior_train, 128, replace=False).astype(int)
    va = np.random.default_rng(1733).choice(candidates, 64, replace=False).astype(int)
    manifest = {
        "prior_valid_sources": sources,
        "prior_valid_union_count": len(exposed),
        "prior_valid_union_row_groups": sorted(exposed),
        "prior_valid_union_seq_ix": [sequence_id(valid, rg) for rg in sorted(exposed)],
        "prior_train_row_groups": prior_train.tolist(),
        "prior_train_seq_ix": [sequence_id(train, int(rg)) for rg in prior_train],
        "train_row_groups": tr.tolist(),
        "train_seq_ix": [sequence_id(train, int(rg)) for rg in tr],
        "valid_row_groups": va.tolist(),
        "valid_seq_ix": [sequence_id(valid, int(rg)) for rg in va],
        "valid_overlap_with_known_own_prior": len(set(va) & exposed),
        "train_prior_exposure": "All 128 were already used in predecessor's 400-sequence fit.",
        "valid_prior_exposure": "Disjoint from recoverable own scripted samples; unrecorded/campaign exposure unknown.",
        "feature_order": FEATURES, "training_stride": STRIDE,
        "datasets": {split: {"row_groups": pf.metadata.num_row_groups,
                              "rows": pf.metadata.num_rows,
                              "bytes": (ROOT / f"datasets/{split}.parquet").stat().st_size}
                     for split, pf in [("train", train), ("valid", valid)]},
        "source_sha256": {name: digest(OUT / name) for name in
                          ["cadence_probe.py", "gram_v1.npz", "univariate.npz"]},
    }
    save_json("cadence_manifest.json", manifest)
    hypothesis = json.loads((OUT / "HYPOTHESES.json").read_text())
    hypothesis["frozen_manifest_sha256"] = digest(OUT / "cadence_manifest.json")
    save_json("HYPOTHESES.json", hypothesis)
    update_state("samples_and_hypotheses_frozen_before_scoring",
                 inspected_sequence_ids="cadence_manifest.json contains all recoverable old and new seq_ix IDs",
                 commands=["run_cpu.sh eda-linear .venv/bin/python -B research/campaign_2026-10-03/eda/linear/cadence_probe.py prepare"],
                 artifacts=["RECOVERY_SUMMARY.txt", "STATE.json", "HYPOTHESES.json", "cadence_manifest.json", "cadence_probe.py"])
    print("Frozen manifest: 128 exposed training sequences, 64 own-disjoint validation sequences;",
          len(exposed), "prior valid sequences excluded.", flush=True)


def ema(x: np.ndarray, alpha: float) -> np.ndarray:
    y, _ = lfilter([alpha], [1.0, -(1.0 - alpha)], x, axis=0,
                  zi=(1.0 - alpha) * x[:1].astype(np.float64))
    return y.astype(np.float32)


def phase_ema(x: np.ndarray, alpha: float, period: int) -> np.ndarray:
    result = np.empty_like(x)
    effective = 1.0 - (1.0 - alpha) ** period
    for phase in range(period):
        if phase < len(x):
            result[phase::period] = ema(x[phase::period], effective)
    return result


def trailing_mean(x: np.ndarray, period: int = 8) -> np.ndarray:
    padded = np.concatenate([np.repeat(x[:1], period - 1, axis=0), x])
    cumulative = np.concatenate([np.zeros((1, x.shape[1])),
                                 np.cumsum(padded, axis=0, dtype=np.float64)])
    return ((cumulative[period:] - cumulative[:-period]) / period).astype(np.float32)


def transform(x: np.ndarray, name: str) -> np.ndarray:
    if name == "levels":
        return x
    if name == "ema":
        smooth = [ema(x, 1 / horizon) for horizon in (32, 256)]
    elif name in ("phase8", "phase7"):
        period = int(name[-1])
        smooth = [phase_ema(x, 1 / horizon, period) for horizon in (32, 256)]
    elif name == "cycle8":
        averaged = trailing_mean(x)
        smooth = [ema(averaged, 1 / horizon) for horizon in (32, 256)]
    else:
        raise ValueError(name)
    return np.concatenate([x] + smooth, axis=1)


def online_transform(x: np.ndarray, name: str) -> np.ndarray:
    """Independent row-at-a-time reference: no targets, future data or mask."""
    if name == "levels":
        return x.copy()
    period = int(name[-1]) if name.startswith("phase") else 1
    state = np.zeros((2, period, x.shape[1]), dtype=np.float64)
    seen = np.zeros(period, dtype=bool)
    buffer = np.repeat(x[:1].astype(np.float64), 8, axis=0)
    outputs = []
    for step, row in enumerate(x):
        phase = step % period
        value = row.astype(np.float64)
        if name == "cycle8":
            buffer[step % 8] = value
            value = buffer.mean(axis=0).astype(np.float32).astype(np.float64)
        for j, horizon in enumerate((32, 256)):
            alpha = 1.0 - (1.0 - 1.0 / horizon) ** period
            state[j, phase] = (value if not seen[phase] else
                               alpha * value + (1.0 - alpha) * state[j, phase])
        seen[phase] = True
        outputs.append(np.concatenate([row, state[0, phase], state[1, phase]]))
    return np.array(outputs, dtype=np.float32)


def load(pf: pq.ParquetFile, rg: int, scored: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    columns = FEATURES + ["t0", "t1", "need_prediction"] + (["is_scored"] if scored else [])
    table = pf.read_row_group(rg, columns=columns, use_threads=False)
    x = np.column_stack([table.column(name).to_numpy() for name in FEATURES]).astype(np.float32)
    y = np.column_stack([table.column(name).to_numpy() for name in ("t0", "t1")]).astype(np.float64)
    mask = table.column("need_prediction").to_numpy()
    if scored:
        mask = mask & table.column("is_scored").to_numpy()
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError(f"Nonfinite data in row group {rg}")
    return x, y, mask


def design(x: np.ndarray, name: str, selection: np.ndarray) -> np.ndarray:
    selected = transform(x, name)[selection].astype(np.float64)
    return np.column_stack([selected, np.ones(len(selected))])


def solve(g: np.ndarray, b: np.ndarray) -> np.ndarray:
    total = g[-1, -1]
    scale = np.sqrt(np.maximum(np.diag(g) / total, 1e-12))
    scale[-1] = 1.0
    penalty = np.eye(len(b)) * (LAMBDA * total)
    penalty[-1, -1] = 0.0
    return np.linalg.solve(g / np.outer(scale, scale) + penalty, b / scale) / scale


def moments(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    p = np.clip(prediction, -2, 2).astype(np.float64)
    y = np.clip(target, -2, 2).astype(np.float64)
    w = np.abs(y)
    return np.array([w.sum(), w @ y, w @ p, w @ (y * y), w @ (p * p), w @ (y * p)])


def correlation(moment: np.ndarray) -> np.ndarray:
    sw, sy, sp, syy, spp, syp = np.moveaxis(moment, -1, 0)
    covariance = syp - sy * sp / sw
    variance = (syy - sy * sy / sw) * (spp - sp * sp / sw)
    return covariance / np.sqrt(np.maximum(variance, 1e-30))


def run() -> None:
    hypothesis = json.loads((OUT / "HYPOTHESES.json").read_text())
    if digest(OUT / "cadence_manifest.json") != hypothesis["frozen_manifest_sha256"]:
        raise RuntimeError("Frozen sample manifest changed.")
    manifest = json.loads((OUT / "cadence_manifest.json").read_text())
    if digest(OUT / "cadence_probe.py") != manifest["source_sha256"]["cadence_probe.py"]:
        raise RuntimeError("Frozen script changed before scoring.")
    if (OUT / "cadence_results.json").exists():
        raise RuntimeError("Existing results must be preserved.")
    start = time.perf_counter()
    state = json.loads((OUT / "STATE.json").read_text())
    commands = state["commands"] + ["run_cpu.sh eda-linear .venv/bin/python -B research/campaign_2026-10-03/eda/linear/cadence_probe.py run"]
    update_state("fitting_frozen_causal_filter_comparison", commands=commands,
                 active_command_budget="15 minutes, one core, 2 GiB; one streamed row group at a time")
    train = pq.ParquetFile(ROOT / "datasets/train.parquet")
    grams = {name: np.zeros((2, 113 if name == "levels" else 337, 113 if name == "levels" else 337)) for name in NAMES}
    rhs = {name: np.zeros((2, 113 if name == "levels" else 337)) for name in NAMES}
    training_rows = 0
    parity = {}
    for index, rg in enumerate(manifest["train_row_groups"]):
        x, y, required = load(train, rg)
        if index == 0:
            for name in NAMES:
                error = float(np.max(np.abs(transform(x[:600], name) - online_transform(x[:600], name))))
                # Prefix agreement checks that extending the input cannot change past features.
                prefix_error = float(np.max(np.abs(transform(x[:800], name)[:600] - transform(x[:600], name))))
                parity[name] = {"online_max_abs_error": error, "future_extension_max_abs_error": prefix_error}
                if error > 2e-5 or prefix_error > 2e-5:
                    raise AssertionError(f"Causal parity failed for {name}: {parity[name]}")
            save_json("cadence_parity.json", parity)
        selected = np.arange(99 + index % STRIDE, len(x), STRIDE)
        if not required[selected].all():
            raise AssertionError("Training row sampler selected warm-up.")
        yc = np.clip(y[selected], -2, 2)
        weights = np.abs(yc)
        training_rows += len(selected)
        for name in NAMES:
            a = design(x, name, selected)
            for target in range(2):
                weighted = a * weights[:, target, None]
                grams[name][target] += a.T @ weighted
                rhs[name][target] += weighted.T @ yc[:, target]
        if index % 16 == 0:
            print("fit", index, "elapsed_seconds", round(time.perf_counter() - start, 1), flush=True)
    models = {name: np.column_stack([solve(grams[name][target], rhs[name][target]) for target in range(2)]) for name in NAMES}
    np.savez(OUT / "cadence_models.npz", **models)
    np.savez(OUT / "cadence_grams.npz", **{f"G_{name}": value for name, value in grams.items()},
             **{f"b_{name}": value for name, value in rhs.items()})
    update_state("fitted_models_evaluating_frozen_own_disjoint_sample", training_rows=training_rows)
    valid = pq.ParquetFile(ROOT / "datasets/valid.parquet")
    seq_moments = np.zeros((len(manifest["valid_row_groups"]), len(NAMES), 2, 6))
    scored_counts = []
    example = None
    # One target-only descriptive diagnostic; never used in model inputs or routing.
    cadence_diagnostic = []
    for index, rg in enumerate(manifest["valid_row_groups"]):
        x, y, scored = load(valid, rg, scored=True)
        selected = np.flatnonzero(scored)
        scored_counts.append(len(selected))
        dy = np.diff(y, axis=0)
        diagnostic = [[float(np.corrcoef(dy[:-lag, target], dy[lag:, target])[0, 1])
                       for target in range(2)] for lag in (7, 8)]
        cadence_diagnostic.append(diagnostic)
        if index == 0:
            example = {"step": np.arange(1000, 1600), "targets": y[1000:1600],
                       "scored": scored[1000:1600]}
        for model_index, name in enumerate(NAMES):
            prediction = design(x, name, selected) @ models[name]
            for target in range(2):
                seq_moments[index, model_index, target] = moments(prediction[:, target], y[selected, target])
            if index == 0:
                assert example is not None
                example[name] = design(x, name, np.arange(1000, 1600)) @ models[name]
        if index % 16 == 0:
            print("eval", index, "elapsed_seconds", round(time.perf_counter() - start, 1), flush=True)
    pooled_target = correlation(seq_moments.sum(axis=0))
    pooled = pooled_target.mean(axis=1)
    within = correlation(seq_moments).mean(axis=2)
    loo = correlation(seq_moments.sum(axis=0)[None] - seq_moments).mean(axis=2)
    scores = {}
    for index, name in enumerate(NAMES):
        delta = within[:, index] - within[:, 1]
        loo_delta = loo[:, index] - loo[:, 1]
        scores[name] = {
            "wp": float(pooled[index]), "t0": float(pooled_target[index, 0]),
            "t1": float(pooled_target[index, 1]), "delta_vs_ema": float(pooled[index] - pooled[1]),
            "within_sequence_delta_median": float(np.median(delta)),
            "within_sequence_delta_positive_count": int((delta > 0).sum()),
            "leave_one_sequence_out_pooled_delta_range": [float(loo_delta.min()), float(loo_delta.max())],
        }
    ema_gain = float(pooled[1] - pooled[0])
    retention = float((pooled[2] - pooled[0]) / ema_gain) if ema_gain > 0 else None
    phase8_deficit = float(pooled[1] - pooled[2])
    cycle8_deficit = float(pooled[1] - pooled[4])
    period_advantage = float(pooled[2] - pooled[3])
    result = {
        "scores": scores, "training_sequences": 128, "training_rows": training_rows,
        "evaluation_sequences": 64, "evaluation_scored_rows": sum(scored_counts),
        "lambda": LAMBDA, "stride": STRIDE, "feature_count": 112,
        "phase8_ema_gain_retention": retention,
        "H1_cadence_only_rejected_by_frozen_rule": bool(retention is not None and retention >= 0.8 and phase8_deficit < 0.005),
        "H1_phase8_replacement_rejected_by_frozen_rule": bool(phase8_deficit >= 0.005),
        "H1_cadence_averaging_dependence_suggested_by_frozen_rule": bool(phase8_deficit >= 0.01 and cycle8_deficit < 0.005),
        "H2_phase8_minus_phase7": period_advantage,
        "H2_specific_period8_advantage_rejected_by_frozen_rule": bool(period_advantage <= 0.002),
        "descriptive_difference_acf_mean_lag7_lag8": np.mean(cadence_diagnostic, axis=0).tolist(),
        "uncertainty_limit": "Paired sequence contributions/leave-one-sequence-out sensitivity only, not campaign connected-component confidence intervals; dependence/deduplication unresolved.",
        "interpretation_limit": "Filter variants differ in sample support, response lag and sampling aliasing despite equal decay/size. This falsifier cannot uniquely identify market microstructure or prove economic causality.",
        "hypotheses_sha256": digest(OUT / "HYPOTHESES.json"),
        "manifest_sha256": digest(OUT / "cadence_manifest.json"),
        "elapsed_seconds": time.perf_counter() - start,
    }
    np.savez(OUT / "cadence_eval_moments.npz", moments=seq_moments, names=np.array(NAMES),
             seq_ix=np.array(manifest["valid_seq_ix"]), row_groups=np.array(manifest["valid_row_groups"]),
             scored_counts=np.array(scored_counts), difference_acf=np.array(cadence_diagnostic))
    assert example is not None
    np.savez(OUT / "cadence_example.npz", **example)
    save_json("cadence_results.json", result)
    for name, score in scores.items():
        print(name, json.dumps(score), flush=True)
    print("frozen verdicts", json.dumps({key: value for key, value in result.items() if key.startswith("H")}), flush=True)
    artifact_names = ["cadence_models.npz", "cadence_grams.npz", "cadence_eval_moments.npz",
                      "cadence_example.npz", "cadence_results.json", "cadence_parity.json"]
    save_json("cadence_provenance.json", {"artifacts_sha256": {name: digest(OUT / name) for name in artifact_names},
                                       "manifest_sha256": result["manifest_sha256"],
                                       "hypotheses_sha256": result["hypotheses_sha256"]})
    update_state("matched_probe_completed_pending_render_and_report",
                 artifacts=state["artifacts"] + artifact_names + ["cadence_provenance.json"],
                 pending_work=["Render and inspect figures", "Update runnable notebook", "Write REPORT.txt"],
                 last_failure=None)


def render() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    result = json.loads((OUT / "cadence_results.json").read_text())
    saved = np.load(OUT / "cadence_eval_moments.npz")
    within = correlation(saved["moments"]).mean(axis=2)
    labels = ["Levels", "EMA", "Phase 8", "Phase 7", "Cycle 8 + EMA"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for target, color in enumerate(["#245ea8", "#d16b35"]):
        values = [result["scores"][name][f"t{target}"] for name in NAMES]
        axes[0].plot(range(5), values, "o-", label=f"t{target}", color=color)
    axes[0].set_xticks(range(5), labels, rotation=25)
    axes[0].set_ylabel("Global clipped weighted Pearson")
    axes[0].legend()
    axes[0].set_title("64 own-disjoint validation sequences")
    for index in (2, 3, 4):
        axes[1].plot(np.sort(within[:, index] - within[:, 1]), label=labels[index])
    axes[1].axhline(0, color="grey", linewidth=0.7)
    axes[1].set_xlabel("Sequences sorted separately for each variant")
    axes[1].set_ylabel("Within-sequence mean WP minus EMA")
    axes[1].set_title("Paired descriptive differences; not pooled score")
    axes[1].legend()
    acf = saved["difference_acf"]
    axes[2].scatter(acf[:, 0, 0], acf[:, 1, 0], s=12, label="t0", color="#245ea8")
    axes[2].scatter(acf[:, 0, 1], acf[:, 1, 1], s=12, label="t1", color="#d16b35")
    axes[2].set_xlabel("Target difference correlation at lag7")
    axes[2].set_ylabel("Target difference correlation at lag8")
    axes[2].set_title("Cadence diagnostic; targets never inference inputs")
    axes[2].legend()
    fig.tight_layout()
    fig.savefig(OUT / "fig05_cadence_comparison.png", dpi=130)
    plt.close(fig)
    example = np.load(OUT / "cadence_example.npz")
    fig, axes = plt.subplots(2, 1, figsize=(13, 5), sharex=True)
    for target, ax in enumerate(axes):
        ax.plot(example["step"], np.clip(example["targets"][:, target], -2, 2), color="black", linewidth=0.7, label="target")
        for name, color in [("ema", "#245ea8"), ("phase8", "#d16b35"), ("cycle8", "#23875e")]:
            ax.plot(example["step"], np.clip(example[name][:, target], -2, 2), linewidth=0.7, label=name, color=color)
        ax.set_ylabel(f"t{target}")
        ax.legend(ncol=4, fontsize=8)
    axes[0].set_title(f"First frozen validation sequence {int(saved['seq_ix'][0])}; illustrative window")
    axes[1].set_xlabel("Step within sequence")
    fig.tight_layout()
    fig.savefig(OUT / "fig06_cadence_prediction_example.png", dpi=130)
    plt.close(fig)
    print("Rendered fig05_cadence_comparison.png and fig06_cadence_prediction_example.png", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "run", "render"])
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("Use the authorized CPU wrapper with CUDA unavailable.")
    try:
        {"prepare": prepare, "run": run, "render": render}[args.command]()
    except Exception:
        failure = traceback.format_exc()
        (OUT / "cadence_failure.log").write_text(failure)
        update_state("probe_failed", last_failure=failure)
        raise


if __name__ == "__main__":
    main()
