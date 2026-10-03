"""Score fixed predecessor models on the already-exposed cadence evaluation sample."""
from __future__ import annotations

import json
import os
from pathlib import Path
import time
import traceback

import numpy as np
import pyarrow.parquet as pq

from cadence_probe import OUT, ROOT, correlation, design, digest, load, moments, save_json, update_state


def main() -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("Use authorized CPU wrapper.")
    hypothesis = json.loads((OUT / "CONTEXT_HYPOTHESIS.json").read_text())
    manifest = json.loads((OUT / "cadence_manifest.json").read_text())
    primary = json.loads((OUT / "cadence_results.json").read_text())
    state = json.loads((OUT / "STATE.json").read_text())
    update_state("evaluating_archived_models_context_frozen",
                 commands=state["commands"] + ["run_cpu.sh eda-linear .venv/bin/python -B research/campaign_2026-10-03/eda/linear/cadence_context.py"])
    archive = np.load(OUT / "models_v1.npz")
    names = hypothesis["models"]
    indices = np.r_[np.arange(112), np.arange(224, 448), 560]
    models = {name: archive[name][indices] for name in names}
    # L-only models embed zero coefficients in unselected blocks; this extracts identical predictions.
    for name in names:
        discarded = np.setdiff1d(np.arange(561), indices)
        if np.any(archive[name][discarded] != 0):
            raise AssertionError("Archived model has nonzero omitted coefficients.")
    pf = pq.ParquetFile(ROOT / "datasets/valid.parquet")
    saved = np.zeros((64, len(names), 2, 6))
    quantum = np.zeros(64)
    classes = np.zeros(64, dtype=int)
    scored_counts = np.zeros(64, dtype=int)
    started = time.perf_counter()
    for index, rg in enumerate(manifest["valid_row_groups"]):
        x, y, scored = load(pf, rg, scored=True)
        selected = np.flatnonzero(scored)
        a = design(x, "ema", selected)
        nonzero = np.abs(y[y != 0])
        quantum[index] = nonzero.min()
        classes[index] = np.argmin(np.abs(np.array([0.1254, 0.2413, 0.4685, 0.6569]) - quantum[index]))
        scored_counts[index] = len(selected)
        for model_index, name in enumerate(names):
            prediction = a @ models[name]
            for target in range(2):
                saved[index, model_index, target] = moments(prediction[:, target], y[selected, target])
    pooled = correlation(saved.sum(axis=0))
    scores = {name: {"wp": float(pooled[index].mean()), "t0": float(pooled[index, 0]),
                     "t1": float(pooled[index, 1])} for index, name in enumerate(names)}
    class_summary = []
    for group in range(4):
        mask = classes == group
        scores_class = correlation(saved[mask].sum(axis=0))
        class_summary.append({"class": group, "quantum": [0.1254, 0.2413, 0.4685, 0.6569][group],
                              "sequences": int(mask.sum()), "scored_rows": int(scored_counts[mask].sum()),
                              "scored_weight_by_target": saved[mask, 0, :, 0].sum(axis=0).tolist(),
                              "scores": {name: float(scores_class[index].mean()) for index, name in enumerate(names)}})
    old_eval = json.loads((OUT / "eval_v1.json").read_text())
    same_lambda = scores["L+E__lam0.001__w"]["wp"]
    result = {
        "scores": scores, "class_diagnostic_only": class_summary,
        "same_lambda_archived_minus_new_fit": same_lambda - primary["scores"]["ema"]["wp"],
        "same_lambda_new_sample_minus_old_sample": same_lambda - old_eval["L+E|lam0.001|w"]["wp"],
        "smaller_training_as_sole_explanation_rejected": bool(same_lambda < 0.60),
        "evaluation_exposure": "Same64 already scored in cadence experiment; contextual follow-up, not independent confirmation.",
        "old_training_stride": "not recovered; cannot attribute same-sample gap solely to training sequence count",
        "hypothesis_sha256": digest(OUT / "CONTEXT_HYPOTHESIS.json"),
        "source_sha256": digest(Path(__file__).resolve()), "archived_models_sha256": digest(OUT / "models_v1.npz"),
        "manifest_sha256": digest(OUT / "cadence_manifest.json"),
        "elapsed_seconds": time.perf_counter() - started,
    }
    save_json("cadence_context_results.json", result)
    np.savez(OUT / "cadence_context_moments.npz", moments=saved, names=np.array(names),
             quantum=quantum, diagnostic_classes=classes, seq_ix=np.array(manifest["valid_seq_ix"]))
    update_state("primary_and_context_probe_completed_pending_figures_report",
                 artifacts=state["artifacts"] + ["CONTEXT_HYPOTHESIS.json", "cadence_context.py",
                                                 "cadence_context_results.json", "cadence_context_moments.npz"],
                 pending_work=["Render/inspect figures", "Headless notebook export", "REPORT.txt"], last_failure=None)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        failure = traceback.format_exc()
        (OUT / "cadence_context_failure.log").write_text(failure)
        update_state("context_probe_failed", last_failure=failure)
        raise
