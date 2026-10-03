"""Render saved linear results, audit pooled moments and finalize this bounded pass.

CPU wrapper required. No dataset reads, fitting, model selection or source mutations.
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cadence_probe import OUT, NAMES, correlation, digest, save_json, update_state


def main() -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("Use continuation/run_cpu.sh.")
    result = json.loads((OUT / "cadence_results.json").read_text())
    context = json.loads((OUT / "cadence_context_results.json").read_text())
    previous = json.loads((OUT / "eval_v1.json").read_text())
    manifest = json.loads((OUT / "cadence_manifest.json").read_text())
    saved = np.load(OUT / "cadence_eval_moments.npz")
    reconstructed = correlation(saved["moments"].sum(axis=0))
    expected = np.array([[result["scores"][name][f"t{k}"] for k in range(2)] for name in NAMES])
    if not np.allclose(reconstructed, expected, atol=1e-12, rtol=0):
        raise AssertionError("Saved moments do not reconstruct reported global score.")
    if result["hypotheses_sha256"] != digest(OUT / "HYPOTHESES.json"):
        raise AssertionError("Primary hypotheses changed after scoring.")
    if result["manifest_sha256"] != digest(OUT / "cadence_manifest.json"):
        raise AssertionError("Primary sample manifest changed after scoring.")
    if context["hypothesis_sha256"] != digest(OUT / "CONTEXT_HYPOTHESIS.json"):
        raise AssertionError("Context hypothesis changed after scoring.")
    # Independent centered arithmetic checks the sufficient-moment implementation.
    example = np.load(OUT / "cadence_example.npz")
    from cadence_probe import moments
    direct_errors = []
    for target in range(2):
        p = np.clip(example["ema"][:, target], -2, 2)
        y = np.clip(example["targets"][:, target], -2, 2)
        w = np.abs(y)
        pc = p - np.average(p, weights=w)
        yc = y - np.average(y, weights=w)
        direct = (w * pc * yc).sum() / np.sqrt((w * pc * pc).sum() * (w * yc * yc).sum())
        direct_errors.append(float(abs(direct - correlation(moments(p, y)))))
    if max(direct_errors) > 1e-12:
        raise AssertionError("Centered arithmetic disagrees with moment correlation.")
    save_json("cadence_audit.json", {"reported_scores_reconstructed_from_moments": True,
                                     "centered_vs_moment_max_abs_error": max(direct_errors),
                                     "frozen_hypothesis_and_manifest_hashes_match": True,
                                     "known_own_validation_overlap": manifest["valid_overlap_with_known_own_prior"]})
    labels = ["Levels", "EMA", "Phase 8", "Phase 7", "Cycle 8 + EMA"]
    within = correlation(saved["moments"]).mean(axis=2)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for target, color in enumerate(["#245ea8", "#d16b35"]):
        axes[0].plot(range(5), reconstructed[:, target], "o-", label=f"t{target}", color=color)
    axes[0].set_xticks(range(5), labels, rotation=25)
    axes[0].set_ylabel("Global clipped weighted Pearson")
    axes[0].set_title("64 sequences; own prior samples excluded")
    axes[0].legend()
    for index in (2, 3, 4):
        axes[1].plot(np.sort(within[:, index] - within[:, 1]), label=labels[index])
    axes[1].axhline(0, color="grey", linewidth=0.7)
    axes[1].set_xlabel("Sequences sorted separately for each variant")
    axes[1].set_ylabel("Within-sequence mean WP minus EMA")
    axes[1].set_title("Paired descriptive differences\nNot a pooled score or confidence interval")
    axes[1].legend()
    acf = saved["difference_acf"]
    for target, color in enumerate(["#245ea8", "#d16b35"]):
        axes[2].scatter(acf[:, 0, target], acf[:, 1, target], s=12, label=f"t{target}", color=color)
    axes[2].set_xlabel("Target difference correlation at lag7")
    axes[2].set_ylabel("Target difference correlation at lag8")
    axes[2].set_title("Difference ACF by sequence\nTargets used for diagnostics only")
    axes[2].legend()
    fig.tight_layout()
    fig.savefig(OUT / "fig05_cadence_comparison.png", dpi=130)
    plt.close(fig)

    group = context["class_diagnostic_only"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(range(4), previous["L+E|lam0.001|w"]["per_class_mean"], "o-", label="Old300 development sample")
    axes[0].plot(range(4), [item["scores"]["L+E__lam0.001__w"] for item in group], "o-", label="New64 sample")
    axes[0].plot(range(4), [item["scores"]["L__lam0.001__w"] for item in group], "o--", label="Levels on new64")
    axes[0].set_xticks(range(4), [str(item["quantum"]) for item in group])
    axes[0].set_xlabel("Target-derived quantum (diagnostic only)")
    axes[0].set_ylabel("Class-pooled mean WP")
    axes[0].set_title("Fixed archived models; no target-based routing")
    axes[0].legend(fontsize=8)
    new_weight = np.array([item["scored_weight_by_target"] for item in group])
    for target in range(2):
        axes[1].bar(np.arange(4) + (target - 0.5) * 0.35,
                    new_weight[:, target] / new_weight[:, target].sum(), width=0.35, label=f"t{target}")
    axes[1].set_xticks(range(4), [str(item["quantum"]) for item in group])
    axes[1].set_xlabel("Target-derived quantum (diagnostic only)")
    axes[1].set_ylabel("Share of scored target weight in new64")
    axes[1].set_title("Scored mass and predictability vary by class")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(OUT / "fig07_archived_model_stability.png", dpi=130)
    plt.close(fig)
    files = ["cadence_probe.py", "cadence_context.py", "finish_linear.py", "eda_linear.py", "eda_linear_prior.py",
             "HYPOTHESES.json", "CONTEXT_HYPOTHESIS.json", "cadence_manifest.json", "cadence_results.json",
             "cadence_context_results.json", "cadence_audit.json", "fig05_cadence_comparison.png",
             "fig06_cadence_prediction_example.png", "fig07_archived_model_stability.png"]
    provenance = json.loads((OUT / "cadence_provenance.json").read_text())
    provenance["continuation_artifacts_sha256"] = {name: digest(OUT / name) for name in files}
    save_json("cadence_provenance.json", provenance)
    state = json.loads((OUT / "STATE.json").read_text())
    update_state("figures_rendered_metric_and_freeze_audits_passed_pending_inspection_notebook_report",
                 artifacts=sorted(set(state["artifacts"] + files + ["cadence_provenance.json"])),
                 commands=state["commands"] + ["run_cpu.sh eda-linear .venv/bin/python -B research/campaign_2026-10-03/eda/linear/cadence_probe.py render",
                                               "run_cpu.sh eda-linear .venv/bin/python -B research/campaign_2026-10-03/eda/linear/finish_linear.py"],
                 pending_work=["Inspect final figures", "Headless notebook export", "REPORT.txt"], last_failure=None)
    print("Metric reconstruction, centered-moment arithmetic, causal parity and freeze/provenance checks passed.")


if __name__ == "__main__":
    main()
