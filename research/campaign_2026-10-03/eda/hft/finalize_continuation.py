from __future__ import annotations

import json
import sys

import marimo
import numpy as np
import pandas as pd
import pyarrow
import scipy
import sklearn

from continuation_common import OWN, ROOT, save_json, set_phase, sha256


def main() -> None:
    """Verify freeze hashes and save the final reproducibility inventory."""
    first = json.loads((OWN / "sampling_manifest.json").read_text())
    second = json.loads((OWN / "sampling_manifest_h3.json").read_text())
    if sha256(OWN / "HYPOTHESES.json") != first["hashes"]["HYPOTHESES.json"]:
        raise ValueError("The H1/H2 hypothesis freeze changed")
    if sha256(OWN / "HYPOTHESES_H3.json") != second["hypothesis_sha256"]:
        raise ValueError("The H3 hypothesis freeze changed")
    if set(first["selected_valid_row_groups"]) & set(
        second["selected_valid_row_groups"]
    ):
        raise ValueError("Continuation cohorts overlap")
    artifact_names = [
        "RECOVERY_SUMMARY.txt",
        "HYPOTHESES.json",
        "HYPOTHESES_H3.json",
        "REPORT.txt",
        "sampling_manifest.json",
        "sampling_manifest_h3.json",
        "decoder.pkl",
        "regime_stats.csv",
        "warmup_scale_model_v1.pkl",
        "warmup_scale_training_v1.json",
        "continuation_common.py",
        "probe_quote_identity.py",
        "probe_causal_sign_scale.py",
        "probe_spread_ablation.py",
        "render_continuation.py",
        "eda_hft.py",
        "quote_identity_v1.csv",
        "quote_identity_v1.log",
        "quote_identity_summary_v1.json",
        "quote_identity_witnesses_v1.json",
        "causal_sign_scale_v1.csv",
        "causal_sign_scale_v1.log",
        "causal_sign_scale_summary_v1.json",
        "spread_ablation_v1.csv",
        "spread_ablation_v1.log",
        "spread_ablation_summary_v1.json",
        "fig06_identity_falsifier_v1.png",
        "fig07_causal_sign_scale_v1.png",
        "fig08_spread_ablation_v1.png",
        "eda_hft_continuation_v1.html",
        "notebook_export_v1.log",
        "format_check_v1.log",
        "lint_check_v1.log",
        "finalize_continuation.py",
    ]
    commands = json.loads((OWN / "STATE.json").read_text())["commands"]
    commands.extend(
        [
            "run_cpu.sh eda-hft .venv/bin/ruff format --isolated --no-cache --check <new own scripts>",
            "run_cpu.sh eda-hft .venv/bin/ruff check --isolated --no-cache --config <local imports> <new own scripts>",
            "run_cpu.sh eda-hft .venv/bin/marimo export html research/campaign_2026-10-03/eda/hft/eda_hft.py -o research/campaign_2026-10-03/eda/hft/eda_hft_continuation_v1.html",
            "run_cpu.sh eda-hft .venv/bin/python -B research/campaign_2026-10-03/eda/hft/finalize_continuation.py",
        ]
    )
    set_phase(
        "complete_report_frozen",
        commands=commands,
        artifacts=artifact_names + ["STATE.json", "artifact_provenance_v1.json"],
        prior_exposure="Both cohorts disjoint from own documented prior VALID exposure; same previously exposed 300 TRAIN prefixes reused for fitting. Campaign exposure and paired connected components unknown.",
        pending_work=[],
        next_experiment="Freeze TRAIN-only regime intercept/sign/amplitude calibration and common-sign/constant-regime controls on a fresh cohort; controller deduplication and baseline integration come later.",
        last_failure=None,
        figures_inspected=[
            "fig06_identity_falsifier_v1.png",
            "fig07_causal_sign_scale_v1.png",
            "fig08_spread_ablation_v1.png",
        ],
        hypothesis_verdicts={
            "H1": "Exact shared-reference spread rejected20/20; exact transition mapping rejected15/20. Statistical transition association survives.",
            "H2": "Warm-up classes correct20/20; universal sign and universally useful quantum multiplier rejected. Exact input-prefix causality checks pass.",
            "H3": "Current spread loses0.033146 meanWP to regime-only control; material incremental contribution rejected. Classes correct12/12.",
        },
    )
    artifact_names.append("STATE.json")
    provenance = {
        "artifact_sha256": {name: sha256(OWN / name) for name in artifact_names},
        "dataset_metadata_only": {
            name: {
                "bytes": (ROOT / "datasets" / name).stat().st_size,
                "mtime_ns": (ROOT / "datasets" / name).stat().st_mtime_ns,
            }
            for name in ["train.parquet", "valid.parquet"]
        },
        "versions": {
            "python": sys.version,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "pyarrow": pyarrow.__version__,
            "scipy": scipy.__version__,
            "sklearn": sklearn.__version__,
            "marimo": marimo.__version__,
        },
        "script_changes_after_measurement": "Only formatter/import/logger fixes, notebook empty-return cleanup and native-bool serialization; probe procedures and parameters unchanged.",
        "resource_limits": "Mandated run_cpu.sh; 1 core,2 GiB,15 minute max per command; CUDA unavailable; row groups streamed individually.",
        "prior_decoder_provenance_limit": "Existing model hash is preserved; original training selection reconstructed from script, not authenticated by an old run log.",
        "freeze_hashes_verified": True,
        "cohorts_disjoint_verified": True,
        "baseline_comparison": "Not performed in this independent phase.",
    }
    save_json("artifact_provenance_v1.json", provenance)


if __name__ == "__main__":
    main()
