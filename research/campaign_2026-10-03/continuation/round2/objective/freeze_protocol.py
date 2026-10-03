from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = Path.cwd()


def main() -> None:
    """Freeze training-only row groups and all rules before outcome computations."""
    if (HERE / "PREREG.json").exists():
        raise FileExistsError("preregistration exists; never overwrite it")
    manifest_path = ROOT / "research/campaign_2026-09-29/artifacts/splits/fold0.json"
    manifest = json.loads(manifest_path.read_text())
    allowed = np.asarray(manifest["fit"]["train"], dtype=np.int64)
    rng = np.random.default_rng(173101)
    groups = rng.permutation(allowed)[:256].tolist()
    protocol = {
        "status": "frozen before synthetic or actual scoring",
        "version": 1,
        "recipe_run": "c20261003_combo_s3_f0",
        "secondary_recipe": "c20261003_w320_s3_f0",
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "sample": {
            "split": "TRAIN only",
            "row_groups_in_order": groups,
            "selection": "numpy.default_rng(173101).permutation(fold0.fit.train)[:256]",
            "n": 256,
            "batch": 32,
            "rows_per_sequence": 20000,
            "warmup": 99,
            "no_label_or_class_based_selection": True,
            "evaluation_rows_read": False,
        },
        "batch_permutations": [
            "identity(primary)",
            "numpy.default_rng(173102)",
            "numpy.default_rng(173103)",
        ],
        "chunks": [1000, 2000],
        "target_clip": [-2, 2],
        "actual_weights": "valid*(cached_propensity**2+.03)*max(.25,1/(1+since_i0px/32)); labels use |clip(y)|",
        "active_target": "half clipped current teacher average + half (teacher at t+8 if raw label pair stays constant strictly beyond t+8, else clipped y)",
        "secondary_target": "half clipped y + half clipped teacher average (no bootstrap)",
        "aux_denominators": [
            "structural sum(v)",
            "next-change sum(v*has_future_D_change)",
            "EMA sum(v), broadcast over 2 targets",
        ],
        "fixed_input_bins": {"propensity_edges": [0.25, 0.5, 0.75], "since_edges": [16, 64, 256]},
        "gradient_proxies": [
            "constant-zero prediction",
            "cached clipped teacher-average prediction; output derivatives + two-coordinate intercept/input head proxy",
        ],
        "normalizer_gate": "primary chunk1000 CV(W)<.05 closes claim; analyze CV separately after excluding first warmup chunk and within chunk positions",
        "predictability": "ridge of logW from chunk input summaries, then input+label summaries; leave-one-32-sequence-batch-out CV; no validation scored; observational association only",
        "synthetics": {
            "clusters": 240,
            "rows_per_cluster": 16,
            "seeds": [101, 102, 103, 104],
            "positive": "three observed x states, cluster-correlated stochastic labels and mass; equal-chunk weighted regression vs pooled clipped-weight regression",
            "negative1": "fixed outcome-determined masses per observed x, correct saturated model: large W variation but identical conditional fit",
            "negative2": "identical W every cluster: fitted estimators and gradients identical",
            "evaluation": "enumerated independent population for counterexamples; fresh seeded clusters for simulation; official clipped |y| weighted Pearson",
            "minimum_positive_effect": "population WP optimum differs by >.01 and positive-control pooled fit recovers expected optimum",
            "falsifiers": [
                "positive control fails to recover analytic optimum",
                "negative equal-mass estimators differ >1e-10",
                "large W alone declared evidence of bias",
            ],
        },
        "limitations": [
            "TRAIN-only subset omits true-mask validation-in-fit branch; that branch retains batch aggregate propensity mass with mask-dependent within-sequence distribution",
            "cached propensity is same active cache but no new GPU numerical parity",
            "gradient proxies do not equal full GRU parameter gradients or EMA/aux gradients",
            "input summaries see full chunk; not necessarily available at its first row",
            "no direct train-time intervention efficacy score",
        ],
        "next_step_gate": "at most one scientific-estimator contrast proposed if actual variation large and controls pass; same batches/chunks/updates/teacher/aux/EMA schedule and gradient-size matched sham",
    }
    (HERE / "PREREG.json").write_text(json.dumps(protocol, indent=2) + "\n")
    (HERE / "sample_ids.json").write_text(
        json.dumps(
            {"row_groups": groups, "split": "train", "seq_ix": "will be recorded by streamed read"},
            indent=2,
        )
        + "\n"
    )
    print(
        json.dumps(
            {
                "status": "frozen",
                "n": len(groups),
                "prereg_sha256": hashlib.sha256((HERE / "PREREG.json").read_bytes()).hexdigest(),
            }
        )
    )


if __name__ == "__main__":
    main()
