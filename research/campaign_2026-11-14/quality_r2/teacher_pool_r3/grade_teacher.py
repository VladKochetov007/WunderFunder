"""Score the frozen original TL3C target pool against the current single teacher."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT.parent / "trajectory_mean_r1"))
from grade_local import CACHE, RUNS, digest, predictions, score, validate_identity  # noqa: E402

POOL = {"q14t_fbc_s5": 0.34, "q14t_tlead_s5": 0.33, "q14t_tlead_s6": 0.33}
BASELINE = "tt29rep_fbc1k_s5"


def main() -> None:
    """Freeze a target-quality diagnostic without selecting epochs or coefficients."""
    assert digest(CACHE / "valid_targets.npz") == (
        "dbb7a1a48dd0af9fa524a80c2694008a56cb91d5dae80e8d0d4984c0c5622955"
    )
    with np.load(CACHE / "valid_targets.npz") as targets:
        groups, mask, labels = targets["row_group"], targets["mask"], targets["y"]
    assert len(groups) == 1873 and int(mask.sum()) == 3528273
    labels = labels[mask]
    values, scores, bindings = {}, {}, {}
    for name in [*POOL, BASELINE]:
        path = CACHE / f"pred_{name}_valid.npy"
        side = json.loads(path.with_suffix(".json").read_text())
        provenance = json.loads((RUNS / name / "provenance.json").read_text())
        validate_identity(
            name,
            side,
            provenance,
            checkpoint_file="model.pt" if name == BASELINE else "model_ep3.pt",
        )
        assert side["inference_precision"] == "fp32" and side["feature_only"] is True
        assert side["campaign_export_scope"] == "full-valid-feature-only"
        assert side["prediction_sha256"] == digest(path)
        assert side["run_provenance_sha256"] == digest(RUNS / name / "provenance.json")
        values[name] = predictions(name, groups, mask)
        assert np.isfinite(values[name]).all()
        scores[name] = score(labels, values[name])
        bindings[name] = {
            key: side[key]
            for key in ("prediction_sha256", "checkpoint_sha256", "run_provenance_sha256")
        }
    pool = sum(values[name] * np.float32(weight) for name, weight in POOL.items())
    scores["fixed_pool"] = score(labels, pool)
    delta = scores["fixed_pool"]["wp"] - scores[BASELINE]["wp"]
    result = {
        "status": "complete_fixed_teacher_target_grade",
        "scores": scores,
        "pool_minus_current_teacher": delta,
        "target_quality_trigger": delta >= 0.0005,
        "coefficients": POOL,
        "bindings": bindings,
        "scored_rows": int(mask.sum()),
        "fits_launched": 0,
        "requires_resource_admission_and_review": True,
        "scope": "Repeatedly exposed full-valid target diagnostic; no student quality, new seed replication or public-score claim.",
    }
    (OUT / "TEACHER_GRADE_R1.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"scores": scores, "delta": delta, "target_quality_trigger": delta >= 0.0005}))


if __name__ == "__main__":
    main()
