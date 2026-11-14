"""Score frozen train-only endpoints and check final-package numerical parity."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from wnn.gmetric import global_wp, reference_global_wp

ROOT = Path(__file__).resolve().parents[3]
CACHE = Path("/home/vlad/development/wunder/research_tmp/chrono_root/cache")
REPLAY = Path("/home/vlad/development/wunder/research_tmp/pkg_20261114")
OUT = Path(__file__).parent / "submission_checks"


def predictions(run: str, groups: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Read predictions by source row-group identity, keeping scored rows only."""
    side = json.loads((CACHE / f"pred_{run}_valid.json").read_text())
    positions = {int(g): i for i, g in enumerate(side["row_groups"])}
    assert len(positions) == len(side["row_groups"])
    array = np.load(CACHE / f"pred_{run}_valid.npy", mmap_mode="r")
    assert array.shape[0] == len(positions)
    return np.concatenate([array[positions[int(g)]][m] for g, m in zip(groups, mask, strict=True)])


def score(y: np.ndarray, pred: np.ndarray) -> dict:
    """Check the official score against its independent reference implementation."""
    selected = np.ones(len(y), dtype=bool)
    result = global_wp(y, pred, selected)
    assert abs(result["wp"] - reference_global_wp(y, pred, selected)) < 1e-10
    return result


def main() -> None:
    """Write all comparisons without searching mixture coefficients or epochs."""
    with np.load(CACHE / "valid_targets.npz") as targets:
        groups, mask = targets["row_group"], targets["mask"]
        y = targets["y"].astype(np.float32)
    assert len(groups) == 1873 and int(mask.sum()) == 3528273
    selected_y = y[mask]
    runs = ("l14_kd50_s3", "l14_kd50_s4", "l14_kd90_s3")
    pred = {run: predictions(run, groups, mask) for run in runs}
    scores = {run: score(selected_y, pred[run]) for run in runs}
    pair = (pred[runs[0]] + pred[runs[1]]) * 0.5
    scores["kd50_pair_equal"] = score(selected_y, pair)
    # The pair is a fixed recipe already declared before prediction inspection.
    scores["pair_minus_mean_single"] = scores["kd50_pair_equal"]["wp"] - np.mean(
        [scores[runs[0]]["wp"], scores[runs[1]]["wp"]]
    )
    (OUT / "FULL_VALID_R1.json").write_text(json.dumps(scores, indent=2) + "\n")
    plan = json.loads(
        (
            ROOT
            / "research/campaign_2026-10-24/resume4h_r1/quota_submission_r1/REPLAY_PLAN_R1.json"
        ).read_text()
    )
    selected_groups = np.array([member["row_group"] for member in plan["members"]])
    positions = {int(g): i for i, g in enumerate(groups)}
    indices = [positions[int(g)] for g in selected_groups]
    panel_mask, panel_y = mask[indices], y[indices]
    packages = {
        "final_kd50_s3s4_pair": ("l14_final_kd50_s3", "l14_final_kd50_s4"),
        "final_kd50_s3_single": ("l14_final_kd50_s3",),
        "kd90_s3_single": ("l14_kd90_s3",),
    }
    checks = {}
    for name, members in packages.items():
        served = np.load(REPLAY / f"replay_{name}" / "predictions.npy")[panel_mask]
        reference = np.mean(
            [predictions(run, selected_groups, panel_mask) for run in members], axis=0
        )
        native_score = score(panel_y[panel_mask], served)["wp"]
        torch_score = score(panel_y[panel_mask], reference)["wp"]
        rms = float(np.sqrt(np.mean((served - reference) ** 2) / np.mean(reference**2)))
        assert np.isfinite(served).all() and rms < 0.02
        assert abs(native_score - torch_score) < 0.0005
        checks[name] = {
            "native_wp": native_score,
            "torch_wp": torch_score,
            "delta": native_score - torch_score,
            "rms_relative_error": rms,
            "sequences": len(indices),
            "scored_rows": int(panel_mask.sum()),
            "fit_exposure": "train+valid; parity only"
            if name.startswith("final")
            else "train-only; exposed valid",
        }
    (OUT / "NATIVE_PARITY_R1.json").write_text(json.dumps(checks, indent=2) + "\n")
    print(json.dumps({"full_valid": scores, "native_parity": checks}, indent=2))


if __name__ == "__main__":
    main()
