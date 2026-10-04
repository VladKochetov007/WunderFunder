"""Family-held-out sensitivity for user-reported public-score calibration.

This reads the frozen local-score matrix. It cannot infer how a new mechanism
transfers from clean OOF folds to a public test set.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from campaign_public_calibration import inner_error, linear_fit

ROOT = Path(__file__).resolve().parents[1]
FAMILY = {
    "baseline_x025": "early",
    "ab18": "early",
    "final_tv_e2": "early",
    "blend_base_final_tv": "early",
    "final_s4_base35": "s4",
    "final_s4ft_base35": "s4",
    "final_s4lpft_base35": "s4",
    "final_ema4_lpft_base35": "ema4",
    "final_since2_gclp": "since",
    "final_mixed_ema6_shared": "ema6",
    "final_ema6_shared": "ema6",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenario", choices=("pair", "single"))
    args = parser.parse_args()
    source = ROOT / "research" / "campaign_2026-09-29" / "artifacts" / f"nested_public_calibration_{args.scenario}.json"
    report = json.loads(source.read_text())
    names = report["submission_order"]
    if set(names) != set(FAMILY) or report["ema_membership_scenario"] != args.scenario:
        raise ValueError("unexpected public package inventory or membership scenario")
    y = np.asarray(report["public_scores_user_reported"], dtype=np.float64)
    x = np.asarray(report["local_scores_by_variant_by_submission"], dtype=np.float64)
    if x.shape != (21, len(names)) or not np.isfinite(x).all():
        raise ValueError("invalid frozen local proxy matrix")
    families = np.array([FAMILY[name] for name in names])
    rows = []
    for family in dict.fromkeys(families):
        held = np.flatnonzero(families == family)
        train = np.flatnonzero(families != family)
        inner = np.array([inner_error(proxy, y, train) for proxy in x])
        chosen = int(np.argmin(inner))
        slope, intercept = linear_fit(x[chosen, train], y[train])
        for i in held:
            predicted = float(intercept + slope * x[chosen, i])
            rows.append({
                "name": names[i],
                "family": family,
                "public": float(y[i]),
                "predicted": predicted,
                "error": predicted - float(y[i]),
                "selected_variant": report["proxy_variants"][chosen],
                "inner_package_loso_rmse": float(inner[chosen]),
            })
    errors = np.array([row["error"] for row in rows])
    output = {
        "scenario": args.scenario,
        "source": str(source.relative_to(ROOT)),
        "n_submissions": len(names),
        "n_families": len(set(families)),
        "family_heldout_rmse": float(np.sqrt(np.mean(errors**2))),
        "family_heldout_mae": float(np.mean(np.abs(errors))),
        "family_heldout_bias": float(np.mean(errors)),
        "family_heldout_rows": rows,
        "caveat": "families were defined after historical research; related public observations, selection bias, and unknown novel-mechanism transport remain",
    }
    destination = source.with_name(f"nested_public_family_holdout_{args.scenario}.json")
    destination.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({key: output[key] for key in ("scenario", "family_heldout_rmse", "family_heldout_mae", "family_heldout_bias")}))


if __name__ == "__main__":
    main()
