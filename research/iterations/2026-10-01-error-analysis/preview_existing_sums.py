"""Descriptive first error budget from already archived sufficient statistics."""
import hashlib
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SOURCE = ROOT / "research/iterations/2026-10-01-continuation/student/evaluation-01/sufficient.npz"
CLASSES = ROOT / "research/artifacts/market-classes-v1.npz"


def budget(s):
    total = s.sum(0)
    w, sy, sp, syy, spp, syp = total.T
    my, mp = sy / w, sp / w
    vy, vp = syy / w - my * my, spp / w - mp * mp
    rho = (syp / w - my * mp) / np.sqrt(vy * vp)
    wg, yg, pg, yyg, ppg, ypg = np.moveaxis(s, -1, 0)
    loss = .25 / w * (
        (yyg - 2 * my * yg + my * my * wg) / vy
        + (ppg - 2 * mp * pg + mp * mp * wg) / vp
        - 2 * (ypg - my * pg - mp * yg + my * mp * wg) / np.sqrt(vy * vp)
    )
    assert np.all(loss >= -1e-12)
    assert abs(loss.sum() - (1 - rho.mean())) < 1e-12
    mass = .5 * wg / w
    return rho, loss, mass


def main():
    z = np.load(SOURCE)
    classes = np.load(CLASSES)["valid"][z["row_groups"]]
    result = {
        "scope": "Exploratory descriptive preview from preexisting fixed-profile sufficient sums, no new predictions or candidate scores.",
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "classes_sha256": hashlib.sha256(CLASSES.read_bytes()).hexdigest(),
        "class_scope": "Retrospective target-derived metadata, not a serving feature.",
        "profiles": {},
    }
    for name in ("candidate4", "reference4", "reference3"):
        rho, loss, mass = budget(z[name])
        per_class = {}
        for c in range(4):
            take = classes == c
            l, m = loss[take].sum(0), mass[take].sum(0)
            per_class[str(c)] = {"sequences": int(take.sum()), "loss_contribution": float(l.sum()),
                                 "loss_share": float(l.sum() / loss.sum()), "metric_weight_share": float(m.sum()),
                                 "loss_by_target": l.tolist(), "mass_by_target": m.tolist()}
        chain_loss = np.array([loss[z["chains"] == c].sum() for c in np.unique(z["chains"])])
        order = np.sort(chain_loss)[::-1]
        result["profiles"][name] = {
            "WP": float(rho.mean()), "rho_by_target": rho.tolist(),
            "deficit_by_target": loss.sum(0).tolist(), "deficit": float(loss.sum()),
            "class_budget": per_class,
            "chain_error_concentration": {str(k): float(order[:k].sum() / loss.sum()) for k in (1, 5, 25, 50)},
            "note": "Error concentration is descriptive; perfect knowledge and reducibility are not assumed.",
        }
    (HERE / "preview-existing-sums.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["profiles"]["candidate4"], indent=2))


if __name__ == "__main__":
    main()
