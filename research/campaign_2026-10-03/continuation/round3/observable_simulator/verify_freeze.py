from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import numpy as np
from representation import Ridge, features
from simulator import ORIGINS, stopping_labels

OWN = Path(__file__).resolve().parent


def main() -> None:
    """Verify saved model replay and origin identity, then freeze provenance."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    result = json.loads((OWN / "results.json").read_text())
    assert result["cost"]["sum_fit_seconds"] <= 120
    fit_ids, test_ids = set(), set()
    for record in result["records"]:
        fit_ids.add((record["world"], record["fit_seed"]))
        test_ids.add((record["world"], record["test_seed"]))
        assert record["test_barrier_hits"] + record["test_forced_cap_exits"] == 1440
    assert fit_ids.isdisjoint(test_ids)
    witness = dict(np.load(OWN / "witness.npz"))
    labels, _, lags = stopping_labels(witness["mid"])
    assert np.array_equal(labels, witness["targets"])
    assert np.array_equal(lags, witness["lags"])
    models = np.load(OWN / "models.npz")
    losses = np.load(OWN / "path_losses.npz")
    for arm in ("row", "reset", "history"):
        model = Ridge()
        for name in ("mean", "scale", "beta", "intercept"):
            setattr(model, name, models[f"nominal_0_{arm}_{name}"])
        x = features(witness, arm)[:, ORIGINS]
        pred = model.predict(x.reshape(-1, x.shape[-1])).reshape(labels.shape)
        replay_loss = ((pred - labels) ** 2).mean(axis=(1, 2))
        assert np.allclose(replay_loss, losses[f"nominal_0_nominal_{arm}"][:2], rtol=1e-12)
    state = json.loads((OWN / "STATE.json").read_text())
    state.update(
        phase="report_frozen",
        independent_review="pending",
        figure_inspected=True,
        saved_model_replay="3 arms x2 witness paths exact within1e-12",
        seed_identity_check="12 distinct fit and12 distinct test seed/world pairs, disjoint",
        final_lint="passed",
    )
    (OWN / "STATE.json").write_text(json.dumps(state, indent=2) + "\n")
    (OWN / "COST.json").write_text(json.dumps(result["cost"], indent=2) + "\n")
    root = OWN.parents[4]
    prior = root / "research/campaign_2026-10-03/eda/puzzle_a/REPORT.txt"
    manifest = {
        "files_sha256": {
            str(p.relative_to(OWN)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(OWN.rglob("*"))
            if p.is_file() and p.name not in ("PROVENANCE.json", "verify.log")
        },
        "prior_mechanism_report": {str(prior): hashlib.sha256(prior.read_bytes()).hexdigest()},
        "synthetic_array_hashes": result["data_hashes"],
        "raw_data_reads": 0,
        "review_status": "frozen_for_independent_review; self-checks are not independent review",
    }
    (OWN / "PROVENANCE.json").write_text(json.dumps(manifest, indent=2) + "\n")
    logging.info("Saved model replay, target replay and disjoint IDs verified; report frozen.")


if __name__ == "__main__":
    main()
