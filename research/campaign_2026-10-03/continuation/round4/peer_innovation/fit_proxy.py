from __future__ import annotations

import json
import time

import numpy as np
from common import ROOT, dump, predict, ridge_fit, ridge_inputs

NAMES = ["base", "base_plus_peer", "base_plus_shifted_peer"]


def experiment(
    base: np.ndarray,
    peer: np.ndarray,
    shift: np.ndarray,
    y: np.ndarray,
    component: np.ndarray,
    age: np.ndarray,
    fit_components: list[int],
    test_components: list[int],
) -> tuple[dict, dict]:
    fit = np.isin(component, fit_components)
    test = np.isin(component, test_components)
    used_fit = [c for c in fit_components if np.any(fit & (component == c))]
    used_test = [c for c in test_components if np.any(test & (component == c))]
    weight = np.array(
        [fit.sum() / (len(used_fit) * np.sum(fit & (component == c))) for c in component[fit]]
    )
    matrices = [base, np.c_[base, peer], np.c_[base, shift]]
    preparations = [ridge_inputs(x[fit], weight) for x in matrices]
    df = max(1, min(16, int(fit.sum()) - 2, min(p["rank"] for p in preparations) - 1))
    models = [ridge_fit(p, y[fit], weight, df) for p in preparations]
    trainpred = np.column_stack([predict(m, x[fit]) for m, x in zip(models, matrices, strict=True)])
    testpred = np.column_stack([predict(m, x[test]) for m, x in zip(models, matrices, strict=True)])
    losses = (testpred - y[test, None]) ** 2
    test_c = component[test]
    percomponent = np.array([losses[test_c == c].mean(0) for c in used_test])
    mse = percomponent.mean(0)
    rng = np.random.default_rng(614405)
    draws = rng.integers(0, len(used_test), size=(1000, len(used_test)))
    bootstrap = percomponent[draws].mean(1)
    contrasts = {}
    for bi in [0, 2]:
        improvement = 1 - mse[1] / mse[bi] if mse[bi] > 0 else 0.0
        interval = np.quantile(bootstrap[:, bi] - bootstrap[:, 1], [0.025, 0.975]).tolist()
        contrasts[NAMES[bi]] = dict(
            relative_improvement=float(improvement),
            mse_reduction=float(mse[bi] - mse[1]),
            paired_component95_interval=interval,
            positive_components=int(np.sum(percomponent[:, bi] > percomponent[:, 1])),
        )
    strata = []
    for name, selected in [("peer_age0", age[test] == 0), ("peer_age1+", age[test] > 0)]:
        n = int(selected.sum())
        count = len(set(test_c[selected]))
        means = []
        for c in used_test:
            m = selected & (test_c == c)
            if m.any():
                means.append(losses[m].mean(0))
        smse = np.mean(means, 0) if means else np.full(3, np.nan)
        adequate = n >= 12 and count >= 2
        reductions = [float(smse[bi] - smse[1]) for bi in [0, 2]] if means else [None, None]
        strata.append(
            dict(
                stratum=name,
                queries=n,
                components=count,
                adequate=adequate,
                mse=smse.tolist(),
                peer_mse_reductions=reductions,
            )
        )
    gate = all(
        c["relative_improvement"] >= 0.02 and c["paired_component95_interval"][0] > 0
        for c in contrasts.values()
    ) and all(s["adequate"] and all(v >= 0 for v in s["peer_mse_reductions"]) for s in strata)
    geometry = []
    for name, x, p, m in zip(NAMES, matrices, preparations, models, strict=True):
        geometry.append(
            dict(
                arm=name,
                columns=x.shape[1],
                numerical_rank=p["rank"],
                effective_df=m["edf"],
                intercept_df=1,
                lambda_=m["lambda_"],
                fit_queries=int(fit.sum()),
                test_queries=int(test.sum()),
                train_component_balanced_mse=float(
                    np.average((trainpred[:, NAMES.index(name)] - y[fit]) ** 2, weights=weight)
                ),
                hat_frobenius_from_base=float(np.linalg.norm(m["hat"] - models[0]["hat"])),
                coefficient_l2=float(np.linalg.norm(m["beta"])),
            )
        )
    result = dict(
        gate_passed=gate,
        fit_components=used_fit,
        test_components=used_test,
        fit_queries=int(fit.sum()),
        test_queries=int(test.sum()),
        input_only_effective_df_target=df,
        geometry=geometry,
        arm_test_component_mse={name: float(mse[a]) for a, name in enumerate(NAMES)},
        contrasts=contrasts,
        input_freshness_strata=strata,
        per_test_component=[
            dict(component=c, mse=percomponent[i].tolist()) for i, c in enumerate(used_test)
        ],
        target_summary=dict(
            fit_mean=float(y[fit].mean()),
            fit_sd=float(y[fit].std()),
            test_mean=float(y[test].mean()),
            test_sd=float(y[test].std()),
            test_zero_fraction=float(np.mean(y[test] == 0)),
            test_max_abs=float(np.abs(y[test]).max()),
        ),
        peer_shift_fit_correlation=float(np.corrcoef(peer[fit], shift[fit])[0, 1]),
        edf_control_max_abs=max(abs(m["edf"] - df) for m in models),
        uncertainty="1000 paired test-component draws conditional on exposed fixedpanel/split; no rowIID or population gain",
        null_hypothesis_scope="no useful addition for this fixed ridge protocol/observable, not complete conditional information",
    )
    saved = dict(
        fit=fit,
        test=test,
        component=component,
        target=y,
        prediction=testpred,
        train_prediction=trainpred,
        loss=losses,
        test_component=test_c,
        bootstrap_indices=draws,
        percomponent_loss=percomponent,
    )
    for name, m in zip(NAMES, models, strict=True):
        for field in ["mean", "scale", "block", "beta", "intercept", "lambda_"]:
            saved[f"{name}_{field}"] = m[field]
    return result, saved


def main() -> None:
    started = time.monotonic()
    panel = json.loads((ROOT / "samples.json").read_text())
    bases = []
    peers = []
    shifts = []
    targets = []
    ages = []
    components = []
    queries = []
    ledger = []
    for i, s in enumerate(panel["samples"]):
        meta = json.loads((ROOT / f"seq_{i:02d}.json").read_text())
        ledger.extend(meta["ledger"])
        with np.load(ROOT / f"seq_{i:02d}.npz") as z:
            bases.extend(z["base"])
            peers.extend(z["peer"])
            shifts.extend(z["shift"])
            targets.extend(z["target"])
            ages.extend(z["peer_age"])
            queries.extend(z["queries"])
            components.extend([s["component"]] * len(z["target"]))
    base = np.array(bases)
    peer = np.array(peers)
    shift = np.array(shifts)
    y = np.array(targets)
    age = np.array(ages)
    component = np.array(components)
    controls = json.loads((ROOT / "controls.json").read_text())
    result, saved = experiment(
        base, peer, shift, y, component, age, panel["fit_components"], panel["test_components"]
    )
    result.update(
        controls_passed=controls["passed"],
        promotion_allowed=result["gate_passed"] and controls["passed"],
        ledger=dict(
            original_selected=len(ledger),
            eligible=len(y),
            missing_future=sum(r["reason"] == "missing_future_end" for r in ledger),
            nonfinite=sum(r["reason"] == "nonfinite" for r in ledger),
            shift_missing=sum(not r["shift_history_available"] for r in ledger),
            original_total_query_inverse_probability_mass=sum(
                r["original_inverse_query_probability"] for r in ledger
            ),
            fit_inverse_query_mass=sum(
                r["original_inverse_query_probability"]
                for r in ledger
                if r["eligible"] and r["split"] == "fit"
            ),
            test_inverse_query_mass=sum(
                r["original_inverse_query_probability"]
                for r in ledger
                if r["eligible"] and r["split"] == "test"
            ),
        ),
        seconds=time.monotonic() - started,
        fits_executed=3,
        verdict="proxy gate passed, only future-observable candidate investigation warranted"
        if result["gate_passed"] and controls["passed"]
        else "frozen peer innovation proxy gate failed; no proposal promotion, not saturation",
    )
    saved.update(base=base, peer=peer, shift=shift, peer_age=age, queries=np.array(queries))
    np.savez(ROOT / "panel_models.npz", **saved)
    dump("results.json", result)
    dump("origins.json", dict(origins=ledger))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
