from __future__ import annotations

import json
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from common import ROOT, dump, rms


def ci(x: np.ndarray) -> list[float]:
    return np.quantile(x[np.isfinite(x)], [0.025, 0.975]).tolist()


def main() -> None:
    started = time.monotonic()
    samples = json.loads((ROOT / "samples.json").read_text())["samples"]
    summaries = []
    detail = []
    weights = []
    components = []
    response = []
    identity = []
    hidden = []
    component_weight = []
    component_squares = []
    all_records = []
    for i, s in enumerate(samples):
        m = json.loads((ROOT / f"seq_{i:02d}.json").read_text())
        summaries.append(m)
        w = 1 / s["query_pi"]
        all_records.extend(
            dict(**r, component=s["component"], design_weight=w) for r in m["records"]
        )
        if not m["accepted"]:
            component_weight.append(0)
            component_squares.append(0)
            detail.append(
                dict(
                    component=s["component"], accepted=0, rejected=m["rejected"], response_rms=None
                )
            )
            continue
        with np.load(ROOT / f"seq_{i:02d}.npz") as z:
            delta = z["predictions"].astype(float) - z["identity_predictions"].astype(float)
            numeric = z["identity_predictions"].astype(float) - z[
                "original_reference_query"
            ].astype(float)
            h = z["hidden_rms"]
        squares = np.mean(delta**2, axis=1)
        response.extend(delta)
        identity.extend(numeric)
        hidden.extend(h)
        weights.extend([w] * m["accepted"])
        components.extend([s["component"]] * m["accepted"])
        component_weight.append(w * m["accepted"])
        component_squares.append(w * squares.sum())
        detail.append(
            dict(
                component=s["component"],
                accepted=m["accepted"],
                rejected=m["rejected"],
                response_rms=rms(delta),
            )
        )
    accepted = sum(m["accepted"] for m in summaries)
    total = sum(s["m_selected"] for s in samples)
    gate_counts = {
        name: sum(r["reason"] == name for r in all_records)
        for name in [
            "accepted_exact_suffix",
            "engineered_suffix_mismatch",
            "nonfinite",
            "unmatched",
        ]
    }
    gate_mass = {
        name: sum(r["design_weight"] for r in all_records if r["reason"] == name)
        for name in gate_counts
    }
    result = dict(
        panel=dict(
            components=24,
            original_eligible_scored_after511=sum(s["n_eligible"] for s in samples),
            selected=total,
            input_donor_eligible=sum(r["lag"] is not None for r in all_records),
            accepted=accepted,
            rejected=total - accepted,
            unmatched=gate_counts["unmatched"],
            nonfinite=gate_counts["nonfinite"],
            components_with_acceptance=sum(m["accepted"] > 0 for m in summaries),
            empty_accepted_components=[
                m["sample"]["component"] for m in summaries if not m["accepted"]
            ],
            no_origins_replaced=True,
        ),
        feature_gate=dict(
            atol=0.0,
            rtol=0.0,
            domain="all211 inputs t-32..t,33rows, exact float32 value equality",
            counts=gate_counts,
            original_query_inverse_probability_mass=gate_mass,
            rejected_clock_mismatch_origins=sum(
                r.get("suffix_clock_unequal_entries", 0) > 0
                for r in all_records
                if not r["accepted"]
            ),
            rejected_held_mismatch_origins=sum(
                r.get("suffix_held_unequal_entries", 0) > 0
                for r in all_records
                if not r["accepted"]
            ),
        ),
        sampling_scope="original fixed exposed24component/288query panel, gate conditional on input history, inverse original query probabilities; retained/rejected mass is an estimate of original after511 domain, not a census or official score",
        component_results=detail,
        state_cache=dict(
            exact_old_boundary_states_reused=sum(
                m.get("exact_prior_boundary_states_reused", 0) for m in summaries
            ),
            new_boundary_states=sum(m.get("new_boundary_states", 0) for m in summaries),
            original_sequence_forward_count=sum(
                m.get("original_sequence_forward_count", 0) for m in summaries
            ),
            maximum_original_sequence_forwards_per_sequence=max(
                m.get("original_sequence_forward_count", 0) for m in summaries
            ),
        ),
        cost=dict(
            body_seconds=sum(m["seconds"] for m in summaries),
            max_sequence_body_seconds=max(m["seconds"] for m in summaries),
            peak_body_rss_mib=max(m["peak_rss_mib"] for m in summaries),
        ),
    )
    if not accepted:
        result.update(
            verdict="useful null: no origins pass exact engineered suffix gate; no memory claim",
            response=None,
        )
    else:
        response = np.array(response)
        identity = np.array(identity)
        weights = np.array(weights)
        hidden = np.array(hidden)
        cw = np.array(component_weight)
        cs = np.array(component_squares)
        rng = np.random.default_rng(614305)
        draws = rng.integers(0, 24, size=(1000, 24))
        denominator = cw[draws].sum(1)
        boot = np.sqrt(
            np.divide(
                cs[draws].sum(1), denominator, out=np.full(1000, np.nan), where=denominator > 0
            )
        )
        weighted = float(np.sqrt(cs.sum() / cw.sum()))
        numeric = rms(identity)
        floor = max(0.001, 100 * numeric)
        accepted_records = [r for r in all_records if r["accepted"]]
        real_records = [r for m in summaries for r in m.get("accepted_results", [])]
        result.update(
            response=dict(
                raw_query_rms=rms(response),
                per_target_query_rms=np.sqrt(np.mean(response**2, axis=0)).tolist(),
                design_weighted_query_rms=weighted,
                conditional_component95_interval=ci(boot),
                bootstrap_draws=1000,
                valid_bootstrap_draws=int(np.isfinite(boot).sum()),
                seed=614305,
                identity_query_rms=numeric,
                identity_max_window_abs=max(r["identity"]["max_abs"] for r in real_records),
                hidden_layer_rms=np.sqrt(np.average(hidden**2, axis=0, weights=weights)).tolist(),
                usage_floor=floor,
                passed_usage_floor=weighted > floor,
                uncertainty_scope="paired components conditional on fixed exposed panel, realized queries and input acceptance; not rowIID or survey-design confidence interval",
            ),
            gates=dict(
                all_identity_pass=all(r["identity"]["passed"] for r in real_records),
                future_canary_count=sum(m["model_forward_executed"] for m in summaries),
                future_canary_max_abs=max(
                    m["future_canary"]["max_abs"] for m in summaries if m["model_forward_executed"]
                ),
                all_future_pass=all(
                    m["future_canary"]["passed"] for m in summaries if m["model_forward_executed"]
                ),
                accepted_suffix_max_abs=max(r["suffix_max_abs"] for r in accepted_records),
            ),
            support=dict(
                accepted_raw_amplitude_rms=rms(
                    np.array([r["amplitude"] for r in accepted_records])
                ),
                accepted_amplitude_range=[
                    min(r["amplitude"] for r in accepted_records),
                    max(r["amplitude"] for r in accepted_records),
                ],
                all_chosen_lags={
                    str(lag): sum(r["lag"] == lag for r in all_records) for lag in [64, 128, 256]
                },
                unavailable_lag_candidates=sum(3 - len(r["candidates"]) for r in all_records),
                accepted_seam_original_median=float(
                    np.median([r["seam_original"] for r in real_records])
                ),
                accepted_seam_altered_median=float(
                    np.median([r["seam_altered"] for r in real_records])
                ),
                accepted_seam_return_median=float(
                    np.median([r["seam_return"] for r in real_records])
                ),
                market_support="unproven; nearestlag does not certify conditional exchangeability",
            ),
            verdict="carried recurrent sensitivity survives identical33row model-input suffix"
            if weighted > floor
            else "no material carried sensitivity above frozen threshold on accepted origins; not a saturation claim",
        )
        np.savez(
            ROOT / "accepted_panel.npz",
            response=response,
            identity_error=identity,
            query_design_weight=weights,
            component=np.array(components),
            hidden_rms=hidden,
            bootstrap_component_indices=draws,
            component_design_weight=cw,
            component_response_squares=cs,
        )
    result["cost"]["summary_seconds_before_plot"] = time.monotonic() - started
    dump(ROOT / "results.json", result)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.7), constrained_layout=True)
    x = np.arange(24)
    n = np.array([d["accepted"] for d in detail])
    axes[0].bar(x, n, label="Exact suffix accepted")
    axes[0].bar(x, 12 - n, bottom=n, label="Rejected", alpha=0.5)
    axes[0].set(
        xlabel="Fixed component index",
        ylabel="Queries (12 each)",
        title="No replacement of rejected origins",
    )
    axes[0].legend(fontsize=8)
    axes[1].plot(
        x, [d["response_rms"] if d["response_rms"] is not None else np.nan for d in detail], "o"
    )
    axes[1].axhline(0.001, color="gray", ls="--", label="Frozen .001 usage floor")
    axes[1].set(
        xlabel="Fixed component index",
        ylabel="Accepted prediction response RMS",
        title="Identical final33 model inputs",
    )
    axes[1].legend(fontsize=8)
    fig.savefig(ROOT / "washout.png", dpi=160)
    plt.close(fig)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
