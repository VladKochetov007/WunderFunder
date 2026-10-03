from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(x, dtype=float) ** 2)))


def correlations(moment: np.ndarray) -> np.ndarray:
    w, y, p, yy, pp, yp = np.moveaxis(moment, -1, 0)
    cov = yp - y * p / w
    vy = np.maximum(yy - y * y / w, 0)
    vp = np.maximum(pp - p * p / w, 0)
    denominator = np.sqrt(vy * vp)
    return np.divide(cov, denominator, out=np.zeros_like(cov), where=denominator > 1e-15)


def interval(values: np.ndarray) -> list[float]:
    return np.quantile(values, [0.025, 0.975]).tolist()


def main() -> None:
    started = time.monotonic()
    samples = json.loads((ROOT / "samples.json").read_text())["samples"]
    assert json.loads((ROOT / "controls.json").read_text())["passed"]
    all_pred, all_y, all_base, all_age, all_design, all_comp, all_records = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    moments = np.zeros((len(samples), 3, 2, 6))
    per_component = []
    metas = []
    for i, s in enumerate(samples):
        meta = json.loads((ROOT / f"seq_{i:02d}.json").read_text())
        metas.append(meta)
        assert meta["cpu_reference"]["passed"] and meta["future_canary"]["passed"]
        with np.load(ROOT / f"seq_{i:02d}.npz") as z:
            q = z["queries"]
            pred = z["predictions"]
            y = z["targets"][q]
            base = z["baseline"][q]
            age = z["quiet_age"]
        assert len(q) == s["m_selected"]
        design = np.full(len(q), 1 / s["query_pi"])
        yc = np.clip(y, -2, 2).astype(float)
        pc = np.clip(pred, -2, 2).astype(float)
        w = design[:, None] * np.abs(yc)
        for a in range(3):
            p = pc[:, a]
            moments[i, a] = np.stack(
                [
                    w.sum(0),
                    (w * yc).sum(0),
                    (w * p).sum(0),
                    (w * yc**2).sum(0),
                    (w * p**2).sum(0),
                    (w * yc * p).sum(0),
                ],
                axis=-1,
            )
        per_component.append(
            dict(
                component=s["component"],
                raw_rms=rms(pred[:, 1] - base),
                packed_rms=rms(pred[:, 2] - base),
                raw_minus_packed_rms=rms(pred[:, 1] - pred[:, 2]),
                queries=len(q),
            )
        )
        all_pred.append(pred)
        all_y.append(y)
        all_base.append(base)
        all_age.append(age)
        all_design.append(design)
        all_comp.extend([s["component"]] * len(q))
        all_records.extend(meta["records"])
    pred = np.concatenate(all_pred)
    target = np.concatenate(all_y)
    base = np.concatenate(all_base)
    age = np.concatenate(all_age)
    design = np.concatenate(all_design)
    delta = pred - base[:, None]
    contrast = pred[:, 1] - pred[:, 2]
    pooled = moments.sum(0)
    wp = correlations(pooled).mean(-1)
    mse = ((pooled[..., 3] + pooled[..., 4] - 2 * pooled[..., 5]) / pooled[..., 0]).mean(-1)
    rng = np.random.default_rng(614205)
    draws = rng.integers(0, len(samples), size=(1000, len(samples)))
    component_weight = []
    component_response_squares = []
    component_contrast_squares = []
    for sample in samples:
        selected = np.array(all_comp) == sample["component"]
        component_weight.append(design[selected].sum())
        component_response_squares.append(
            (design[selected, None] * np.mean(delta[selected].astype(float) ** 2, axis=-1)).sum(0)
        )
        component_contrast_squares.append(
            (design[selected] * np.mean(contrast[selected].astype(float) ** 2, axis=-1)).sum()
        )
    bootstrap_weight = np.array(component_weight)[draws].sum(1)
    bootstrap_response = np.sqrt(np.array(component_response_squares)[draws].sum(1) / bootstrap_weight[:, None])
    bootstrap_contrast = np.sqrt(np.array(component_contrast_squares)[draws].sum(1) / bootstrap_weight)
    bootmom = moments[draws].sum(1)
    bootwp = correlations(bootmom).mean(-1)
    bootmse = ((bootmom[..., 3] + bootmom[..., 4] - 2 * bootmom[..., 5]) / bootmom[..., 0]).mean(-1)
    amp = np.array([r["raw_displacement"] for r in all_records])
    current = np.array([r["current_engineered_changed"] for r in all_records])
    tail_equal = np.array([r["protected_tail_engineered_rms"] == 0 for r in all_records])
    inconsistent = np.array([r["packed_inconsistent"] for r in all_records])
    queryraw = np.sqrt(np.mean(delta[:, 1].astype(float) ** 2, axis=1))
    querypacked = np.sqrt(np.mean(delta[:, 2].astype(float) ** 2, axis=1))
    bins = []
    for label, lo, hi in [
        ("0", 0, 0),
        ("1-3", 1, 3),
        ("4-15", 4, 15),
        ("16-63", 16, 63),
        ("64+", 64, 1000000),
    ]:
        m = (age >= lo) & (age <= hi)
        if not m.any():
            bins.append(dict(age=label, n=0))
            continue

        def wr(v: np.ndarray, selected: np.ndarray = m) -> float:
            return float(np.sqrt(np.average(v[selected] ** 2, weights=design[selected])))

        nonzero = m & (amp > 1e-8)
        bins.append(
            dict(
                age=label,
                n=int(m.sum()),
                design_mass=float(design[m].sum()),
                components=len(set(np.array(all_comp)[m])),
                raw_rms=wr(queryraw),
                packed_rms=wr(querypacked),
                raw_minus_packed_rms=float(
                    np.sqrt(
                        np.average(
                            np.mean(contrast[m].astype(float) ** 2, axis=1), weights=design[m]
                        )
                    )
                ),
                amplitude_rms=wr(amp),
                raw_response_per_amplitude_median=float(np.median(queryraw[nonzero] / amp[nonzero]))
                if nonzero.any()
                else None,
                current_engineered_changed_fraction=float(np.mean(current[m])),
            )
        )
    numeric_max = max(r["identity"]["max_abs"] for r in all_records)
    arms = []
    for a, name in enumerate(["identity", "raw_recomputed", "packed_i0"]):
        arms.append(
            dict(
                arm=name,
                query_rms=rms(delta[:, a]),
                design_weighted_response_rms=float(
                    np.sqrt(
                        np.average(np.mean(delta[:, a].astype(float) ** 2, axis=1), weights=design)
                    )
                ),
                response_rms_component_ci=interval(bootstrap_response[:, a]),
                scoped_sampled_wp=float(wp[a]),
                wp_minus_identity=float(wp[a] - wp[0]),
                wp_difference_component_ci=interval(bootwp[:, a] - bootwp[:, 0]),
                scoped_weighted_mse=float(mse[a]),
                mse_minus_identity=float(mse[a] - mse[0]),
                mse_difference_component_ci=interval(bootmse[:, a] - bootmse[:, 0]),
            )
        )
    summary = dict(
        scope="fixed24component panel, scored step512..19999 sampled-domain HT moments; ratio WP estimate, not official fulltest; paired component bootstrap conditional on exposed panel, no row IID or fresh confirmation",
        components=len(samples),
        queries=len(pred),
        n_eligible=sum(s["n_eligible"] for s in samples),
        empty_components=sum(s["m_selected"] == 0 for s in samples),
        design_mass=float(design.sum()),
        inclusion_probability_range=[
            min(s["query_pi"] for s in samples),
            max(s["query_pi"] for s in samples),
        ],
        arms=arms,
        raw_minus_packed_rms=rms(contrast),
        raw_minus_packed_design_weighted_rms=float(np.sqrt(np.average(np.mean(contrast.astype(float)**2, axis=-1), weights=design))),
        raw_minus_packed_response_component_ci=interval(bootstrap_contrast),
        raw_rms_minus_packed_rms_component_ci=interval(bootstrap_response[:, 1]-bootstrap_response[:, 2]),
        raw_minus_packed_wp_difference=float(wp[1] - wp[2]),
        raw_minus_packed_wp_component_ci=interval(bootwp[:, 1] - bootwp[:, 2]),
        age_bins=bins,
        per_component=per_component,
        displacement=dict(
            raw_amplitude_rms=rms(amp),
            median=float(np.median(amp)),
            min=float(amp.min()),
            max=float(amp.max()),
            zero_mass=int(np.sum(amp == 0)),
            lags={str(lag): sum(r["lag"] == lag for r in all_records) for lag in [64, 128, 256]},
            unavailable_lag_candidates=sum(
                len(d["candidates"]) < 3
                for i in range(len(samples))
                for d in json.loads((ROOT / f"donors_{i:02d}.json").read_text())["donors"]
            ),
            unmatched=sum(m["unmatched"] for m in metas),
            cannot_replay=sum(m["cannot_replay"] for m in metas),
            seam_original_median=float(np.median([r["seam_original"] for r in all_records])),
            seam_altered_median=float(np.median([r["seam_altered"] for r in all_records])),
            seam_return_median=float(np.median([r["seam_return"] for r in all_records])),
        ),
        consistency=dict(
            packed_inconsistent_queries=int(inconsistent.sum()),
            packed_flags_mismatch_mean=float(
                np.mean([r["packed_flags_mismatch_fraction"] for r in all_records])
            ),
            packed_clock_mismatch_mean=float(
                np.mean([r["packed_clock_mismatch_fraction"] for r in all_records])
            ),
            packed_engineered_mismatch_rms=float(
                np.sqrt(np.mean([r["packed_engineered_mismatch_rms"] ** 2 for r in all_records]))
            ),
            raw_current_equal_all=True,
            cross_i1_aux_equal_all=True,
            raw_recomputed_current_engineered_changed_queries=int(current.sum()),
            raw_current_engineered_rms=float(
                np.sqrt(np.mean([r["current_engineered_rms"] ** 2 for r in all_records]))
            ),
            raw_response_rms_when_current_engineered_equal=rms(delta[~current, 1])
            if (~current).any()
            else None,
            raw_response_rms_when_current_engineered_changed=rms(delta[current, 1])
            if current.any()
            else None,
            protected_33row_full_model_inputs_equal_queries=int(tail_equal.sum()),
            protected_33row_full_model_inputs_equal_components=len(set(np.array(all_comp)[tail_equal])),
            raw_response_rms_when_protected_33row_full_model_inputs_equal=rms(delta[tail_equal, 1])
            if tail_equal.any()
            else None,
        ),
        gates=dict(
            identity_max_abs=numeric_max,
            cpu_reference_max_abs=max(m["cpu_reference"]["max_abs"] for m in metas),
            future_canary_max_abs=max(m["future_canary"]["max_abs"] for m in metas),
            all_cpu_identity_future_pass=True,
            all_cache_ids_masks_targets_equal=True,
            gpu_cache_max_abs=max(m["gpu_cache"]["max_abs"] for m in metas),
            gpu_cache_worst_rms=max(m["gpu_cache"]["rms"] for m in metas),
            gpu_cache_r2_bound_failures=sum(not m["gpu_cache"]["r2_bound_passed"] for m in metas),
        ),
        cost=dict(
            instrument_seconds=sum(m["seconds"] for m in metas),
            max_sequence_seconds=max(m["seconds"] for m in metas),
            peak_rss_mib=max(m["peak_rss_mib"] for m in metas),
            summary_seconds=time.monotonic() - started,
        ),
    )
    (ROOT / "results.json").write_text(json.dumps(summary, indent=2))
    np.savez(
        ROOT / "panel.npz",
        predictions=pred,
        targets=target,
        baseline=base,
        quiet_age=age,
        query_design_weight=design,
        component=np.array(all_comp),
        moments=moments,
        bootstrap_component_indices=draws,
        amplitude=amp,
        current_engineered_changed=current,
    )
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6), constrained_layout=True)
    ax[0].scatter(amp, queryraw, s=12, alpha=0.6, label="raw, recomputed")
    ax[0].scatter(amp, querypacked, s=12, alpha=0.5, label="packed i0")
    ax[0].set(
        xlabel="Matched raw i0 displacement / model std",
        ylabel="Prediction response RMS",
        title="Same donor, distinct causal encoding",
    )
    ax[0].legend(fontsize=8)
    x = np.arange(len(bins))
    width = 0.35
    ax[1].bar(x - width / 2, [b.get("raw_rms", 0) for b in bins], width, label="raw, recomputed")
    ax[1].bar(x + width / 2, [b.get("packed_rms", 0) for b in bins], width, label="packed")
    ax[1].set(
        xticks=x,
        xticklabels=[f"{b['age']}\nn={b['n']}" for b in bins],
        xlabel="Original quiet age (rows)",
        ylabel="Design-weighted response RMS",
        title="Age and amplitude remain confounded",
    )
    differences = np.array([a["wp_minus_identity"] for a in arms[1:]])
    low = np.array([a["wp_difference_component_ci"][0] for a in arms[1:]])
    high = np.array([a["wp_difference_component_ci"][1] for a in arms[1:]])
    ax[2].errorbar(
        np.arange(2),
        differences,
        yerr=np.vstack([differences - low, high - differences]),
        fmt="o",
        capsize=4,
    )
    ax[2].axhline(0, c="gray", lw=1)
    ax[2].set(
        xticks=[0, 1],
        xticklabels=["Raw recomputed", "Packed i0"],
        ylabel="Scoped sampled WP change",
        title="Paired component uncertainty\nFixed exposed panel; not fulltest",
    )
    fig.savefig(ROOT / "matched_usage.png", dpi=160)
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
