"""Assemble engine/results.json from the run artifacts (profile, calibration, parity)."""
import json
from pathlib import Path

E = Path(__file__).resolve().parent
runs = E / "runs"
prof = json.loads((runs / "profile_f16d.json").read_text())["results"]
calib = [json.loads(l) for l in (runs / "profile_f16d.log").read_text().splitlines() if l.startswith('{"zip"')]
inc = [c for c in calib if "since2_gclp" in c["zip"]]
mix = [c for c in calib if "mixed" in c["zip"]]
kc_inc = sum(c["kcycles_per_row"] for c in inc) / len(inc)
kc_mix = sum(c["kcycles_per_row"] for c in mix) / len(mix)
us_inc = sum(c["us_per_row"] for c in inc) / len(inc)
PUBLIC_INC, PUBLIC_MIX = 2187.0, 1552.0  # seconds: final_since2_gclp (36m27s), final_mixed_ema6_shared (25m52s)
GATE_MIN = 52.0


def project(kc):
    by_inc = PUBLIC_INC * kc / kc_inc / 60
    by_mix = PUBLIC_MIX * kc / kc_mix / 60
    return round(by_inc, 1), round(by_mix, 1)


table = {}
for name, r in prof.items():
    kc = r["real_kcycles_per_row"]
    a, b = project(kc)
    table[name] = {
        **r,
        "public_min_by_incumbent_ratio": a,
        "public_min_by_mixed_ratio": b,
        "fits_52min_gate_conservative": a <= GATE_MIN,
    }
kc_budget = GATE_MIN * 60 / PUBLIC_INC * kc_inc
parity = json.loads((runs / "parity_f16c_v3.json").read_text())
torch_parity = json.loads((runs / "torch_parity_f16d_v3.json").read_text())

results = {
    "date": "2026-10-01",
    "host": "i9-11900H laptop, 1 pinned logical CPU (taskset -c 15), throttled 1.9-3.3 GHz; timings in user "
    "core cycles via perf_event_open, wall-clock us also given",
    "harness": "scripts/engine_fp16_check.py replays utils.ScorerStepByStep per-row loop (DataPoint, "
    "np.asarray, isfinite, copy) around the callback; Python 3.11.16, numpy 2.2.6, pyarrow 19.0.1",
    "calibration": {
        "incumbent_zip": "submissions/final_since2_gclp/final_since2_gclp.zip",
        "incumbent_public_s": PUBLIC_INC,
        "incumbent_local_kcycles_per_row_runs": [c["kcycles_per_row"] for c in inc],
        "incumbent_local_us_per_row_runs": [c["us_per_row"] for c in inc],
        "mixed_zip": "submissions/final_mixed_ema6_shared/final_mixed_ema6_shared.zip",
        "mixed_public_s": PUBLIC_MIX,
        "mixed_local_kcycles_per_row_runs": [c["kcycles_per_row"] for c in mix],
        "local_ratio_mixed_over_incumbent": round(kc_mix / kc_inc, 3),
        "public_ratio_mixed_over_incumbent": round(PUBLIC_MIX / PUBLIC_INC, 3),
        "model": "public_s = 2187 * kcyc(candidate) / kcyc(incumbent), same session, intercept a = 0 "
        "(conservative: parquet I/O ~35-45 ms/sequence locally does not scale with model cost); "
        "the mixed-zip ratio gives ~7% lower minutes",
        "kcycles_budget_for_52min": round(kc_budget, 1),
    },
    "bench": table,
    "parity_valid_3seq": parity,
    "parity_torch_reference": torch_parity,
}
results["ledger_entry"] = {
    "id": "E-ENG-2026-10-01",
    "date": "2026-10-01",
    "kind": "engine",
    "title": "Native 16-bit weight engine (F16 via F16C, I16 per-row scale), LSTM cell, wide merged heads, native callback + governor",
    "result": {
        "i16_parity_max_abs_valid3seq_worst": max(c["i16"]["max_abs_diff_all_rows"] for c in parity["cases"].values()),
        "f16_parity_max_abs_valid3seq_worst": max(c["f16"]["max_abs_diff_all_rows"] for c in parity["cases"].values()),
        "i16_delta_wp_worst_abs": max(abs(c["i16"]["delta_wp"]) for c in parity["cases"].values()),
        "kcyc_incumbent": round(kc_inc, 1),
        "kcyc_2xh128_f32_i16_f16": [prof[f"2xh128_{p}"]["real_kcycles_per_row"] for p in ("f32", "i16", "f16")],
        "kcyc_2xh128+h256_f32_i16_f16": [prof[f"2xh128+h256_{p}"]["real_kcycles_per_row"] for p in ("f32", "i16", "f16")],
        "public_min_2xh128+h256_i16_f16": [table["2xh128+h256_i16"]["public_min_by_incumbent_ratio"], table["2xh128+h256_f16"]["public_min_by_incumbent_ratio"]],
        "public_min_3xh128+h256_i16_f16": [table["3xh128+h256_i16"]["public_min_by_incumbent_ratio"], table["3xh128+h256_f16"]["public_min_by_incumbent_ratio"]],
    },
    "verdict": "I16 passes the 1e-4 parity gate (WP-neutral) and cuts per-row cost ~27% (2xh128) to ~27% (2xh128+h256) vs our fp32 ensemble; F16 is 10-15% faster than I16 but fails 1e-4 (max 3.9e-4-5.8e-4, dWP ~1e-7). Within a conservative 52-min gate: 2xh128+h256 (i16 51.5 / f16 45.0 min), 6xh128 f16 (49.1); not 3xh128+h256 i16 (60.5), 4xh128+h256, or 3xh128+lstm128+h256.",
    "artifacts": "research/campaign_2026-10-01/engine/{REPORT.md,results.json,runs/}",
}
# Merged multi-head members (head width k*H), clean session with its own calibration.
wide_prof = json.loads((runs / "profile_wide2.json").read_text())["results"]
wide_cal = [json.loads(l) for l in (runs / "profile_wide2.log").read_text().splitlines() if l.startswith('{"zip"')]
w_inc = sum(c["kcycles_per_row"] for c in wide_cal if "since2_gclp" in c["zip"]) / 2
w_mix = sum(c["kcycles_per_row"] for c in wide_cal if "mixed" in c["zip"]) / 2
results["bench_wide_heads"] = {
    "members": "h128w2 = real since trunk + [own head, other since member's head] (256-wide head.0); "
    "h256w2 = synthetic trained-scale h256 trunk + 2 heads (512-wide); A = 2 x h128w2; B = A + h256w2",
    "calibration_kcycles": {"incumbent": round(w_inc, 1), "mixed": round(w_mix, 1)},
    "kcycles_budget_for_52min": round(GATE_MIN * 60 / PUBLIC_INC * w_inc, 1),
    "configs": {
        name: {
            **r,
            "public_min_by_incumbent_ratio": round(PUBLIC_INC * r["real_kcycles_per_row"] / w_inc / 60, 1),
            "public_min_by_mixed_ratio": round(PUBLIC_MIX * r["real_kcycles_per_row"] / w_mix / 60, 1),
        }
        for name, r in wide_prof.items()
    },
    "parity_merged_2head": json.loads((runs / "parity_merged_final_v3.json").read_text())["cases"],
}
(E / "results.json").write_text(json.dumps(results, indent=1))
print(f"wide session: inc {w_inc:.1f} mix {w_mix:.1f}")
for name, r in results["bench_wide_heads"]["configs"].items():
    print(f"{name:28s} {r['real_kcycles_per_row']:7.1f} kcyc  public {r['public_min_by_incumbent_ratio']:5.1f} / {r['public_min_by_mixed_ratio']:5.1f} min")
print(f"kc_inc {kc_inc:.1f} kc_mix {kc_mix:.1f} us_inc {us_inc:.1f} budget {kc_budget:.1f}")
for name, r in table.items():
    print(f"{name:34s} {r['real_kcycles_per_row']:7.1f} kcyc {r['real_us_per_row']:7.2f} us  public {r['public_min_by_incumbent_ratio']:5.1f} / {r['public_min_by_mixed_ratio']:5.1f} min {'OK' if r['fits_52min_gate_conservative'] else '--'}")
