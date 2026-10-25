# Saturation panel — 2026-10-25 R1

**Evidence (zero-fit E0, [gate_saturation.py](../optimizer_panel_r1/diagnostics/gate_saturation.py) →
[R1 json](../optimizer_panel_r1/diagnostics/gate_saturation_R1.json), 10 F0 evaluation sequences,
exact recurrence replay):**

| Model | z-gate saturated (<.02 or >.98) | r-gate saturated | τ=1/(1−z) q50 / q90 / q99 | z preact RMS (input / recurrent) |
|---|---:|---:|---|---|
| AdamW F0 | 45.2% | 23.2% | 7.9 / 3,010 / 546,035 | 4.24 / 3.67 |
| Muon F0 | 3.1% | 1.2% | 2.5 / 10.8 / 166 | 1.16 / 0.75 |
| AdamW F2 | 48.4% | 21.5% | 22.6 / 4,322 / 660,418 | 3.73 / 4.05 |
| Muon F2 | 3.3% | 1.1% | 2.5 / 11.2 / 213 | 1.14 / 0.76 |

AdamW's GRU splits into frozen-memory units (effective timescales far beyond the 20k-row
sequence) and instant units; Muon keeps gates soft with short graded timescales. This
corroborates the gate-saturation mechanism (Grok) and fails the Opus falsifier (AdamW
saturation is ~15× Muon's), correlationally.

**Arms (F0/s1001, F2/s1002; frozen source `eb369bd3c`):**

| Arm | Change | Control | Question |
|---|---|---|---|
| S1 | AdamW + `--spectral-cap 4 --spectral-cap-every 10` (new) | AdamW | does preventing gate-block spectral growth reproduce Muon's gain? |
| S2 | Muon + `--rnn-init chrono` | Muon | can soft gates plus long initial timescales add memory Muon lacks (q99 τ≈200)? |
| S3 | Muon H512×2 | Muon H512×1 | does Muon unlock depth/scale that AdamW could not use? |

Queue: clip panel → mechanism panel → this panel → Muon stage-6 production-lineage fit.

**Amendment (before launch):** S2 Muon+chrono replaced by S2 Muon `--muon-rms 0.3` after the mechanism panel showed chrono init −0.0057 under AdamW and Muon half-step −0.0024 (the step-size direction to test is up, not long timescales).

## Interim F0 ([grade](grade_f0_partial_R1.json), [saturation](../optimizer_panel_r1/diagnostics/gate_saturation_sat_R1.json))

| Arm | WP | Paired Δ [95% CI] | z-sat | τ q99 | z preact RMS in/rec |
|---|---:|---|---:|---:|---|
| S1 AdamW + spectral cap 4 | 0.66061 | +0.00107 vs AdamW [−0.00213, +0.00431]; Muon − S1 **+0.00274 [+0.00080, +0.00472]** | 25.4% | 61,591 | 2.91 / 1.80 |
| S2 Muon rms 0.3 | 0.66385 | +0.00050 vs Muon 0.2 [−0.00164, +0.00290] | 5.4% | 414 | 1.36 / 0.93 |

Post-hoc spectral capping of the GRU gate blocks halves saturation (same rung as AdamW 5e-4:
25% / 32k / .6608) but leaves the update gate's input drive large; Muon keeps a significant
lead. Weight-norm control alone does not reproduce update-side spectral equalization. Muon's
step is on a plateau (0.1 −0.0024; 0.3 +0.0005 vs 0.2). Dose-response (z-sat / τ q99 / WP,
F0): AdamW 1e-3 45% / 546k / .6595; M1 14% / 229k / .6594; AdamW 5e-4 25% / 32k / .6608;
S1 25% / 62k / .6606; Muon .2 3% / 166 / .6633; Muon .3 5% / 414 / .6639.
