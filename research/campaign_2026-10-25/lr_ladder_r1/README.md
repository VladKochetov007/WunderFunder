# AdamW LR ladder point 2.5e-4 — 2026-10-25 R1

Prediction from the frozen-tail dose-response (see [mechanism panel](../mechanism_panel_r1/README.md)):
lower AdamW LR shrinks the frozen-unit tail and narrows the Muon gap; if AdamW 2.5e-4 matches
Muon, the Muon claim is an LR-tuning artifact for this recipe. Two fits (F0/s1001, F2/s1002)
vs reused AdamW 1e-3 / 5e-4 and Muon controls; [plans](PLANS_R1.json), [runner](run_panel.sh).
Queued after the saturation panel, before the Muon stage-6 fit.

## Result ([F0](grade_f0_R1.json), [F2](grade_f2_R1.json), [saturation](../optimizer_panel_r1/diagnostics/gate_saturation_lr25e5_R1.json))

| Comparison | F0 | F2 |
|---|---|---|
| AdamW 2.5e-4 − AdamW 5e-4 | −0.00129 [−0.00310, +0.00056] | **−0.00141 [−0.00259, −0.00011]** |
| Muon 1e-3 − AdamW 2.5e-4 | **+0.00385 [+0.00134, +0.00624]** | **+0.00487 [+0.00330, +0.00638]** |

AdamW's LR optimum on this recipe is ≈5e-4 (1e-3 and 2.5e-4 both worse), so **Muon beats the
best AdamW by +0.0026 / +0.0035**: the LR-fairness question is closed. F0 AdamW ladder
(z-sat / τ q99 / WP): 1e-3 45% / 546k / .6595; 5e-4 25% / 32k / .6608; 2.5e-4 13% / 1.6k /
.6595; Muon 3% / 166 / .6633. **AdamW faces a trade-off** — high LR learns more but collapses
gates into frozen units, low LR avoids collapse but under-trains; Muon escapes it (full-strength
updates with soft gates). The frozen tail is a cost, not the sole determinant.
