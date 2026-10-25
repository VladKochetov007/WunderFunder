# Optimizer diagnostics and root hypotheses (committed before external reviews), 2026-10-25

[state_spectra.py](state_spectra.py) → [state_spectra_R1.json](state_spectra_R1.json): final
weights and optimizer state of matched AdamW vs Muon fast-recipe fits (F0, F2).

Findings (root):
1. W_hh gate blocks under AdamW: top sv 12.4 / 25.4–29.2 / 5.0, stable rank 3.8–4.1 / 2.0–2.4 /
   10.8–11.8 (r, z, n). Under Muon: top sv 1.8 / 3.1–3.2 / 2.2–2.3, stable rank 82–85 / 34–37 /
   47–49. W_ih similar contrast. GRU biases end ~2× larger under Muon.
2. Pre-clip gradient norm is dominated by the head (median 1.3–1.8 vs W_hh 0.04–0.11); clip 1.0
   active on 80–94% of steps → global clipping is head-driven and shrinks recurrent gradients.
3. AdamW end-state |m|/sqrt(v) median 0.07–0.15 in every family (≤ ~0.23 pure-noise level for
   β1=0.9; caveats: end of anneal, OneCycle β1 cycling).
4. **Known bug:** for MuonAdamW runs the state→name mapping in `state_spectra.py` is wrong
   (param groups are reordered); Muon spectra are valid (read from weights), Muon state rows are
   not.

Root mechanism hypotheses for Muon's +0.004 (pre-registered):
- H-a **effective step**: Adam's per-coordinate step ∝ SNR (≈0.1–0.2 here) and self-anneals in
  a noise-dominated regime; Muon's step RMS is fixed (0.2·sqrt(max(m,n))·lr) regardless of SNR
  → much larger recurrent steps. Test D2: AdamW with body LR ×3 (`--lr 3e-3 --head-lr-mult 0.333`).
- H-b **clip decoupling**: orthogonalization discards magnitude, so Muon's recurrent updates
  ignore head-driven clipping. Test D1: AdamW `--clip-per-module head:1.0` (each module
  clipped separately; zero code).
- H-c **conditioning**: Muon prevents the low-rank, large-sv update-gate collapse; would show
  as a gain that neither D1 nor D2 reproduces. Follow-up: spectral-norm-capped AdamW.
- D3 Muon + per-module clip and D4 Muon `--muon-scope hidden` probe additivity.

## Zero-fit EDA, round 2 (CPU only)

**Where Muon's gain lives** ([muon_gain_decomposition.py](muon_gain_decomposition.py) →
[R1 json](muon_gain_decomposition_R1.json); within-slice pooled WP, Muon − AdamW):

| Slice | F0 | F2 |
|---|---|---|
| class 0 / 1 / 2 / 3 | +.0041 / +.0078 / +.0121 / +.0022 | +.0042 / +.0060 / +.0032 / +.0048 |
| run age 0 / 1–3 / 4–15 / 16–63 / 64+ rows | +.0080 / +.0074 / +.0043 / +.0026 / +.0005 | +.0062 / +.0063 / +.0032 / +.0037 / +.0041 |
| position 99–999 / 1k–5k / 5k–12k / 12k+ | +.0017 / +.0084 / +.0023 / +.0031 | +.0067 / +.0069 / +.0044 / +.0028 |
| top |y| quintile | +.0075 | +.0090 |

The gain is broad across classes and largest right after label changes (run age 0–3) and on
the largest-|y| rows (the lowest quintile is noise-dominated within-slice). Consistent with
the saturation mechanism: frozen/instant AdamW units adapt poorly when the label steps.

**When AdamW collapses** ([gate_saturation_epochs_R1.json](gate_saturation_epochs_R1.json), 6 F0
sequences): AdamW z-saturation 40.8% / 47.5% / 45.4% at epoch ends 0 / 1 / 2 (τ q90 723 →
3,526 → 3,161 rows); Muon 1.4% / 2.9% / 3.2%. The collapse is established during epoch 0 —
the high-LR phase — so the LR-fairness arm (AdamW 5e-4) and the from-step-10 spectral cap
are the right causal tests.
