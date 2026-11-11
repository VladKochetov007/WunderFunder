# External optimizer reviews (2026-10-25)

Shared [brief](BRIEF_OPT_REVOLUTION_R1.md) with measured spectra, gradient norms, Adam state
and optimizer-panel results.

- Gemini-4-Argon via OpenCode: **not run** — `Upstream request failed: Insufficient account
  funds` ([stderr](argon_R1.stderr)); replaced by Grok.
- [Grok 4.7](grok_4.7_R1.md): rejects the effective-LR explanation (Muon step RMS ≈2× Adam's
  median, not 10×); proposes a gate-saturation "collapse loop" (saturating update gate +
  slow β2 suppresses corrective steps) with Muon as an update-side spectral equalizer.
  Discriminators: RMS-matched momentum without Newton–Schulz (B2), update-gate spectral cap
  (B3), β2 .95 on GRU (B4); next steps: update-gate bias init, per-family clipping with Muon
  blocks unclipped, longer TBPTT; diagnostics: gate saturation, stable rank, per-run offset
  fraction φ, epoch-1 (not end-of-anneal) SNR.
- [Codex Astra xhigh, R2](codex_astra_xhigh_R2.md) (R1 killed by the 12:52 reset): geometry-vs-size 2×2 (Adam/Muon direction × Adam/Muon norm via shadow states); per-block recurrent clipping; norm-matched Muon on input/head; periodic spectral projection of W_hz; norm-matched Shampoo; gradient accumulation with matched EMA time scales. Corrects the SNR median benchmark (≈0.155 at β1 .9).
- [Opus red-team](opus_redteam_R1.md) (Claude subagent, rerun after the first died with the session): arms matched; NS is FP32; **cuDNN GRU ran in TF32** (verified); H-a contradicted (equal per-entry steps, AdamW moved W_hh 2.4× more; Muon caps spectral growth); D1 premise false; SNR claim a miscalculation (β1 ends at .95); prior LR trio shrinks best-vs-best gap to +0.0004; state→name mapping bug confirmed with the correct indices; proposes LR fairness, RMS ablation, spectral-capped AdamW, Muon H512×2, fixed-momentum Muon, zero-fit gate saturation and ΔW spectra.
