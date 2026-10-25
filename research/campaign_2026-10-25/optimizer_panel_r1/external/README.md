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
- Codex Astra xhigh: R1 killed by the 12:52 reset ([partial stderr](codex_astra_xhigh_R1_killed_by_reset.stderr)); R2 running.
- Opus red-team (Claude subagent): first run died with the session; rerun in progress →
  `opus_redteam_R1.md`.
