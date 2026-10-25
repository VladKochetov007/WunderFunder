# Mechanism panel: why does Muon win? — 2026-10-25 R1

Pre-registered after the [optimizer panel](../optimizer_panel_r1/README.md) (Muon +0.0038/+0.0043)
and the [diagnostics](../optimizer_panel_r1/diagnostics/README.md) (AdamW W_hh update-gate block
collapses to stable rank ≈2 with top sv 25–29; head-driven global clipping on 80–94% of steps;
Adam |m|/√v at the noise floor), and Grok's [review](../optimizer_panel_r1/external/grok_4.7_R1.md)
(gate-saturation collapse loop). Same fast recipe, F0/s1001 + F2/s1002, reused controls
(AdamW = i1-lead zero-block fits; Muon-NS = optimizer-panel fits). Frozen source `26218559d`.

| Arm | Change | If it reproduces most of Muon's gain → |
|---|---|---|
| M1 `muon --muon-update rms` | Muon with the polar factor replaced by RMS-normalized Nesterov momentum (step RMS 0.2, singular spread kept) | fixed step / no second moment is enough; orthogonalization is not needed |
| M2 AdamW `--beta2 0.95` | faster second moment | stale-v collapse loop |
| M3 AdamW `--clip-per-module head:1.0` | each module clipped separately (recurrent no longer shrunk by head norm) | head-driven clipping |
| M4 AdamW `--rnn-init chrono` | chrono gate biases (b_z = log(T−1), T~U[2,2000]) | gate saturation / timescale initialization |

Comparisons: M1 vs Muon-NS and vs AdamW; M2–M4 vs AdamW. Effect of interest ≥ +0.002 in both
cells. Queue: after the clip panel, before the Muon stage-6 production-lineage fit. Units
`wnn-mechpanel`, `wnn-mechpanel-thermal` (freeze ≥92 °C).
