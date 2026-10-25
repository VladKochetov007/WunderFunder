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

## Revision R2 before launch (after Codex Astra, Opus red-team)

The [Opus red-team](../optimizer_panel_r1/external/opus_redteam_R1.md) and
[Codex Astra xhigh](../optimizer_panel_r1/external/codex_astra_xhigh_R2.md) reviews changed the
panel before any arm ran:

- Root hypothesis H-a (Muon = larger recurrent step) is contradicted: per-entry steps are
  ≈equal (Muon ≈0.18·lr, AdamW ≈0.17·lr) and AdamW displaced W_hh ≈2.4× more; Muon instead caps
  cumulative spectral growth per block (≈13 at this LR) — an implicit, LR-coupled spectral
  constraint.
- H-b (clip decoupling) has a false premise: both optimizers are invariant to a per-step
  gradient scale; clipping only reweights history, for both. M3 dropped.
- "Adam SNR below noise floor" was a miscalculation (OneCycle ends at β1 0.95: pure-noise
  median 0.108 ≈ measured 0.106).
- LR fairness: historical fold-1 trio had AdamW best at 5e-4; best-vs-best gap +0.0004.
- **TF32 finding, verified:** torch 2.14 defaults `cudnn.allow_tf32=True`,
  `cudnn.rnn.fp32_precision='tf32'`; the main trainer never disabled them, so local/B300
  "strict FP32" GRU fits ran cuDNN recurrences in TF32 (Kaggle T4 runs were true FP32). Paired
  comparisons remain matched. New flag `--strict-fp32-kernels` (commit `7829859e4`).

Final 12 fits (F0, F2; frozen source `7829859e4`): M1 Muon `--muon-update rms`; E1 AdamW
`--lr 5e-4`; E1b Muon `--muon-rms 0.1`; M2 AdamW `--beta2 0.95`; M4 AdamW `--rnn-init chrono`;
TF AdamW `--strict-fp32-kernels`. Key comparisons: M1 vs Muon-NS; Muon-NS vs AdamW-5e-4
(best-vs-tuned); TF vs AdamW (TF32 effect).
