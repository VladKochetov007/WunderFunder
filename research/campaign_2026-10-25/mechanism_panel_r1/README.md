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

## Interim: M1 (RMS-momentum, no orthogonalization), F0 ([grade](grade_f0_m1_only_R1.json))

AdamW 0.659530, Muon-NS 0.663349, **M1 0.659367**: M1 − Muon **−0.003982 [−0.005734, −0.001994]**,
M1 − AdamW −0.000163 [−0.002746, +0.003316]. Same step RMS, no second moment, singular spread
kept → AdamW-level quality: on this cell the entire Muon gain needs the orthogonalization.
M1 gate saturation ([json](../optimizer_panel_r1/diagnostics/gate_saturation_m1_R1.json)): z 13.9% /
r 14.1% (AdamW 45/23, Muon 3/1), τ q50/q90/q99 = 2.9 / 70 / 228,643 rows — frozen units
persist as in AdamW, absent under Muon (q99 ≈170). Quality tracks the frozen-unit tail, not the
mean saturation fraction. F2 cell pending.

## Interim: E1 AdamW lr 5e-4, F0 ([grade](grade_f0_e1_only_R1.json))

AdamW 1e-3 0.659530 → **AdamW 5e-4 0.660790** (+0.001259 [−0.000990, +0.003972]); **Muon 1e-3
− AdamW 5e-4 = +0.002559 [+0.000271, +0.004814]**. Part of the original gap was AdamW LR; the
Muon advantage survives against the better-tuned AdamW on this cell. Saturation of AdamW 5e-4
([json](../optimizer_panel_r1/diagnostics/gate_saturation_e1_R1.json)): z 25.2%, τ q99 32,435.

Dose-response across F0 models (z-saturation / τ q99 / WP): AdamW 1e-3 45% / 546k / .6595;
M1 14% / 229k / .6594; AdamW 5e-4 25% / 32k / .6608; Muon 3% / 166 / .6633 — WP tracks the
frozen-unit tail better than mean saturation. Follow-up: [AdamW lr 2.5e-4 ladder](../lr_ladder_r1/).

## Interim: batch 2, F0 ([grade](grade_f0_batch2_R1.json))

| Arm | WP | Paired Δ [95% CI] |
|---|---:|---|
| E1b Muon `--muon-rms 0.1` (half step) | 0.660990 | −0.002359 vs Muon [−0.004337, −0.000506] |
| M2 AdamW β2 .95 | 0.660539 | +0.001009 vs AdamW [−0.002029, +0.004077] |
| M4 AdamW chrono init | 0.653809 | **−0.005721 vs AdamW [−0.010013, −0.000942]** |

Muon is not over-stepping at rms 0.2 (half step loses). β2 .95 at most a small effect. Long
initial gate timescales (chrono) hurt — consistent with the frozen-unit finding that long,
never-updating memory is harmful here. Consequence: the saturation panel's Muon+chrono arm
was replaced before launch by Muon `--muon-rms 0.3` (larger step).

## Interim: M1 F2 replicates; TF32 F0 ([M1 F2](grade_f2_m1_only_R1.json), [TF F0](grade_f0_tf_only_R1.json))

- **M1 F2:** AdamW 0.683091, Muon 0.687427, M1 0.683388 → M1 − Muon **−0.004039 [−0.005899, −0.002123]**,
  M1 − AdamW +0.000297 [−0.001387, +0.002293]. **Two-cell supported:** removing only the
  orthogonalization returns Muon to AdamW level; the gain needs spectrally equalized recurrent updates.
- **TF arm F0:** strict FP32 kernels 0.658476 vs TF32 AdamW 0.659530 → −0.001054
  [−0.002546, +0.000285]. No quality benefit from true FP32 on this cell; F2 pending before
  deciding whether `--strict-fp32-kernels` becomes the default (policy compliance argues yes).

## Final panel ([F0](grade_f0_R1.json), [F2](grade_f2_R1.json)) — all effects replicate across both cells

| Comparison | F0 Δ [95% CI] | F2 Δ [95% CI] |
|---|---|---|
| **Muon − AdamW 5e-4 (tuned)** | **+0.00256 [+0.00027, +0.00481]** | **+0.00346 [+0.00171, +0.00530]** |
| **M1 (no orthogonalization) − Muon** | **−0.00398 [−0.00573, −0.00199]** | **−0.00404 [−0.00590, −0.00212]** |
| M1 − AdamW | −0.00016 [−0.00275, +0.00332] | +0.00030 [−0.00139, +0.00229] |
| AdamW 5e-4 − AdamW 1e-3 | +0.00126 [−0.00099, +0.00397] | +0.00088 [−0.00080, +0.00300] |
| Muon half step − Muon | −0.00236 [−0.00434, −0.00051] | −0.00211 [−0.00373, −0.00025] |
| AdamW β2 .95 − AdamW | +0.00101 [−0.00203, +0.00408] | +0.00047 [−0.00076, +0.00180] |
| AdamW chrono − AdamW | −0.00572 [−0.01001, −0.00094] | −0.00561 [−0.00808, −0.00321] |
| strict FP32 kernels − TF32 | −0.00105 [−0.00255, +0.00028] | −0.00011 [−0.00066, +0.00053] |

**Conclusions.** (1) Muon beats an LR-tuned AdamW (mean +0.0030) — the claim survives the LR-
fairness critique. (2) The gain requires orthogonalized (spectrally equalized) recurrent
updates; fixed-step normalized momentum without Newton–Schulz equals AdamW. (3) Muon's step is
not too large (half step loses); a larger step is tested in the saturation panel. (4) Long
initial gate timescales hurt (chrono −0.0057), consistent with harmful frozen units. (5) True
FP32 kernels cost no measurable quality — new recipes should use `--strict-fp32-kernels` for
policy compliance. Limits: 10% data, cold H512×1, exposed folds, one seed per cell; the
independent Kaggle reproduction (fresh seeds, T4, true FP32) and the full-scale production
test (Muon stage-6) are pending.
