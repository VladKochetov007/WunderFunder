# [Results] Strict-FP32 Muon for GRU recurrent matrices: evidence, mechanism and transfer limits

1. **Status, lineage and owner.** Completed evidence, published for the team. No new fits in this
   PR. Owner: Vlad / Claude Code. H-OPT25-MUON-FP32-R1, H-OPT25-MUON-MECHANISM-R1;
   E-OPT25-FAST-PANEL-R1 plus the mechanism, saturation, LR-ladder and Kaggle R3 panels. Source
   lives on `experiment/next-cycle-recovery-20261021` (commit `0ea63b982`): `src/wnn/optim.py`
   (`Muon`, `muon_param_groups(scope="rnn")`), `src/wnn/optim_variants.py`,
   `src/wnn/training/cli_optimizer.py` and `--strict-fp32-kernels`. The code is not copied here:
   research-team's `optim.py` diverged, and a port needs its own reviewed change.
2. **Why it works.** Under AdamW the GRU update gates collapse during epoch 0: 45–48% of
   update-gate activations saturate, and frozen units get memory timescales around 5×10⁵ rows.
   Muon orthogonalizes each gate block's update with Newton–Schulz and scales it to
   `0.2·sqrt(max(rows, cols))·lr`. Gates stay soft (3% saturated, q99 timescale ≈ 170 rows).
   - Discriminating ablation: the RMS-matched update without orthogonalization falls back to
     AdamW level (−0.0040 vs Muon in both cells). The orthogonalization itself is the active
     ingredient.
   - Spectral capping of AdamW's gate blocks recovers only about 40% of the gap.
3. **Compute advantage.** Same FLOPs as AdamW; Newton–Schulz is negligible next to the
   recurrence. The gain is in **cold** training only. On an already-warm distilled student the
   gain shrinks to +0.00026 [−0.00041, +0.00101] (×0.07), and production parents were already
   Muon. The useful place is any cold GRU stage: teachers, parents, and labels-only screens such
   as the PR79/PR80 L4 control.
4. **Frozen comparison.** Cold fast recipe: H512×1 GRU, 10% frozen memberships, 3 epochs, fixed
   CP2, folds F0/s1001 and F2/s1002. Arms: AdamW 1e-3 / 5e-4 / 2.5e-4, Muon (rnn scope, AdamW
   elsewhere), Cautious AdamW, AdEMAMix. Strict FP32 throughout.
5. **Fits used.** Optimizer panel 6, mechanism panel 12, saturation panel 6, LR ladder 2, Kaggle
   R3 4 (fresh seeds 2001/2002, T4, true FP32), stage-6 transfer 1.
6. **Results.** All on exposed folds, one seed per cell; paired component-bootstrap intervals are
   conditional on the fitted models.

   | Comparison | F0 | F2 |
   |---|---|---|
   | Muon − AdamW 1e-3 | +0.0038 | +0.0043 |
   | Muon − AdamW 5e-4 (AdamW optimum) | +0.0026 | +0.0035 |
   | Kaggle R3 Muon − AdamW (true FP32) | +0.0036 | +0.0033 |
   | Cautious AdamW − AdamW | −0.0014 | −0.0011 |
   | AdEMAMix − AdamW | −0.0045 | −0.0038 |
   | Strict cuDNN FP32 − TF32 recurrences | −0.0011 | −0.0001 |
   | Stage-6 warm student Muon − AdamW | +0.00026 | — |

   Muon students correlate 0.9995 with their AdamW siblings (seed pairs: 0.994).
7. **Leakage and coverage.** Fold-clean memberships, causal inputs, fixed endpoints. Folds are
   exposed development data. The cold-recipe gain is replicated; transfer to warm stages is
   measured and small.
8. **Decision.** Use strict-FP32 Muon (rnn scope) for every **cold** GRU stage. Do not expect a
   gain from switching an already-warm student. For runtimes whose sequence Muon is
   Transformer-only (PR79/PR80), the port is the first step; see PR #82 card 01.
9. **Fidelity finding.** Torch 2.14 cuDNN GRU used TF32 recurrences in recipes labelled
   strict FP32 until the RNN precision was set explicitly. Every GRU runtime should log
   `torch.backends.cudnn.rnn.fp32_precision`.
10. **Evidence.** [Oct 25 synthesis](../../campaign_2026-10-25/SESSION_SYNTHESIS_R1.md) ·
    [optimizer panel](../../campaign_2026-10-25/optimizer_panel_r1/README.md) ·
    [mechanism panel](../../campaign_2026-10-25/mechanism_panel_r1/README.md) ·
    [saturation panel](../../campaign_2026-10-25/saturation_panel_r1/README.md) ·
    [LR ladder](../../campaign_2026-10-25/lr_ladder_r1/README.md) ·
    [Kaggle R3](../../campaign_2026-10-25/kaggle_muon_repro_r1/README.md) ·
    [Oct 26 transfer, gate saturation and Muon students](../../campaign_2026-10-26/README.md).
    The same campaigns also hold the loss-family, clipping, class-3 and i1-lead negatives
    (synthesis §1.3–1.4).
11. **Trace.** Copied unchanged from `experiment/next-cycle-recovery-20261021` at `0ea63b982`.
    No independent review of this PR; the Opus/Grok reviews linked in the synthesis cover the
    original analysis.
