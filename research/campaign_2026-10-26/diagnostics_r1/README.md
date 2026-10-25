# Diagnostics — 2026-10-26

## Production lineage already uses Muon in its parents (reframe)

[gate_sat_h390_vs_parent.json](gate_sat_h390_vs_parent.json): zero-fit gate replay
(`optimizer_panel_r1/diagnostics/gate_saturation.py`, layer 0, 6 F0 evaluation sequences, CPU).

| Checkpoint | Optimizer history | z saturated | τ q99 (rows) |
|---|---|---:|---:|
| `vt_all_fin_mu256p0e4_s3@ep3` (H390 parent) | Muon scope all, FP16, 4 ep | 10.4% | 741 |
| `vt_all_h390_ds_homogeneous_s3_r3@ep5` | parent + 6 ep AdamW distill | 10.7% | 597 |
| `c20261003_combo_s3_f0@ep3` (stage-6 parent) | Muon scope rnn, FP16, 4 ep | 10.5% | 432 |
| fast-recipe AdamW (Oct 25) | AdamW from scratch | 45–48% | ~5×10⁵ |
| fast-recipe Muon (Oct 25) | Muon from scratch | ~3% | ~170 |

Finding: both production parents were trained with (bf16-NS) Muon; the AdamW distillation stage
does not re-collapse the gates (+0.3 pp). The Oct-25 mechanism (AdamW gate collapse) therefore has
little left to fix in this lineage; the h390m/muon6 change is "Muon also in the distill stage +
strict FP32". Prior for the stage-6 transfer test: small effect. A null there does not contradict
the cold-recipe result. Larger headroom would need the parents themselves retrained under
strict-FP32 Muon (cold, full data) — candidate next experiment.
