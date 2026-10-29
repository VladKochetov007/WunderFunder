# TabM-mini GRU ensemble and NorMuon — cold-recipe screen (R1)

**Question.** (a) Does a TabM-style parameter-efficient ensemble of GRUs (Gorishniy et al. 2024,
[arXiv 2410.24210](https://arxiv.org/abs/2410.24210)) beat a single Muon GRU at equal recipe, and
how close does it get to a deep ensemble? (b) Does NorMuon's neuron-wise normalization after
Newton–Schulz ([arXiv 2510.05491](https://arxiv.org/abs/2510.05491)) improve on Muon?

**Design (TabM-GRU, `--ensemble-k K`, [code](../../../src/wnn/tabm_gru.py)).** K members share the
input projection, GRU and hidden readout; member *i* scales normalized inputs by its own
`r_i` (TabM-mini adapter, random ±1 init) and has its own output layer. Training repeats each
sequence K times in the batch (row j → member j mod K), so the unchanged row-weighted losses
optimize the **mean of member losses** (TabM's prescription, not the loss of the mean); eval
returns the member mean. Each member is exactly a plain `RNNModel` (`r_i` folds into
`inp.weight`), so the existing native ZIP engine can serve members as a fixed blend — verified
by an exact materialization test with carried state ([tests](../../../tests/test_tabm_gru.py)).
**NorMuon** (`--muon-update normuon`): per gate block, divide each output row of the polar
factor by its running RMS (β2 0.95), then restore the block's Frobenius norm so the Muon step
convention is unchanged.

**Ceiling first (zero-fit, [F0](ceiling_f0.out.json) / [F2](ceiling_f2.out.json)).** Existing
Muon predictions on the same panel: mean of two seeds +0.0011 over the mean single; four Muon
siblings (two seeds, rms 0.3, H512×2) **+0.0015 / +0.0013** over the best single (intervals > 0).
So an implicit ensemble of this recipe is bounded near **+0.002** unless joint training beats a
deep ensemble. At the production stage the ceiling shrinks further (student slot ≈ +0.0006
public; warm optimizer transfer ×0.07). This is bounded exploration, not the main line.

**Blend-value law (zero-fit EDA, [notebook](ensemble_eda.py), [plot](plots/pair_gain_vs_corr.png)).**
Across all 28 pairs of eight cold-panel models per fold (Muon seeds/variants, AdamW, Cautious,
AdEMAMix), equal-weight pair WP minus mean single WP ≈ **0.168 × (1 − ρ)**, where ρ is the
|y|-weighted prediction correlation; the line passes through the origin and fits both folds and
both same-optimizer and mixed pairs tightly. Muon siblings sit at ρ 0.995–0.998 (gain
+0.0004–0.0008); AdamW/AdEMAMix partners reach ρ ≈ 0.98 (+0.0034 over the mean, but negative
over the better member). Rule: a new member helps the better one iff 0.168·(1−ρ) > ½·score gap.
For TabM this means member ρ ≲ 0.988 at unchanged member quality is needed to reach +0.002.

**NorMuon pre-check.** For square 512×512 gate blocks the polar factor of a *random* matrix has
equal row norms (cv 0.011) → near no-op; for low-rank gradients with hot rows the 5-step NS
factor leaves rows uneven (cv 1.6), and normalization boosts updates of quiet neurons — a
plausible link to the gate-saturation mechanism. Kept as an arm.

**Fits (6 new; controls reused).**

| Lane | Cell | Arm | Control |
|---|---|---|---|
| Kaggle T4 ([specs](specs/), [builder](build_specs.py)) | F0/s2001, F2/s2002 | TabM k=8 (Muon) | `krepro_*_muon_r3` (R3, same seeds/hardware) |
| Kaggle T4 | F0/s2001, F2/s2002 | NorMuon | same |
| Local RTX 3060 ([runner](run_local_k4.sh)) | F0/s1001 | TabM k=4 vs strict Muon k=1 | matched local pair |

Profile ([k=4](smoke_k4.log), [k=1](smoke_k1.log)): k=4 costs 1.8× per step, 3.5 GB peak; k=8
OOMs at the local 4.6 GB cap ([log](smoke_k8.log)) and runs on Kaggle. Kaggle dataset
`vladkochetov07/wnn-tabm-gru-20261029-r1` (READY, 741,898 B), kernel `…-r1-runner` v1 pushed
00:05 UTC, RUNNING. Code commit `2a412f1ec`.

**Decision rule (declared before results).** TabM is interesting if its ensemble beats the single
Muon control on both cells and reaches ≥ the four-sibling deep-ensemble gain; then the next step
is a member-materialized package check and a teacher-side use (cheap ensemble teacher). NorMuon
continues only if positive on both cells.

## Results

### Local F0/s1001, TabM k=4 (signs) vs matched strict Muon ([grade](grade_local_f0.json))

| | WP F0 |
|---|---:|
| Matched single Muon (`tabm29_f0_s1001_muon_local_r1`) | 0.663340 |
| TabM k=4 ensemble | 0.662129 — **−0.00121 [−0.00276, +0.00033]** |
| Its four members, materialized and exported as plain GRUs ([splitter](split_members.py)) | 0.66077 / 0.65950 / 0.65993 / 0.65994 (mean 0.660035) |
| Mean member-pair ρ | 0.9914 (seed pairs: 0.995–0.998) |

Ensemble − mean member = +0.0021, matching the blend law (0.168·(1−ρ) scaled to k=4: +0.0022).
Members are **more diverse than independent seeds but each is −0.0033 weaker** than the single
model; weakness outweighs diversity. TabM's "weak but diverse" regime, but with a net loss here.
Hypothesis: random ±1 input signs force one shared H512 GRU to serve k different input
encodings. **Prediction before the Kaggle k=8 results:** weaker members, lower ρ, net negative.
Mechanism test running: k=4 with `--ensemble-init ones` (members differ only by output layer
and learned scale drift), control reused.
