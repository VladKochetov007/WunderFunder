# Red-team review: "strict-FP32 Muon beats AdamW by ~+0.004" (Opus 5.5, 2026-10-25, R1)

Read-only review; nothing was launched or modified apart from this file. Historical LR trio:
`campaign_2026-10-06/bandit_continuation/prediction_bank/GRADE_FOLD1_20261011T023029Z_8abce483b2227831.json`.

## Verdict

The two arms really do differ only in the optimizer. The measured effect is real at **AdamW's untuned lr=1e-3**.
Four of the root's supporting interpretations are wrong or untested:

1. **H-a (larger recurrent effective step) contradicts the root's own data.** Muon's per-entry step RMS is about equal to AdamW's, and AdamW moved W_hh about 2.4× *more* in Frobenius norm.
2. **H-b (clip decoupling) has a false premise.** Both optimizers are invariant to per-step gradient scale. Muon also feeds clipped gradients into its momentum buffer.
3. **The pre-registered tests skip the most important prior evidence.** The repo already has an LR trio for both optimizers. At each optimizer's best LR, that trio shrinks the gap from +0.0033 to +0.0004.
4. **"Strict FP32" is probably not true for the GRU kernels in either arm** (TF32 through cuDNN RNN; details below).

## A. Audit of the Muon claim

**Matching.** From PLANS argv the two arms differ only by `--optim adamw` vs `--optim muon --muon-scope rnn` plus the run name, in both cells. The control ran on frozen worktree `96d31cf4d` and the arms on `7161d0eef`. Their diff is limited to `optim_variants.py`, the CLI choices and the new branch in `build_optimizer` (training_dynamics.py:100-112). The AdamW path, model, data and order are unchanged. Non-recurrent parameters use `MuonAdamW`'s own Adam loop (optim.py:106-118), mathematically equal to `torch.optim.AdamW`. Muon covers 1.57M of 2.24M parameters.

**Newton–Schulz precision.** The NS iteration is FP32. `orthogonalize` casts to float32 (optim.py:27). Its matmuls go through `cuda.matmul`, whose `fp32_precision` is `none`, so legacy `allow_tf32=False` and full FP32 apply.

**TF32 in the GRU (new finding, both arms).** In this `.venv` (torch 2.14), `torch.backends.cudnn.rnn.fp32_precision` defaults to `"tf32"` and `cudnn.allow_tf32` defaults to True. The trainer disables TF32 only in special paths: setup_lineage.py:162-163 (student-frontier) and validation_inputs.py:42-43 (dynamics aux). Neither applies to these argv. The GPU is an RTX 3060, sm_86, which supports TF32. So the cuDNN GRU forward and backward very likely ran in TF32 in **both** arms. The comparison stays matched, but the "strict FP32" label is probably false, campaign-wide; provenance does not record the flags. Verify by printing the flag in a live fit; fix by setting `cudnn.allow_tf32=False` whenever `--precision fp32`.

**Effective-step quantification (H-a).** All shapes from the checkpoints:

- Every gate block is 512×512, because W_ih's input is the encoder width of 512.
- Training runs 4,620 steps with OneCycle (5% warmup, final div 2,500), so Σlr ≈ 2.31.
- Muon per block: update = 0.2·√512·lr·Q, with ‖Q‖_F ≈ 0.9·√512 for a noise-dominated, near-full-rank momentum. That gives **per-entry RMS ≈ 0.18·lr** and ‖ΔW‖₂ ≤ 0.2·√512·1.13·lr ≈ 5.1·lr.
- AdamW per entry: lr·|m̂|/√v̂. The measured end-state mean for gru.weight_hh is 0.134, which as a half-normal means σ ≈ 0.17, so **per-entry RMS ≈ 0.17·lr**. The mid-run noise floor at β1 = 0.85–0.9 is 0.23–0.28·lr.
- **Ratio Muon/AdamW ≈ 0.65–1.1, not ≫1.** This is exactly what the Liu et al. RMS matching is designed to produce.
- Measured displacement: the default init is U(±1/√512), RMS 0.0255. Final W_hh RMS is 0.055–0.058 for AdamW and 0.033 for Muon. That gives displacement RMS ≈ 0.050 vs ≈ 0.021 (assuming the change is near-orthogonal to init). The same holds for W_ih (0.047 vs 0.034).

So **AdamW took larger recurrent steps.** The difference is in where the steps go, not how big they are:

- For a coherent rank-1 gradient, Adam's sign-like update has spectral norm up to lr·√(mn) = 512·lr, about 100× Muon's bound.
- Muon's cumulative spectral growth is capped at ≈ 1.15 + 5.1·Σlr ≈ 13 per block at this LR. AdamW's z block reached 25–29, so **Muon cannot reach the AdamW solution in this budget**.
- The mechanism is therefore an implicit, LR-coupled spectral-norm constraint (H-c), not step size. A spectral cap is only one candidate mechanism, though.

**Fairness of the LR.** The claim should read "Muon beats AdamW at lr 1e-3". The historical fold-1 trio used h256, b32, chunk 1000, 4 epochs, seed 3 and bf16 NS:

| LR | AdamW | Muon |
|---|---:|---:|
| 5e-4 | **0.66515** | 0.66425 |
| 1e-3 | 0.66231 | **0.66559** |
| 2e-3 | 0.66132 | 0.66346 |

At matched lr 1e-3 Muon leads by +0.0033. Comparing the best LR for each, the lead is only +0.0004. AdamW's optimum was *below* 1e-3. This is one seed on a different recipe, but it points the opposite way from D2.

**Statistics.** The CIs resample evaluation components and condition on one training seed per arm. The 0.0007–0.0014 seed SD comes from full-data endpoints; 10%-data fits are probably noisier (unmeasured). The real support is a consistent sign across three matched-LR contexts (F0, F2, fold 1), not significance against seed variance.

**The pre-registered tests.**

- **D1** (`--clip-per-module head:1.0`)
  - The flag works. Families are top-level modules `inp`, `gru` and `head`, each clipped at 1.0 (training_dynamics.py:357-361). The gru module norm is ≈ 0.2–0.3 and inp ≈ 0.45, so in practice D1 means "clip only the head".
  - The premise fails. Adam's update is scale-invariant per coordinate up to eps: eps = 1e-8 against √v ≈ 1.5–3e-5 for gru coordinates. Global clipping therefore does not shrink AdamW's recurrent step; it only reweights steps over time inside m and v.
  - Muon shares that reweighting because `buf.lerp_(g)` accumulates c_t·g_t (optim.py:94).
  - D1 does not discriminate Muon's mechanism, and it largely duplicates the no-clip arm that is already running. Prediction: about 0.
- **D2** (`--lr 3e-3 --head-lr-mult 0.333`): flags work (training_dynamics.py:121-145, :192-193), but it also triples LR on `inp.*` and all GRU biases, makes AdamW's step about 3× Muon's, and history says higher AdamW LR loses. Expect a negative, low-information result.
- **D3** only makes sense if D1 is positive. Defer it.
- **D4** (`hidden`) moves `head.0.weight` [512,1024] onto Muon with scale 6.4. That is exploration, not a discriminator. The historically robust recipe was `all` scope (including `inp`), which the current tree lacks (optim.py:140). It exists in frozen v6.
- **Missing:** a Muon-groups-only LR ladder (`rms`, optim.py:55, has no CLI flag; `--lr` also moves Adam groups), an AdamW LR multiplier on gru matrices only, AdamW at 5e-4, a spectral-cap AdamW, and a seed replicate of the control.
- The working tree's new `--muon-update rms` ablation (uncommitted diff in optim.py and cli_optimizer.py) is the right discriminator. Its per-block step RMS equals that of a full-rank polar factor and it keeps the momentum's singular spread.

## B. Diagnostics and interpretations

**State→name mapping (confirmed bug).**

- `muon_param_groups` returns `[{adam params, named_parameters order minus gru.weight_*}, {gru.weight_ih_l0, gru.weight_hh_l0}]` (optim.py:166).
- The saved optimizer state indexes parameters by group order. state_spectra.py:23-24 zips those indices with model order instead.
- The correct mapping for the Muon runs is: 0 inp.weight, 1 inp.bias, 2 gru.bias_ih_l0, 3 gru.bias_hh_l0, 4 head.0.weight, 5 head.0.bias, 6 head.2.weight, 7 head.2.bias, **8 gru.weight_ih_l0 (momentum_buffer), 9 gru.weight_hh_l0 (momentum_buffer)**.
- The JSON confirms it: the "head.2.weight" row has only `m_rms` (a momentum buffer), and the "head.0.*" rows' √v (1.08e-2, 9.4e-2) match the control's head.2.* (1.24e-2, 1.03e-1).
- AdamW runs map correctly. Weights and spectra are read by name, so they are valid.
- Corrected reading: Muon's raw momentum RMS is 9.97e-5 (ih) and 5.50e-5 (hh), about 4× AdamW's m, consistent with the larger gradients.

**"Adam SNR below noise floor".**

- 0.23 is the noise *RMS* for β1 = 0.9; it is not a median.
- OneCycle runs β1 back up to **0.95** at the end (training_dynamics.py:200). Pure noise at β1 = 0.95 gives a half-normal with σ = 0.160: median 0.108, mean 0.128, q95 0.31.
- Measured AdamW gru.weight_hh on F0: median 0.106, mean 0.134, q95 0.353. That is **exactly pure noise**, not below it.
- At the end of an anneal to lr ≈ 4e-7, per-step SNR is near 0 by construction, so this says nothing about Adam "self-annealing". Measure at mid-epoch-1 instead.

**Spectra across optimizers.**

- Muon's top singular values (1.8–3.2) sit under its hard growth cap (≈ 13). "Muon prevents collapse" is partly a consequence of that cap, not evidence that collapse hurts.
- Muon's high stable rank (34–85) may be accumulated noise. Compare ΔW = W − W0 spectra (W0 reproducible from the seed) against a same-RMS random walk.

**"Global clipping is head-driven".**

- True of norms: the head carries 79–92% of the squared norm.
- The consequence is reweighting over time, not shrinking. The clip factor tracks head-gradient magnitude, so it tracks |y|-mass. That is an implicit down-weighting of high-|y|-mass batches, and it is **shared by both optimizers**.
- Brief corrections: the head input is 1024-wide (GRU output plus `inp` skip), and the input has 263 features, not 211.

**Gradients 3–4× larger under Muon, and biases 2× larger.** Neither is evidence of "more signal". The more likely explanation is that AdamW's sv-25 z block saturates the gates, so σ′ → 0 and recurrent gradients become small, while under Muon the biases carry the gate set-points. Check this with a zero-fit gate-statistics pass.

## C. Experiments that could be AdamW→Muon-sized

All run on the same fast recipe (about 12 min per fit), with the existing AdamW/Muon fits reused as controls.

**E0 — zero-fit diagnostics, run first (CPU-light).**

- Forward 20 validation sequences through both checkpoints. Record the fraction of z/r gates beyond 0.02 or 0.98, and the distribution of memory timescales 1/(1−z).
- Compute the ΔW spectra.
- Fix state_spectra.py.
- Falsifier for the saturation story: AdamW's saturated-gate fraction is no higher than Muon's.

**E1 — LR fairness (decides whether the claim stands; 3 fits).**

- AdamW `--lr 5e-4`, F0 and F2 (no code change).
- Muon with a new one-line `--muon-rms 0.1` flag (half the Muon-group LR), F0.
- Falsifier: AdamW at 5e-4 comes within 0.001 of Muon at 1e-3 in both cells. Then the "Muon gain" is mostly AdamW being over its LR optimum, and the cheaper win is a tuned LR.
- If Muon at rms 0.1 beats rms 0.2, Muon also had headroom.

**E2 — orthogonalization vs normalization (2 fits).**

- `--muon-update rms`, F0 and F2.
- Prediction if spectral equalization drives the gain: rms ≈ AdamW, and its top singular value grows toward AdamW's.
- Falsifier: rms comes within 0.001 of NS in both cells. Then the gain comes from per-block normalized momentum, and the polar factor is unnecessary.

**E3 — spectral-norm-capped AdamW (2 fits, about 25 lines).**

- After every 10th `step()`, run an SVD on each 512×512 gate block of gru.weight_ih/hh and clamp σ ≤ τ = 4 (Muon's observed max is 3.2). This adds about 1 min per fit.
- Proxy: wd 0.3 on gru matrices only (wd 0.01 currently shrinks by just exp(−0.023)).
- Falsifier: Δ ≤ 0.001 in both cells. Then norm regularization is not the mechanism, and per-step geometry is.

**E4 — capacity unlocked by Muon (2 fits).**

- Muon H512×2 (≈ 3.8M parameters, fits in 20 MB) vs the existing Muon H512×1, F0 and F2.
- Rationale: AdamW cold runs got worse beyond width 384, plausibly through unconstrained low-rank gate growth. Muon's bound may restore the benefit of scale (3 layers was already slightly better).
- Falsifier: ≤ 0 in both cells.

**E5 — Muon schedule (2 fits).** Muon with fixed momentum 0.95 (no OneCycle β cycling; needs a flag). Never tuned. Falsifier: |Δ| < 0.001 in both cells.

**Stop:**

- D2 as written.
- D1/D3 unless no-clip turns out positive.
- End-of-anneal SNR arguments.
- Comparisons at a single LR (AdEMAMix included).
- Calling the runs strict FP32 until the cuDNN RNN TF32 flag is fixed.

The warm stage-6 Muon run starts from AdamW weights (z-block σ ≈ 25); do not read it as a test of the cold mechanism.
