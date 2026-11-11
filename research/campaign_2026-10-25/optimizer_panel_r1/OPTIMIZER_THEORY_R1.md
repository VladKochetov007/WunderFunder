---
kind: idea
status: active
created_utc: 2026-10-25T08:20:00Z
hypothesis_ids: [H-OPT25-MUON-FP32-R1, H-OPT25-CAUTIOUS-R1, H-OPT25-ADEMAMIX-R1, H-OPT25-CLIP-REGIME-R1]
experiment_ids: []
evidence_paths:
  - research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/OPTIMIZERS_SCHEDULES_STABILITY_R1.md
  - research/experiment_drafts/own-optimizer-muon-20261024/README.md
---

# Optimizer theory for this task: what regime we are in, what can still move

User request (2026-10-25): "think a lot about theory and optimization methods; check literature
and intuition behind optimizers; which ideas can we improve?". This note fixes the regime from
our own measurements, maps optimizer families onto it, and states the arms now running.

## 1. The regime, measured

| Fact | Evidence | Consequence |
|---|---|---|
| Very low signal-to-noise labels: best weighted R² ≈ 0.46 (WP 0.68), class 3 ≈ 0.17 | weighted target variance 1.80 (Oct-25 receipt) vs model WMSE ≈ 0.97 | Stochastic gradients are dominated by label noise, not curvature |
| Small model, large data: ~1 M params (H256×2) vs 10.6 k sequences × 20 k rows | trainer logs | Not interpolating; capacity/optimization still matter, importance weighting keeps its effect (Byrd & Lipton 1812.03372) |
| Early peak, then a noisy plateau: long runs peak at epoch 1–3, oscillate 0.658–0.665; a 3-epoch annealed run is best | Oct-10 long ladder (AdamW and Muon-all) | The endpoint is set by how the final iterate averages noise (annealing), not by reaching a sharper minimum |
| Schedules tie at full scale (one-cycle ≈ WSD ≈ trapezoid ≈ SGDR within noise) | Oct-9 synthesis | Schedule shape is not a lever; the averaging it implies is already near-optimal |
| Almost-always-clipped: 94% of steps clip at norm 1.0, mean factor 0.56 (fast cold recipe); 7% in warm students; 100% under NLL | Oct-25 `GRAD_STATS`, Oct-20 NLL receipt | Clipping is an active, untuned part of the update rule |
| Batch weight mass swings 0.1×–9.7× | Oct-24 batch-mass audit | Per-step gradient scale is heavy-tailed |
| Muon on recurrent matrices beat AdamW by ~0.004 in one full-scale teacher pair; Muon-all H384 robust lead (3–5 seeds) — all with the bf16 Newton–Schulz cast | Oct-9/10 logs | The best optimizer evidence we have is policy-violating and must be re-measured in strict FP32 |
| Synthetic/small screens disagree with full scale (WSD +0.013 small, 0 at scale; Muon −0.005 synthetic) | Oct-9, Oct-24 | Optimizer conclusions need real-data, real-scale arms |

## 2. Intuition per family, mapped onto this regime

- **Adam / AdamW** = steepest descent under an ℓ∞-type geometry with a diagonal running preconditioner
  (Bernstein & Newhouse, "Old Optimizer, New Norm", 2409.20325). Invariant to a constant gradient scale.
  In a noise-dominated problem its per-coordinate normalization mostly equalizes noisy coordinates.
- **Muon** = steepest descent under the spectral norm for matrix parameters: the update is the
  orthogonalized momentum UVᵀ, so every singular direction moves equally (2409.20325; Jordan 2024;
  RMS-matched scaling, Liu et al. 2502.16982 — already in our `MuonAdamW`). For a GRU this matters
  because W_hh's spectrum governs memory: equalizing directions trains rarely used slow modes. We
  measured exactly that (stable rank 0.07→0.41, Oct-24 C5) but no small-scale quality gain. The
  question is whether the full-scale gain survives strict FP32 — the bf16 cast perturbs the
  orthogonalization most for small singular values, which is where Muon's benefit lives.
- **Shampoo / SOAP** (2409.11321) = Kronecker-factored full-matrix preconditioning; Muon is
  Shampoo without accumulation. Our matrices are tiny (256×768), so SOAP is cheap here, but it
  needs QR/eigendecompositions every k steps and more hyperparameters. Candidate only if Muon
  survives in FP32 (it would indicate geometry, not noise, is limiting).
- **Momentum variants for noise** — AdEMAMix (2409.03137) mixes a fast EMA with a very slow one;
  Schedule-Free and AdEMAMix are shown to be accelerated SGD in the *noise-dominated* regime
  (2502.02431). Our regime is noise-dominated, so a slow gradient average that keeps old
  information is the most theory-aligned new update rule. Risk: our runs are short (4.6 k–45 k
  updates), so β3 must be modest (0.999, half-life ≈ 700 steps) and warmed up.
- **Cautious masking** (2411.16085): zero update coordinates whose sign disagrees with the current
  gradient. Interpretation: with heavy noise, momentum often points against the fresh gradient;
  masking reduces overshoot without changing the fixed points. One line, no new state.
- **Variance reduction** — MARS (2411.10438) and STORM-style corrections need a second gradient at
  the previous parameters on the same batch. With TBPTT and recurrent state this doubles cost and
  complicates state handling; deprioritized.
- **Gradient clipping** — theory: clipping accelerates under relaxed smoothness (Zhang et al. 2019)
  and is a robust estimator under heavy-tailed noise, but introduces bias. With Adam, a per-step
  clip factor c_t rescales that step's contribution inside m and v, i.e. it down-weights steps
  with large gradient norm. At 94% clipping this is an implicit batch reweighting toward
  low-mass batches — directly interacting with the measured 0.1–9.7× batch-mass swings. Never
  varied in this repo.
- **Weight averaging (EMA/SWA/LAWA)** is Polyak averaging, the rate-optimal response to noise —
  but our one-cycle anneal already averages, and SWA25 and EMA 0.999 at 4 epochs were null
  (Oct-10). Not re-run.
- **Implicit regularization of label-noise SGD** (2106.06530): noise drives SGD toward flat
  minima; very aggressive noise reduction can lose that. Predicts that variance-reduction-style
  changes may help optimization speed but not generalization — another reason to measure WP,
  not train loss.

## 3. Arms now running (fast cold recipe, reused exact AdamW controls)

Recipe: cold H512×1 GRU, 10% frozen memberships, 3 epochs, fixed CP2, F0/s1001 and F2/s1002,
deterministic seeded order; controls are the i1 lead-block screen's zero-block fits, so every
arm differs by exactly one optimizer flag. Frozen source `7161d0eef`.

| Arm | Flag | Prediction | Falsifier |
|---|---|---|---|
| Muon (rnn), strict FP32 | `--optim muon --muon-scope rnn` | if the Oct-9 gain was real, ≥ +0.001 in both cells | ≤ 0 in both cells |
| Cautious AdamW | `--optim cautious_adamw` | small positive (overshoot reduction) | ≤ 0 in both cells |
| AdEMAMix | `--optim ademamix --ademamix-alpha 4 --ademamix-beta3 0.999 --ademamix-warmup T` | positive if old-gradient averaging helps under label noise | ≤ 0 in both cells |
| No clip | `--no-gradient-clip` | sign tells whether clipping's implicit reweighting helps | — (direction finding) |
| Clip 0.3 | `--clip 0.3` | more normalized updates; if robust reweighting helps, ≥ clip 1 | — |

Running loss normalization (C4) was planned and dropped: it was already tested on Oct 10
(0.66604 vs 0.6660, null) — a dedup catch. Two cells per arm on exposed folds give direction
evidence only; a positive arm earns a full-scale, multi-seed follow-up, not a submission claim.

## 4. Ideas held for after these results

1. If Muon survives FP32: SOAP on recurrent matrices; Muon on the input projection (`all` scope
   from the historical trainer); per-gate NS vs whole-matrix.
2. If clipping direction matters: per-sequence (per-sample) clipping as a robust estimator
   instead of global-norm clipping — bounds the influence of extreme-|y| sequences without
   changing the target.
3. If AdEMAMix helps: β3/α ladder; combine with Muon (slow momentum fed to orthogonalization).
4. Second-order view of the head: the readout is a linear least-squares problem given hidden
   states — closed-form/ridge refit of the head at the end of training (zero-fit on saved
   checkpoints) removes head optimization noise entirely. Cheap zero-fit test.
