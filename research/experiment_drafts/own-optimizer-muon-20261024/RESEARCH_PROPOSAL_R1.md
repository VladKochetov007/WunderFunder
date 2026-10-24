---
kind: proposal
status: draft-proposal (scout, one-shot; no ledger rows appended yet)
created_utc: 2026-10-24
owner: independent ML-research scout (Luna line, one-shot)
proposed_hypothesis_family: H-OPT-OWN-MUON-20261024
competition_deadline: 2026-11-15
---

# R1 proposal: an "own optimizer" for the GRU / weighted-clipped-metric setting, Muon-inspired

Read-only scout proposal: no fit, ledger append or branch created. In-repo claims carry file
paths; literature claims are abstract-level and labelled. Every number is copied from a cited
path or the assigned task brief; none is invented.

## 1. Main judgment

**Promising only in a sharpened form.** The naive version ("swap AdamW for Muon") is refuted
in-repo at small scale: matched 350-sequence D02 quartet, Muon-WMSE 0.600653672 vs AdamW-WMSE
0.606001923 (Muon ≈ −0.0054, one seed, 220 updates;
`research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/grading_r2/D02_QUARTET_FINDINGS_R2.md`).
The complacent version ("Muon is already in all parents, nothing to do") is also wrong: scope,
scaling, precision, normalization interaction and schedule were never systematically decomposed.
Sharpest version of the question:

1. **Geometry.** Does per-gate orthogonalization change the effective rank / feature quality of
   GRU gate and hidden matrices here (Parallax/effective-rank claim), and does that pass through
   the weighted-clipped metric?
2. **Mass (ours alone).** Per-batch loss normalization (`src/wnn/training/losses.py`,
   `(w*err²).sum()/w.sum()`) gives batches an effective reweighting of 0.109–9.743× vs pooled
   rows (Oct24 audit, `research/campaign_2026-10-24/resume4h_r1/CAMPAIGN_SYNTHESIS_R1.md`);
   momentum/second moments mix these differently-scaled gradients. "Own optimizer" for THIS
   competition arguably means fixing the optimizer-measure interaction first.
3. **Fidelity.** Our CUDA Newton–Schulz casts to bfloat16 (`src/wnn/optim.py` line 27), as does
   upstream PyTorch Muon — already flagged as violating the strict-FP32 contract
   (`research/unified/sources/TRANSFORMER_B300_SCALE_SOURCES_20261019_R2.md`). A strict-FP32 NS
   variant has never been fitted.

Honest prior: headline Muon gains are LLM/transformer results; our one matched screen was
negative; positive parent evidence is confounded (width/propensity/scope/epochs co-vary — see
`specs/current-state-history-20261017.md` line 457 on the H382 comparison). Expected endpoint
value is small and uncertain, ±0.001–0.005 WP class.

## 2. What exists in-repo

### 2.1 Implementation (ours, not teammate code)

`src/wnn/optim.py` — self-maintained `MuonAdamW` (called "independently maintained project
MuonAdamW" in `scripts/build_gru_capacity_v2_packet.py` line 345):

- NS: quintic coefficients `(3.4445, -4.7750, 2.0315)`, default 5 steps, Frobenius
  pre-normalization, SVs pushed toward ~[0.7, 1.2] (lines 18–37).
- **Precision: bfloat16 on CUDA, FP32 on CPU** (line 27) — every historical GPU Muon fit ran
  bf16 NS regardless of `--precision fp32`; CPU branch already strict-FP32.
- Momentum: buffer `buf.lerp_(g, 1-beta1)`, Nesterov-style
  `orthogonalize(g.lerp(buf, beta1))` (lines 75–79); `betas[0]` shared with Adam beta1 and cycled
  by OneCycleLR (docstring). Champion parent used momentum 0.95 (see 2.2).
- Update scaling: `-lr * rms * sqrt(max(rows, cols))`, `rms=0.2` (lines 48, 77–81). **The
  Kimi/Moonlight RMS-matching scaling is ALREADY present** (0.2·sqrt(max(m,n)), citing Liu et
  al. 2025, arXiv 2502.16982 in the module docstring), applied per gate block after splitting
  stacked 3H×H (GRU) / 4H×H (LSTM) matrices (`_gates`, `muon_param_groups`).
- Weight decay: decoupled `p.mul_(1 - lr*wd)` on both branches (lines 80, 89) — Kimi fix #1 also
  present.
- Groups (`muon_param_groups`, lines 107–146): scope `rnn` = `gru.weight_ih_l*`/`weight_hh_l*`
  (+`experts.*`) on Muon, `inp`/heads/biases on AdamW; scope `hidden` adds `head.0.weight` /
  `head.extra.weight` on Muon (`blocks=1`). Hand-rolled AdamW branch with bias correction
  (lines 83–95).
- `src/wnn/training/cli_optimizer.py`: `--optim {adamw,muon,sgd}` (default adamw),
  `--muon-scope {rnn,hidden}` (default rnn), `--sched {onecycle,wsd,sgdr}`, `--clip 1.0`,
  `--grad-stats`. **`ns_steps`/`rms` are constructor defaults with no CLI flag found** — an NS
  schedule experiment needs a small flag addition.
- `src/wnn/training_dynamics.py` `build_optimizer` (lines 73–131): `head_lr_mult` split
  (champion "LRmult3" mechanism), `optim_switch`; `switch_optimizer` (lines 187–199) rebinds
  muon↔adamw mid-run with cold starts.
- `src/wnn/training/losses.py`: batch-local weight-sum normalization (lines 87, 107, 115);
  a `loss_norm == "running"` EMA (0.99/0.01 of batch wsum) variant already exists (lines
  108–113) — coded, never quality-graded.
- `GradStats` (`optim.py` lines 212–263): per-family pre-clip gradient norms (rnn_hh/rnn_ih/inp/
  head/bias) per epoch — reusable instrumentation.

### 2.2 Evidence history (mixed, never decomposed)

- **Positive-ish (unmatched).** Fold1 scratch: AdamW 0.66231/0.66515/0.66132 at lr 1e-3/5e-4/2e-3
  vs Muon-rnn base ~0.6662-class, single runs, one-run SD ~0.0022
  (`.../SCALING_DEEP_DIVE_20261011.md`); stability note: "Muon … beat AdamW by about 0.004–0.005
  at the same nominal LR in the one pair we have (single seed each, LR not tuned for AdamW)"
  (`.../OPTIMIZERS_SCHEDULES_STABILITY_R1.md`). Clean F0 H384 Muon-all/prop2 teachers s3/4/5:
  0.67898918/0.67836691/0.67969849, trio 0.68090199 (`specs/current-state-history-20261017.md`
  line 194). h384+all best scratch single 0.67021, n=1 (SCALING_DEEP_DIVE). Champion Z4 parent
  embeds hidden-scope Muon LRmult3/momentum.95
  (`.../IDEA_STATUS_AND_EXPLOITATION_PLAN_20261011_R1.md`).
- **Negative (matched, tiny).** D02 quartet above; 50/50 AdamW-WMSE+Muon-WMSE blend 0.604142139,
  paired delta −0.001859784, interval [−0.002617604, −0.001124842]; Muon-Huber 0.596056129;
  schedule semantics audited to match every step
  (`grading_r2/D02_QUARTET_FINDINGS_R2.md`, `D02_SCHEDULE_SEMANTIC_AUDIT_R2.json`).
- **Confounds.** `.../MUON_BATCH_EPOCH_MATH_20261011.md`: historical single-seed hybrids, batch
  and epoch effects entangled; frozen v6 producers (which had a muon-lr-mult flag) must be
  audited, not assumed equal to the working tree.
- **FP32 policy.** Upstream PyTorch Muon `_zeropower_via_newtonschulz` "explicitly converts the
  working matrix to bfloat16 … violates our strict-FP32 contract"
  (`research/unified/sources/TRANSFORMER_B300_SCALE_SOURCES_20261019_R2.md`); B300 plan requires
  all-FP32 Muon (`research/unified/ideas/LARGE_DYNAMICS_TRANSFORMER_B300_20261019_R2.md`);
  teammate PR32 "BF16-emulated runs are not strict-FP32 quality evidence"
  (`research/campaign_2026-10-24/resume4h_r1/FRONTIER.md` line 23). Our CUDA branch shares this.
- **Rank-adjacent (unmatched).** Per-gate 95%-energy ranks: H256 control 148–154 (H=256) vs H384
  Muon teacher 212–229 (H=384); explicitly "does not isolate Muon"
  (`research/campaign_2026-10-14/cpu_student_frontier_r1/WEIGHT_GEOMETRY_R1.md` + R2 receipt;
  diagnostic: 560–621 MB RSS, 0.41–0.66 s, one core).
- **Batch-mass audit (Oct24).** Multiplier 0.109–9.743 (F0) / 0.104–8.291 (F1);
  pooled/equal-batch WMSE 1.211032/1.058217 (F0), 1.298912/1.100586 (F1); inverse-HHI
  118.94→175.69, 125.41→177.11 (`resume4h_r1/CAMPAIGN_SYNTHESIS_R1.md`; plots in
  `resume4h_r1/eda/`). In-sample objective audit only; any normalization comparison must hold
  membership, order, update count, init and LR progress fixed.
- **Budget.** Kaggle 108,000 GPU-seconds remaining, zero reserved, verified 10:58:44 UTC Oct24
  (`resume4h_r1/FRONTIER.md` line 11). Local: 1 CPU core, ≤1 GiB, no swap (task brief; sibling
  agent occupies ~900 MB). Tiny-screen endpoint noise ~0.0007–0.001 WP (task brief; consistent
  with measured one-SD 0.0006–0.001 in OPTIMIZERS_SCHEDULES_STABILITY_R1.md).

## 3. Design space: candidate cards

Common constraints: strict FP32 everywhere (never cast NS to bf16, even on CUDA), causal inputs
and official metric unchanged, tiny CPU screens only locally, honest seed counts (single seeds =
hypothesis generation; paired deltas with 2–3 seeds minimum for a screen claim).

### C1. Strict-FP32 Newton–Schulz + scaling-convention audit
- **Source.** Kimi fixes (transcript [09:56]–[10:57]; arXiv 2502.16982 per `optim.py` docstring).
- **Analogy.** Our blocks are small (H×H, H×F, H≤512); only the CUDA path violates policy.
- **Intervention.** (i) Remove the CUDA bf16 cast (always FP32); (ii) audit scaling convention —
  0.2·sqrt(max(m,n)) vs sqrt(m/n) conventions diverge exactly on H×F weight_ih blocks; (iii)
  expose `ns_steps`/`rms` as flags.
- **Assumptions.** bf16 NS noise is small but nonzero; convention matters most for ih/head.
- **Predicted observable.** FP32-vs-bf16 NS divergence on saved gradients (small, nonzero
  relative error); endpoint sign unknown; compliance gain regardless.
- **Cheapest test.** Zero-fit: run both NS precisions on one batch's real gradients / cached
  gradients on CPU; record max/rms divergence and update angle. Minutes, <200 MB.
- **Falsifier.** Agreement <1e-6 relative on real shapes AND no endpoint change in the C3 panel —
  fix is compliance-only.
- **Cost.** CPU minutes; ~20-line patch. **Artifacts.** `optim.py`, GradStats, D02 caches.
- **Family.** numerical-fidelity-optimizer.

### C2. DeepSeek-style per-group hybrid for GRUs (which matrices get Muon)
- **Source.** DeepSeek-V4 recipe (transcript [11:22]–[12:36]): Muon for matrix-heavy modules,
  AdamW for embeddings/heads/biases/gates/norm.
- **Analogy.** A GRU is nearly all matrix (ih, hh, head). Untested splits: hh-only (pure
  dynamics), ih-only, per-gate (z/n on Muon, r on AdamW), head-only. DeepSeek keeps heads on
  AdamW — our champion's hidden scope contradicts exactly there.
- **Intervention.** Extend `muon_param_groups` with scopes `hh`, `ih`, `gates:zn`; matched panel
  vs existing `rnn`/`hidden`/AdamW.
- **Assumptions.** H×H recurrent blocks benefit from orthogonalized updates; input projection
  benefits from per-coordinate adaptivity.
- **Predicted observable.** Endpoint WP per scope (2 seeds); prior from ancestry: hidden ≥ rnn ≥
  hh-only; DeepSeek prior: head-on-AdamW ≥ head-on-Muon at matched budget.
- **Cheapest test.** CPU 350-seq screens are the wrong scale (D02); one Kaggle session, 8 matched
  cells {adamw, rnn, hh, hidden} × 2 seeds, H256, 4 epochs, all else identical. Well inside
  108k GPU-s (three H384 fits completed in one corrected v2 kernel,
  `specs/current-state-history-20261017.md` line 220).
- **Falsifier.** All Muon arms ≤ AdamW + noise on 2 seeds × 2 folds, or scopes
  indistinguishable within 0.001 — scope decomposition dead; parent Muon success confounded.
- **Cost.** ~8 small GPU fits; CPU only for packet prep. **Artifacts.** `muon_param_groups`,
  `build_optimizer`, existing Kaggle packet builders. **Family.** update-geometry-per-module.

### C3. Two-stage NS coefficient schedule
- **Source.** DeepSeek-V4 two-stage NS (transcript [12:36]–[13:11]): ~8 fast iterations then ~2
  stabilizing coefficients.
- **Analogy.** 5 fixed quintic steps may over/undershoot depending on gradient spectral spread,
  which changes across training phases.
- **Intervention.** `--ns-schedule {fixed5, deepseek10, staged}`: fast coefficients for k steps,
  then stabilizing set; optionally adaptive to Gram condition.
- **Assumptions.** Residual SV spread after 5 steps changes update direction, not just magnitude,
  at H 256–512.
- **Predicted observable.** Post-NS SV histograms per gate family over training; prediction:
  fixed-5 leaves spread >0.1 early.
- **Cheapest test.** Zero-fit replay of `orthogonalize` on saved/one-batch gradients at 5 vs
  8+2 vs 10 steps; compare update subspaces via principal angles. Fit arm only if angles diverge
  materially (>few degrees).
- **Falsifier.** Principal angles <2° across phases — schedule cosmetic at our sizes.
- **Cost.** CPU minutes; optional one GPU cell in the C2 session. **Artifacts.** `orthogonalize`.
  **Family.** update-geometry-per-module (spectral).

### C4. Batch-mass-corrected optimizer (normalization interaction)
- **Source.** Ours: 0.109–9.743× audit + already-coded `loss_norm=="running"` EMA branch
  (`losses.py` lines 108–113).
- **Analogy.** Per-batch normalization makes a heavy and a light batch contribute comparably
  normalized loss but differently-scaled per-row gradients; momentum then weights recent batches
  by arrival/mass regime, not pooled measure. Multiplies any base optimizer; needs no LLM story.
- **Intervention.** Arms in rising code cost: (i) `--loss-norm running` (zero new code);
  (ii) rescale gradient by running_wsum/batch_wsum BEFORE the momentum update (pooled-mass
  buffer); (iii) mass-aware second moment.
- **Assumptions.** The 100× spread reaches optimizer state (plausible: it scales the gradients
  the EMA averages); WP alignment is a population result, not a finite-run guarantee
  (`resume4h_r1/astra_fundamentals_r1/INDEPENDENT_CLAIM_REVIEW_R1.md`).
- **Predicted observable.** Endpoint sign unknown honestly (audit proves no harm); mechanism
  observable: reduced cross-batch variance of per-family pre-clip norms (GradStats measures it).
- **Cheapest test.** Matched 350-seq CPU pair, batch vs running normalizer, identical
  membership/order/updates/init/LR (exactly the synthesis's requirement), 3 seeds/arm paired by
  seed. Fits the 1-core/1-GiB envelope.
- **Falsifier.** Paired seed-mean delta within ±0.001 AND GradStats variance unchanged —
  normalization axis closed for the competition.
- **Cost.** 6 tiny CPU fits (hours at 1 core), zero GPU. **Artifacts.** `losses.py` running
  branch, GradStats, D02 harness/grading scripts. **Family.** optimizer-measure interaction.

### C5. Matched effective-rank diagnostics (zero-fit)
- **Source.** Transcript [16:36]–[17:40] (Muon features: robustness/transfer, larger logit
  margins, higher effective rank) + Parallax [14:17]–[16:02].
- **Analogy.** Recurrent state collapse / low-rank W_hh is a classic GRU failure on 20k-row
  sequences; if Muon raises gate-matrix effective rank, that is the mechanism by which it could
  help parents — measurable on disk.
- **Intervention.** Run `scripts/diagnose_student_compression.py` on MATCHED pairs (D02
  AdamW-WMSE vs Muon-WMSE checkpoints: same architecture/seed/updates; wider-recipe pairs from
  `research/unified/inference/f3_qat_ancestry_r1/INVENTORY_R1.md` as fallback); optionally add
  hidden-state effective rank on one cached sequence.
- **Assumptions.** Quartet checkpoints preserved with hashes (verify; else fallback pairs).
- **Predicted observable.** Per-gate 95%-energy rank / stable rank of W_hh/W_ih: Muon > AdamW at
  matched width+seed. Existing unmatched numbers (148–154/256 vs 212–229/384) are ~equal
  relatively and do NOT isolate the optimizer.
- **Cheapest test.** The diagnostic itself: 0.41–0.66 s, 560–621 MB RSS, one core — fits today.
- **Falsifier.** No rank/stable-rank separation on matched pairs — the Parallax/rank story does
  not transfer here; C2 loses its main mechanism.
- **Cost.** Minutes, CPU-only, zero fits. **Artifacts.** `diagnose_student_compression.py`,
  WEIGHT_GEOMETRY receipts, `grading_r2/QUARTET_GRADE_R2.moments.npz`. **Family.**
  representation-geometry diagnostics.

### C6. Metric-aware update shaping
- **Source.** Ours: official metric = weighted Pearson on targets clipped ±2 with weights
  |clip(Y)| (`src/wnn/gmetric.py`); training loss = per-batch normalized weighted MSE.
- **Analogy.** Pearson is shift/scale invariant per scored set; MSE is not; clip-boundary rows
  carry maximal weight. The optimizer can shape WHERE update mass goes without touching the loss.
- **Intervention.** Component-balanced updates: per-output-row (t0/t1) second-moment
  normalization of head+recurrent gradients so both targets' update magnitudes track their
  weighted gradient mass; or weight-mass-scaled per-family clip caps instead of the single
  global `--clip 1.0`.
- **Assumptions.** t0/t1 and heavy-row gradients are persistently imbalanced in update norm.
- **Predicted observable.** GradStats familySq-share and t0/t1 update-norm ratio; small positive
  endpoint on t1-heavy folds IF imbalance exists.
- **Cheapest test.** Zero-fit: run GradStats on one existing 350-seq recipe for a few hundred
  CPU updates; quantify imbalance; fit only if >2× persistent.
- **Falsifier.** Update-norm ratios ~1 and stable — drop the card.
- **Cost.** CPU minutes; conditional 2×2-seed screens. **Artifacts.** GradStats, D02 harness,
  `gmetric`. **Family.** optimizer-measure interaction (metric).

### C7. "Adamuon": second-moment blend on orthogonalized updates
- **Source (abstract-level).** SOAP (Adam in preconditioner eigenbasis; >40% iteration reduction
  on 360m/660m LMs); KL-Shampoo (competitive without Adam grafting); "A New Perspective on
  Shampoo's Preconditioner" (Shampoo² ≈ one power-iteration step toward optimal Kronecker).
- **Analogy.** Muon fixes direction distribution; Adam fixes per-coordinate scale. Keep both:
  NS first, then element-wise bias-corrected normalization of the ORTHOGONALIZED update —
  SOAP-like without eigendecomposition.
- **Intervention.** In `MuonAdamW`, keep `exp_avg_sq` of the post-NS update; final update =
  ns_update/(sqrt(v)+eps), scaling reduced to a tunable constant. Extra state ≈ 5 MB FP32 at
  H256 — CPU/RAM/GPU negligible.
- **Assumptions.** Post-NS updates retain per-coordinate imbalance worth normalizing.
- **Predicted observable.** Endpoint unknown; early diagnostic: coefficient of variation of
  post-NS update entries (if ~flat, the Adam stage is a no-op — cheap kill).
- **Cheapest test.** Zero-fit variability diagnostic, then ONE cell (2 seeds) appended to the C2
  Kaggle session; do not spend CPU-screen slots (D02 showed tiny screens punish Muon variants).
- **Falsifier.** Post-NS CV < ~0.1, or Kaggle cell ≤ Muon − noise.
- **Cost.** ~30 lines + one GPU cell. **Artifacts.** `MuonAdamW`. **Family.** hybrid-first/
  second-order.

### C8. Optimizer sandwich via existing `optim_switch`
- **Source.** Ours: `switch_optimizer` (`src/wnn/training_dynamics.py` lines 187–199), WSD/SGDR.
- **Analogy.** If Muon builds features early (rank story) and AdamW fine-tunes late under noisy
  batch measures, a fixed-fraction switch is the minimal test; champion's momentum.95/LRmult3
  hints winning regimes are far from defaults.
- **Intervention.** Muon-first/AdamW-late and reverse arms in the C2 Kaggle session.
- **Assumptions.** Phase-dependent optimizer advantage exists at our 4–24-epoch horizons.
- **Predicted observable.** Endpoint vs both pure arms; GradStats family-share drift after the
  switch.
- **Cheapest test.** Two cells (1–2 seeds) in the C2 session; CPU screens under-powered.
- **Falsifier.** Both sandwiches within noise of the pure arms.
- **Cost.** 2 GPU fits, zero new optimizer code. **Artifacts.** `switch_optimizer`, schedule
  lambdas. **Family.** update-geometry-per-module (phase).

## 4. Prioritized plan (deadline 2026-11-15)

1. **C4 (batch-mass axis) — best local value.** Zero/low code, runs in today's 1-core/1-GiB
   envelope, unique to our setting, with the audit already quantifying the mechanism input.
   Paired 350-seq screens batch vs running, 3 seeds/arm, per-seed deltas against the
   0.0007–0.001 floor; if seed-mean paired delta > +0.0015 with consistent sign, promote to a
   2×2 Kaggle confirmation (normalizer × optimizer, 2 seeds) before the deadline.
2. **C5 (matched rank diagnostics) — best zero-fit value.** Minutes of CPU; either gives C2/C7 a
   mechanism or removes the Parallax justification; gates the C2 panel's expected direction.
3. **C1+C2+C3(+optional C7/C8 cells) as ONE Kaggle session** — the never-run decomposition:
   strict-FP32 NS patch first (compliance), then 8–10 matched H256 4-epoch cells ({adamw, rnn,
   hh, hidden} × 2 seeds, optional adamuon/sandwich probes). Within 108k GPU-s given the
   historical three-H384-fit kernel; still require a fresh shared-quota check and actual-device
   timing before admission (repo rule). Gate: any scope arm beating AdamW by >0.002 seed-mean
   earns a 2-fold × 2-seed confirmation (~4 fits) before any pipeline adoption.

**Wholesale falsifier:** C4 paired |delta| ≤ noise on 3 seeds AND C5 shows no matched rank
separation AND every Muon arm ≤ AdamW + noise across seeds/folds ⇒ "own optimizer" closes for
this competition; keep the FP32 NS patch (policy), archive the rest, spend remaining quota on
blend/serving endpoints. A positive on any axis does NOT authorize scope creep beyond the
confirmation cells before 2026-11-15.

Honesty: label endpoint claims with seed counts and exposure; single-seed results stay
hypothesis-generation; no public-score forecasting from local deltas; D02/successor panels are
exposed fold0 and must be labeled as such.

## 5. Literature log (abstract-level; local arxiv-search CLI)

- `Muon optimizer Newton-Schulz` — **used**: "Convergence of Muon with Newton-Schulz" (few-step
  NS matches SVD-polar up to a constant, doubly-exponentially small in steps; removes sqrt(rank)
  loss vs SGD-momentum); "Muon with Finite Newton-Schulz: The Smoothing Benefit in Nonsmooth
  Nonconvex Optimization" (finite NS smooths the polar map; log-depth suffices); "Hierarchical
  Muon (HiMuon)" (tiled NS: O(r²sK)→O(HWTK), relevant to cheap CPU/tiled variants). Discarded:
  three particle-physics papers.
- `Muon scalable language model training`, `Muon optimizer memory efficient distributed`,
  `Muon is Scalable for Speech Language Model Training`, `Muon generalization feature learning
  effective rank` — **failed**: literal "muon" floods results with g-2/spectrometer papers; the
  Moonlight paper is instead cited from the verified in-repo register
  (`research/unified/sources/TRANSFORMER_B300_SCALE_SOURCES_20261019_R2.md`, 2502.16982 row).
- `SOAP Shampoo Adam preconditioner` — **used**: "SOAP: Improving and Stabilizing Shampoo using
  Adam" (Adam in preconditioner eigenbasis; >40% iteration reduction, 360m/660m LMs; basis for
  C7); "Understanding and Improving Shampoo and SOAP via Kullback-Leibler Minimization"
  (KL-Shampoo, no Adam grafting); "Reparametrizing Shampoo and SOAP for Subspace Basis Updates
  and BFloat16 Storage" (precision/cost of QR-based preconditioners); "A New Perspective on
  Shampoo's Preconditioner" (Shampoo² ≈ one power-iteration step; Gauss-Newton/Adagrad view).
- `spectral descent matrix factorization optimization` — **failed** (NMF/conic/PDE; nothing
  usable).
- `momentum orthogonalized update matrix optimizer neural network` — mostly **failed**; kept one
  tangential item (MomentumRNN; momentum-in-cell, not an optimizer result).
- `second order Kronecker recurrent neural network optimization` — **used**: "Eva: A General
  Vectorized Approximation Framework for Second-order Optimization" (Kronecker-factorized
  stochastic-vector second moments + Sherman–Morrison; 2.05×/2.42× speedups incl. vs Shampoo —
  candidate cheap CPU preconditioner family beyond this proposal's horizon). Rest noise.
- `adaptive gradient methods time series forecasting deep learning` — **failed**: no
  optimizer-vs-optimizer evidence for forecasting; nothing used.

All used items are abstract-level; none is a demonstrated GRU/weighted-clipped time-series
result; none transfers numerically to our endpoint.

## 6. Sources inspected

Repository (read-only): `src/wnn/optim.py`; `src/wnn/training/README.md`;
`src/wnn/training/cli_optimizer.py`; `src/wnn/training/optimization.py`;
`src/wnn/training/losses.py`; `src/wnn/training_dynamics.py` (lines 60–205);
`specs/current-state-history-20261017.md`; `grading_r2/D02_QUARTET_FINDINGS_R2.md` (+ schedule
audit JSON, grade moments); `OPTIMIZERS_SCHEDULES_STABILITY_R1.md`; `SCALING_DEEP_DIVE_20261011.md`;
`IDEA_STATUS_AND_EXPLOITATION_PLAN_20261011_R1.md`; `MUON_BATCH_EPOCH_MATH_20261011.md`;
`resume4h_r1/FRONTIER.md` + `CAMPAIGN_SYNTHESIS_R1.md` + `astra_fundamentals_r1/`;
`TRANSFORMER_B300_SCALE_SOURCES_20261019_R2.md`; `LARGE_DYNAMICS_TRANSFORMER_B300_20261019_R2.md`;
`EXPERIMENT_DRAFT_WORKFLOW_20261017_R1.md` (FP32 definition); `WEIGHT_GEOMETRY_R1.md` (R2
receipt); `f3_qat_ancestry_r1/INVENTORY_R1.md`. User-supplied:
`/home/vlad/Downloads/Muon_ The Most Underrated Breakthrough in LLMs - TranscriptX.ai.txt`
(Kimi fixes [09:56]–[11:22] incl. ~52% FLOPs claim; DeepSeek-V4 hybrid/two-stage NS
[11:22]–[13:11]; Parallax [14:17]–[16:02]; features/effective-rank [16:36]–[17:40]). External:
abstract-level arxiv-search results (section 5); PyTorch Muon API/source claims from the
verified in-repo register, not re-fetched.
