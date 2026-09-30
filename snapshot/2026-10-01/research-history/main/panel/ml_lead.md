# ML lead: scaling, learning, augmentation, data weighting, diversity (panel, 2026-09-26)

Lens: deep-learning lead. Verification tier B (chain-grouped folds on validation, stress panel, 7 LB
points; the private LB is "a separate, hidden dataset", docs/rules.md). All new numbers below come from
read-only CPU probes (`scripts/limited`, `WNN_MEM_MAX=3G WNN_CPU_PCT=10`, cached `pred_*`/targets
via mmap, at most 2 parquet row groups). Probe scripts live in the session scratchpad; each is
described well enough to rerun. Nothing was trained and the ledger was not edited: section 6 lists
entries for the controller to log.

## 0. Contract and cost units

- Question: which DL levers can still add >= +0.001 LB under 1 vCPU / 60 min / 20 MB, and at what
  GPU cost?
- Runtime budget: 36.3 min for 2 members on the scorer. One more h128x2 member (~+7 min, ~44 min)
  fits without new engineering. Two more need the int8 engine (opus_arch A1). Keep the final
  package <= 45 min, because nobody has said the private "Final Dataset" is the same size as the
  public test.
- GPU cost at 55 seq/s:
  - train-only 2-epoch fit: 0.11 h;
  - train+valid fit: 0.12 h;
  - 1-epoch valid fine-tune on ~1.4k sequences: <0.01 h training + ~0.02 h eval;
  - full-valid predict: 0.04 h;
  - 4-fold chain K-fold of the final recipe: 0.6 h.
- Conclusion: GPU is not the binding resource. The binding resources are evaluation noise (paired
  SE 0.0003-0.0007, seed noise ~0.001, proxy-vs-LB error +-0.001-0.002) and LB slots (5/day until
  Nov 15; the private set is separate, so public-LB tuning carries overfit risk).

## 1. New measurements (this report)

| id | probe | result | consequence |
|---|---|---|---|
| M1 | Dense-scored gap (E-132) decomposed. Within-class scoring-rate terciles x causal-propensity bins (prop_v1), and within-sequence (demeaned) WP vs level. Full valid, ab30 and baseline | The gap **persists at matched propensity**: class 3, p>=0.5 bin, WP 0.604 / 0.496 / 0.407 (sparse/mid/dense); class 2 0.904 / 0.870 / 0.814. It lives **within sequences**: class-3 within-seq WP 0.530 / 0.454 / 0.379 (pooled 0.547 / 0.461 / 0.385). **The baseline shows the same pattern** (0.520 / 0.445 / 0.370) | Not row composition, not level, not our weighting. Dense sequences are intrinsically less predictable row by row, so loss reweighting toward them has no support (section 4) |
| M2 | Organizer-baseline output scale, 200 valid seqs, rows >= 99 | Raw \|p\|: median 1.34, mean 1.76. Mean \|y\| is 0.63. For any conditional mean, E\|E[y\|x]\| <= E\|y\| (Jensen), so the baseline is **not an MSE fit to these targets** | Its training objective (scale-free, or rescaled targets) differs from ours. That is the most likely source of its diversity (residual corr 0.956), since E-120/E-125 ruled out the input view and clipped plain MSE → X2 |
| M3 | Did the baseline see train? All-rows WP (no mask), 300 random seqs per split | Baseline: train 0.4451, valid 0.5052. Our train-only GRUs (in-sample on train): train 0.4318-0.4327, valid 0.5039-0.5041. The baseline's lead over ours is +0.013 on train and +0.001 on valid | The baseline behaves like a model fit on train (it has an in-sample-like edge there). There is no sign of extra or later data. Diversity comes from its training procedure (M2) |
| M4 | Directional priors: \|y\|-weighted mean of D = t0 − t1 on scored valid rows, per class | True +0.038 / −0.185 / −0.282 / +0.087. ab30 predicts +0.040 / −0.291 / −0.384 / +0.095. Baseline predicts +0.27 / +0.12 / −0.005 / +0.29 | Our model carries, and in classes 1/2 amplifies, a directional prior. Plain mirror augmentation would erase it, which can help or hurt. So use a mirror-flag input (X7) |
| M5 | Mirror structure, class 3 (valid rg 3 + global table `cache/class3_moves_full.pkl`) | Every book event changes both the bid and the ask half-code (codes are relative to a joint reference). In the sequence, pairing sorted half-codes i↔i maps all 8 joint states onto observed states. It also maps transitions onto transitions with t0↔t1 swapped: ask-up from the symmetric state 609 ↔ bid-down 337; completing an up move 451 ↔ completing a down move 457. The mirror is **not** a signed column permutation (column matches are non-unique). The global table has 95 bid half-codes, 85 ask half-codes, 119 joint states and 320 labelled transitions | A class-3 involution is a small labelled-graph automorphism problem. Classes 0-2 and the volume codes need a learned or ICA map (X7) |
| M6 | Loader | Parquet read of one sequence takes 31-76 ms (warm/cold), about 13-32 seq/s per worker. Runs use 2 workers at 46 seq/s, so the loader is at or near the bound. **fp16 caching is lossy for event features**: 69/112 columns merge distinct values, and row-change flags disagree on 1.2% of cells | Throughput comes from more workers or a lossless cache. An fp32 cache of train is 95 GB against 86 GB free (X5) |

## 2. Answers to the six questions

1. **Scaling.** There is no evidence-backed case for more width, depth or compute: width
   128/192/256, depth 1/2/3 and LSTM are within noise (E-081/E-083); 3 epochs gives +0.0006; the
   recurrence has effective rank 16-32 (opus_arch D3); +18% generic data gives +0.001 on holdout.
   The one data increase that moved the LB is domain-matched: the valid true-mask fine-tune gave
   +0.0014 LB (E-144) from 1,505 sequences.
   - **Big-teacher distillation: no.** A 3-model teacher gives +0.0013 alone, ~0 on top of
     floor + aux (E-102) and 0 in the final recipe (E-116). The tower property keeps the minimiser
     E[y\|x], so a teacher can only reduce variance, and variance is small (seed residual corr
     0.994; infinite seeds ≈ +0.0018, opus_independent S3). A bigger causal teacher ≈ the student
     (width is flat). Keep the non-causal event-clock teacher (opus_arch A8) only as a class 0-2
     long shot. Distillation to free runtime is not needed: one extra member fits today.
2. **Learning.** lr 2e-3 = 1e-3 (ab23: 0.6712 vs 0.6707); SWA over the tail is flat (E-052);
   longer TBPTT hurts (E-083); no warm-up deficit (P4); memory <= ~1000 rows (D1). Two levers
   remain: more sequences per step with a shorter recurrence (X5; chunk 500 was +0.0016 stress,
   E-123) and weight soups of fine-tune children that share a parent (X3). Dense-regime loss
   reweighting: not supported (M1).
3. **Augmentation.** Mirror symmetry is real in class 3 (M5), but it needs an involution for
   price, volume, trade and i1 codes; X7 gives a staged protocol with an equivariance gate on the
   existing model before any training. Time-warp, input noise and random crops have low priors:
   per-sequence normalisation was flat (E-118), there is no warm-up deficit (P4), and event flags
   are exact-equality features, so noise must be added after flag computation.
4. **Filtering and domain weighting.** E-144 is one noisy point: FT +0.0014 LB against +0.0005
   clean-fold and +0.0009 leaky-holdout predictions. It fits **H-sibling: the public test is
   sequence-level siblings of the valid streams** (a random sequence split of the same pool): the
   test is ~1,850 sequences, the size of valid, and the chain-leaky holdout predicted the LB better
   than clean folds. X1/X4 test H-sibling cheaply. The private set is "separate", so every final
   pick must also be >= 0 on clean chain folds. Valid oversampling and adversarial importance
   weights are ranked last (X6): class-mass weights (E-104) and online norm (E-118) were flat.
5. **Diversity.** Diversity came from the objective and the data, not from the cell. Residual
   correlation with our GRU:

   | member | differs from ours in | residual corr |
   |---|---|---|
   | our recipe variants (seeds, heads, epochs, floor, data) | — | 0.990-0.995 |
   | ab12 | unweighted | 0.918 |
   | organizer baseline | objective, M2 | 0.956 |

   So objective changes come first (X2), then a zero-cost change to the existing diverse member
   (X1), then cells only behind a residual-corr gate (X8).
6. **Pipeline.** There are 2-3x from batch/chunk rebalancing plus loader workers (X5). Spend the
   speed on 2 seeds per arm, which halves seed noise; that matters more than more arms. Add the
   residual-correlation readout (opus_arch D7) to `decide.py`.

## 3. Top 8, ranked by expected LB gain per GPU-hour

E[ΔLB] = P(passes gates) × gain if it passes. All costs are small, so the ranking mostly reflects
expected gain and odds.

| rank | id | experiment | E[ΔLB] (range) | P | GPU-h | E/GPU-h | inference cost |
|---|---|---|---|---|---|---|---|
| 1 | X1 | Fine-tune the organizer-baseline member on valid rows (true masks) | +0.0005 (−0.0003..+0.0015) | 0.45 | 0.3-0.5 | ~0.0012 | 0 |
| 2 | X2 | Pooled weighted-Pearson ("gcorr") objective member | +0.0004 (0..+0.0015) | 0.35 | 0.35 | ~0.0012 | +1 member (~7 min) |
| 3 | X3 | Weight soup of fine-tune children from one parent | +0.0002 (0..+0.0006) | 0.5 | 0.2 | ~0.001 | 0 |
| 4 | X4 | Sibling-fold protocol, FT schedule, FT on all 1,873 valid, 1-2 discriminating LB slots | +0.0004 (0..+0.0012) | 0.5 | 0.5-0.8 | ~0.0006 (+ decides valid emphasis) | 0 |
| 5 | X5 | Batch 128 × chunk 500 + 4-6 loader workers (throughput enabler) | +0.0002 (0..+0.0008) | 0.4 | 0.3 | ~0.0008 + 2-3x speed | 0 |
| 6 | X8 | Cell-family member (TCN or minGRU) behind a residual-corr gate | +0.0002 (0..+0.001) | 0.2 | 0.3 | ~0.0007 | +1 member |
| 7 | X7 | Mirror involution: equivariance gate, then conditional augmentation and a TTA member | +0.0003 (0..+0.0015) | 0.25 | 0.5 (+1-2 agent-days CPU) | ~0.0006 | TTA +1 member |
| 8 | X6 | Domain-weighted main fit (valid ×2 + adversarial importance weights on train sequences) | +0.0002 (0..+0.001) | 0.3 | 0.6 | ~0.0004 | 0 |

**Execution order (dependencies).** (1) X4a, the random-fold flag (code only). (2) X1, X3 and the
X4 sweep: all fine-tune based, about 1 GPU-h together, one evening. (3) X5, so later fits are fast
and seeded. (4) X2 or X8: only one third member fits, so take whichever passes its gate with the
lower residual corr. (5) X7 gates on CPU in parallel; X6 last.

### X1. Fine-tune the organizer-baseline member on valid (the 0.35-weight member has never seen valid)
- **Mechanism.** The valid true-mask FT is the only lever with measured LB transfer (+0.0014,
  E-144), and so far it is applied only to our GRU. The baseline is the shift regulariser: its blend
  gain rises with train-unlikeness (E-129). Adapting it to the valid/test regime at low lr, with the
  raw view frozen, should improve the member without collapsing its diversity, which comes from
  its objective (M2).
- **Exact change.**
  - Write `artifacts/runs/base_rnn/model.pt` from `scripts/convert_baseline.py`: fold the 0.25
    output scale into `head.2`; args hidden 128, layers 2, cell gru; n_in 211.
  - `train_rnn.py`: add `--freeze inp,head.0` (requires_grad False) so the identity-plus-shift
    layers stay exact.
  - Run with `--init base_rnn --raw-only --train-source valid --valid-fold k --mask-weight
    --aux-struct 0 --aux-next 0 --epochs 1 --lr 1e-4 --wd 0`, with loss `wmse` in one arm and
    `gcorr` (X2) in the other.
  - Evaluate on chain folds 0/1 (parents `ab39_ftm_f*` / `ab29_mask_f*` for ours) and on random
    folds (X4a).
- **Readouts.** Residual corr of baseFT vs ourFT (D7 method); blend 0.65·ourFT + 0.35·baseFT vs
  0.65·ourFT + 0.35·base at λ ∈ {0, 0.05}, full and stress.
- **Falsifier.** The blend delta is <= 0 on either fold type, or residual corr rises above 0.985
  (diversity collapsed).
- **Cost.** 2 losses × 2 folds × 2 fold types × ~0.03 h, plus blends ≈ 0.3-0.5 GPU-h; one LB slot
  if it passes.

### X2. Pooled weighted-Pearson objective ("gcorr") as the diversity member
- **Mechanism.** M2/M3 say the baseline's diversity comes from its objective geometry, not from its
  data or its view. Our earlier `corr` loss is per-sequence-chunk (`weighted_corr_loss` computes ρ
  per batch element), which drops between-sequence levels. The metric pools globally, so a loss
  pooled over all B×T rows of a step is both scale-free and level-aware. It is the metric itself,
  so it fits a different function than |y|-weighted MSE.
- **Exact change.**
  - `src/wnn/nn.py`: add `pooled_wcorr_loss(pred, target, row_w)`. Per target, flatten (B,T); set
    w = |clip y|·row_w; loss = 1 − mean_k ρ_w.
  - `train_rnn.py`: add `--loss gcorr` (optional `--gcorr-mse 0.1` anchor against scale drift).
  - Arm A: train-only ab30 flags with `--loss gcorr`, seed 3. Arm B: the same with `--raw-only`,
    only if A passes.
  - Fit the member's output scale for blending by one scalar per target on chain folds.
- **Readouts and gate.** Alone WP >= 0.672 (the baseline's point); residual corr vs ab30 <= 0.97;
  3-way blend (ab30 + baseline×0.25 + gcorr) beats the 2-way on chain folds and stress.
- **Falsifier.** Residual corr >= 0.985, or 3-way <= 2-way + 0.0003. Arm A also tests M2: if it is
  not more diverse than ab9 (0.982), the scale-free explanation of the baseline is refuted.
- **Cost.** 0.35 GPU-h; runtime +7 min (fits).

### X3. Weight soup of fine-tune children (zero inference cost)
- **Mechanism.** The soup of independently trained arms failed because their normalisation
  buffers differed (E-119). SWA along one OneCycle tail had no diversity (E-052). Children
  fine-tuned from one parent share buffers and a basin (Wortsman et al. 2022, arXiv:2203.05482;
  Frankle et al. 2020, arXiv:1912.05671). Averaging them removes FT data-order and subset noise at
  no runtime cost.
- **Exact change.**
  - Parent: `ab33_final_s4` for the final; `ab29_mask_f{0,1}` for evaluation.
  - Four children with seeds 3-6, lr {1e-4, 2e-4, 2e-4, 3e-4}, each on an 80% sequence bootstrap
    (`--ft-subsample 0.8`, new flag).
  - Average `state_dict`s. EMA variant: decay 0.995 per FT step.
  - Before averaging, check that the loss on the linear path between two children has no barrier.
- **Falsifier.** The soup is <= the best child on both fold types.
- **Cost.** 0.2 GPU-h.

### X4. Sibling-fold protocol, FT schedule, FT on all valid, discriminating LB pair (tests H-sibling)
- **Mechanism.**
  - If the public test is a random sequence split of the valid pool, test sequences have chain
    neighbours in valid. FT on valid then acts like the leaky holdout (+0.0009), not like clean
    folds (+0.0005), and the LB (+0.0014) agrees.
  - In that case more valid-stream data (the 368 holdout sequences, +24%) and a stronger FT pay
    off on the public LB. The private set may not share streams, so this is a public-vs-private
    trade-off to manage explicitly.
- **Exact change.**
  - `train_rnn.py`: add `--valid-fold-mode random` (fold = seeded permutation of valid row groups
    mod 4; `chain` stays the default private proxy).
  - Replicate the ab39 FT on random folds.
  - Add random-fold arms to the running ab41 sweep: epochs {1,2,3} × lr {1e-4, 2e-4, 5e-4}, plus
    `--ft-replay 0.1` (10% train sequences mixed in).
  - Final: FT on dev + holdout. The `--train-source valid` pool must include the holdout when
    `--valid-fold -1 --ft-all`.
  - LB: submit two FT variants that the two proxies rank oppositely (for example e3 lr 5e-4 vs e1
    lr 2e-4).
- **Falsifier.** The random-fold FT gain is within 1 SE of the chain-fold gain (no sibling effect):
  then choose FT settings on chain folds only and stop valid-emphasis work. Selection rule for final
  picks: chain-fold delta >= 0 (private proxy) AND random-fold delta > 0.
- **Cost.** 0.5-0.8 GPU-h; 1-2 LB slots (high information value: it decides X6 and the final
  valid emphasis).

### X5. Batch/chunk rebalancing and loader throughput
- **Mechanism.**
  - B32×C2000 and B128×C500 process the same rows per step and the same number of steps. B128×C500
    shows 4x more sequences per step, which decorrelates updates: B32 feeds the same 32 streams for
    10 consecutive steps.
  - It also has 4x shorter sequential depth per kernel call. A GRU at h128 with B32 is
    latency-bound, not FLOP-bound.
  - Chunk 500 alone was +0.0016 on stress and +0.0006 full (E-123).
- **Exact change.**
  - Bench first: synthetic GPU-only step timing for B ∈ {32, 64, 128, 256} × C ∈ {2000, 1000,
    500}, then the loader at `WNN_LOADER_WORKERS` 2/4/6 (M6: ~30 seq/s per worker).
  - Quality run: `--batch 128 --chunk 500 --lr 2e-3` (ab23 showed 2e-3 is safe), 2 seeds.
  - Host RAM: 1.2 GB per batch, within the 15 GB job cap with prefetch 1-2 and 4 workers.
  - Do not cache in fp16 (M6). If a cache is needed, use a lossless per-sequence zstd npy.
- **Falsifier.** Speed-up < 1.5x, or the full-valid delta < 0 on both chain halves.
- **Cost.** 0.3 GPU-h.
- **Payoff.** 2-3x throughput, spent on 2 seeds per arm.

### X6. Domain-weighted main fit
- **Mechanism.** Propensity weighting was worth +0.05 (E-069), so our GRU is a "global learner"
  whose training distribution matters (Shimodaira 2000). Train and valid separate at AUC 0.99
  (E-094). Weighting train sequences toward valid-likeness corrects the same kind of shift at
  sequence level.
- **Exact change.**
  - Emit OOF p(valid | seq) for train sequences from `scripts/adversarial_validation.py`.
  - w = clip((p/(1−p))^γ / mean, 0.25, 4), γ ∈ {0.5, 1}.
  - Add `--seq-weights cache/adv_train_w.npy` (WeightedRandomSampler, as `--class-balance`
    already does) and `--valid-repeat 2`.
  - Train-only arms, then the train+valid arm on chain K-fold.
- **Falsifier.** <= 0 in the most train-unlike tercile (opus_independent V2) and on stress.
- **Cost.** 0.6 GPU-h. Prior evidence against: E-104 (class-mass weights), E-118.

### X7. Bid/ask mirror: involution discovery → equivariance gate → conditional augmentation → TTA
- **Mechanism.** Mirroring (price p → −p, bid ↔ ask, t0 ↔ t1, since t0 = F − ask and t1 = bid − F)
  doubles the data, including valid-stream data for X4. A TTA member (same weights, mirrored stream,
  swapped outputs) is a diversity member whose errors come from the model's non-equivariance.
- **Exact change.**
  1. **Class-3 price codes (exact).** Nodes: the 119 joint states of `cache/class3_moves_full.pkl`,
     labelled by modal spread (`wnn.ticks.learn_spreads`). Edges: labelled (d_ask, d_bid). Find an
     involutive isomorphism G → G' where G' maps edge labels (d_ask, d_bid) → (−d_bid, −d_ask)
     (networkx `DiGraphMatcher` with node/edge match). Break ties by symmetry of train visit
     frequencies. Sequences with unmatched rare states are marked non-mirrorable.
  2. **Volume (22), trade dp/dv (8) and i1 codes.**
     - Per class and block, run ICA (E-090 recipe).
     - Pair bid/ask ICs by Hungarian matching on signatures: kurtosis, lag-1 autocorrelation, and
       response to own-side price events (known from step 1).
     - Mirror = unmix → swap pairs (sign by response) → remix.
     - a-columns are invariant unless a2/a3 respond side-specifically.
     - Classes 0-2 price codes use the same ICA route (grammar only 65-69% exact).
  3. **Gate A** (CPU, before any training):
     - Run `ab33_final_s4` on 20 held-out class-3 valid sequences, original and mirrored.
     - Require corr(f(Mx)_t0, f(x)_t1) >= 0.9 and WP(swap f(Mx)) >= WP(f(x)) − 0.02.
     - A wrong M gives corr well below 0.5.
     - Test the price map alone first, with a price-only class-3 GRU (0.05 GPU-h), then add blocks
       one at a time.
  4. **Training.** `Sequences.__getitem__`: `--mirror-p 0.5` applies M and swaps y columns, and
     appends a mirror-flag input (n_in + 1) so that the directional priors of M4 survive (flag 0
     at test).
  5. **TTA member.** Same weights on the mirrored stream with flag 1, outputs swapped.
- **Falsifier.** Gate A fails; or the augmented arm is < 0 on chain folds; or the TTA member's
  residual corr is >= 0.99 (no better than a seed, E-111).
- **Cost.** GPU 0.5 h. The real cost is 1-2 agent-days of CPU discovery work.

### X8. Cell-family member behind a gate (only if X2 fails)
- **Exact change.** One of opus_arch A3, with the ab30 recipe: minGRU with a log-space parallel
  scan, or a causal TCN (kernel 2, dilations 1..128, 64 channels, receptive field 256 per the D1
  memory horizon, incremental ring-buffer inference, ~65k MACs).
- **Gate.** Alone WP >= 0.672 and residual corr <= 0.975; then the 3-way blend.
- **Falsifier.** Residual corr >= 0.985.
- **Prior.** Low. Every change inside our recipe lands at 0.990-0.995 regardless of cell (the LSTM
  and all widths behave the same). The two diverse members differ in objective or weights.
- **Cost.** 0.3 GPU-h.

## 4. Not recommended now (evidence)

| idea | why not |
|---|---|
| Wider/deeper/LSTM, longer training | E-081/E-083, ab23 (3 epochs +0.0006, lr 2e-3 = 1e-3), D3 rank 16-32 |
| Large causal teacher → student | Width flat (teacher ≈ student); E-102/E-116 distillation adds 0 on top of the recipe; variance cap S3 |
| Non-causal teacher (general) | Minimiser unchanged by the tower property; reduces label-noise variance, which is not binding. Keep only opus_arch A8 (event-clock teacher, classes 0-2) as a long shot |
| Loss reweighting toward dense-scored sequences | M1: gap persists at matched propensity and within sequences, and the baseline shows the same gap: intrinsic difficulty, not misallocated capacity. Density gating of blends already failed (E-138); propensity as input was flat (E-045) |
| Time-warp of clock columns, input noise, random crops/state resets | E-118 per-sequence normalisation flat; P4 no warm-up deficit; D1 memory <= ~1000 rows; clocks are quantile codes whose warp map is unknown; noise breaks exact event flags unless added after flag computation |
| SWA/EMA along one trajectory, LR schedule tweaks | E-052 flat; ab23 lr 2e-3 = 1e-3 |
| D/S reparameterisation (predict F and spread) | A linear change of the output head; under wMSE it differs only in per-target weights, so little diversity is expected. Do it only as an aux head if X2 fails |
| fp16 data cache | M6: corrupts 1.2% of event flags |

## 5. Decision protocol for these arms

- Two proxies for every arm: chain folds = private / new-stream proxy; random sibling folds (X4a)
  = public proxy; plus the stress panel at λ 0.05. Promote an arm only if it is >= 0 on chain
  folds and > 0 on sibling folds.
- Seeds: 2 per arm whenever the predicted |Δ| < 0.002. X5 makes this affordable.
- Diversity members: read residual corr (D7) before any blend. It is the cheapest and least noisy
  diversity signal.
- LB: one slot per mechanism family, preferably as discriminating pairs (X4). Final picks must pass
  the chain-fold rule, because the private LB is a separate dataset. Keep packages <= 45 min.

## 6. For the controller's ledger (not written by me)

Hypotheses [mechanism family]: **H-ML1** [member domain adaptation] a valid FT of the baseline
member adds on top of the FT'd GRU (X1). **H-ML2** [objective geometry] the baseline's diversity
comes from a scale-free objective; a pooled-Pearson member reproduces it more accurately (X2; M2,
M3). **H-ML3** [weight averaging] a soup of FT children beats the best child (X3). **H-sibling**
[data provenance] the public test is sequence-level siblings of the valid streams (X4; E-142,
E-144). **H-ML5** [optimisation] more sequences per step at equal rows per step helps, and
B128×C500 is 2-3x faster (X5). **H-ML6** [cell diversity] a TCN/minGRU member reaches residual corr
<= 0.975 (X8). **H-ML7** [symmetry] a class-3 mirror involution exists as a labelled-graph
automorphism; conditional augmentation or TTA adds a member (X7; M4, M5). **H-ML8** [importance
weighting] adversarial sequence weights help the train-unlike tercile (X6).

Observations to log as E-entries: M1 (dense gap intrinsic), M2 (baseline |p| > |y|, not an MSE
fit), M3 (baseline in-sample-like on train), M4 (directional priors), M5 (class-3 mirror closure;
global table size), M6 (loader bound; fp16 cache lossy).
