# Architectures and training dynamics under the latency budget (Opus researcher, 2026-09-26)

Roles: scout, ideator and specification critic (conduct-creative-research). Verification tier B
(chain-grouped folds on validation; the test mask and test instruments are unobservable).

Tags: **[read]** = read in the primary source at the cited location; **[web]** = abstract or
listing checked online in this session; **[gk]** = general knowledge, not re-read;
**[ours]** = measured here with tiny read-only checks (one process, `scripts/limited`,
`WNN_MEM_MAX=2G WNN_CPU_PCT=10`, CPU only, at most 96 sequences through `SequenceFile` or mmap
of cached predictions). Probe scripts live in the session scratchpad; the appendix describes
each one well enough to rerun it.

Independence note. I committed twelve hypotheses before reading `research/tree.md` and
`research/lit/*.md`. Overlaps I found afterwards: the lookahead teacher (tree F3, scout M9), fused
int8 inference (tree E1, scout M5), new cells (tree C4), the reset-state memory-length probe
(tree triage H2-7, run here as D1), and volume-level invariance (shift scout M5/M6). Where I
overlap, I add the new evidence or a cheaper implementation and point to the other report.

## 0. New measurements [ours]

All model probes use `artifacts/runs/ab30_eps_aux/model.pt` (the best single recipe: GRU h128x2,
wMSE, aux struct + next-change heads, p²+0.03 weights; devB 0.6736), run on CPU with the exact
torch weights.

| id | probe | data | result |
|---|---|---|---|
| D1 | **Memory horizon.** Predictions from a context of only K/2..K rows (two staggered runs that reset the state every K rows), compared with the full 20k-row stream on scored rows | 11 valid seqs (3 class 0, 3 class 1, 2 class 2, 4 class 3) | Output corr with the full stream: K=64 0.990-0.997 (classes 0-2), 0.940-0.991 (class 3); K=256 0.997-0.9995 / 0.978-0.996; K=1024 ≥0.9992 / 0.994-0.9987; K=4096 ≥0.9991 everywhere. Per-sequence WP change at K=256: classes 0-2 within ±0.007; class 3 +0.009, −0.011, −0.023, −0.049. **Classes 0-2 need ≤256 rows of memory; class 3 uses up to ~1000** |
| D2 | **Gate timescales** τ = 1/(1 − mean update gate) per unit | 6 seqs, all classes | Layer 1: median τ 2.8-4.0 rows, 90th pct 11-29, 1-4% of units with τ>100. Layer 2: median 1.8-6.9, 90th pct 40-2146, **6-34% near-frozen units (τ>100)**, most in classes 2-3. Per-sequence mean state (32 seqs, 8/class): 93-96% of its variance is between classes. The net is a fast event filter plus a class/regime latch |
| D3 | **Recurrent rank.** SVD-truncate W_hh of both layers post hoc | 7 seqs | r=16: output corr ≥0.9984; r=32: ≥0.9996; r=64: ≥0.9999; per-seq WP change −0.004..0 (r=16), −0.006..+0.001 (r=32; single-sequence WP is noisy, the −0.006 is a class-3 sequence at output corr 0.9997). W_ih at r=32: corr 0.9978-0.9991. **Input projection at r=32: corr 0.987-0.997, WP −0.019..+0.025 per seq.** The input layer is the only place where the model uses rank |
| D4 | Weight spectra | checkpoint | Stable rank ‖W‖²_F/‖W‖²_2 of W_hh gate blocks 2.8-11 (update gate lowest); top-32 singular values hold 74-88% of the energy of each 128×128 block |
| D5 | Post-hoc weight quantization | 7 seqs | int8 per-row symmetric (every matrix): output corr ≥0.99998, WP change ≤0.0006; bf16 weights: corr 1.00000 |
| D6 | Rows identical to the previous row | 7 seqs | 0.0% over all 112 columns; 0-1.2% if a0-a7 are ignored. Event-driven skipping saves nothing (agrees with scout P1) |
| D7 | **Member diversity.** Residual (y − p) correlation with ab30, \|y\|-weighted, scored rows, mean of t0/t1 | 96 random valid seqs, 187k rows, cached `pred_*_valid.npy` (mmap) | seed 4 (ab23) 0.9945, OOF train+valid (ab26) 0.9949, ab16 0.9904, ab18 0.9911, old 1-epoch mix-loss ab9 0.982, **organizer baseline 0.956**, GBM member 0.953, unweighted GRU ab12 0.918. Class 3: seeds 0.997, baseline 0.976. 50/50 blend gain over ab30 on these rows: seed +0.0006, baseline +0.0010, GBM −0.0020 |
| D8 | **GEMV microbenchmark** (C, AVX2, one pinned core, 384×128 = one GRU W_hh), hot cache and with an 8 MB sweep between calls (a proxy for the Python callback between rows) | synthetic | Native-style kernel (one 16-lane accumulator chain per output row): 4.45 us hot / 6.59 us cold (22 / 15 GFLOPS). Four rows interleaved (4 independent FMA chains, one hadd tree): 2.79 / 5.80 us. **int8 weights + 7-bit dynamically quantized input via `maddubs`/`madd` into int32: 2.60 / 2.29 us** (38-43 GFLOPS), output rmse 1e-3 at output scale ~0.3. The native kernel is FMA-latency-bound when hot and bandwidth-bound when cold, not FLOP-bound |
| D9 | Rows from a scored row to the 1st and 4th next i0 price-code event; to the next target change | 6 seqs per class | 1st event median 2/5/7/32 rows (classes 0/1/2/3); 4th event median 7/18/25/73, 90th pct 26/80/163/421; next target change median 3/7/8/33 |
| D10 | Hygiene | cache | **`cache/pred_ab25_onlinenorm_valid.npy` is all zeros** (max \|p\| = 0 in all 1873 sequences), and `cache/seqsums_ab25_onlinenorm.npz` exists. Any `decide.py` comparison against ab25_onlinenorm is invalid. The devB number from its training log (0.6702) is unaffected |

## 1. Diagnosis: why the architecture saturated

**1.1 The trained GRU is a low-rank, short-memory event filter with a class latch.** D1-D3 give the
mechanism behind the flat sweeps (E-081/E-083). The network does four things:
1. it decodes the nonlinear book codes row by row, which is the only full-rank computation (D3);
2. it filters events over about 3-30 rows (D2);
3. it latches a class/regime code (D2);
4. in class 3 only, it uses context out to about 1000 rows (D1).

Widths 192/256 add recurrent dimensions the dynamics do not use (effective rank 16-32). A second
layer adds latches that one layer can also hold (E-081: one layer 0.6675 vs two 0.6669). LSTM
adds a memory cell the task does not need. TBPTT chunks of 4000 rows add gradient paths into a
horizon that carries little signal and halve the optimizer steps (E-083: −0.003).

**1.2 The remaining error is shared by the whole family.** In D7 every variant of our recipe has
residuals correlated 0.990-0.995 with ab30: seeds, 1→2 epochs, aux heads, the propensity floor,
OOF train+valid data. Each of them moves less than 1% of the error variance. Only a different
family decorrelates: the organizer GRU (0.956), which uses raw inputs and was trained by someone
else, and the GBM (0.953), which is too weak to help. This agrees with scout S1-S3: the error is
information-limited and cross-domain (E-094). Repeating the recipe can add about +0.001 in total.

**1.3 What architecture can still do.** Capacity, cell type and depth are not levers. Three
things are:
- (a) Make members cheap enough that different families fit the budget.
- (b) Build inductive biases whose errors correlate ≤0.97 with the GRU at ≥0.67 WP. The baseline
  sits at (0.956, 0.6728); a new member has to beat that point.
- (c) Change what the fixed-capacity model is asked to fit. Every past win was of this kind:
  p² weights +0.05, the floor +0.002, aux-next +0.0014, distillation +0.0013.

**1.4 Latency binds less than assumed.** The profiling note calls inference FLOP-bound at 12-15
GFLOPS. D8 says otherwise: the kernel is limited by one dependent FMA chain per output row (hot)
and by bandwidth (cold). int8 plus multi-row blocking gives 1.7-2.9x per GEMV. One layer and a
rank-32 W_hh halve a member's MACs (256k → 125k). Together, a member costs roughly a quarter of
today's.

## 2. Budget arithmetic

Planning factor: int8 blocked kernels run 2x the current per-MAC speed (D8 measured 1.7-2.9x).
The limit is ≤ 75 us/row in local `docker --cpus 1`. This follows from the E-077 ratio of
0.48-0.68 LB-minutes per local us, the 60-minute limit, and a 15% margin.

| component | MACs/row | now (us) | int8 + blocking (us) |
|---|---:|---:|---:|
| features + PyO3 + scorer overhead | – | ~10 | ~10 (one fused call for all members) |
| ours, GRU h128×2 (inp 27k + 4×49k + head 33k) | 256k | 31-35 | ~15 |
| organizer GRU h128×2 on 112 raw (43k + 3×49k) | 190k | ~52 (ONNX; 94.5 − 42.3, E-085) | ~11 (native, exact conversion exists) |
| GRU h128×1, W_hh rank 32 + diag | 125k | ~16 | ~7 |
| minGRU 3×h128 (A3) | ~150k | ~19 | ~9 |
| additive group-GRU 6×h32 (A5) | ~55k | ~7 | ~3.5 |

Example: ours (15) + baseline (11) + three cheap members (3×7) + overhead (10) = 57 us, against
94.5 us for today's two-member blend.

## 3. Proposals, ranked by expected LB gain × probability / cost

| rank | id | family | cheapest test | predicted Δ (full valid) | inference cost | unconventional |
|---|---|---|---|---|---|---|
| 1 | A1 | int8 multi-member engine (enabler) | R1 (5 min GPU) + Rust bench | 0 alone; enables A3-A5/A7 | −50% | |
| 2 | A2 | TBPTT chunk vs step count | 3 runs, existing flag | +0.0005..0.0015 | 0 | |
| 3 | A4 | view-diverse member (organizer view, our recipe) | 1 run + blends | +0.0005..0.0015 as 3rd member | ~11 us | |
| 4 | A3 | cell-diverse member (minGRU / TCN) | 1 run + blends | +0.0005..0.0015 as 3rd member | ~9 us | |
| 5 | A5 | additive group-GRU with summed partial forecasts | 1 run + blends | +0.001..0.002 blend with ab30 | ~3.5 us | ★ |
| 6 | A8 | event-clock lookahead teacher, tilt-preserving targets | 2 teachers + 1 student | +0.0005..0.002 | 0 | ★ |
| 7 | A6 | deep row encoder + 1-layer rank-32 recurrence | 1 run | ±0.001 at −28% MACs | −28% | |
| 8 | A7 | class-routed specialist fine-tunes | 4 × 5 min | +0.0005..0.0015 | +1 member | |
| 9 | A10 | multi-objective BatchEnsemble as a skinny GEMM | 1 run | +0.0003..0.001 | 1.5x for 4 | ★ |
| 10 | A9 | damped signed-event accumulators (event lexicon) for class 3 | CPU lift probe first | 0..+0.002 | <1 us | ★ |
| 11 | A11 | instrument-adversarial state (gradient reversal) | 1 run | −0.001..+0.001 | 0 | ★ |
| 12 | A12 | real-time state decay (GRU-D) | 1 run | ±0.001 | ~1 us | |

A realistic total from this list is +0.002..0.004 on full valid. Most of it would come from a
3-4 member blend (A1 + two of A3/A4/A5) and from A2/A8. That total is additive with the
mask/selection levers in `opus_independent.md` (M1-M3), because those change the weights, not
the architecture.

### A1. Native int8 multi-member engine (enabler)
- **Source.**
  - N-BEATS §3.4 p.5 [read]: "ensembling is a much more powerful regularization technique than
    ... dropout or L2-norm penalty. The addition of those methods improved individual models,
    but was hurting the performance of the ensemble. The core property of an ensemble is
    diversity." Diversity comes from three losses, six input windows (2H-7H) and bagging.
    App. B.2 p.15, Fig. 3: 18 members are nearly as good as 180.
  - Brownlee, *Deep Learning for Time Series Forecasting*, ch. 10 §10.6-10.8 pp.167-170 [read]:
    combinations (Comb, Theta) dominate the M-study; LSTMs "report better model fitting but worse
    forecasting accuracy".
  - XTREND (Fieberg et al.) §2.3 eq. (7) p.10 [read]: an equal-weight combination shrinks slopes
    by 1/J; §5.2 p.24: "regularization and forecast selection does not help ... forecast
    combination is essential".
  - Hedges 2026, arXiv:2606.25986 [web, abstract]: for LOB models "latency is not merely noisy
    compute".
  - Our D3, D5, D8.
- **Mechanism.** Once members cost a quarter of today's, the budget holds the organizer GRU plus
  2-4 diverse members instead of a single extra full GRU.
- **Exact change** (`native/src/rnn_fast.rs`, `native/src/python.rs`):
  1. Store each matrix as int8 row-major with a per-row f32 scale and 32-byte-aligned rows.
  2. Quantize each layer input once per step to 7 bits (scale = max|x|/127) and move its sign onto
     the weights with `_mm256_sign_epi8`.
  3. Compute 4 output rows per iteration: `maddubs_epi16` + `madd_epi16` into int32, then one hadd
     tree per 4 rows. Keep activations in f32 with the existing `approx.rs`.
  4. Add a `W_hh = U Vᵀ + diag(d)` path (two small GEMVs, 32×128 then 384×32).
  5. Add an `Ensemble` PyO3 class: one `step(row)` computes the 99 event features once, runs K
     member nets (each with an input index list, so views can differ), and returns the weighted
     sum.
  6. Move the organizer GRU into the same engine through the exact baseline→RNNModel conversion
     (E-085, max abs diff 2.9e-5).
- **Cheapest discriminating test.**
  - R1: full-valid predictions of ab30 with post-hoc int8 weights and rank-32 W_hh (a modified
    state dict through `predict_run.py`, about 5 min GPU), then `decide.py compare`.
  - R2: Rust bench of the fused K-member step inside `docker --cpus 1` with
    `verify_submission.py` on `cache/valid_subset20.parquet`.
- **Predicted effect.** |ΔWP| ≤ 0.0003 from quantization plus truncation (D3/D5). Speed-up
  1.7-2.9x per member. The current blend drops from 94.5 to about 45-50 us/row.
- **Falsifier.** Speed-up < 1.5x in the docker bench, or |ΔWP| > 0.0005 on full valid.
- **Notes.**
  - Integer accumulation is exact, so v2 and v3 builds give bit-identical outputs, which float
    paths do not.
  - `maddubs` saturates at 32767. With |x_q| ≤ 127 and |w_q| ≤ 127 a pair sums to at most 32258,
    so do not use the full u8 range.

### A2. TBPTT chunk length versus optimizer step count (cheapest training-dynamics test)
- **Source.** D1 (≤256 rows of memory in classes 0-2); E-083 (chunk 4000 −0.003 at 1 epoch);
  E-083/E-084 (2 epochs +0.004, 3 epochs +0.0006). Kapturowski et al., ICLR 2019, R2D2
  [web]: replayed windows with a burn-in prefix recover recurrent state without stored state.
- **Mechanism.** Under a fixed OneCycle schedule, both the epoch gain and the chunk-4000 loss can
  be explained by the number of optimizer steps rather than by data passes or gradient horizon.
  Shorter chunks give more steps per pass and credit paths that match the memory the model
  actually uses. A second effect: each batch shows the same 32 instruments for 10 consecutive
  steps, which correlates the updates.
- **Exact change.** No code: run `--chunk 1000` and `--chunk 500` with the ab30 recipe (2 epochs).
  The discriminator is `--chunk 1000 --epochs 1`: same steps as ab30, half the data passes. If
  positive, follow up with random 1024-row windows plus a 256-row loss-free burn-in, 64 different
  sequences per batch (a loader change).
- **Test.** 3 runs of about 15 min, then `decide.py compare X ab30_eps_aux`.
- **Predicted effect.** chunk 1000 / 2 epochs +0.0005..0.0015. chunk 1000 / 1 epoch ≈ ab30
  (±0.001), which would mean the epoch gain was a step-count effect.
- **Falsifier.** Chunks 1000 and 500 both ≤ 0: the E-083 chunk effect was noise or specific to
  the horizon.
- **Latency.** 0.

### A3. A cell-diverse member: minGRU stack trained with a parallel scan (tree C4), or a TCN
- **Source.**
  - Feng et al. 2024, arXiv:2410.01201 [web]: minGRU drops the hidden-state dependence of the
    gates, so it trains with a parallel scan (175x faster at length 512).
  - Orvieto et al. 2023, LRU, arXiv:2303.06349 [gk].
  - N-BEATS App. B.1 pp.14-15 [read]: stacks with different bases win because their "function
    classes ... have small overlap".
  - Brownlee §24.5 p.517 [read]: heads that read the sequence at different resolutions, merged by
    a dense interpreter.
- **Mechanism.** D1-D3 say the task needs short-memory, low-rank dynamics. A stack of
  input-gated linear recurrences expresses that with a different inductive bias: no
  state-dependent gating, and exact long memory in its slow channels. Its errors should
  decorrelate more than seeds do (D7: 0.994) at similar accuracy.
- **Exact change.** Add `MinGRUModel` to `src/wnn/nn.py`:
  - input: Linear(211→128) + GELU;
  - three blocks of [minGRU(128): z = σ(W_z u), h̃ = W_h u, h = (1−z)h + z·h̃, a log-space scan
    within each 2000-row chunk with state carried across chunks] + residual;
  - the same head and aux heads, trained with the ab30 recipe.

  TCN alternative for the same slot: causal dilated conv (kernel 2, dilations 1..128, 64 channels,
  receptive field 256 as D1 suggests), run incrementally at inference from a ring buffer
  (8×2×64×64 = 65k MACs).
- **Test.** 1 run. Report WP alone, residual correlation with ab30 (D7 method), and
  `decide.py blend` for ab30 + member and for ab30 + member + baseline×0.25.
- **Predicted effect.** Alone 0.674-0.680. Residual corr 0.97-0.985. 3-way blend
  +0.0005..0.0015 over ab30 + baseline.
- **Falsifier.** Residual corr ≥ 0.99, or the 3-way blend ≤ the 2-way blend.
- **Latency.** About 150k MACs, ~9 us with int8 (A1).

### A4. A view-diverse member: our recipe on the organizer's input view
- **Source.** D7: the baseline is the only strong member that decorrelates. N-BEATS §3.4 p.5
  [read]: diversity from input windows and metrics. Brownlee §19.8 pp.385-386 [read]: a separate
  sub-model per input variable. XTREND §2.2-2.3 pp.8-11 [read]: information from another asset's
  indicators enters as its own forecast before the combination.
- **Mechanism.** Two things could explain why the organizer GRU decorrelates: its input view (raw
  112 columns, no event features) or its training (data, loss, no propensity weights). If the
  view is responsible, our recipe on that view gives a partner that is more accurate than the
  baseline (our weighting was worth +0.05 historically) and still decorrelated, so a strictly
  better blend partner.
- **Exact change.** Add `--inputs raw|full` to `train_rnn.build_inputs` (raw = the 112 columns
  only), train with the ab30 recipe, and initialize from the converted baseline weights in a
  second arm (`--init`). The level-free view is the shift scout's M6 (`opus_shift.md`); run it
  there, not here.
- **Test.** 2 runs (random init; baseline init), then residual corr and blends (member + ab30;
  member + ab30 + baseline) on chain folds at λ ∈ {0, 0.02}.
- **Predicted effect.** Alone 0.672-0.677, residual corr 0.96-0.97. The blend with ab30 is at
  least the baseline blend (+0.0017 full valid, E-084). As a third member +0.0005..0.0015.
- **Falsifier.** Residual corr with ab30 ≥ 0.985. Then the baseline's diversity comes from its
  training, not its view. In that case keep the baseline as is and spend the slot on A3/A5.
- **Latency.** Same as the baseline (190k MACs, ~11 us with int8).

### A5. ★ Additive group-GRU: per-group recurrent sub-models whose partial forecasts are summed
- **Source.**
  - N-BEATS [read]: §3.2 p.4 (doubly residual stacking; the forecast is the sum of partial
    forecasts); §3.3 pp.4-5 (interpretable stacks); p.6 (the N-BEATS-I+G ensemble); App. B.1
    pp.14-15 (different stacks have little functional overlap).
  - XTREND §2.3 eqs. (6)-(9) pp.9-11 [read]: univariate forecasts are fitted separately and
    combined.
  - Brownlee §19.8 p.385 [read].
  - Neural additive models, Agarwal et al. 2021, arXiv:2004.13912 [gk].
  - Our E-100/E-101: four additive closed-form terms reproduce 84-92% of the GRU's direction
    correlation, and each term comes from one feature group (trade flow, book shape, queue
    imbalance, i1 trades).
- **Mechanism.** The GRU's useful function is mostly additive over feature groups. Cross-group
  interactions are the part most likely to fingerprint instruments (E-094). An additive
  recurrent model keeps each group's dynamics but forms interactions only through a small joint
  channel. Its errors should therefore differ from the GRU's, and it should have lower variance
  under shift.
- **Exact change.**
  - Groups, each with its event deltas and counters: G1 i0 price codes (+Δ, held Δ,
    since_i0px); G2 i0 volume codes (+Δ); G3 i0 trade columns (+trade flags and counters);
    G4 all i1 columns (+i1 flags and counters); G5 a0-a7.
  - Each group: Linear(|G|→32) + GELU → GRU(32) → Linear(32→2), a partial forecast.
  - A joint level channel: Linear(211→32) → GRU(32) → Linear(→2).
  - Output = Σ partials. Aux heads read the concatenated states.
  - Implementation: one cuDNN GRU with hidden 192 (6 × 32), with fixed block masks applied to
    W_ih/W_hh (multiply in forward, or re-mask after each step).
- **Test.** 1 run. Report WP alone, residual corr, and blends with ab30 and with ab30 + baseline;
  per-class and novelty-tercile deltas (scout V2).
- **Predicted effect.** Alone −0.004..−0.010 vs ab30. Residual corr 0.96-0.98. Blend with ab30
  +0.001..0.002 over ab30 alone; as a third member +0.0005..0.001.
- **Falsifier.** Residual corr ≥ 0.99, or the blend gain ≤ the seed-pair gain
  (+0.0006..0.0009).
- **Latency.** About 55k MACs if zero blocks are skipped, ~3.5 us with int8.

### A6. Rebalanced single model: a deeper row encoder and a one-layer rank-32 recurrence
- **Source.** D2/D3 (only the input layer uses rank; the recurrence has effective rank 16-32;
  layer 2 is fast units plus latches); E-081 (one layer = two).
  - Livan, Novaes, Vivo, *Introduction to Random Matrices*, §14.3 pp.88-90 [read]: compare an
    empirical spectrum with the Marčenko-Pastur law; eigenvalues outside the bulk carry the
    structure and the bulk is noise. So size the recurrent rank by the spikes, not by the width
    (D4).
  - Martin & Mahoney 2018, arXiv:1810.01075 [gk].
- **Mechanism.** Move parameters from where the model does not use rank (the recurrence) to where
  it does (per-row decoding of nonlinear codes), and regularize the recurrence by construction.
- **Exact change.** Add `RNNModel(cell="lrgru")`:
  - encoder Linear(211→256) + GELU + Linear(256→128) + GELU;
  - one GRU layer h128 with W_hh = U Vᵀ + diag(d), r=32 (a custom cell).

  Cheaper variant with no custom cell: train one layer normally, SVD-truncate W_hh to r=32, then
  fine-tune U, V for 0.3 epoch.
- **Test.** 1-2 runs vs ab30.
- **Predicted effect.** ±0.001 WP at 185k MACs instead of 256k (−28%); +0.001 if encoder capacity
  binds (weak prior: heuristics and GBM lift were flat, E-088/E-089).
- **Falsifier.** Below −0.0015. Then keep two layers and only quantize.
- **Latency.** −28% vs today, ~11 us with int8.

### A7. Class-routed specialists with conditional compute
- **Source.** E-055: a class 0-2 expert trained from scratch gave +0.0025/+0.0009/−0.0014 in
  classes 0/1/2. E-075: the aux class head is 100% accurate. E-099: p² puts 60% of the loss mass
  on class 3. Jacobs et al. 1991, mixtures of experts [gk].
- **Mechanism.** The class is known almost immediately and never changes within a sequence. A
  specialist fine-tuned from the generalist on one class moves the shared compromise toward that
  class. Averaging generalist and specialist costs two member evaluations; the zip ships four
  specialists.
- **Exact change.** From ab30: four fine-tunes of 0.5 epoch, each on one class's sequences, lr
  2e-4, same weights. At inference, route by the class head's argmax from row 99 on; before that
  use the generalist alone (warm-up rows are not scored).
- **Test.** 4 fine-tunes of about 5 min, OOF on chain folds; per class, `decide.py compare` of
  0.5·generalist + 0.5·specialist against 0.5·ab30 + 0.5·seed4.
- **Predicted effect.** +0.0005..0.0015, mostly in classes 0/1.
- **Falsifier.** The routed blend is ≤ the seed-pair blend in every class.
- **Latency.** One extra member (~7-15 us with int8).

### A8. ★ Event-clock lookahead teacher with tilt-preserving targets (overlaps tree F3 / scout M9)
- **Source.**
  - Bi-LS-AttM (Xie et al., Appl. Sci. 2023, 13, 7916) §3.3-3.4, eqs. (9)-(11) [read]: a forward
    state from t=1 and a backward state from t=T merged by bilinear attention. Here the backward
    context is used only at training time, as privileged information.
  - Lopez-Paz et al. 2016, generalized distillation, arXiv:1511.03643 [gk].
  - E-097: a causal 3-model teacher gives +0.0013 and passes.
- **Mechanism and a caveat the other drafts miss.** With student target ỹ = E[y | 𝓕_τ], the
  noise in the student's target is Var(ỹ − f\*) = Var(y − f\*) − Var(y − ỹ). The privileged
  information removes only the variance it does *not* explain. A teacher that knows the first
  move's direction has ỹ ≈ y in class 3 (near-move AUC 0.96-0.99, E-082), so it removes almost
  nothing there. The useful privileged information is what happens *after* the first future price
  event: events 2-4 of the 4-event average in the label (E-079).
- **Exact change.** Let τ(t) be the first i0 price-code event after row t, a stopping time, so the
  tower property still holds. Targets are constant between events, so every row in [e_{k−1}, e_k)
  shares y and shares τ = e_k.
  1. Teacher: the same GRU, trained with loss only on event rows, where the label at row e_k is
     y(e_k − 1), the label of the interval that just ended. Its two heads per target predict
     E[|y|·y] and E[|y|].
  2. Broadcast the teacher's outputs back to each interval's rows.
  3. Student: loss weight ŵ = E[|y|] and target E[|y|y]/E[|y|], mixed with y by α = 0.5. The
     tilt is preserved, so the student's minimizer stays E[|y|y|x]/E[|y||x].
  4. Cross-fit the teachers on chain halves.

  D9 is why a fixed row shift does not work: resolution takes a median 7/18/25/73 rows by class.
- **Test.** 2 teacher runs + 1 student (about 45 min); `decide.py compare` against ab30 and
  against the distill3 student.
- **Predicted effect.** +0.0005..0.002, concentrated in classes 0-2, where F averages four moves
  over 7-25 rows.
- **Falsifier.** The teacher's WP on its interval labels is < 0.8 in classes 0-2 (the
  information is too weak), or the student is ≤ the distill3 student (E-097).
- **Latency.** 0.

### A9. ★ Damped signed-event accumulators: an event lexicon integrated at long timescales (class 3)
- **Source.**
  - Uhr, Zenkert, Fathi, "Sentiment analysis in financial markets", IEEE SMC 2014,
    `SentimentAnalysis.pdf` §II.B [read]. CIMAWA association combines an asymmetric and a
    symmetric co-occurrence part inside a text window with a damping factor of 0.4-0.6 (eq. 1).
    Sentiment is made dynamic over a moving window "similar to a moving average"; "fast or slow
    changing sentiment values can be easily created by adjustment of the time window" (eq. 2).
  - Orvieto et al. 2023 (LRU) [gk].
  - Bouchaud et al., *Trades, Quotes and Prices*, ch. 10, long memory of order-flow sign [gk].
- **Mapping.**
  - words → discrete book/trade event tokens;
  - lexicon polarity → a learned embedding per token;
  - association → co-occurrence of a token with the recent context;
  - window and damping → a bank of decays.
- **Mechanism.** D1: class 3 uses context beyond 256 rows. D2: the GRU's units are either fast
  (τ ≤ 30 rows) or frozen latches, with no well-trained integrators at τ = 100-5000. Scout P3:
  the class-3 per-sequence level correlates only 0.53 with the truth (0.84-0.90 in classes 0-2).
  A multi-timescale integrator of signed events is cheap and adds exactly that missing channel.
- **Exact change.**
  - Tokens per row (multi-hot): i0 bid/ask price up/down; bid- and ask-side queue up/down (sign
    of Δ of the E-090 ICA best-queue proxies); trade at bid or ask; i1 up/down.
  - Embedding E ∈ R^{tokens×8}, four decays τ ∈ {64, 512, 2048, 8192} rows:
    s_t = λ·s_{t−1} + (1−λ)·Σ E[tok] (32 channels), concatenated to the GRU input.
  - Optional association term: the bilinear product of the current token embedding with s_{t−1}.
- **Cheapest test (CPU, no GPU).** R5: a fixed-polarity version (±1 per token, the four decays)
  in `lift_probe.py` on ab30's class-3 residuals with chain folds. Only then one run.
- **Predicted effect.** Probe lift ≤ +0.002 in class 3; run 0..+0.002 globally.
- **Falsifier.** Probe lift < 1 SE in class 3. E-088 (trade-flow and rate families, no lift) and
  E-020 (level deviations ~0) are prior evidence against, so this ranks low.
- **Latency.** Table lookups + 128 FMAs + a 32×128 input block ≈ 4k MACs, < 1 us.

### A10. ★ Multi-objective BatchEnsemble run as a skinny GEMM
- **Source.** Wen, Tran, Ba 2020, BatchEnsemble, arXiv:2002.06715 [web]: W_k = W ∘ r_k s_kᵀ; "the
  speedup at test time is 3X and memory reduction is 3X at an ensemble of size 4" with accuracy
  competitive with full ensembles. Diversity caveat: arXiv:2601.16936 [web, title]. Our D8.
- **Mechanism.** Our GEMV is chain- and bandwidth-bound (D8). K members that share W turn K
  GEMVs into one W·[x₁..x_K] product: each weight load feeds K independent FMA chains, so four
  members cost about 1.5 members. BatchEnsemble members are usually more alike than seeds, so the
  diversity has to come from member-specific objectives (N-BEATS §3.4: members fit to different
  metrics).
- **Exact change.** Add `BERNNModel`: shared GRU and projection weights, per-member r_k, s_k on
  inp/W_ih/W_hh, per-member heads. Repeat the batch K times with member-specific loss weights:
  prop-eps ∈ {0, 0.03, 0.1, 0.03}, aux-next ∈ {0.3, 0, 0.3, 0.6}. At inference, a K-column int8
  GEMM.
- **Test.** 1 run (K=4): member-mean vs best member vs ab30; residual corr among members.
- **Predicted effect.** +0.0003..0.001 over ab30.
- **Falsifier.** Member residual corr ≥ 0.997, or the member mean ≤ ab30 + 0.0005.
- **Latency.** 1.4-1.6x one member for four members.

### A11. ★ Instrument-adversarial hidden state (low prior; see shift scout M5-M7)
- **Source.** Ganin et al. 2016, DANN, arXiv:1505.07818 [web]: a gradient-reversal layer makes
  features domain-invariant, shown on document sentiment domains (the SentimentAnalysis paper's
  field) and on images. E-094: train and valid separate with AUC 0.99 within every class.
- **Mechanism.** Strip instrument fingerprints from the state but keep the class/level
  information, which is legitimate.
- **Exact change.** A head on the layer-2 state predicts one of 16 within-class instrument clusters
  (k-means of per-sequence volume and a-column statistics) through gradient reversal, with λ
  ramped 0→0.1. The class head is untouched.
- **Test.** 1 run; the novelty-tercile delta (scout V2) is the readout.
- **Predicted effect.** Overall −0.001..+0.001; novelty tercile 0..+0.002.
- **Falsifier.** Novelty-tercile delta ≤ 0.
- **Why low.** The a5/a6 clocks are both drivers of the separation and informative, so an
  adversary may remove signal. Online z-scores, a crude invariance, were flat (ab25; re-check
  that comparison because of D10).

### A12. Real-time state decay (GRU-D style; low prior)
- **Source.** Che et al. 2018, GRU-D, arXiv:1606.01865 [gk]; E-091 (a6 = real time since the last
  book change, a5 = since the last trade).
- **Mechanism.** Labels live on the event clock, but events arrive in real time. A per-unit decay
  by elapsed time would let the state forget faster when the book is busy.
- **Exact change.** h ← h ⊙ exp(−softplus(w ⊙ φ(a6, a5) + b)) before each GRU step, where φ is a
  learned monotone 1-D map of the codes.
- **Test.** 1 run.
- **Predicted effect.** ±0.001.
- **Falsifier.** ≤ 0.
- **Why low.** The gates already receive a5/a6, and the fast units adapt within about 3 rows (D2).
- **Latency.** 128 exps, about 1 us.

### Considered and rejected, with evidence
- **Event-driven or skip updates.** D6 (0-1.2% unchanged rows) and scout P1.
- **Bilinear attention over the last k price events** (Bi-LS-AttM §3.4). It would retrieve
  near-move information (the pre-move queue state), where the GRU already has AUC 0.96-0.99
  (E-082). D1: 64 rows of context already recover most of the output. Keep it only as a
  diversity member if A3-A5 fail.
- **More width, depth, LSTM cells or longer TBPTT.** §1.1 explains why these are flat.
- **Structured readouts** (outcome-code CE with an analytic readout). Tested and lost
  (E-071/E-072).
- **Online test-time adaptation from rebuilt labels.** E-080 (it hurts); E-078's ceiling assumes
  labels that are not yet resolved at the lags it uses.

## 4. Specification critic: failure modes to check for every arm
1. **Seed noise.** Paired full-valid deltas carry about 0.001 of seed noise (scout P8). Run arms
   with predicted |Δ| < 0.002 with 2 seeds.
2. **Blend decisions must be OOF.** Fit blend weights on chain folds and report them at
   λ ∈ {0, 0.02}. Any fold model trained on valid (ab29_mask_f*) is in-sample on 3/4 of valid.
3. **Class 3 swings.** Per-sequence class-3 WP moves by ±0.03 under harmless perturbations (D1).
   Read class-3 effects only from pooled chain-fold deltas.
4. **Burn-in windows (A2 follow-up).** Latches trained on ≤1.3k-row windows may drift over 20k
   rows. Check WP by step bucket (scout P4) before adopting.
5. **Low-rank cells (A6).** Keep the diagonal term d. Otherwise the near-frozen units of D2 (update
   gate ≈ 1) lose their dynamics.
6. **int8 (A1).** Keep 7-bit activations so `maddubs` cannot saturate. Check the quantized engine
   against torch on full valid (R1), not only on synthetic data.
7. **New members must beat the baseline's point in the diversity/accuracy plane** (residual corr
   0.956 at 0.6728). A member that is more accurate but correlated 0.99 adds nothing (D7).

## 5. Requests (bigger than my limits)
- **R1** (A1/A6 gate, ~5 min GPU). Full-valid predictions of ab30 with (a) int8 per-row weights,
  (b) rank-32 W_hh, (c) both, through `predict_run.py` with a modified state dict; then
  `decide.py cache` and `compare` vs ab30.
- **R2** (A1). Rust: int8 4-row-blocked GEMV and a fused K-member `Ensemble.step`; bench inside
  `docker --cpus 1` with `verify_submission.py`.
- **R3** (A2). ab30 recipe with `--chunk 1000`, `--chunk 500`, and `--chunk 1000 --epochs 1`.
- **R4** (A3/A4/A5). One run each, then blends with ab30 and ab30 + baseline×0.25 on chain folds.
  Add the D7 residual-correlation readout to `decide.py`: it is the cheapest diversity metric and
  much less noisy than blend WP.
- **R5** (A9). `lift_probe.py` with fixed-polarity multi-timescale event accumulators on ab30's
  class-3 residuals, chain folds.
- **R6** (hygiene, D10). Regenerate `cache/pred_ab25_onlinenorm_valid.npy` and delete
  `cache/seqsums_ab25_onlinenorm.npz`; re-check any decision that used them.
- **R7** (decides A9's priority). D1 on full valid for K ∈ {256, 1024}: per-class WP of
  context-limited predictions (GPU, ~10 min). If class 3 loses < 0.003 at K=1024, long memory is
  not a lever and A9 drops.

## 6. Literature used (location → transferable mechanism → where used)

| source | location | mechanism | used in |
|---|---|---|---|
| Brownlee, *Deep Learning for Time Series Forecasting* (2018) | ch. 10 §10.6-10.8 pp.167-170 | combinations beat single ML models; LSTMs fit better and forecast worse | A1 |
| same | §19.8 pp.385-386; §24.5 p.517 | a separate sub-model per variable; heads at different resolutions merged by a dense layer | A3, A4, A5 |
| Oreshkin et al., N-BEATS (ICLR 2020) | §3.2 p.4, §3.3 pp.4-5, §3.4 p.5, p.6 (I+G), App. B.1-B.3 pp.14-16 | sums of partial forecasts; interpretable stacks; ensemble diversity from metrics, windows, bagging; synergy of stacks with small overlap | A1, A3, A5, A10 |
| Xie et al., Bi-LS-AttM (Appl. Sci. 2023) | §3.3-3.4, eqs. (9)-(11) | backward context and bilinear attention; here backward context is used as a training-only privileged teacher | A8; attention rejected |
| Fieberg, Liedtke, Schlag, Zaremba, XTREND | §2.3 eqs. (3)-(9) pp.9-11; §5.2 p.24 | cross-asset signals as separately fitted univariate forecasts; equal-weight combination (1/J shrinkage); "forecast combination is essential" | A1, A4, A5 |
| Uhr, Zenkert, Fathi, sentiment (IEEE SMC 2014) | §II.B eqs. (1)-(2) | lexicon polarity × damped windowed co-occurrence, fast/slow via the window | A9 |
| Livan, Novaes, Vivo, random matrices (arXiv:1712.07903) | §14.1, §14.3 pp.88-90 | spikes outside the Marčenko-Pastur bulk = structure; bulk = noise | D4, A6 |
| Web: Wen et al. 2020; Feng et al. 2024; Kapturowski et al. 2019; Ganin et al. 2016; Wood et al. 2024 (X-Trend, arXiv:2310.10500: cross-attention over a context set of regimes, zero-shot to new assets, not pursued because S1 caps the gain and the context bank would be train regimes); Prata et al. 2024 (LOBCAST, arXiv:2308.01915: all 15 LOB models drop sharply on new data, which matches E-094); Hedges 2026 | abstracts | see the proposals | A1-A3, A10, A11 |

## Appendix: probes (session scratchpad; rerun recipe)
- **`arch_diag.py RUN SEQS`.** Loads `model.pt` and prints SVD spectra (D4). Reruns the exact
  model on CPU for the listed valid row groups, first with `torch.nn.GRU` for the variants
  (low-rank W_hh/W_ih/inp, int8/bf16 weights), then with a float64 NumPy loop for the gates and
  the duplicate-row hold. Reports output corr vs the full model and per-sequence WP (D2, D3, D5,
  D6).
- **`memory_horizon.py RUN SEQS`.** Two resetting runs offset by K/2; each row takes the run with
  the longer context (K/2..K rows) (D1).
- **`latches.py`.** 8 sequences per class. Update gates of layer 2 from the layer-1 outputs and the
  previous state; between-class share of the per-sequence mean state (D2).
- **`member_diversity.py N`.** mmap of `cache/pred_*_valid.npy` for N random rows of
  `valid_targets.npz["row_group"]`; targets and masks from `SequenceFile` (D7).
- **`gemv/bench.c`.** `gcc -O3 -march=x86-64-v3`; `taskset -c 7`; optional 8 MB sweep between
  calls (D8).
- **`resolve_horizon.py`.** Price-code events from the first bid/ask price column (D9).

## Sources
- [N-BEATS, arXiv:1905.10437](https://arxiv.org/abs/1905.10437) (local PDF read)
- [BatchEnsemble, arXiv:2002.06715](https://arxiv.org/abs/2002.06715)
- [Is BatchEnsemble a Single Model?, arXiv:2601.16936](https://arxiv.org/abs/2601.16936)
- [Were RNNs All We Needed?, arXiv:2410.01201](https://arxiv.org/abs/2410.01201)
- [R2D2, ICLR 2019](https://openreview.net/pdf/387fb2fcee8f74c53cf707a9856f40c458f33933.pdf)
- [DANN, arXiv:1505.07818 / JMLR 17](https://jmlr.org/papers/v17/15-239.html)
- [X-Trend, arXiv:2310.10500](https://arxiv.org/abs/2310.10500)
- [LOBCAST, arXiv:2308.01915](https://arxiv.org/abs/2308.01915)
- [Hedges 2026, arXiv:2606.25986](https://arxiv.org/abs/2606.25986)
- [Introduction to Random Matrices, arXiv:1712.07903](https://arxiv.org/abs/1712.07903) (local PDF read)
