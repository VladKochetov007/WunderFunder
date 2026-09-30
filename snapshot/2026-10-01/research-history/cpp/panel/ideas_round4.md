# Ideator round 4: hypothesis cards (2026-09-27 03:20 EEST)

Role: ideator (conduct-creative-research). I committed these cards before any `research/panel/critic*.md` existed.
Tier B: chain-grouped folds, OOS models. I trained nothing. Four read-only CPU probes ran under `scripts/limited`
(WNN_CPU_PCT=6, 1 thread, 3G) on cached arrays. The scripts are in the session scratchpad at `ideator_r4/probes.py`
and `p4.py`. Literature: 3 random files from ~/Documents/Trading (Wooldridge *Introductory Econometrics*;
`present.pdf`; *How to DeFi: Advanced*), plus Kalman/robust-estimation skims and arXiv. Only Wooldridge ch. 8/14
transferred: WLS efficiency (P3) and the between/within panel split (P4). Cost unit: EMA-4 fold run ≈ 0.33 GPU-h,
2-epoch fold run ≈ 0.17 GPU-h, fold LP-FT ≈ 0.08 GPU-h (after the throughput work). E[ΔLB] = P(pass) × LB gain if it
passes, taking LB = 0.5-0.8× the fold gain (E-163/E-165).

## 1. New measurements (these drive the ranking)
| id | probe (OOS: ab56_ema_f{k} on its own held-out chain fold k; Q-weights \|clip y\|·scored) | result, fold 0 / fold 1 |
|---|---|---|
| P1 | **TD target** T = p(j) with j = last row before the next label change (the label is identical on (t, j]) | covers **85% of Q-weight**. Noise Var(T−p_t)/Var(y−p_t) = **0.30 / 0.29**. corr(y, p_t) 0.63 → corr(y, T) **0.83**. Slope y~p_t 1.04, T~p_t 0.92 (T about 11% under-dispersed) |
| P1b | same with a fixed k: 8 / 32 / 128 rows | Q-share 0.53 / 0.34 / 0.15. Noise ratio 0.05 / 0.10 / 0.12. Slopes matched (k=32: 0.72 vs 0.72) |
| P2 | label episodes per sequence, n_eff = (Σw)²/Σ_ep W², training weight \|y\|(p²+.03); × causal m = 1/(1+since_last/c) | class 3 median **39.5 → 99 (c=32), 63 (c=128)**. Classes 0-2: 97-161 → 147-181 (c=32). With metric weights, class 3: 28.8 → 53.7 |
| P3 | residual variance by causal bins (Wooldridge WLS): \|D\| deciles; since_last | WLS efficiency factor 1.04 / 1.04 / 1.19 / 1.04 by \|D\| and 1.04 / 1.06 / 1.02 / 1.12 by since_last (classes 0-3). Class-3 s² rises from 0.6 to 1.6 with stretch age. **Not a standalone lever** |
| P4 | between-sequence, within-class slope of y on p (panel "between" regression) | ref 2-ep 1.37/1.34 (f0), 1.29/1.24 (f1). **EMA 1.46/1.34, 1.38/1.34**. conv 1.37/1.31. EMA and conv leave the sequence-level component **~25-30% under-dispersed** |

Reading. The EMA win says label-noise memorization binds (gap_closer M1). P1 gives a way to remove about 70% of that
noise on 85% of the metric weight, without any target law, in every class. The P1 slopes say the bias is small and
roughly uniform, and Pearson is invariant to a uniform scale.

## 2. Ranking by E[ΔLB] per GPU-hour
| # | card | bucket | P | LB gain if pass | GPU-h | E/GPU-h | inference | code |
|---|---|---|---|---|---|---|---|---|
| 1 | B1b TD-τ targets in the EMA-4 main fit | b | 0.40 | +0.0010..0.0020 | 0.67 | 9e-4 | 0 | ~40 lines |
| 2 | B2 causal uniqueness weights | b | 0.35 | +0.0005..0.0010 | 0.34 | 8e-4 | 0 | ~10 lines |
| 3 | A2 event-clock FIR front-end (retry E-056) | a | 0.25 | +0.0005..0.0015 | 0.34 | 7e-4 | +~1% | ~60 + Rust |
| 4 | B1a TD targets in the valid LP-FT (gate for B1b) | b | 0.40 | +0.0003..0.0008 | 0.33 | 6e-4 | 0 | shares B1 |
| 5 | A3 class-routed heads in LP-FT (retry E-109/E-127) | a | 0.25 | +0.0003..0.0006 | 0.16 | 6e-4 | +1 head | ~40 + Rust |
| 6 | C3 fuse the baseline's hidden state into our head (valid LP only) | c | 0.20 | +0.0004..0.0010 | 0.20 | 6e-4 | +16k MAC | ~50 + Rust |
| 7 | B3 dilated conv bank with chunk-state carry | b | 0.30 | +0.0003..0.0010 | 0.34 | 5e-4 | +9% | ~40 + Rust |
| 8 | B5 self-supervised next-event heads | b | 0.20 | +0.0003..0.0008 | 0.34 | 3.5e-4 | 0 | ~50 |
| 9 | B4 slow channel (sequence-level aux + slow pool) | b | 0.20 | +0.0003..0.0008 | 0.34 | 3.5e-4 | +3% | ~50 + Rust |
| 10 | A1 weight-dropped GRU under EMA-6 (retry E-156/E-083) | a | 0.30 | +0.0004..0.0010 | 1.0 | 2e-4 | 0 | ~25 |
| 11 | A4 augmented mean teacher (retry E-118/E-154) | a | 0.25 | +0.0003..0.0008 | 0.67 | 2e-4 | 0 | ~30 |
| 12 | C2 chain-contrastive instrument embedding + FiLM | c | 0.10 | +0.0005..0.0015 | 0.55 | 2e-4 | +8k MAC | ~120 + Rust |
| 13 | C1 last-layer gradient-alignment reweighting (L2RW-lite) | c | 0.10 | +0.0005..0.0015 | 0.80 | 1e-4 | 0 | ~40 |

Suggested order before noon. Wave 1: B1b f0, B2 f0, A2 f0 in the queue, plus the LP-FT batch (B1a f0/f1 with its
control, A3, C3). Wave 2: fold-1 and seed replications of the fold-0 passers, then B3. Wave 3: B1 + B2 stacked in
one EMA-4 final. Every flag named below is new unless stated; REC = the q22 recipe (`artifacts/logs/q22A.sh`).
EMA4 = `REC --epochs 4 --ema-teacher 1.0 --ema-decay 0.999 --eval-ema`. The controls already exist:
ab56_ema_e4_f0 / ab58_ema_e4_f1 (EMA-4) and ab46_ref_f0/f1 (2-epoch).

## 3. Cards

**B1b. TD-τ: k-step bootstrapped targets from the EMA teacher** [b; label denoising / temporal difference]
- Mechanism. The label is constant on (t, τ), where τ is the next label change. So E[y_t | F_j] with j = τ−1 is a valid
  target by the tower property. The causal EMA teacher at row j estimates it. It keeps the resolvable information
  (corr 0.83 vs 0.63) and drops ~70% of the race-outcome noise (P1). Keep the true \|clip y\| weight and swap only the
  target. If T is the tilted mean given F_j, then E[\|y\|T\|x] = E[\|y\|y\|x], so the minimizer is unchanged. This is
  Sutton-1988 TD for prediction: TD beats Monte Carlo when noise is correlated along the path (arXiv:2404.15518). The
  labels anchor the endpoints (at j = t the target is y), and TD carries the accurate near-event beliefs back to far rows.
- Not a rerun of E-097/E-116. Those distilled a *same-time* teacher, whose target carries no information beyond the
  student's own row. This teacher has seen up to τ−1 more rows. It is also not gap_closer C3, which needs the exact
  law and a Markov kernel and so works only for class 3.
- Change. `next_change_labels` (train_rnn.py:254) returns `nxt`, the first change strictly after t. In the chunk loop
  after :675: `j = min(nxt-1, s+chunk-1)-s`, `ok = j > t_local`,
  `T = gather(out_ema[...,:2], j).clamp(-2,2)`, `tgt = where(ok, (1-a)*yc + a*T, yc)`. Add
  `nn.weighted_mse_target(pred, y, tgt, valid)` with w = \|clip y\|·valid, which replaces :690. Flags
  `--td-alpha a --td-horizon tau|K --td-ramp 0.5`: a ramps up from 0 over the first half-epoch, while the teacher is
  still poor.
- Test. `EMA4 --valid-fold 0 --td-alpha 0.5 --td-horizon tau --td-ramp 0.5` vs ab56_ema_e4_f0, then fold 1 vs ab58.
  Arm B: a = 1.0. Readouts: per-epoch curve (TD should still rise at epochs 3-4), slice_compare by rows-to-next (gain
  in 9-128), P4 slope (TD predicts it falls toward ≤ 1.25).
- Predicted. Fold +0.001..+0.003 at λ 0.02. **Falsifier**: a = 0.5 below +0.0008 pooled over folds 0/1 (skeptic 80%
  threshold), or the slope rises (the teacher's shrinkage propagates). Cost 0.67 GPU-h. EMA forward already runs.

**B1a. TD targets in the valid LP-FT** [b; same family, fast gate]
- Mechanism. The FT is noise-limited too: ~1.4k sequences, FT-child seed spread 0.0008 (E-156). TD from the frozen parent
  denoises the FT targets. It also pulls toward the parent the way WiSE 0.5 does (E-147), where that is harmless.
- Change. `--td-teacher RUN` loads a frozen model and runs a no-grad forward per chunk, like the EMA path.
- Test. Head stage `--init ab56_ema_e4_f0 --train-source valid --valid-fold 0 --eval-source vfold --full-valid
  --mask-weight --loss wmse --aux-struct 0.3 --aux-next 0.3 --hidden 128 --epochs 1 --freeze inp,gru --lr 5e-4`, then the
  full stage at `--lr 2e-4`, each ± `--td-alpha 0.5 --td-teacher ab56_ema_e4_f0`. Same on fold 1 with ab58.
- **Falsifier**: TD-LP-FT is not above LP-FT on either fold. Caveat: the parent saw these rows, so its T carries some
  memorized noise and the test is conservative. Cost 0.33 GPU-h.

**B2. Causal uniqueness weights** [b; effective sample size]
- Mechanism. One label episode is replicated over every row of its quiet stretch. Late rows add nearly the same gradient
  noise again (AFML ch. 4 uniqueness). m = max(f, 1/(1+since/c)) is x-measurable, so the minimizer is unchanged.
  Only capacity moves. It lifts class-3 n_eff 2.5x at c=32 (P2), and the down-weighted rows are also the high-variance
  ones (P3).
- Change. After train_rnn.py:649: `since = expm1(inputs[..., N_FEATURES+FEATURE_NAMES.index("since_i0px")])`. Then
  `main_valid *= m`, rescaled so that Σmain_valid per batch is unchanged. Apply it before the aux weights (:656). Flags
  `--since-weight c --since-floor f`. Note: ev since_i0px uses the both-quote flag, and P2 used rte (any of cols 0-21).
- Test. `REC --valid-fold 0 --since-weight 32 --since-floor 0.25` vs ab46_ref_f0, then f1. Arm c=128. Slice by
  rows-to-next: expect a gain at 1-32 and a loss capped at 129+.
- **Falsifier**: below +0.0003 pooled, or the 33+ buckets lose more than the near buckets gain. Cost 0.34 GPU-h.

**A2. Event-clock FIR front-end** [a; retry E-056 dual clock (flat); representation / time]
- Escape. E-056's event GRU fed only the head. It ran in the old recipe (1 epoch, devB, no floor/aux) and had to
  learn event dynamics from scratch. E-168 now shows a learned local FIR helps (+0.0018). Class 3 uses ~1000 rows of
  context (opus_arch D1), which is ~16-30 price events: an event-indexed FIR covers that range for the price of a
  16-row one.
- Change. `--enc evconv --conv-k 16`: depthwise conv over z at the last 16 ev_i0px rows (the gather and scatter of
  `DualClockModel.forward`), held until the next event, added residually to z. The state carries (B,16,H) across chunks.
  Rust: ring buffer, updated only on event rows.
- Test. `REC --valid-fold 0 --enc evconv` vs ab46_ref_f0 and ab61_conv_gru_f0; residual corr. Expect the gain in class-3
  33-128-row buckets. **Falsifier**: ≤ conv16 on fold 0 with no class-3 far-row gain. Cost 0.34 GPU-h.

**A3. Class-routed heads, LP stage only** [a; retry E-109/E-127 (per-class calibration and blend weights refuted), A7]
- Escape. E-109/E-127 fit per-class affine maps by per-segment criteria, which pooled WP charges for (algebra §1). Here
  four nonlinear heads are trained jointly on the pooled wMSE, so the objective stays the pooled one. They are trained
  only where the FT gain lives (E-148), with an L2 tie to the shared head against the 4-way data split.
- Change. `--class-heads --head-tie 1e-2`: at training, route by `sequence_class(y)`; at inference, by argmax of the
  aux class logits (100%, E-075). The heads start as copies of the parent head.
- Test. B1a's LP-FT commands + `--class-heads` vs the B1a control. **Falsifier**: ≤ control on either fold, or class 3
  loses (E-127 pattern). Cost 0.16 GPU-h. Rust: one extra head eval, ~+1.5 min.

**C3. Representation stacking with the organizer baseline (crazy)** [c; fusion]
- Mechanism. The package already computes the baseline's layer-2 state (member 2, residual corr 0.956). A 0.35 linear
  blend is the weakest way to use it. Our head can read h_base directly. Train the fusion **only on valid** (OOS for the
  baseline; on train it has an in-sample edge, ml_lead M3) in the LP stage: head.0 widened by 128 zero-init columns, h_base
  from an fp32 baseline forward (audit 2.2: the fp16 copy errs by up to 8.5).
- Test. LP-FT ± fusion, both then blended 0.65/0.35 with the baseline. **Falsifier**: fused blend ≤ plain blend on either
  fold, or residual corr with the baseline > 0.985. Cost 0.2 GPU-h. Rust: pass h_base between members.

**B3. Dilated conv bank with chunk-state carry** [b; representation / multi-scale]
- Mechanism. E-168's conv pads zeros at every chunk start (nn.py:71): 15 of 2000 training rows, and every 4000 rows
  in predict, while the native stream sees real context. Dilations 1/4/16 (k=16, RF ≈ 241, matching the D1 memory
  horizon) give multi-scale FIRs.
- Change. `--conv-dil 1,4,16` (branches summed before a shared `conv_pw`) and `--conv-carry` (state = (h, zbuf of the
  last (k−1)·d_max rows)), both in forward_chunk and predict. Test: REC f0 conv16+carry (isolates carry, expect ~0) and
  the dilated bank vs ab61_conv_gru_f0, then EMA-4 f0/f1. **Falsifier**: dilated ≤ conv16 + 0.0003. Cost 0.34 GPU-h.

**B5. Self-supervised next-event heads** [b; dense auxiliary supervision, L11]
- Mechanism. Labels give ~30-160 effective outcomes per sequence (P2). Input-clock events give thousands. The one
  shipped scout idea was an aux head (aux-next +0.0014); E-107 only added *label* horizons.
- Change. `--aux-ssl 0.3`: heads for log1p rows to the next ev_i0px (input clock), trade within 8 rows, i1 event within
  16 rows, and the sign of the next i0 volume-code change of the first ICA pair (E-090). Targets come from future
  inputs in the batch, training only. Test: REC f0/f1 vs ab46_ref. **Falsifier**: ≤ +0.0003 pooled. Cost 0.34 GPU-h.

**B4. Slow channel** [b; sequence-level de-shrinkage; algebra card C mechanism]
- Mechanism. P4: the sequence-level component is 25-30% under-dispersed, and EMA makes it worse (1.46). Early stopping
  and averaging shrink low-SNR directions, and the sequence level has n = #sequences. Post-hoc causal maps got ≤ +0.0004
  (algebra §3), so the fix has to be in training. Option: causal EMAs (τ 500, 4000 rows) of a 32-dim projection of the
  layer-2 output feed the head, plus an aux head on the forward 2000-row Q-mean of y (lower noise per sample).
- Change. `--slow-pool 500,4000 --aux-slow 0.3`. The target comes from reverse cumsums of \|y\|y and \|y\| over valid rows.
- Test. REC f0/f1. **Falsifier**: the P4 slope stays ≥ 1.25 and Δ ≤ 0. Cost 0.34 GPU-h. Stack with B1 only if both pass.

**A1. Weight-dropped GRU under a longer EMA run** [a; retry E-156 dropout 0.1 and E-083 "epochs > 2 flat"]
- Escape. E-156 used activation dropout between layers at 2 epochs, and E-154 shows every regularizer loses at 2 epochs
  (EMA −0.0011) and pays only in longer runs (E-153/E-164). DropConnect on W_hh (AWD-LSTM, arXiv:1708.02182) targets
  the recurrent latches, where sequence-identity memorization can live (D2: 6-34% near-frozen units). It keeps cuDNN,
  with one mask per chunk.
- Change. `--weight-drop 0.2`: before each forward_chunk, `functional_call(model, {"gru.weight_hh_l{0,1}": W*mask/0.8})`.
- Test. `EMA4 --epochs 6 --weight-drop 0.2` vs EMA-6 without it (the gap_closer C1 control), f0 then f1.
  **Falsifier**: ≤ control + 0.0003 on either fold. Cost ~1.0 GPU-h.

**A4. Augmented mean teacher** [a; retry E-118 online norm (flat) and E-154 consistency without perturbation]
- Escape. E-118 removed level information deterministically. E-154's student and teacher saw identical inputs, so the
  consistency term acted only as a proximal pull. Mean Teacher's gain comes from perturbing the student
  (arXiv:1703.01780). Here the perturbations run along the train/valid shift axes of E-094: volume-code means and i1 stats.
- Change. `--aug-groups i1,vol --aug-p 0.15 --aug-shift 0.2 --cons-rows all`. The student's inputs (after the flags
  are computed) get the group set to model.mean with probability p, and per-sequence offsets of 0.2·std. The teacher
  stays clean. Consistency is weighted by `valid`, not `main_valid`.
- Test. EMA4 f0/f1 vs ab56/ab58. Readouts: train-unlike tercile (E-129), stress λ .05. **Falsifier**: ≤ 0 overall and
  no train-unlike gain. Cost 0.67 GPU-h.

**C2. Chain-contrastive instrument embedding + FiLM (crazy)** [c; explicit regime identity]
- Mechanism. The 3,288 chains (E-095) are free labels for "same instrument, same period". An InfoNCE encoder over the
  causal prefix (GRU h32 on raw rows; positives = same chain, negatives = other chains of the same class) gives z_t.
  FiLM (z ← z(1+γ(z_t)) + β(z_t)) hands the GRU a de-noised regime code, attacking P4 from a new direction.
- Change. `scripts/train_chain_encoder.py` + `--film-enc RUN`. Test: REC f0; P4 slope and Δ. **Falsifier**: the encoder's
  held-out chain retrieval top-1 < 0.3, or Δ ≤ 0. Cost 0.55 GPU-h.

**C1. Last-layer gradient-alignment reweighting (crazy)** [c; meta-reweighting, Ren et al. 2018 arXiv:1803.09050]
- Mechanism. Upweight train rows whose head gradient aligns with the gradient on true-mask valid rows in the same batch.
  This is selection and denoising in one step, and it is cheap because the last-layer gradient is w(p−y)⊗φ. It needs no
  double backward, which cuDNN RNNs lack.
- Change. `--l2rw-eta 0.5`: m = clip(1 + η·a/scale, 0, 2), applied detached to train rows only. Risk: the weights depend
  on the labels, which biases the minimizer (E-103). Test: EMA4 f0. **Falsifier**: ≤ 0, or the gain appears only on
  unscored rows. Cost 0.8 GPU-h.

## 4. Closures, stacking and cautions
- Closed by P3: inverse-variance (WLS) row weights as a standalone lever. Efficiency factor ≤ 1.19, ≤ 1.12 in class 3.
  B2 already captures the stretch-age variance axis.
- Closed by E-146: a runtime buyback (distilling the baseline into h64×1) to fit a 3rd member. The slot is worth only
  +0.0001..0.0002 on the holdout, below the fidelity risk.
- B1 and B2 both attack the label-noise channel. Expect sub-additivity: test B1+B2 together only after each passes
  alone. Under TD targets (B1), extra width (E-083 flat) may start to pay, because memorization is what caps capacity.
  That is a follow-up, not a card: the h>128 backward is slow (throughput F9).
- Every card with a Rust note needs a stream-vs-batch parity check (corr ≥ 0.9999, E-143) before any LB slot.
  B3/A2 must carry conv state, or the training/serving mismatch grows with the receptive field.
- Sources (primary unless marked): Sutton 1988, *Learning to predict by the methods of temporal differences* [gk];
  arXiv:2404.15518 (TD for supervised learning with correlated noise) [web]; arXiv:2606.20357 (variance of TD) [web
  listing]; arXiv:1708.02182; arXiv:1703.01780; arXiv:1803.09050 [gk]; López de Prado, AFML ch. 4 [gk]; Wooldridge,
  ch. 8 and 14 [read, TOC].
