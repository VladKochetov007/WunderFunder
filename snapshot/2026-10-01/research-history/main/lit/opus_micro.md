# Microstructure information scout (Opus, 2026-09-26)

Role: scout, ideator and specification critic for one question: which causal market-microstructure
quantities might a GRU on raw encoded rows still be missing, and how to compute each as an O(1)
streaming feature from the anonymized columns. Verification tier B. All numbers below are screening
evidence unless a full-valid chain-fold probe is cited.

Tags: **[read]** read in the primary source at the cited location; **[web]** checked online;
**[gk]** general knowledge, not re-read; **[ours]** measured in this session. Probes ran one
process at a time under `scripts/limited` with `WNN_MEM_MAX=2G WNN_CPU_PCT=10`, read 1-45 valid
sequences each through `wnn.data.SequenceFile`, and took under a minute. No training, no GPU, no
file over 400 MB loaded (GRU predictions `cache/pred_ab20_gru128_e2_valid.npy` were memory-mapped
and indexed per sequence). Probe scripts and the two lookup tables are in the session scratchpad,
`.../scratchpad/opus_micro/`.

I committed hypotheses H1-H11 before reading `research/tree.md` and `research/lit/*.md`. Afterwards
I dropped nothing, re-ranked two items, and marked where `opus_independent.md` already covers
ground (M10 race features, S2 class-3 ceiling).

## 0. New measurements [ours]

| id | question | data | result |
|---|---|---|---|
| N1 | what are `i0_dp0..3` | 2 class-3 seqs | dp is a 4-dim code of the **last trade price relative to the current quotes**. It changes on 4371 rows but the trade-volume code on only 1343, because every book move shifts the relative position. At mid-price events without a trade, the pair (dp_prev, dp_new) determines the decoded mid change on 2226/2226 and 2996/2996 events. On 12 seqs per class, the target jump at trade-free book events is 99.2% determined by the dp pair in class 3 and 85-88% in classes 0-2, about as well as the price-code pair (joint 86-90%). That gives a second, independent move decoder |
| N2 | trade side | 1 class-3 seq; 24 folds-0/1 seqs | each dp code fixes the sign pattern of the 4 dv columns: 256/256 trades on code A have pattern 0000, 239/241 on code B have 1101. A table from dp code to the sign of the next mid move within 50 rows gives \|score\| > 0.5 for 170 of 302 codes, a usable Lee-Ready substitute (Hautsch §3.1.5 [gk]). This is why symreg finds the *unsigned-looking* `tradevol_EMA16` directional (E-101): the dv sum carries the side |
| N3 | do i0 and i1 share the code dictionary | 4 seqs | yes. Class 3: all 8 of i0's bid-price codes also occur among i1's 45 (and 20); class 0: 217 of 530 overlap. A move table learned from i0 targets on 56 folds-0/1 seqs decodes i1 book transitions with 1-2% unknown in class 2, 15-25% in class 1, 14-37% in class 3 and 45-64% in class 0. **i1's price path in i1 ticks is recoverable with the i0 decoder** |
| N4 | i1 → i0 lead-lag, event level (corr of the i1 decoded mid move since the last i0 mid move with the sign of the next i0 move) | 10+5+5+5 seqs, folds 2-3 | class 2: 0.30/0.34, 0.04/0.21, 0.26/0.27, 0.27/0.26, 0.25/0.28 (half 1/half 2, all positive); class 1 ≈ 0.10; class 0 ≈ 0.07 with half of i1 transitions undecoded; class 3 ≈ 0.01, because only 1-9% of the intervals between i0 moves contain any i1 move |
| N5 | is N4 already in the GRU (scored rows, \|y\|-weighted, residual after a pooled linear fit of D=t0−t1 on the GRU's p0−p1) | 10 seqs per class | class 2: corr(f,D) 0.56, corr(f,P) 0.68, residual 0.063; class 1: 0.33 / 0.37 / 0.045; class 0: 0.08 / 0.13 / −0.05; class 3: 0.19 / 0.23 / 0.12 in one sample of 10, 0.02 in a fresh sample of 24. The GRU already learns most of the lead from raw i1 codes |
| N6 | a-column semantics (corrects E-091 for a0/a1/a4) | 3 seqs | **a0 and a4 encode the per-row real-time increment Δt.** On rows with no book change, Spearman(Δa6, a0) = 0.75-0.89 and (Δa6, a4) = 0.71-0.90; a1 0.48-0.49; a7 −0.45 to −0.68 (an intensity). a0 has atoms (quantized clock). i0 price-change rows have short Δt (a0 median −1.1 vs +0.15 for volume-only rows: flicker bursts). a1 alternates (lag-1 autocorrelation −0.4 to −0.7, lag-2 positive), which fits paired trade/reset rows |
| N7 | is the class-3 sequence-level direction a persistent property | 40 class-3 seqs | no. The \|y\|-weighted mean of D in the first half vs the second half has corr −0.036. Each half's level has corr 0.60-0.64 with that half's realized mid drift. The class-3 level headroom (opus_independent P3: oracle level +0.034) is **realized drift**, so it cannot be learned online. This explains E-080 |
| N8 | feature screens vs GRU residual (class 3 unless noted; n = sequences; ± = sequence-cluster bootstrap sd) | 20-45 folds-2/3 seqs per draw | see §0.1 |

### 0.1 Screening table (residual = weighted corr with D − (a + b·P); "far" = ≥ 50 rows since the last mid move, 45% of class-3 weight)

Each cell is a separate random draw. In the MO0/MO1 rows, draws 3 and 4 exclude the earlier draws;
draws 1 and 2 may share a few sequences. In the other rows, draws with different seeds can overlap.

| feature (streaming definition in §2) | draw 1 (n=20-30) | draw 2 (n=40) | draw 3 (n=30-40) | draw 4 (n=45) | verdict |
|---|---|---|---|---|---|
| signed trade-flow EMA16, VNET-since-move, same-side run | resid 0.02 / 0.00 / −0.01; corr(f,P) > corr(f,D) | | | | captured by GRU (also class 1) |
| long-memory sign EMA, half-life 2048 trades | −0.068; −0.088 ± 0.032 | | −0.008 ± 0.044 | | inconclusive |
| price-level departure memory (keyed by decoded mid) | +0.085 (24) | −0.099 ± 0.062 (30) | | | sign flips, likely noise |
| iceberg proxy (trades absorbed at the current bid vs ask level) | −0.083 ± 0.041; corr(f,P) ≈ corr(f,D) | | | | mostly captured |
| **MO1 flow** (signed trades that moved the mid, EMA16 in trade time) | −0.132 ± 0.068 | −0.058 ± 0.042 (far −0.121 ± 0.063) | −0.066 ± 0.045 (far −0.125 ± 0.063) | +0.006 ± 0.045 (far +0.031) | pooled −0.051 ± 0.024; GRU corr with it ≈ 0 far from moves |
| **MO0 flow** (signed trades that did not move the mid) | +0.095 ± 0.051 | +0.138 ± 0.040 (far +0.182) | −0.022 ± 0.040 | | pooled +0.067 ± 0.025, heterogeneous |
| quote-caused mid moves (EMA16 over mid events without a trade at t or t−1) | | | | −0.102 ± 0.053 (far −0.115) | one sample |
| real-time decayed MO0 flow (decay driven by a0) | +0.045 ± 0.031 | | | | weak |

Lesson for the specification critic: class-3 residual correlations from ≤ 45 sequences swing by more
than their bootstrap sd between disjoint samples, because a few sequences dominate. With about 30
screens run, a couple of 2-sigma results are expected by chance. Only a full-valid, chain-fold probe
(697 class-3 sequences) can decide. §4 specifies it.

## 1. What a GRU on raw codes can and cannot compute (first principles)

The null results (E-088/E-089; N5; §0.1 row 1) say the GRU already forms EMAs, recency counters and
signed flows of anything visible in one or two consecutive rows. A GRU fails when it must do one of
the following:

1. **Use a lookup table it gets no direct supervision for.** i0 code-pair → tick move is supervised
   by target jumps. i1 moves (N3) and trade side (N2) are supervised only through weak downstream
   effects, and in classes 0-1 code pairs are numerous (22.5k keys in 12 class-0 seqs).
2. **Keep keyed (associative) memory**, for example "what happened last time the mid was at this
   level". A fixed-size recurrent state cannot address memory by a decoded price.
3. **Integrate decoded, additive quantities over reset windows**, such as net signed volume since the
   last price change (VNET, Hautsch §9.2.1 eq. 9.15, p.229 [read]) or queue depletion since the level
   formed.
4. **Match events across rows**: did the trade at row t cause the mid change at row t+1 (MO1) or not
   (MO0)? Was a price change caused by a trade or by a cancellation? That is the six-event taxonomy of
   TQP §14.4 (pp.275-282 [read]) and Eisler-Bouchaud-Kockelkoren 2012 ([arXiv:0904.0900](https://arxiv.org/abs/0904.0900) [web]).
5. **Reach horizons beyond its TBPTT chunk**, for example power-law order-sign memory over thousands
   of trades (TQP §10.1 eq. 10.2, p.188 [read]).
6. **Estimate per-instrument parameters online.** N7 removes the most tempting target, the class-3
   drift level.

Items 1, 2 and 4 are where microstructure theory still points at information the GRU cannot easily
reach. Items 3 and 5 screened as captured or inconclusive.

## 2. Causal quantities: status and O(1) streaming recipes

Shared primitives, all causal and O(1) per row:
- `book_ev(t)`: any of cols 0-21 changes. `trade(t)`: cols 48-51 change to a value other than the
  resting code, taken as the running per-sequence mode (0.025 in i0 class-3 seqs; i1 of class 0 rests
  at 0.076, so use the per-sequence mode rather than a constant).
- `dmid(t)` in half-ticks: `T_px[(codes 0-21 at t−1, at t)] → (d_ask, d_bid)` (`wnn.ticks.learn_moves`,
  fitted on **train**). Fallback when no trade is in the row: `T_dp[(dp_{t−1}, dp_t)] → Δmid` (N1).
  Otherwise 0, plus an `unknown` counter. For i1, apply the same tables to cols 52-73 / 96-99 (N3).
- `side(t) = T_side[dp code at t] ∈ [−1, 1]` (N2; fit on train).
- `Δt(t)`: monotone in a0 (and a4) (N6). A usable proxy is `exp(k·a0)`; an exact inverse needs the
  additivity fit in H2.
- Trade-time EMA: on trade rows only, `e ← e + α(s − e)` with `α = 1 − 2^(−1/h)`.

| quantity | likely in the GRU? | evidence | streaming recipe |
|---|---|---|---|
| OFI (Cont-Kukanov-Stoikov) at the touch | mostly | symreg `tanh(v0 − EMA32(v0))` (E-100), E-088 depth family flat | needs best-queue sizes; use ICA bid/ask factors (E-090): `e = Δq_b·1{bid ≥ prev} − q_b,prev·1{bid ≤ prev} − …`, then EMA in event time |
| signed trade flow, VNET, runs | yes | §0.1 row 1 | trade-time EMA of `side`; VNET = Σ side since the last `dmid ≠ 0` |
| MO0 vs MO1 split | **no** (GRU corr ≈ 0 far from moves) | §0.1; TQP §10.2 p.190 [read]: in large-tick stocks MO0 signs keep power-law persistence, while "the sign of market orders in MO1 is negatively correlated with that of the next market order"; §14.3 p.273 [read]: R_MO0(ℓ) grows with lag, R_MO1 plateaus at once for large ticks | when row t+1 arrives, label the trade at t MO1 if `dmid(t)≠0 or dmid(t+1)≠0`, else MO0; separate trade-time EMAs (h = 4, 16, 64) |
| quote-caused vs trade-caused moves (six-event) | probably not | §0.1 last rows; TQP §14.3 p.274 [read]: MO1 propagator larger but "substantially stronger impact decay" | event-time EMA of `dmid` split by whether a trade occurred at t or t−1 (better: in the same Δt≈0 message, H2) |
| propagator / long-memory sign flow | inconclusive | §0.1 row 2 | Σ_k w_k·EMA_k(side), h_k ∈ {64, 256, 1024, 4096} trades, w_k ∝ h_k^(−β), β ≈ 0.25 (TQP ch.13 [gk]) |
| Hawkes intensities by side and type | partly (a2/a3 are unsigned book intensities) | E-091, N6 | per type j: `λ_j ← λ_j·exp(−Δt/τ) + 1{event j}` with Δt from a0; types: buy/sell × MO0/MO1, bid/ask × cancel/add |
| queue race with activity asymmetry | partly | TQP §7.3.1 p.123 [read]: "for queues of the same size, the fastest-changing queue is more likely to empty first" | `log(λ_askchg/λ_bidchg)` in real time × ICA imbalance; overlaps opus_independent M10 |
| durations / excess-volume durations | weak prior | Hautsch §3.2.2 p.37 [read] | time (Σ Δt) to accumulate \|Σ side\| ≥ v̄ since the last reset; ratio to its running median |
| cross-instrument decoded path | partly (class 2 captured) | N4, N5 | `c1 = Σ dmid_i1`; `lead = c1(t) − c1(last i0 mid move)`; plus i1 MO flow |
| price-level memory | no, but the signal is unstable | §0.1 row 3 | hash map level → (EMA of departure sign, trades absorbed per side, visits); levels bounded (~15-50 per seq) |

## 3. Ranked hypotheses

Families: EV = event taxonomy, XI = cross-instrument, MEM = keyed memory, FL = flow/propagator,
RT = real time, QR = queue race, VOL = magnitude. **U** marks the unconventional ones (H2, H3, H4,
H10, H11).

**H1 (EV). Split trade flow by price impact: MO0 continues, MO1 reverses (class 3, far from moves).**
- *Mechanism.* In large-tick books a non-price-changing buy MO is part of persistent (split
  meta-order) flow and forecasts more buying and an eventual ask depletion. A price-changing MO
  exhausts one side, the new opposite queue is small and the flow inverts (TQP §10.2 p.190, §7.5
  pp.130-131 [read]). Pooled over all trades the two effects cancel, which is why the all-trade EMA is
  "captured" but useless beyond the GRU.
- *Feature.* `mo0_h`, `mo1_h` for h ∈ {4, 16, 64}, their contrast, and `mo0_since_move`.
- *Screen.* MO1 pooled −0.051 ± 0.024; MO0 +0.067 ± 0.025 (heterogeneous); far-from-move up to
  ±0.12-0.18 in two of three samples. The GRU's corr with `mo1_16` far from moves is ≈ 0.
- *Test.* `lift_probe.py --npy` (chain grouping), class-3 delta. Then a GRU arm with the 6 inputs.
- *Predicted.* +0.0005..0.002 global (class-3 direction corr +0.005..0.02).
- *Falsifier.* Class-3 lift ≤ 1 sd, or the sign of the MO1 coefficient differs between chain-fold
  halves.
- *Cost.* ~15 min CPU for features, 1 min per probe.

**H2 (RT, U). Rebuild exchange messages from Δt and classify the six events exactly.**
- *Representation change.* The rows are exchange events at quantized timestamps (a0 atoms, N6). Rows
  with Δt at the lowest atoms belong to one message: a sweep writes a trade row and a book row at the
  same time. Grouping rows into messages turns H1's one-row heuristic into exact MO0/MO1/CA0/CA1/LO0/
  LO1 labels (TQP §14.4 [read]; EBK 2012 [web]) and gives real-time kernels for Hawkes intensities.
- *Feature.* Six signed event EMAs in event time and in real time.
- *Test.* (a) Fit a monotone g with `g(a6_t) − g(a6_{t−1}) = h(a0_t)` on quiet rows (isotonic
  alternating fit) to validate Δt; (b) rerun H1 with message-based labels. H2 wins if its lift beats
  H1's by more than 1 sd.
- *Predicted.* H1 + 0..0.001.
- *Falsifier.* The Δt atoms do not separate trade→book pairs (share of trade-followed book rows at
  the lowest atom not above the base rate).

**H3 (XI, U). Decode i1 with i0's dictionary; give the GRU a shared (siamese) book encoder.**
- *Mechanism.* N3 shows both instruments use one global code map. The GRU has separate input
  weights, so it must learn i1's decoder from weak lead-lag gradients. That is the unsupervised-table
  failure of §1.1.
- *Feature path.* `lead_i1 = c1 − c1[last i0 move]`, i1 moves over the last 16/64 i1 events, and i1
  MO0/MO1 flow.
- *Architecture path.* One shared MLP embeds (code_t, code_{t−1}) for both price blocks, and the i0
  target supervises the i1 decoder through weight tying.
- *Screen.* Lead corr 0.25 (class 2), 0.10 (class 1), 0.07 (class 0, half undecoded); the GRU
  already carries most of it in class 2; residual +0.06 (class 2) and +0.045 (class 1) on 10 seqs.
- *Test.* Lift probe on classes 0-2 with a train-fitted move table (measure the unknown rate first).
  Then one GRU arm with tied encoders.
- *Predicted.* +0.001..0.003 (classes 0-2 carry 52% of the weight).
- *Falsifier.* Lift ≤ 1 sd in classes 1-2 and a tied-encoder arm within seed noise (0.0014).

**H4 (MEM, U). Price-level associative memory (support/resistance, hidden liquidity).**
- *Mechanism.* Large resting or iceberg interest at a price makes departures from that level
  one-sided (CJP §4.5 hidden orders [gk]). A GRU cannot address memory by a decoded price level.
- *Feature.* A per-level hash map of the EMA of departure signs, trades absorbed at the level while
  it was best bid or ask, and the visit count.
- *Screen.* +0.085 vs −0.099 in two samples, i.e. no stable sign.
- *Why keep it.* It is the one quantity a GRU is structurally unable to form, and the test costs a
  minute once features exist.
- *Predicted.* 0..+0.002.
- *Falsifier.* Lift ≤ 1 sd in every class.

**H5 (FL). Quote-caused moves revert; trade-caused moves persist (class 3).**
- *Mechanism.* A cancellation that opens the spread is often refilled on the same side (φ0 in TQP
  §7.5 p.130 [read]). A trade-driven depletion reflects persistent aggressive flow. Eisler et al.
  report nearly permanent bare impacts for large ticks [web], so any predictability must come
  through event cross-correlations.
- *Feature.* `ret_q16`, `ret_mo16`, `ret_q64`, `ret_mo64`.
- *Screen.* −0.102 ± 0.053 (quote-caused), +0.036 ± 0.036 (trade-caused), one sample.
- *Test.* Same probe as H1 (share the feature file).
- *Predicted.* +0..0.0015.
- *Falsifier.* Lift ≤ 1 sd.

**H6 (FL). Power-law propagator residual beyond the TBPTT horizon.**
- *Feature.* Σ_k w_k·EMA_{h_k}(side) with h_k up to 4096 trades, plus the "surprise"
  side − E[side | past] from a DAR fit (TQP §10.4 [gk]).
- *Screen.* −0.088, −0.068, −0.008 across samples.
- *Predicted.* 0..0.001.
- *Falsifier.* Lift ≤ 1 sd.
- *Risk.* The slow EMA is nearly a per-sequence constant, so any gain would come from sequence
  level. N7 says class-3 level is drift, so expect little.

**H7 (QR). Queue race with activity asymmetry and depletion progress.**
- *Feature.* `log(λ_a/λ_b)` (per-side volume-change intensities in real time) × ICA imbalance, and
  the fraction of the formation-time queue proxy consumed by same-side MO0 flow since the level
  formed.
- *Source.* TQP §7.3.1 p.123 and Fig. 7.3 [read]; flow field universal in u = V/V̄, §7.4.2 p.128
  [read].
- *Overlap.* Complements opus_independent M10 (min-queue).
- *Predicted.* ≤ +0.001.
- *Falsifier.* Lift ≤ 1 sd.

**H8 (FL/RT). Impact-weighted flow: `side / (liquidity proxy)`.**
- *Source.* Bank-Cartea-Körber ([arXiv:2306.00621](https://arxiv.org/abs/2306.00621), §1-2
  [read], the "Execution.pdf"). Impact of a market order falls with the liquidity process (posted
  volume minus MO volume and cancellations); LOs and MOs mutually excite.
- *Feature.* Trade-time EMA of `side/(1 + softplus(depth_bid_IC + depth_ask_IC))` and an EMA of
  depth change, the book's own resilience.
- *Predicted.* ≤ +0.001.
- *Falsifier.* Lift ≤ 1 sd.

**H9 (RT). Excess-volume duration (informed-trading clock).**
- *Source.* Hautsch §3.2.2 p.37, §9.2.2 p.230 [read]: short excess-volume durations signal
  one-sided, informed demand.
- *Feature.* Real time needed to accumulate \|Σ side\| ≥ v̄ (v̄ = running median VNET per mid
  move), signed by the direction of the excess, and its ratio to the running median.
- *Predicted.* ≤ +0.001, mostly classes 0-1.
- *Falsifier.* Lift ≤ 1 sd.

**H10 (VOL, U). "Make your own VIX": a realized-straddle index from rebuilt labels.**
- *Transfer.* The article replicates the VIX with a rolling re-centred ATM straddle (Quant Galore,
  pp.2-4 [read]). The class-3 label is a 4-event forward average of mid (E-079), so its magnitude
  \|F − mid\| is a 4-event straddle payoff that becomes observable online.
- *Feature.* The running mean of realized \|F − mid\| over the last resolved rows, and its ratio to
  the sequence median. It targets the t0+t1 "volatility" component (opus_independent P3: model corr
  0.40) and does not recalibrate the direction level, which N7 shows is futile.
- *Test.* Lift probe on the t0+t1 residual.
- *Predicted.* +0.0003..0.001.
- *Falsifier.* Lift ≤ 1 sd.

**H11 (XI/RT, U). Cross-venue "signal before the order" (Meyer-σ-field idea).**
- *Transfer.* In Bank-Cartea-Körber the trader learns about an imminent order just before it
  arrives (§1-2 [read]). Here an i1 event on the same timestamp atom (Δt ≈ 0) right before an i0
  trade marks a common message, for example an arbitrageur hitting both books.
- *Feature.* Signed i1 events within one Δt atom of the current row, and the EMA of i1-then-i0
  same-atom sequences.
- *Focus.* Class 2, where i1 trades matter (E-101).
- *Predicted.* ≤ +0.001.
- *Falsifier.* Share of i0 trades preceded by a same-atom i1 event not above its base rate.

**Ranking rationale.** H1 and H3 have mechanism, a demonstrated GRU blind spot (corr(f,P) ≈ 0, or a
missing supervised decoder) and positive screens. H2 is H1 made exact. H4 is the purest
"GRU-impossible" quantity but screened as noise. H5-H9 are textbook quantities with weak or captured
screens. H10-H11 are cheap, creative long shots.

**Deprioritize (evidence):**
- Online learning of the class-3 sequence direction level (N7, E-080).
- All-trade signed flow, VNET, runs and iceberg counts (captured, §0.1).
- i1 lead-lag for class 3 (N4 ≈ 0.01).

**Overall expectation.** Consistent with opus_independent S1-S2, the microstructure features left
are worth +0.002..0.005 global in total, not +0.03. Class-3 far-from-move direction sits near the
TQP §7.3 ceiling (p.126, as quoted in opus_independent S2: for long queues p+ depends only weakly on
imbalance), and the class-3 level is drift (N7).

## 4. Requests (larger than my limits)

- **R1. Train-fitted tables.** From ~2000 train sequences (all classes, row group 528 excluded),
  build `T_px` (code pair → d_ask, d_bid) with a per-class quantum, `T_dp` (dp pair → Δmid at
  trade-free events) and `T_side` (dp code → mean sign of the next mid move within 50 rows).
  Report the unknown rate on valid i0 and i1 per class. CPU ~5-10 min, < 2 GB. Fitting on train
  keeps the valid lift probes clean. My tables were fitted on valid folds 0-1 and scored on folds
  2-3, which is fine for screening but not for decisions.
- **R2. One feature file for all hypotheses.** A numba kernel over all 1873 valid sequences, one
  pass per sequence, writing the H1/H3/H4/H5/H6/H10 columns (about 24) for the scored rows in
  `heur_valid` order to `cache/micro_valid.npy` (~3.5M × 24 × 4 B ≈ 340 MB; split per family if
  400 MB is a limit). Single core, ~15 min. My reference implementations: `mo01c.py` (H1),
  `six.py` (H5), `leadlag5.py` (H3), `levelmem.py`/`combo.py` (H4), `prop.py` (H6), in
  `.../scratchpad/opus_micro/`.
- **R3. Lift probes.** `scripts/lift_probe.py --npy cache/micro_valid.npy --name micro_<family>`
  per family (chain grouping), then `--gbm` for H1+H5 together. Report `delta_by_class`. The
  decision rule is a class delta > 2 sd with a consistent sign on both chain halves.
- **R4. GRU arms, only for survivors.** An H1(+H2) input arm and an H3 tied-encoder arm, each one
  2-epoch run judged by `scripts/decide.py`.
- **R5. Validate H2's clock.** The isotonic additivity fit of a6 vs a0 on 50 sequences (minutes).

## 5. Specification-critic notes (predicted failure modes)

- **Leakage through the tables.** `T_side` is fitted on future moves. It must come from train (R1).
  `lift_probe` itself fits on chain folds 0-1 only.
- **One-row delay.** MO1 labels resolve at t+1. Features written at row t must use only trades up to
  t−1 as classified, plus the pending trade as "unclassified".
- **Resting-code drift.** The resting dv code differs by instrument; use the per-sequence running
  mode, not 0.025.
- **Unknown decodes.** Unknown code pairs contribute 0 to paths. Carry the running unknown rate as a
  feature so the model can discount noisy i1 paths (46-64% unknown in class 0 with a 56-seq table).
- **Heavy-tailed sequences.** Report per-chain-fold deltas, not only the pooled number (§0.1
  lesson).
- **Streaming cost.** Every recipe is a few EMAs, two dict lookups per book event and one hash-map
  update per mid move, well inside the 80 µs/row budget.

## Sources

Local [read]:
- Bouchaud, Bonart, Donier, Gould, *Trades, Quotes and Prices* (CUP 2018): §7.3.1 p.123 and
  Fig. 7.3; §7.4.2 pp.128-129; §7.5 pp.130-131; §10.1 eq. 10.2 p.188; §10.2 p.190;
  §14.3 pp.273-274; §14.4 p.275.
- Hautsch, *Econometrics of Financial High-Frequency Data* (Springer 2012): §3.2.2 p.37 (excess
  volume durations); §9.2.1 eq. 9.15 pp.228-229 (VNET); §9.2.2 p.230.
- Cartea, Jaimungal, Penalva, *Algorithmic and High-Frequency Trading* (CUP 2015): §3.5
  pp.52-53, Table 3.5 (57-64% of quote changes reverse); ch. 12 pp.295-296 (imbalance regimes
  modulate MO arrival rates).
- Bank, Cartea, Körber, "Optimal execution and speculation with trade signals"
  (`Execution.pdf`, [arXiv:2306.00621](https://arxiv.org/abs/2306.00621)), §1-2.
- Quant Galore, "Stop Watching the VIX, Just Make Your Own" (Medium, 2022), pp.2-4.

Web [web]:
- Eisler, Bouchaud, Kockelkoren, *The price impact of order book events*,
  [arXiv:0904.0900](https://arxiv.org/abs/0904.0900) (QF 2012).
- Gould, Bonart, *Queue imbalance as a one-tick-ahead price predictor*,
  [arXiv:1512.03492](https://arxiv.org/abs/1512.03492).
- Kolm, Turiel, Westray, *Deep order flow imbalance*, Math. Finance 33(4) 2023,
  [SSRN 3900141](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3900141): networks fed
  stationary order-flow inputs beat networks fed raw book states. This supports feeding explicit
  flow and event features (H1, H2) rather than hoping the GRU derives them.
- Huang, Lehalle, Rosenbaum, *The queue-reactive model*,
  [arXiv:1312.0563](https://arxiv.org/abs/1312.0563).
- Stoikov, *The micro-price*,
  [SSRN 2970694](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2970694).
