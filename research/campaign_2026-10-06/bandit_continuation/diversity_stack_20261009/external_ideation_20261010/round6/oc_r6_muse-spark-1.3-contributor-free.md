Located your task file — pulling the inputs to build the final summary.# 4 causal recurrent architectures for diversity vs GRU-256x2

Baseline: 2-layer GRU-256, 211 inputs/row, ~1.04M params, ~1M MACs/row, O(1) state 512 floats, int8 VNNI ~26us/row/member. Pairwise teacher error corr 0.95–0.99; static diverse pair +0.003 fold-1. Ctrl fold-1 ~0.6660, one-run sd 0.0022, paired-diff sd 0.0018/0.0022. All WP predictions below are single-seed fold-1 with 95% ≈ ±0.004. Solo parity is not the goal; pair-avg lift at residual corr <0.97 is.

## 1. ClockBank-minGRU-384 (rank 1)

**Layer spec.** Encoder `Linear(211→384)` + GELU (81,408). Partition state into 4 banks × 96 dims tied to measured update-rate groups: fast (100%/70% rows), mid (40%/31%), slow (15%/12%), trade/event (8 extra + event flags). Each bank is a minGRU (no reset gate, candidate has no `h_prev` dependence → fewer matmuls, LUT-friendly sigmoid/tanh only). Clock-open mask `m_b ∈ {0,1}` computed causally from "block changed" flags (already in event features): effective gate `z_eff = m_b·z + (1−m_b)·z_leak`, `z_leak = sigmoid(b_leak)` ≈ 0.01–0.05 learned. Head identical to baseline: `[out;z] 768→384 GELU →2+aux` (~296k). Total ≈ 81k + 4×2×(384×96+96)=4×73,920=296k + 296k ≈ **0.67M**. State 384 floats. Cost/row ≈ 380k MACs + elementwise — ~0.4× GRU-256x2, easily inside 26us.

**Diversity mechanism.** GRU rewrites all 256 dims every row with dense coupling, so stale book blocks still drive state through `W_hh`. Here slow-block inputs physically cannot overwrite fast-bank memory when their clock is shut, and fast trade noise cannot corrupt slow level state except through the learned leak. Error disagreement concentrates on rows where one instrument's book is stale — exactly where the GRU's dense overwrite is most suspect (targets piecewise-constant, mean run 9.5 rows).

**Cell pseudocode (per bank b, dim 96):**
```python
def bank_step(x, h, m, Wz, bz, Wh, bh, leak):
    z = torch.sigmoid(x @ Wz + bz)          # (B,96)
    z_eff = m * z + (1 - m) * torch.sigmoid(leak)
    h_tilde = torch.tanh(x @ Wh + bh)       # no h_prev: minGRU
    return (1 - z_eff) * h + z_eff * h_tilde
# top: x = enc(row211); h = cat(banks); y = head(cat(h, x))
```

**Training with existing recipe.** Drop-in `RNNModel` variant: all `W*` under Muon scope, biases/leak under AdamW, lr 1e-3, wmse clip±2 + struct 0.3 + next-change 0.3, chunk 1000, 3ep, batch 32, truncated-BPTT detach, zero init, reset per stream. Use prop_power 0 + lw_floor 0.18 arm (current lead) for the experimental run but keep one default-propensity seed for comparability. Distill as 5th family (coef from held-out weight fit, or fixed 0.15) into the same 256 student; bank arch need never serve.

**Expected fold-1 WP.** 0.6645–0.6665 (±0.004); P(solo ≥ control) ~30%. Value is pair with GRU: expected pair-avg +0.0015–0.003 if residual corr ≤0.96 (likely: only arch where stale-row updates are structurally zero).

**Cheapest falsifier.** 1 seed, fold-1, short WSD 2ep + SWA (HPO t52 recipe, ~20 GPU-min): kill if (a) solo < 0.660 (TCN-level disaster 0.6598), or (b) residual corr vs matched GRU control > 0.97 *and* static pair-avg ≤ max(solo, control). Pass = solo ≥ 0.662 with residual corr < 0.96.

## 2. Selective diagonal SSM-384 (rank 2)

**Layer spec.** Encoder `Linear(211→192)` + GELU (40,704). Per-row selectivity projections `W_dt, W_B, W_C, W_u`: each `192→384` (~74k, ×4 = 295k). Per-dim learned decay `D` (384 params), `A = −softplus(D)` (time constants ~8–512 rows init log-spread). Recurrence is scalar per dim (no hidden-hidden matmul): `h = exp(A·dt)·h + (dt·B)·u`; read `y = W_out(C⊙h)`, `W_out 384→256` (98,560), then baseline head (~164k). Total ≈ **0.60M**. State 384 floats. Cost ≈ 5×(192×384) ≈ 370k MACs + exp/LUT elementwise — ~0.4× baseline.

**Diversity mechanism.** Only linear-recurrence arm: memory is a bank of 384 learned exponential smoothers with input-dependent step size `dt(x)`, versus GRU's nonlinear coupled overwrite. Long-horizon forecast rows (lookahead ladder: 8 rows ahead already +0.045) should disagree most: SSM holds decaying averages of book imbalance over 100s of rows; GRU compresses the same history through saturating gates. No tanh/sigmoid inside recurrence — the served int8 path is mul/add + one exp LUT.

**Cell pseudocode:**
```python
def ssm_step(z, h, Wdt, Wb, Wu, Wc, D):
    dt = F.softplus(z @ Wdt) + 1e-3           # (B,384) selectivity
    Bm = z @ Wb; u = torch.tanh(z @ Wu)
    Cg = torch.sigmoid(z @ Wc)                 # read gate
    decay = torch.exp(-F.softplus(D) * dt)     # (B,384)
    h = decay * h + (dt * Bm) * u
    return h, (Cg * h)
# y = W_out(state) -> head(cat(y, z_enc))
```

**Training.** Same recipe: Muon on all four projection matrices + `W_out`, AdamW on D/biases (D needs 10× smaller lr — put in AdamW group lr 1e-4, else decay collapses to all-fast). Same loss/mask/chunk/epochs. Init D so `softplus(D)` spans 1/512..1/8. Distill identically (alpha 0.7, family head 0.1).

**Expected fold-1 WP.** 0.6630–0.6660 (±0.0045); P(solo ≥ control) ~25%. Highest decorrelation potential of the four (linear vs gated-nonlinear memory); pair-avg +0.001–0.003 even at solo −0.002.

**Cheapest falsifier.** Same 1-seed 2ep screen: kill if solo < 0.659 (below TCN) or residual corr > 0.97. Extra diagnostic (free): freeze trained model, clamp `dt` to its median — if WP unchanged, selectivity is dead and the arm reduces to a linear smoother; re-init D wider and rerun before promoting.

## 3. Dual-tower Book/Trade late-fusion (rank 3)

**Layer spec.** Split inputs by economics, not index convenience: Tower-B (book levels): 150 cols (both instruments' prices/volumes + price/volume diffs + last-nonzero diffs) → `Linear(150→160)` (24,160) → 1-layer GRU-160 (154,560). Tower-T (trading process): 61 cols (trade prices/volumes, event flags, rows-since-event counters) → `Linear(61→96)` (5,952) → 1-layer GRU-96 (55,872). Fusion: `g = sigmoid(W_g[t_b;t_t])`, `f = g⊙proj_b + (1−g)⊙proj_t`, `W_g,proj 256→256` (~131k). Baseline head on `[f;z]` (~131k). Total ≈ **0.50M**. State 256 floats. Cost ≈ 430k MACs/row.

**Diversity mechanism.** Targets are `F−ask` / `bid−F`: book tower estimates the *level*, trade tower estimates *imminent repricing pressure*. GRU mixes both every row through one coupled state, so a burst of trade prints rewrites book-level memory. Here trade bursts only reach output through the fusion gate, which is shut on quiet rows. Disagreement concentrates on trade-burst → price-move transitions (high-weight rows: |clip(y,2)| weights reward exactly these).

**Cell pseudocode:**
```python
def step(rb, rt, hb, ht, encB, encT, gruB, gruT, Wg, Wpb, Wpt):
    zb, zt = gelu(rb @ encB), gelu(rt @ encT)
    hb, ht = gruB(zb, hb), gruT(zt, ht)       # 1-layer cuDNN GRU each
    f = torch.cat([hb, ht], -1)                # 256
    g = torch.sigmoid(f @ Wg)
    out = g * (hb @ Wpb[:160]) + (1 - g) * (ht @ Wpt[:96])
    return out, (hb, ht)
```

**Training.** Same recipe; Muon on both GRU matrices + fusion/head matrices, AdamW biases. Add auxiliary heads per tower (book tower predicts structure target, trade tower predicts next-change target, weight 0.3 each) — reuses existing aux without new labels; forces the intended split (else both towers learn the same function and diversity dies). Distill as usual.

**Expected fold-1 WP.** 0.6640–0.6670 (±0.004); P(solo ≥ control) ~35% — closest to baseline capacity, smallest decorrelation of the four (still two GRUs inside).

**Cheapest falsifier.** 1-seed 2ep screen + gate audit: kill if solo < 0.660, pair-avg ≤ control, or mean fusion gate saturates (P(g>0.9 or <0.1) > 95% — towers unused). Pass = pair-avg ≥ control + 0.001 with gate active 10–90% of rows.

## 4. Leaky Mixbank-288 + GRU-128 mixer (rank 4)

**Layer spec.** Encoder `Linear(211→256)` + GELU (54,272). Three fixed-tau leaky banks × 96 dims, taus 8/64/512 rows (`a_b = 1−exp(−1/τ_b)`, per-dim learned trim ±20% via sigmoid bias): `W_b 256→96` ×3 (74,016). Mixer: 1-layer GRU-128 on bank concat (288→128, ~160k). Head on `[mixer;enc]` 384→256 (~100k). Total ≈ **0.39M**. State 288+128 = 416 floats. Cost ≈ 340k MACs/row. All recurrence is leaky-average + one small GRU — most integer-friendly arm (leak = multiply by constant + add).

**Diversity mechanism.** Mixture-of-timescales prior: short bank tracks microstructure noise, mid bank tracks the 9.5-row label-run regime, slow bank tracks session drift. GRU must learn timescales from gate biases (chrono init tested negative: 0.6648/0.6605); here they are architectural. Errors decorrelate on slow-drift rows where GRU forgets and the τ=512 bank does not, and on noisy rows where the τ=8 bank smooths what the GRU chases.

**Cell pseudocode:**
```python
def step(z, states, Ws, bs, trims, taus, mixer):
    outs = []
    for h, W, b, tr, tau in zip(states, Ws, bs, trims, taus):
        a = (1 - torch.exp(-1.0 / tau)) * torch.sigmoid(tr) * 2
        cand = torch.tanh(z @ W + b)           # (B,96)
        outs.append((1 - a) * h + a * cand)
    m = torch.cat(outs, -1)                     # 288
    return outs, mixer(m)                       # GRUCell-128
```

**Training.** Same recipe; Muon on `W_b`/mixer/head, AdamW on trims/biases (trim lr 1e-4). Same loss/chunk/epochs/distill. Do not put taus under Muon (fixed constants).

**Expected fold-1 WP.** 0.6620–0.6655 (±0.0045); P(solo ≥ control) ~20%. Weakest standalone (leaky pools underfit sharp moves — the TCN failure mode), kept as a pure-diversity lottery ticket: if residual corr < 0.94 it earns its ensemble slot at solo −0.003.

**Cheapest falsifier.** Same 1-seed screen: kill if solo < 0.658 (worse than TCN 0.6598 — pure smoothing with no signal) or residual corr > 0.97. Pass = solo ≥ 0.660 with residual corr < 0.95.

## Rank and what to run first

1. **ClockBank-minGRU** — best expected pair lift per GPU-min: matched to the multi-rate clock physics, cheapest falsifier, smallest serving cost if ever distilled directly.
2. **Selective SSM-384** — highest decorrelation ceiling (only linear-memory arm); slightly higher implementation risk (D-lr sensitivity).
3. **Dual-tower** — safest solo score, smallest diversity; run third as the pair-stabilizer, not the lead.
4. **Mixbank** — lottery ticket; run only if lane is free after 1–3.

All four: causal, deterministic, zero-init, reset per stream, O(1) state ≤ 512 floats, ≤0.7M params (limit 2.2M), GEMV + LUT-sigmoid/tanh/exp only, trainable under the frozen teacher recipe (Muon matrices / AdamW biases, wmse clip±2 + 0.3/0.3 aux, chunk 1000, 3ep) and distillable with alpha 0.7 into the served 256 student without changing the int8 serving path.
