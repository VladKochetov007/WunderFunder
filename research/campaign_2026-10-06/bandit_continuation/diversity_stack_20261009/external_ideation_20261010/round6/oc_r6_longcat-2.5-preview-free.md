# Four causal recurrent architectures for decorrelated ensemble members

**Shared skeleton (all four):** encoder `Linear(211→H)+GELU` → `z`; two recurrent layers, width `H`, zero initial state, reset per sequence, carried detached across 1000-row chunks (TBPTT); head `[h₂; z]→H→GELU→2`. Baseline GRU-256x2 = 1.04M params, ≈0.94M MACs/row, 1-core budget ≈26 µs/row/member. All cells below are GEMV + elementwise (sigmoid/tanh as LUTs in int8), causal, O(1) state per row.

---

## 1. CW4 — clockwork multi-rate GRU blocks

**Spec.** `H=512 = 4 blocks × 128`. Encoder `Linear(211→512)` (108.0k). Two layers; each layer = 4 cells with periods `p ∈ {1,2,4,8}`. Cell: `z=σ(hz(h)+gz(x)+b)`, `c=tanh(hc(h)+gc(x))`; on rows `t mod p == 0`: `h ← (1−z)h + zc`; else `h ← a⊙h` (per-channel `a=σ(logit)`, init 3.0 ≈ 0.95). Params ≈ 1.37M; MACs/row ≈ 0.75M ≈ 0.80× GRU (one block computes per row).

**Mechanism.** Different temporal granularity per block: slow blocks integrate multi-horizon context, fast blocks react to single-row shocks — a structural match to the multi-rate clocks (blocks update on 12–100% of rows). Decorrelation comes from *when* state changes, not just how.

```python
class CWCell(nn.Module):
    def __init__(self, din, d, period):
        super().__init__()
        self.p = period
        self.gz, self.gc = nn.Linear(din, d, bias=False), nn.Linear(din, d, bias=False)
        self.hz, self.hc = nn.Linear(d, d, bias=False), nn.Linear(d, d, bias=False)
        self.logit_a = nn.Parameter(torch.full((d,), 3.0))
        self.b = nn.Parameter(torch.zeros(d))
    def step(self, x, h, t):
        a = torch.sigmoid(self.logit_a)
        if t % self.p == 0:
            z = torch.sigmoid(self.hz(h) + self.gz(x) + self.b)
            c = torch.tanh(self.hc(h) + self.gc(x))
            return (1 - z) * h + z * c
        return a * h
# 4 cells/layer, periods 1,2,4,8; h = concat of 4 blocks
```

**Training.** Identical recipe: Muon on all weight matrices (`gz,gc,hz,hc`), AdamW enc/head/biases/lr 1e-3/wd 0.01, OneCycle 3 epochs (peak at 3), batch 32 × chunk 1000, fp16 autocast, wmse on clip(y,±2) with `|clip(y,2)|` + propensity weights, aux structure/next-change heads 0.3/0.3.

**Expected fold-1 WP.** 0.664 ± 0.006 (one-run A/B sd 0.0022; small per-block matrices cost some capacity). Cheap WSD+SWA as run 2 (HPO: +0.000–0.0002 pattern).

**Cheapest falsifier.** One fold-1 scratch run; kill if WP < 0.660 or max pairwise prediction corr with the GRU-256x2 member > 0.97. Ablation `all p=1` should drop WP or raise corr ≥ 0.98 if multi-rate is doing anything.

## 2. dSSM-384 — diagonal state-space cell, learned decay & gain

**Spec.** Encoder `Linear(211→384)` (81.0k). Two layers of: per-channel decay `a_c=σ(logit_a)` (init spread over {0.5,0.8,0.9,0.95} → mixture of timescales), per-channel gain `g_c=softplus(raw_g)` init 0.5413 (→1); `B: din→384`, `C: 384→384`. Head as usual. Params ≈ 0.80M; MACs/row ≈ 0.80M ≈ 0.85× GRU.

**Mechanism.** Linear recurrence: `h_t = a⊙h_{t−1} + g⊙(B x_t)`, `y = C h`. No tanh, no gates, no sigmoid in the state path — the opposite inductive bias of GRU. Expected to be the strongest decorrelator; also the highest training-risk (pure linear dynamics may underfit the piecewise-constant mean-reverting PnL).

```python
class DSSMCell(nn.Module):
    def __init__(self, din, d):
        super().__init__()
        self.B = nn.Linear(din, d, bias=False)
        self.C = nn.Linear(d, d, bias=False)
        self.logit_a = nn.Parameter(torch.tensor([0.5,0.8,0.9,0.95]).repeat_interleave(d//4).logit())
        self.raw_g = nn.Parameter(torch.full((d,), 0.5413))
    def step(self, x, h):
        a = torch.sigmoid(self.logit_a); g = F.softplus(self.raw_g)
        h = a * h + g * self.B(x)
        return h, self.C(h)
# 2 cells; layer1 in=211, layer2 in=384; stack per-row outputs
```

**Training.** Same recipe; Muon on `B,C`. If unstable (train loss diverges), lr 5e-4 or freeze `logit_a` for epoch 1 — cheap to detect in the first 10 minutes.

**Expected fold-1 WP.** 0.662 ± 0.007 (widest honest range: linear dynamics could also *win* if GRU overfits short horizons).

**Cheapest falsifier.** Same as CW4 (WP < 0.655 or corr > 0.97 → dead). Training-divergence in epoch 1 also falsifies the init, not necessarily the family.

## 3. QGRU-352 — quasi-recurrent gated cell with depthwise conv memory

**Spec.** Causal depthwise conv (k=8, 211 channels) + pointwise `Linear(211→352)` (76.0k). Two layers of `d=352`: `z=σ(Az(h)+Bz(x)+bz)`, `c=tanh(C(x))`, `h ← z⊙h + (1−z)⊙c`. Head `[704]→256→2`. Params ≈ 1.00M; MACs/row ≈ 1.00M ≈ 1.06× GRU.

**Mechanism.** Quasi-RNN: the candidate is *purely feedforward* — no hidden matmul on the candidate path — plus a short FIR local memory (conv) that GRU lacks entirely. Elementwise gate (not GRU's three additive gates) gives a different nonlinearity path; errors should decorrelate moderately-to-strongly.

```python
class QGRUCell(nn.Module):
    def __init__(self, din, d):
        super().__init__()
        self.Az = nn.Linear(d, d, bias=False)
        self.Bz = nn.Linear(din, d, bias=False)
        self.C  = nn.Linear(din, d, bias=False)
        self.bz = nn.Parameter(torch.zeros(d))
    def step(self, x, h):
        z = torch.sigmoid(self.Az(h) + self.Bz(x) + self.bz)
        c = torch.tanh(self.C(x))
        return z * h + (1 - z) * c
# x = pointwise(dwconv8(raw)); 2 cells, h=352
```

**Training.** Same recipe; Muon on `Az,Bz,C`.

**Expected fold-1 WP.** 0.664 ± 0.006 (quasi gating is the mildest departure from GRU; conv memory should be mildly positive).

**Cheapest falsifier.** Same screen; additionally drop the conv — if WP is unchanged, conv memory is null and the cell reduces to a quasi-GRU.

## 4. 2PATH — book/trade dual-pathway event-gated cells

**Spec.** Pathways read raw (normalization-folded) columns directly, using the known layout: book path = 88 price/volume cols (both instruments), trade path = 16 trade cols + 8 extras. Light encoder `Linear(211→384)+GELU` (81.4k) for the head skip. Book cell `d=224` (EGD: `a_b` init 0.95, tanh candidate `Wh(h)+Wx(x)`, updated every row). Trade cell `d=96`, updated only when a trade-event flag (the 8 event flags already in the 99 event features) fires; between events `a=0.99` hold. Cross-term: on event rows trade input `← +U(h_book)`, `U: 224→96`. Readout `Linear(320→384)` (122.9k) → usual head. Params ≈ 0.50M; MACs/row ≈ 0.50M ≈ 0.54× GRU (widening headroom available).

**Mechanism.** The two instruments/clocks live in separate state that merges only on trade rows — event-gated update + structural split. Decorrelation via pathway independence; exploits the asynchronous-clock structure directly.

```python
class PathCell(nn.Module):
    def __init__(self, din, d, hold):
        super().__init__()
        self.Wx = nn.Linear(din, d, bias=False)
        self.Wh = nn.Linear(d, d, bias=False)
        self.logit_a = nn.Parameter(torch.full((d,), 3.0))
        self.hold = hold
    def step(self, x, h, fire):
        a = torch.where(fire, torch.sigmoid(self.logit_a),
                        torch.full_like(self.logit_a, self.hold))
        u = torch.tanh(self.Wh(h) + self.Wx(x))
        return a * h + (1 - a) * u
# fire = trade-event flags > 0 (from the 99 event features); book path fire=1
```

**Training.** Same recipe; Muon on `Wh,U` (AdamW on `Wx` input-side is fine too — consistent with "Muon on matrices").

**Expected fold-1 WP.** 0.662 ± 0.007 (sparse trades may carry little signal; cross-gate may rarely fire — highest variance of the four).

**Cheapest falsifier.** Same screen; forcing `fire=1` every row isolates whether event gating (vs plain dual EGD) matters.

---

## Ranking (P(member WP ≥ 0.662) × P(pairwise corr ≤ 0.95), ties to cheapness)

| Rank | Arch | Params | Cost vs GRU | Why |
|---|---|---|---|---|
| 1 | **CW4** | 1.37M | 0.80× | Multi-rate matches the data's clock structure; decorrelation via update timing; low training risk |
| 2 | **dSSM-384** | 0.80M | 0.85× | Maximal mechanistic decorrelation (linear, gate-free); highest WP variance |
| 3 | **QGRU-352** | 1.00M | 1.06× | Safe, likely GRU-level WP; conv memory adds a missing FIR path; moderate corr |
| 4 | **2PATH** | 0.50M | 0.54× | Domain-true but sparse-signal risk; worth one run as a cheap wildcard |

**Honest expectations.** Control scratch GRU-256x2 = 0.666 (sd 0.001 seed, 0.0022 one-run A/B). These are single-run T0 screens; expected pair gain ≈ +0.003 requires member WP ≳ 0.66 *and* corr ≲ 0.95 vs the existing member. Prioritize CW4 and dSSM for the first two slots; if both land corr ≤ 0.94, the next student blend (their predictions + the 4-family cache) is the immediate follow-up. Note (HPO): single-trial HPO deltas do not replicate — each arm needs 3 seeds on fold 1 before any lead claim.