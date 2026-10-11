# R6: four decorrelated causal streaming cells (teacher candidates for the diversity stack)

Fit to constraints: 211 inputs/row, 3288-row streams with per-stream state reset, params ≤ 2.2M (before aux/family heads), per-row cost 0.4–0.8× GRU-256x2 (~1.0M MACs/row), O(1) state, int8 GEMV + elementwise + gather only. Shared training: existing trainer, Muon on the recurrent/state matrices, AdamW for biases/head/encoder, OneCycle or WSD 3–4 epochs, wmse + structure/next-change aux (0.3/0.3), propensity row weights, chunk-1000 truncated BPTT, reset per stream. The new causal inputs are first-difference flags and rows-since-event counters derived from raw columns (the trainer's event features and the Rust engine already compute these). All four are teachers: standalone screen, then static pair with the served GRU-256 student, then self-distillation.

## 1. Clock-Block minGRU (multi-rate cells) — rank 1

Spec: input proj Linear(211→128)+SiLU; two stacked cells, state D=384 split by the six observed clock groups {12,15,31,40,70,100}% into dims [24,36,72,72,96,84]; per cell Wg/Wc Linear(128→384), block-diagonal Us (Σd² = 28.5k), row-rate mixer 384→96→384+SiLU; head [384+128]→256→2. Params ~0.56M; ~0.55× GRU MACs/row (blocks evaluate only on their tick; the always-on 100% block plus the mixer dominate).

Mechanism: the data has six asynchronous clocks; a block-diagonal state that integrates only when its own clock ticks is a different time model rather than a reparametrisation of GRU, so decorrelation is structural. Blocks integrate on events, cross-block mixing runs at row rate, so nothing is lost between ticks.

```python
class ClockBlockCell(nn.Module):
    def __init__(self, dims, xd=128):              # dims=[24,36,72,72,96,84] -> 384
        D = sum(dims); self.dims = dims
        self.register_buffer('blk', torch.repeat_interleave(torch.arange(len(dims)), torch.tensor(dims)))
        self.Us = nn.ParameterList([nn.Parameter(0.05*torch.randn(d, d)/d**0.5) for d in dims])
        self.Wg, self.Wc = nn.Linear(xd, D), nn.Linear(xd, D)
        self.mix = nn.Sequential(nn.Linear(D, D//4), nn.SiLU(), nn.Linear(D//4, D))
    def forward(self, x, h, tick):                 # h:[B,D], tick:[B,6] causal flags
        hs = h.split(self.dims, -1)
        g = torch.sigmoid(self.Wg(x) + torch.cat([s @ U for s, U in zip(hs, self.Us)], -1))
        upd = (1-g)*h + g*torch.tanh(self.Wc(x))    # minGRU: no h in candidate
        m = tick[:, self.blk].float()               # block freezes on its own clock
        h = (1-m)*h + m*upd
        return h + self.mix(h)                      # row-rate cross-block mixing
```

Training deltas: Muon covers Wg/Wc/Us/mix; tick flags are constants with no gradient; slow blocks receive gradient through ~8 ticks per 1000-row chunk, an automatic short-BPTT. Everything else identical.

Expected fold-1 WP: standalone 0.664 (90% interval 0.660–0.668; control 0.6662, LSTM-256 sibling 0.6658). Value is the pair: with measured prediction corr vs the GRU-256 student ≤ 0.90, static pair ~0.669–0.671 (prior).

Cheapest falsifier: one fold-1 three-epoch seed (~0.4 GPU-h). Assert the six firing rates on valid rows match 12/15/31/40/70/100%; kill if corr(pred_CB, pred_GRU) ≥ 0.95 or WP ≤ 0.655.

## 2. Decay-Bank diagonal SSM (learned mixture of timescales) — rank 2

Spec: two layers, d=512; per layer A Linear(211→512) decay logits, W Linear(211→512) drive, mixer 512→128→512+SiLU; head [512+128]→256→2. Params ~0.86M; ~0.8× GRU MACs/row; O(1) state of 512 floats; λ = σ(Ax), τ = 1/(1−λ) initialised across dims to span 2–1500 rows.

Mechanism: the opposite inductive bias to a GRU — pure elementwise decaying integration with no cross-dimension gating and no saturation. A bank of ~512 learned EMAs is a natural fit for labels that are piecewise constant over ~9.5 rows plus long price regimes; with no shared nonlinearity except tanh on the drive, decorrelation per FLOP should be the highest of the four, and the recurrence is parallel-scan friendly.

```python
class DecayBankCell(nn.Module):
    def __init__(self, d=512, xd=211):
        self.A, self.W = nn.Linear(xd, d), nn.Linear(xd, d)
        self.mix = nn.Sequential(nn.Linear(d, d//4), nn.SiLU(), nn.Linear(d//4, d))
        with torch.no_grad(): self.A.bias.uniform_(-3.0, 7.3)   # tau ~ 0.05..1500 rows
    def forward(self, x, h):
        lam = torch.sigmoid(self.A(x).clamp(-9, 8.5))           # tau in ~[2,1500] rows
        h = lam*h + (1-lam)*torch.tanh(self.W(x))               # EMA bank, data-dependent decay
        return h + self.mix(h)
```

Training deltas: Muon on A/W matrices, AdamW on the per-dim τ bias so the timescale mixture is learned; the clamp keeps λ < 1 so h stays bounded; otherwise identical. Judge by the pair, not standalone: a linear recursion may sit in the TCN tier (0.6598).

Expected fold-1 WP: standalone 0.660 (0.654–0.666); predicted corr vs GRU 0.84–0.92; pair ~0.669–0.671 if that holds.

Cheapest falsifier: same one-seed screen; kill if corr ≥ 0.95 or WP ≤ 0.655; record the learned τ histogram against the six clock rates (mismatch is informative, not fatal).

## 3. Event-Gated Dual-Path (book vs trade) with between-event drift — rank 3

Spec: input proj Linear(211→128); two stacked cells; book path GRUCell(128→256) gated by the any-book-change flag, trade path GRUCell(128→128) gated by the any-trade-change flag; fusion Linear(386→256)+tanh; head Linear(256→2) plus drift·log1p(rows since last book event) with drift initialised at 0. Params ~1.08M; ~0.7× GRU MACs/row (both gates are event-masked); O(1): two states plus two counters.

Mechanism: the order-book and trade streams have different event rates and roles; giving each its own event-triggered integrator makes the cell's clock data-driven, and between events the logit extrapolates affinely in rows-since-event, matched to labels whose mean run is 9.5 rows. Decorrelation comes from event-triggered rather than per-row integration.

```python
class EventGatedDualCell(nn.Module):
    def __init__(self, xd=128, db=256, dt=128):
        self.bk, self.tr = nn.GRUCell(xd, db), nn.GRUCell(xd, dt)
        self.fuse = nn.Linear(db+dt+2, db)
    def forward(self, x, hb, ht, mb, mt, sb, st):     # m*: event flags, s*: rows-since-event
        hb = torch.where(mb, self.bk(x, hb), hb)       # state frozen between events
        ht = torch.where(mt, self.tr(x, ht), ht)
        sb = torch.where(mb, 0, sb+1); st = torch.where(mt, 0, st+1)
        h = torch.tanh(self.fuse(torch.cat([hb, ht, sb[:,None], st[:,None]], -1)))
        return h, hb, ht, sb, st
# served: out = Linear(256 -> 2)(h) + self.drift * torch.log1p(sb)
```

Training deltas: none beyond architecture 1; GRUCell already exists in the trainer, and the book/trade split is by column offset (book = the 104 book columns, trade = the 16 trade columns).

Expected fold-1 WP: standalone 0.663 (0.658–0.668); predicted corr vs GRU 0.88–0.94; pair ~0.668–0.670.

Cheapest falsifier: same screen; kill if corr ≥ 0.95 or WP ≤ 0.655, and kill if the book flag fires on more than 90% of rows, since the gate then degenerates toward a plain GRU and the diversity mechanism is gone.

## 4. Fast-Weight Bilinear Cell (decayed outer-product state) — rank 4

Spec: two parallel cells; each k Linear(211→256), v Linear(211→96), q Linear(211→256), λ Linear(211→1); state M of shape 256×96 (24.6k, fp16/int16); head [192+128 skip]→256→2. Params ~0.4M; ~0.45× GRU MACs/row; O(1) matrix state.

Mechanism: a matrix state of recency-decayed outer products read by a learned query, i.e. linear attention with learned features and no softmax — the closest attention-like prior in the stack at O(1) cost. Decorrelation vs GRU should be the strongest here because a bilinear write/read is the least gate-like recurrence of the four.

```python
class FastWeightCell(nn.Module):
    def __init__(self, d=256, r=96, xd=211):
        self.k = nn.Linear(xd, d); self.v = nn.Linear(xd, r)
        self.q = nn.Linear(xd, d); self.lam = nn.Linear(xd, 1)
    def forward(self, x, M):                          # M:[B,d,r]
        lam = torch.sigmoid(self.lam(x))              # per-row scalar decay
        M = lam*M + self.k(x)[..., None]*self.v(x)[..., None, :]   # decayed rank-1 write
        y = torch.einsum('bd,bdr->br', self.q(x), M)/r**0.5        # query read
        return y, M
```

Training deltas: Muon on k/v/q, since orthogonal matrix updates stabilise the bilinear form, AdamW on λ; initialise v small (scale 0.1) so early writes stay gentle; otherwise identical. Serving needs a new int16 matrix-state kernel (50 KB, L2-resident) — the long pole for this arm.

Expected fold-1 WP: standalone 0.658 (0.650–0.666, the widest interval given bilinear instability and NaN risk); predicted corr vs GRU 0.80–0.90; pair ~0.670–0.672 if it trains cleanly.

Cheapest falsifier: same screen; kill on any NaN or loss spike, corr ≥ 0.95, or WP ≤ 0.652. One seed only; this is the variance arm.

## Ranking and allocation

| rank | arch | why | compute |
|---|---|---|---|
| 1 | Clock-Block minGRU | mechanism matches the data's actual asynchronous clocks; cheapest to serve; flags exist today | 3 seeds, folds 1+0, after screen |
| 2 | Decay-Bank SSM | highest decorrelation per FLOP, near-zero serving risk; standalone likely weak, so pair-conditional | 3 seeds after screen |
| 3 | Event-Gated Dual-Path | sound mechanism, reuses GRUCell so fastest to build; gate-degeneration risk thins the decorrelation prior | 1 screen + 1 confirm seed |
| 4 | Fast-Weight Bilinear | highest upside and highest variance; new native kernel required | 1 screen seed only |

All pair numbers above are priors, not measurements (control 0.6662; A/B sd 0.0022, seed sd 0.001): they extrapolate the measured +0.003 static-pair gain at corr 0.95–0.99 down to lower corr, and below ~0.85 the member's own errors may be too large to help. The quantity to optimise is therefore measured corr and paired Δ over 3 seeds, not the table. Local pair +0.003 has not reliably transferred publicly (0.6639), so a confirmed member still needs folds 2/3 before any serving claim.

---

Written to `round6/workspace/R6_DIVERSE_STREAMING_ARCHS.md` (~1,500 words).