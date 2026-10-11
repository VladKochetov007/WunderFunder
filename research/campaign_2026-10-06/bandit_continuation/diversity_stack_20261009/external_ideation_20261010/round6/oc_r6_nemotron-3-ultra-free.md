# Four Causal Recurrent Architectures for Decorrelated Errors

## 1. Multi-Rate Gated Recurrent Cell (MR-GRC)

**Layer spec (≈1.9M params)**  
Input: 211 → Linear(211, 256) + GELU → **4 parallel GRU cells** (h=64 each) keyed to the 5 multi-rate clocks (12%, 15%, 31%, 40%, 70%, 100% update rates; 100% cell always updates) → concat 256 → Head: Linear(512, 256) + GELU → Linear(256, 2).  
State: 4 × 64 = 256 floats (O(1)). Each cell updates only on its clock rows; other rows copy state (zero compute).

**Diversity mechanism**  
Separate recurrent dynamics per update rate. The 12% cell learns slow book-drift; the 100% cell learns trade-event reactions. Errors decorrelate because GRU-256 mixes all rates in one cell.

**Cell pseudocode (28 lines)**
```python
class MRGRC(nn.Module):
    def __init__(self, d=256, h=64):
        super().__init__()
        self.cells = nn.ModuleList([nn.GRUCell(h, h) for _ in range(4)])
        self.clock_masks = [0.12, 0.15, 0.31, 0.40]  # frac of rows; 1.0 cell implicit
        self.proj_in = nn.Linear(d, 4*h)
        self.proj_out = nn.Linear(4*h, d)
    def forward(self, x, h_prev, clock_ids):  # x: (B,T,d), clock_ids: (B,T) in {0..4}
        B,T,_ = x.shape
        h = h_prev.view(B,4,-1)  # (B,4,h)
        x_proj = self.proj_in(x).view(B,T,4,-1)
        for i in range(4):
            mask = (clock_ids == i).unsqueeze(-1).float()
            h[:,i] = mask * self.cells[i](x_proj[:,:,i], h[:,i]) + (1-mask) * h[:,i]
        # clock_id==4 (100%) always updates via cell index 3 duplicated or separate
        return self.proj_out(h.view(B,T,-1)), h.view(B,-1)
```

**Training with existing recipe**  
Replace GRU module with MRGRC in `RNNModel`; Muon on all `weight_ih/weight_hh` of the 4 cells; same wmse + aux heads; chunked BPTT works unchanged (state carried per cell).

**Expected fold-1 WP** 0.6675 ± 0.0035 (paired vs GRU-256). Honest: single-run noise 0.0022, seed noise 0.001; mechanism untested.

**Cheapest falsifier**  
T0 screen: fold 1, 1 seed, 2 epochs, width 192 (≈1.1M params). If Δ < +0.0005 paired, discard.

---

## 2. Leaky Mixture-of-Timescales RNN (LMTS-RNN)

**Layer spec (≈2.1M params)**  
Input: 211 → Linear(211, 256) + GELU → **LMTS block**: 256-d state split into 8 timescale groups (32 each) with **learned per-group leak rates** α ∈ (0,1) (softplus-parametrized, initialized log-uniform 0.01..0.99) → update: `h_g = α_g * h_g + (1-α_g) * GRUCell_g(x, h_g)` → concat → Head same as GRU-256.  
State: 256 floats + 8 leaks (O(1)). All groups update every row; leaks are static after init (or slow lr).

**Diversity mechanism**  
Explicit timescale separation via learned leaks. Fast groups (α≈0.01) track trades; slow groups (α≈0.99) integrate book pressure. GRU-256 has implicit timescales via eigenvalues; here they are structural and diverse.

**Cell pseudocode (24 lines)**
```python
class LMTSCell(nn.Module):
    def __init__(self, d=256, n_groups=8):
        super().__init__()
        self.n_groups = n_groups
        self.g = d // n_groups
        self.cells = nn.ModuleList([nn.GRUCell(self.g, self.g) for _ in range(n_groups)])
        self.log_alpha = nn.Parameter(torch.empty(n_groups).uniform_(math.log(0.01), math.log(0.99)))
    def forward(self, x, h_prev):
        B,_ = x.shape
        alpha = torch.sigmoid(self.log_alpha)  # (G,)
        x_g = x.view(B, self.n_groups, self.g)
        h_g = h_prev.view(B, self.n_groups, self.g)
        h_new = []
        for i in range(self.n_groups):
            a = alpha[i]
            h_update = self.cells[i](x_g[:,i], h_g[:,i])
            h_new.append(a * h_g[:,i] + (1-a) * h_update)
        return torch.cat(h_new, dim=1)
```

**Training with existing recipe**  
Muon on GRUCell matrices; AdamW on `log_alpha` (lr 1e-3); same loss/schedule. Leaks can be frozen after epoch 1 for stability.

**Expected fold-1 WP** 0.6680 ± 0.0035. Hypothesis: timescale diversity yields decorrelation; no direct evidence.

**Cheapest falsifier**  
Same T0 screen as MR-GRC. Additional check: inspect learned α spread; if all α > 0.9 or < 0.1, mechanism collapsed.

---

## 3. State-Space Model with Learned Decay (SSM-LD)

**Layer spec (≈1.8M params)**  
Input: 211 → Linear(211, 256) + GELU → **SSM block** (S5-style diagonal state-space):  
State: `x_t ∈ ℝ^256` (complex pairs → 128 complex dims).  
Recurrence: `x_t = Λ x_{t-1} + B u_t`, `y_t = C x_t + D u_t` where `Λ = diag(exp(-exp(log_λ)) * exp(i θ))`, `log_λ ∈ ℝ^128` (learned decay rates), `θ ∈ ℝ^128` (learned frequencies, init uniform 0..π), `B,C ∈ ℂ^{128×256}`, `D ∈ ℝ^{256×256}`.  
Output projection to 256 → Head same.  
Params: Λ(128), θ(128), B(128×256×2), C(128×256×2), D(256×256) ≈ 1.8M.  
State: 256 floats (real-imag interleaved). Pure GEMV + elementwise (integer-friendly via quantization).

**Diversity mechanism**  
Continuous-time dynamics with learned oscillatory/decay modes. GRU is discrete gated; SSM-LD has explicit frequency response. Errors differ on periodic book patterns.

**Cell pseudocode (26 lines)**
```python
class SSMLDCell(nn.Module):
    def __init__(self, d=256):
        super().__init__()
        self.n = d // 2
        self.log_lam = nn.Parameter(torch.empty(self.n).uniform_(-6, 0))  # decay
        self.theta = nn.Parameter(torch.empty(self.n).uniform_(0, math.pi))  # freq
        self.B_re = nn.Parameter(torch.randn(self.n, d) * 0.02)
        self.B_im = nn.Parameter(torch.randn(self.n, d) * 0.02)
        self.C_re = nn.Parameter(torch.randn(self.n, d) * 0.02)
        self.C_im = nn.Parameter(torch.randn(self.n, d) * 0.02)
        self.D = nn.Parameter(torch.eye(d) * 0.1)
    def forward(self, u, x_prev):  # u: (B,d), x_prev: (B,d) real-imag
        B = u.shape[0]
        lam = torch.exp(-torch.exp(self.log_lam))  # (n,)
        cos_t, sin_t = torch.cos(self.theta), torch.sin(self.theta)
        # Λ = lam * (cos + i sin)
        x_re, x_im = x_prev[:,:self.n], x_prev[:,self.n:]
        # x_t = Λ x_{t-1} + B u
        Bu_re = u @ self.B_re.t(); Bu_im = u @ self.B_im.t()
        x_re_new = lam * (cos_t * x_re - sin_t * x_im) + Bu_re
        x_im_new = lam * (sin_t * x_re + cos_t * x_im) + Bu_im
        x_new = torch.cat([x_re_new, x_im_new], dim=1)
        # y = C x + D u
        y_re = x_re_new @ self.C_re.t() + x_im_new @ self.C_im.t() + u @ self.D.t()
        return y_re, x_new
```

**Training with existing recipe**  
Muon on B,C,D (matrices); AdamW on log_lam, theta. Same wmse + aux. Chunked BPTT: state `x` carried across chunks.

**Expected fold-1 WP** 0.6665 ± 0.0040. High uncertainty: SSM on this data untested; may underfit short horizons.

**Cheapest falsifier**  
T0 screen: fold 1, 1 epoch, d=192. If WP < 0.6640, discard. Also check |Λ| max < 1.0 (stability).

---

## 4. Dual-Pathway Book/Trade RNN (DP-RNN)

**Layer spec (≈2.0M params)**  
Input split: **Book pathway** (112 raw cols → Linear(112, 128) + GELU) + **Trade pathway** (99 event feats → Linear(99, 128) + GELU).  
Each pathway: **2-layer minGRU** (quasi-RNN, no gates, ≈50% ops of GRU):  
`h_t = (1 - σ(z_t)) ⊙ h_{t-1} + σ(z_t) ⊙ φ(W x_t + U h_{t-1})` where `z_t = W_z x_t + U_z h_{t-1}`.  
Width 128 each → concat 256 → Cross-pathway attention (1 head, causal, 64 dim) → Head same.  
State: 2 × 2 × 128 = 512 floats (O(1)). minGRU uses only sigmoid + elementwise (int8-friendly).

**Diversity mechanism**  
Structural separation: book pathway sees only slow limit-order dynamics; trade pathway sees only sparse events. Cross-attention merges at readout. GRU-256 entangles both.

**Cell pseudocode (29 lines)**
```python
class MinGRUCell(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.W = nn.Linear(d, d, bias=False)
        self.U = nn.Linear(d, d, bias=False)
        self.Wz = nn.Linear(d, d, bias=False)
        self.Uz = nn.Linear(d, d, bias=False)
    def forward(self, x, h):
        z = torch.sigmoid(self.Wz(x) + self.Uz(h))
        h_cand = torch.tanh(self.W(x) + self.U(h))
        return (1-z) * h + z * h_cand

class DPRNN(nn.Module):
    def __init__(self, d_book=112, d_trade=99, h=128):
        super().__init__()
        self.book_enc = nn.Sequential(nn.Linear(d_book, h), nn.GELU())
        self.trade_enc = nn.Sequential(nn.Linear(d_trade, h), nn.GELU())
        self.book_rnn = nn.ModuleList([MinGRUCell(h) for _ in range(2)])
        self.trade_rnn = nn.ModuleList([MinGRUCell(h) for _ in range(2)])
        self.cross_attn = nn.MultiheadAttention(h, 1, batch_first=True)
    def forward(self, x_book, x_trade, h_book, h_trade):
        b = self.book_enc(x_book); t = self.trade_enc(x_trade)
        for cell in self.book_rnn: b = cell(b, h_book[0]); h_book[0] = b
        for cell in self.trade_rnn: t = cell(t, h_trade[0]); h_trade[0] = t
        merged, _ = self.cross_attn(b.unsqueeze(1), t.unsqueeze(1), t.unsqueeze(1))
        return torch.cat([b, merged.squeeze(1)], dim=-1), (h_book, h_trade)
```

**Training with existing recipe**  
Muon on all `W,U,Wz,Uz` matrices; AdamW on encoders/attn. Same wmse + aux. State tuple carried across chunks.

**Expected fold-1 WP** 0.6685 ± 0.0030. Strongest prior: event features already help; structural separation should decorrelate.

**Cheapest falsifier**  
T0 screen: fold 1, 2 epochs, h=96 (≈1.1M). If paired Δ < +0.001, discard. Also ablate cross-attn (should lose ~0.001).

---

## Ranking (by expected decorrelation × feasibility)

| Rank | Architecture | Expected Δ vs GRU-256 (paired) | Decorrelation Rationale | Implementation Risk |
|------|--------------|-------------------------------|-------------------------|---------------------|
| 1 | **DP-RNN** | +0.0025 ± 0.0030 | Structural book/trade split matches data generation; minGRU proven on long sequences | Low (minGRU simple, encoders exist) |
| 2 | **LMTS-RNN** | +0.0020 ± 0.0035 | Explicit timescales; leaks learned; single recurrent block | Medium (leak init critical) |
| 3 | **MR-GRC** | +0.0015 ± 0.0035 | Clock-rate separation is real; but 4 cells add branching | Medium (clock_id routing at serving) |
| 4 | **SSM-LD** | +0.0005 ± 0.0040 | Continuous-time diversity; but may not capture discrete events | High (complex numbers, stability) |

**Next step**: Run T0 screens for all four in parallel (4 GPU-hours). Promote any with paired Δ > +0.0015 to T1 (folds 1+0, 3 seeds). Target: at least one member with pairwise error correlation ≤ 0.90 vs GRU-256 for ensemble gain.