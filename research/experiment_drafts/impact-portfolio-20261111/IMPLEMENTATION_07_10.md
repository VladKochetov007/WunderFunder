# Cards07–10: executable B300 preparation

The remaining four mechanisms are implemented in this same draft PR. They add
16 fresh fits: two controls and two treatments per card, seeds204101/204102.
All ten cards now have40 planned fits, zero executed or reused control fits.
No teacher/pretraining fit is needed. Source-bound acceptance remains pending.

## Frozen mechanisms

| Card | Control | Treatment |
|---|---|---|
|07 | Current-target Transformer WMSE | Heads for labels at row offsets1/16/64; primary +0.1×mean of three horizon WMSE losses |
|08 | K16 GRU with matched clean prefix/suffix traversal | Every fourth1000-row chunk, perturb the carried state at offset128 and compare the next32 predictions against a detached clean reference |
|09 | K16 GRU mean head | Zero-initialized log-variance head, clamped[-5,3]; add0.05×mean diagonal Gaussian NLL |
|10 | K16 GRU mean head | Three raw-label classes per target: y<−2, −2≤y≤2, y>2; add0.05×mean cross-entropy |

Card07 uses future labels only inside the loss. The full sequence label view lets
a target cross a TBPTT chunk boundary, but never a sequence boundary. A row must
pass the primary source mask, and its future target element must exist and be
finite. Each horizon uses the source-equivalent pooled two-target WMSE denominator,
summed across the complete accumulation window and ranks. Empty horizons contribute
zero to the fixed three-horizon average. Forward inputs and state never see labels.

Card08 selects chunks4/8/12/16/20 of each complete sequence. Both arms process the
same clean prefix/suffix calls; only the treatment runs the noisy fork. Noise has
coordinate std0.01×max(detached hidden RMS,1e−3), computed per sequence over layers
and hidden coordinates. The auxiliary term is0.01×mean squared prediction difference
divided by(0.01²+1e−8), over eligible suffix rows and both targets. Only the clean
trajectory is carried forward. The private noise stream is defined by the immutable
fit seed and an exact integer counter stored in an FP32 buffer; each active branch
derives a local generator seed from that pair. The counter is checkpointed and its
exact representable range is checked. No global training RNG is consumed.
Evaluation does not draw noise. This is not a contraction guarantee or a speedup.

Card09's NLL uses clipped targets and a separate unweighted eligible-element
denominator. It never changes primary WMSE weights. Card10 reads the raw labels
before the primary objective clips them. Train-only preprocessing records counts
for every target/class; both card10 arms are inadmissible when either tail has
fewer than100 rows or32 fitting sequences. Keep the infeasibility receipt and do
not change thresholds after inspecting valid.

## Sources, precision and integration

Card07 retains the PR75 runtime at
`8626000a4b965763a70011a0291a891aa6535d7e`: H768/L24, context1024, gradient
horizon256, batch128, accumulation4, two B300 ranks, four epochs, Muon/AdamW,
BF16/BF16x9. Cards08–10 use PR79 at
`67c3bd818ea243177eeb1dc789cd3d145eba0e81`: K16/H256/L2, chunk1000, batch32,
accumulation1, one B300, six epochs, AdamW and FP32. For GRU, the only native
changes are paired seed and required device L4→B300. Both arms start from scratch.
The old L4 results are historical context, not these new controls.

The launcher selects the exact runtime/world size/precision environment by card
before importing Torch. GRU removes cuBLAS emulation inherited from a Transformer
shell. Both families retain LR3e−4, WD0.01, clipping1,5% warmup, cosine floor0.1
and all-epoch checkpoints. No compile, kernel, batch, optimizer or throughput
optimization is introduced.

All implementation lives in this packet. The existing explicit auxiliary engine
reuses the pinned primary objective, sampler, update, checkpoint, monitor, resume
and scorer components. Canonical `src/`, `scripts/` and parent configs are untouched.
Model-family imports are lazy so the PR75 Transformer does not need PR79 GRU APIs.
Endpoints strictly restore their complete training state before dropping auxiliary
heads; noise branches are disabled in evaluation. Existing jobs stay on their
original checkout. New code requires new acceptance/profile receipts.

## Evidence and receiver obligations

[Launch commands](B300_CARDS07_10.md) cover both source families. Each stage has an
explicit owner, device allocation, timeout and persistent output. There is no
automatic40-fit dispatch. Small receiver probes check masks, future-label alignment,
empty denominators, paired initialization, causal state, auxiliary gradients,
head removal, noise-state replay, and one-update control parity. Each selected
full-size cell then needs its own profile and resource decision before fitting.
None of those model checks has run locally.

Epoch/update records keep primary and combined loss separately. Diagnostics include
future-horizon/target counts and WMSE, perturbation differences at suffix offsets1/8/32,
variance means/bound saturation, and event confusion/per-class CE. Gaussian NLL
can be negative. For card09 treatment profiles, the canonical nonnegative
`losses.total` field is explicitly scoped to the primary objective; the actual
signed combined objective remains in `objective_terms.total_loss` and metric logs.
Both approval and fit admission verify that primary plus auxiliary equals that
signed total. No scientific loss is shifted or clamped. Per-depth or
primary-versus-auxiliary gradient attribution is optional separate receiver analysis;
the automatic gradient norm is the canonical combined preclip norm.

Evaluate every fixed final endpoint (E4 Transformer; E6 GRU) with the unchanged
official scorer and complete coverage. The exposed official split is not a fresh
holdout. Preserve every paired seed/target delta, failure and negative result;
two seeds do not establish significance. A full comparison accepts the two pinned
runtime families but requires exact source/preprocessing agreement within each pair.

MIDAS records a preparation-only research route and validates complete canonical
plus packet source maps when real evidence is exported. No research-agent/GPU run,
evidence import, performance measurement or quality gain is claimed.
