# CTXMEM01 pre-data mathematical and implementation specification

Authority is overnight amendment-01 and the single pre-data projection correction.
This specification describes one fixed memory treatment and its matched current-state
control. No half-life, rank, event detector, projection, precision, or seed search is
authorized. The native cost preflight precedes forecast training.

For a frozen two-layer h128 GRU, let h_t be its top-layer output on the current
causally built input. Set q_t=P h_t, with P the exact FP32 16×128 Hadamard rows
[68,93,98,43,72,42,49,44,7,104,76,95,26,64,41,65], scale
0.0883883461356163 and little-endian FP32 row-major SHA256
0ed6109334d68ec914d5c5e3da6b7f0ed898db05828ace54c76532a4c443f20b.
P is a fixed rank-16 subspace, not a learned representation or a recovered market
state. The selected row labels span all seven GF(2) coordinate bits.

The existing causal AND clock is e_t=(raw_t[0]!=raw_{t-1}[0]) AND
(raw_t[11]!=raw_{t-1}[11]); this is built feature178, read before normalization.
At the first row of every sequence both m4 and m32 equal q_0, regardless of e_0.
Subsequently mH_t=lambda_H*mH_previous+(1-lambda_H)*q_t on clock events,
otherwise it holds. Exact binary64 lambda4=0x1.ae89f995ad3adp-1 and
lambda32=0x1.f50765b6e4540p-1. Both products are separately rounded binary64
before their sum; no FMA in this recurrence. Carry actual m, not the lfilter delay.
Empty chunks preserve state; ordinary chunks never reset. A new sequence resets
the online input construction, GRU state and both memories together.

Treatment head input is [float32(m4),float32(m32)]; control is [q,q]. Both
implementations compute both histories. The 32→128 matrix E has no bias and starts
at exact zero. It adds to the original first head affine output before GELU.
All original head parameters remain trainable. Backbone, normalization and P remain
bitwise fixed. This adds 4096 trainable parameters. The duplicated-q control has
rank16 and symmetric extra-weight gradients; it is equal in parameter count, not
in identifiable feature rank. Its effect can algebraically fold into the existing
current-hidden head map. Memory provides an explicit history path, whose utility
is a forecasting hypothesis rather than a proven physical interpretation.

Training uses original H2 AMP FP16 affine operations, fused AdamW and fixed loss/
auxiliary coefficients. Projection is dense float32 with CUDA TF32 disabled,
outside autocast; h is explicitly converted to FP32 first. CPU compiled SciPy
lfilter runs only the compact event list in float64, with zi=lambda*m_previous.
Returned float32 head features are copied to the model device. No gradient through
the frozen representation or histories. Constructing E preserves ordinary model
initialization and downstream CPU/CUDA RNG streams; P uses no global random draws.
Training remains eight fixed two-epoch screens (current/memory × seeds3/4 × folds0/1),
fixed epoch1, followed only on all amendment gates by two final treatment heads.
The exact training adapter/commands require a separate freeze after native cost
passes; this document does not allocate those fits.

Native serving is derived only from validated critic/native_order_v2, retaining
the exact original feature engine, model order, member accumulation and blend
rounding. P uses the original dense FP32 GEMV kernel. E follows the same storage/
GEMV precision policy as ordinary head weights: FP16 when half_weights=True on
the v3 implementation; FP32 for ordinary fallback/half_weights=False. There is no
zero-weight work skipping. Memory uses the explicit separate binary64 recurrence.
Different BLAS/native projection summation can introduce ordinary float32 differences;
supplied-identical-q recurrence is checked bitexact separately from full-model
Python/native output tolerances. The actual frozen package has FP16 trunks, fast
activations, equal1/3 blend and the unchanged Python callback wrapper.

The cost prototype preserves all original trained arrays of fixed3
[plain128seed3,plain128seed4,c500] and its exact member order/weights. Only the first
two members gain P and E; c500 remains byte-identical. Nonzero E matrices are fixed
PCG64 seed20261009 Gaussian(sd=.03), not fitted, so all new native work executes.
The zero-E qualification must reproduce archived fixed3 predictions bitexact.
Synthetic tests and cached raw-only timing do not open any price labels, scored
masks or OOF prediction caches. Prototype forecasts are used only for exact parity.
An overhead pass provides no score evidence and does not replace the eventual
complete-ZIP correctness and unchanged 50-minute gate on 39.4M projected rows.
