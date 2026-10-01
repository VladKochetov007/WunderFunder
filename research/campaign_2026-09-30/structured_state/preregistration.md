# Card B implementation commitment

This commitment precedes synthetic checks and all competition data/results.
Scope: root Challenge 2, actor `/root/structured_state_actor_0930`; verification
tier C for competitive usefulness. This actor builds; the independent critic and
controller grade. No competition data, scoring mask, target, or sequence identifier
enters this implementation or synthetic smoke. No GPU launch is authorized here.

## Frozen implementation

- Width 96, two residual blocks. Each block: 64 scalar states plus 32 real
  rotation pairs =128 real state coordinates. State tuple is `(s1,s2)`, each
  `[batch,128]`, compatible with the existing detach-state/EMA teacher path.
- Initial projection is GELU of a learned affine map of standardized input.
  Each block uses independent bias-free B/C/D maps, and an affine sigmoid gate:
  `s[t]=A*s[t-1]+sqrt(1-rho²)*B*u[t]`,
  `u_next[t]=u[t]+sigmoid(G*u[t]+b)*GELU(C*s[t]+D*u[t])`.
  Head is affine(192,96), GELU, affine(96,n_out).
- Learned log half-life = `sigmoid(raw)*log(8192)`, so half-life is bounded
  to [1,8192] and `rho=exp(-log(2)/half_life)` is strictly below one.
  Injection uses `sqrt(-expm1(2*log_rho))` to avoid cancellation.
- Multi initial half-life grid is [2,8,32,128,512,2048], repeated in mode order.
  Single initializes every mode to32. No other parameter or buffer differs.
- Phase is frozen and saved as a buffer, with the deterministic six-point grid
  `[-pi/32,-pi/128,-pi/512,pi/512,pi/128,pi/32]`, indexed by `(pair_index//6)%6`.
  This concrete phase grid resolves the card's unspecified "small deterministic
  phase grid" before results. Phases are identical across arms and blocks.
- Training forward uses separate scalar rFFT convolution and complex FFT for
  rotations, zero padded to the next power of two >=2*T-1. Incoming state
  contributes `A^(t+1)*s_in`; final state is carried across chunks. No T² object.
  Complex tensors are only an internal FFT representation; returned states are
  real, and streaming uses independent real scalar/2x2 updates.
- FP32 state/dynamics under autocast; explicit FP64 model supported for synthetic
  accuracy checks. No learned phase, dropout, event clock, mask, or reset at chunk
  boundaries. Native inference remains a later score-gated task.

## Checks and falsifiers

Fixed synthetic seed930. Compare complete model outputs, incoming/final states,
input/parameter/initial-state gradients against the independent real recurrence.
Use FP64 length65 and FP32 lengths/chunks1/17/64/2000. Check a2049-row timeline
with carried random nonzero states, reset and true row stream parity, future
perturbation, bounded extreme half-lives, long quiet recurrence, and finite gradients.
Absolute tolerances: FP64 forward1e-9/backward1e-8; FP32 output/state5e-4;
FP32 causality1e-4; FP32 gradient2e-3. Report measured errors, not just booleans.
An FFT creates globally distributed numerical roundoff, so causality tolerance is
finite rather than bitwise zero. Mathematical prefix dependence remains causal.
Any failed checks are preserved and diagnosed before integration is proposed.

CPU execution: campaign slice, one thread, CPUQuota100%, MemoryMax2G, no swap,
RuntimeMaxSec300. Profile optimized FFT forward/backward at batch2,T2000 and
forward T20000; independent Python loop is a reference, not training backend.
Record parameter count, analytic per-row MAC estimate, observed CPU RSS/time,
and an analytical batch32 working-memory estimate. CPU measurements cannot prove
observed GPU memory <=5GB or deployment <=52min. Compile is optional; retain FFT
if compiler overhead/support is poor under this bound.

## Future competitive claim

Four fixed epochs, one newly frozen clean fold, same source/components, updates,
losses/auxiliaries/EMA and batch policy as matched controls. Cap0.30GPUh per arm;
paired configurations differ only decay initialization. No checkpoint selection
by evaluation. Multi-minus-single tests initialization/optimization, not that
multiple timescales are unavailable to single. Architecture remains exploratory.
Screen and promotion gates remain those in RESUME_CONTRACT.md/card B, including
fixed80/20 blend, >=+0.0005 paired exact WP, long-gap effect and >=20% learned
modes with half-life>128; no choice using synthetic timing or evaluation scores.
