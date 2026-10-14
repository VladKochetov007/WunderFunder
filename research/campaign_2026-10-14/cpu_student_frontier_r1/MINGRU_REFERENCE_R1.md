---
kind: experiment
status: complete
created_utc: 2026-10-14T01:51:58.291280+00:00
updated_utc: 2026-10-14T01:51:58.291280+00:00
hypothesis_ids: [H-STUDENT-LADDER-20261014]
experiment_ids: [E-MINGRU-REFERENCE-20261014-R1]
evidence_paths: [MINGRU_IMPLEMENTATION_TESTS_R1.xml]
---

# minGRU reference with explicit actual-state semantics

## Question and implementation

The cheaper-recurrence arm now has an isolated [Python reference](../../../src/wnn/min_gru.py) for the positive-candidate minGRU variant in [Feng et al., Appendix B](https://arxiv.org/html/2410.01201v3). One packed input projection produces candidate and write gate; there is no hidden-to-gate or hidden-to-candidate matrix. Both sequence and single-row methods receive the actual previous hidden state. Zero means exactly zero, including during chunked continuation and differentiation.

The scan composes affine pairs `(a,b)` with `(c,d)` as `(ca, cb+d)` using parallel doubling. The initial state contributes to the first addition once. This is our own direct affine-composition implementation of the recurrence, rather than the paper's log-space scan. It avoids logarithms or transformations of the initial state. The work and retained autograd intermediates are O(T log T), so this reference does not establish an efficient GPU scan. There is no change to the active GRU fit, trainer CLI, model factory or native callback.

## Evidence

[Nine tests](MINGRU_IMPLEMENTATION_TESTS_R1.xml) pass in1.84seconds under1CPU/1GB after a live aggregate commitment check of15.889GB. Tests compare sequential and parallel recurrence for signed initial states and irregular lengths; compare input, parameter and initial-state gradients in float64 including exact-zero h0; verify a closed-form zero-state fixture; and check causal prefix, chunks, reset replay and finite nonnegative outputs over20,000 synthetic rows. The empty chunk preserves the caller's state. Ruff format/check pass. No GPU work, fit, dataset evaluation or serving benchmark was performed.

The source paper's candidate function is positive, with a linear nonnegative branch and sigmoid negative branch. A zero preactivation yields candidate0.5, while initial state stays0. With zero logits and inputs, the tested recurrence produces0.25,0.375,0.4375. This explicitly prevents the zero-state interpretation mismatch described in the [theory supplement](DISTILLATION_AND_RECURRENCE_THEORY_R1.md).

## Cost hypothesis and limitations

Keeping the planned equal-width encoder and skip readout, L such layers would have `(2L+2)H²+213H` matrix parameters, excluding biases, compared with `(6L+2)H²+213H` for the GRU. Thus H384×2 minGRU would have966,528 matrix parameters, near H256×2 GRU's972,032. This is a shape-derived comparison only: nonlinearities, packing, scan cost, trainability and callback behavior must be measured. The positive candidate, lack of state-dependent gates, and scan arithmetic are all explicit method changes, not exact GRU optimizations.

Root owns the successor: integrate a separately identified training family, verify actual-data forward/backward and native row callback semantics, profile memory/time, then allocate a bounded fit against the fixed teacher/feature/checkpoint protocol. Preserve the exploration arm even if one finite-budget test is inconclusive. Do not divert the already-running eight-shape ladder or promote this code-only result as student quality.

## Full model reference

The separate `MinGRUModel` now preserves the actual baseline encoder semantics: `enc=linear` names the affine encoder family but is followed by GELU. It uses the same normalization buffers, GELU encoder, `[recurrent, projection]` skip readout and GELU hidden head. Independent E/R/M are explicit; state is `[layers,batch,R]`. Its dictionary keys are `mean`, `std`, `inp.*`, `cells.N.input_projection.*`, `head.0.*` and `head.2.*`. The native prototype will use these exact keys and the existing112-raw-to211-causal-feature builder.

[Full-model tests](MINGRU_MODEL_TESTS_R1.xml) pass11/11 in2.14seconds: the prior nine tests plus three-layer full/chunk/reference-row parity with nontrivial normalization, correct all-layer carry/reset, matrix accounting and finite parameter gradients. The1GB test followed a16.927GB aggregate-bound check. This remains a separately defined research model; it is not yet selectable by the production trainer and has no fit or native serving result.
