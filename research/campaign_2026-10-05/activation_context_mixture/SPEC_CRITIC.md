# E484 blind design critique

**Reviewed before implementation.** Design SHA256: `767e25d826ccce594f7549be578d3c407e8aa83fc5aad0b4968885ce2576a4e8`.

The six-arm comparison is a useful small mechanism screen if all six functions share the same frozen trunk/head weights at initialization and differ only in their registered readout activation gate. GELU and fixed 50/50 anchors have 61,058 parameters; learned-static and all dynamic gates have 61,122. State that parameter-count difference explicitly in reports: the zero-added-parameter anchors are descriptive references, while the static-versus-dynamic context comparisons are parameter matched. No result here is a fresh market validation or public-score estimate.

The context normalization must be exactly per sample and time step over the 64 channels:

```text
c_tilde = c / sqrt(mean(c * c, axis=-1, keepdims=True) + 1e-6)
```

Do not center the vector or reduce over batch/time. The epsilon makes an all-zero vector finite; test that case and compare a nonzero vector against an independent numeric calculation. Since every gate coefficient starts at zero, static and dynamic gates must equal `0.5` exactly at initialization, and all four learned-gate arms must initially reproduce the same fixed-mixture function. Verify that creating zero coefficients consumes no RNG, every arm loads identical shared baseline tensor bytes, and all gate coefficients remain finite. Test finite/nonzero coefficient gradients for a synthetic context and retain an unsaturated synthetic example so a detached or incorrectly broadcast gate cannot pass.

Resolve one construction ambiguity before coding: define `z_t` as either the raw `inp(x_t)` linear projection or the actual encoded vector after the existing `GELU(inp(x_t))`. The design calls it the existing input projection, while this model's recurrent input is the activated encoding. I recommend using that existing activated 64-wide encoding and naming it explicitly in the design and implementation; do not add a new projection. Similarly keep `h_t` as the output of the existing head hidden linear before any activation, and `r_t` as the current output of the existing two-layer GRU. These distinctions make the three dynamic sources reproducible.

The mixture belongs after the existing hidden head linear and before its activation:

```text
gate = sigmoid(a * c_tilde)       # dynamic, a=zeros(64)
mixed = gate * GELU(h) + (1-gate) * LeakyReLU(h, 0.01)
prediction = existing_final_linear(mixed)
```

For the learned-static control use `sigmoid(b)` with `b=zeros(64)`. Preserve the original GELU module and exact fixed mixture for their anchor arms. Do not modify the final linear, concat order, GRU, input projection, fit normalizer, dropout, loss, optimizer, scheduler, batch order, or state-carry/detach procedure. Keep gates differentiable: no `.detach()`, `.item()`, or in-place operation on gate/context paths. A row-wise `dim=-1` implementation must produce shape `[B,T,64]` without mixing examples or timesteps.

Require synthetic tests that verify exact parameter counts and initial output equivalence across all six arms, baseline forward/final-state/input-and-parameter-gradient parity against the frozen E477 model, and finite gradients reaching each gate's 64-vector. Test prefix causality by perturbing only rows after a cutoff and confirming all earlier predictions and gates stay unchanged for all three context sources. Test per-sequence independence by changing a different batch element and confirming the first sequence is unchanged. The recurrent gate must use only the current GRU output; the projection and head-self gates must use their specified current-row tensors. No pooling, sequence normalization, future-row statistics, or batch statistics may enter these gates.

Fit controls are two fresh seeds `271834/271835`, one identical initialization and one identical seed-shuffled fit order across all six arms per seed, 24 fit components, B4×256, 474 updates, constant AdamW LR/decay, and zero dropout. Check the all-six shared initial bytes and common row order from receipts, and keep state carried across chunks within each four-component batch and detached at the same update boundary as the frozen fit. Record finite losses, gate maxima/saturation descriptively, exact optimizer groups, rowgroups, chunk offsets, and fixed-step telemetry; do not use those diagnostics to select an arm. Preserve every fit and all 16 registered contrasts.

The three primary comparisons per seed are recurrent minus static, recurrent minus projection, and recurrent minus self-context. Five more preregistered descriptive comparisons per seed are static minus fixed, recurrent minus fixed, projection minus fixed, self-context minus fixed, and fixed minus GELU. Thus there are eight contrasts per seed and 16 total. For each seed, compute official centered weighted-Pearson moments and recompute global WP on the same 2,000 numeric-sorted component bootstrap draws shared across all six arms. Also report both target WPs. Keep seeds and contrast families separate, retain raw selected targets/predictions/masks and component provenance, reject nonfinite raw labels before clipping, and make no public-score or architecture-novelty claim.

An independent source reviewer and fit-only B4×256 GPU probe remain hard gates before any full fits. Use the existing one-worker resource policy and wait for E483 teardown; preserve all attempts and artifacts. Source and probe receipts must bind the exact parent trainer, metric, and dataset/cohort hashes. The preregistered 12-fit and 1,800-second limits are ceilings, not permission to exceed the current global resource or cutoff rules.

**Admission:** implement only after the `z_t` definition is made explicit. Then require the synthetic invariants above, an independent grader that validates the actual source/fit/checkpoint receipts before reading test sequences, and the admitted GPU probe before fit launch.
