# E483 independent source review

Disposition: **GO for the separately controlled B4×256 CUDA probe.** This is not approval for the 18 full fits or a scientific result. I made no GPU calls and did not score or read held-out rows.

The frozen implementation matches the blind design. It creates one E481 state per fresh seed `271831/271832/271833` and loads that exact state into both shape arms and all three momentum settings. It preserves the 42-feature, two-layer H64 GRU and 61,058 parameter count. The only changed optimizer setting is `betas[0]` on the single Muon group, whose exact members are the four recurrent `weight_ih/weight_hh` matrices across two layers with three gate blocks. The disjoint group receipt requires complete unique coverage of trainable parameters; it fixes every AdamW group's betas at `(0.9, 0.999)`, and verifies LR `0.001`, weight decay `0.01`, five orthogonalization steps, and Muon scale `0.2`.

The tests independently exercise two non-collinear Muon gradients and compare the two-step result against the frozen momentum recurrence. They show beta-specific Muon updates while AdamW parameter updates remain bit-identical. A second test compares the actor’s clipped update helper with the frozen E474 helper. Native reset-after GRU output/state/gradient and clipped beta-0.9 optimizer parity also pass on CPU. The GPU probe uses the same parity checks under the registered full-FP32 context, then exercises all 18 shape/beta/seed cells at B4×256 and requires finite loss, output/state, clipping norm, parameters, and at least one changed parameter tensor. The source inspection confirms it configures the CUDA allocator before creating CUDA model tensors.

The fit loop preserves the fixed seed-shuffled fit-node order, six batches of four components, 79 chunks per batch, carried GRU state within each component batch, and detach after every update. It performs 474 updates per arm. The sampled telemetry indices `0, 46, 236, 473` include current batch rowgroups, exact chunk start, loss, state magnitude, and preclip/update RMS. The clipping count increments from the actual detached preclip norm on every update, rather than extrapolating from telemetry. Loss, input, prediction/state, norm, and post-step parameter finiteness are checked; nonfinite gradients fail through `clip_grad_norm_(..., error_if_nonfinite=True)`. The candidate map is monotone on `[-1,1]`, maps the endpoints to `-1` and `1`, and the reset-after state update is a convex combination of the candidate and prior state, so zero initialization gives an exact-arithmetic `|state|≤1` bound. Runtime telemetry records sampled `state_abs_max`, but the code does not separately fail on a tolerance for that bound. Fit loading uses the E470 fit-only loader, requires 24 finite fit sequences, and binds the dataset and cohort bytes. The implementation imports the immutable E481 initializer and does not patch its optimizer factory.

Independent validation used a 1-CPU, 2-GiB limit. All four synthetic tests passed; Ruff, py_compile, and the unmocked `check_parent_sources()` identity check passed. I also inspected the CPU preflight receipt: it contains all 18 unique fit-only cells, one finite B4×256 update per cell, three distinct state hashes, exact parent/data/cohort hashes, and `test_rows_loaded=false`. The receipt is not GPU evidence.

Frozen source identities:

| Object | SHA256 |
|---|---|
| `fit_momentum.py` | `54f4fbc66f69e165a5b21ff740faa556ef368d326ebbeb24736d68a36440bbe8` |
| `tests.py` | `d63b46acc9abae85e6ebac2bdda0c89de01832f197c89b50f926d3fe675e9a42` |
| `gpu_probe.py` | `af3ac0fba649d0027841f07ccd551798222f57f3da569e84b7064393a4f775c4` |
| CPU preflight manifest | `d9fb30e5c809fceded545ab39ff9779def9b16ee467586723381fa868cc4cb13` |

The remaining gate is an independently supervised actual CUDA probe with the registered FP32 policy, memory bounds, teardown checks, and exact source identities. Only if every cell and parity gate passes should the controller consider the full matrix. The preregistered 27 contrasts remain separate by seed; these source checks support no claim about their outcomes or public performance.
