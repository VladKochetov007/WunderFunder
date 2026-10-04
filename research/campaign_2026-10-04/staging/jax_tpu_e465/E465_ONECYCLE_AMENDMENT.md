# E465 OneCycle schedule amendment

This amendment freezes the one-epoch schedule before any fit. It applies to
both the FP32 and BF16 E465 arms, which start from the same fixed epoch-3
parent export and use fresh AdamW optimizer state. It is an AdamW continuation
experiment; it does not claim optimizer equivalence to the parent's Muon run.

The production source is `src/wnn/training_dynamics.py::build_scheduler`.
With E465's `warmup_frac=None`, `lr=0.001`, `optim=adamw`, and
`head_lr_mult=1`, this resolves to PyTorch `OneCycleLR(total_steps=7460,
pct_start=0.05, div_factor=25, final_div_factor=100,
cycle_momentum=True)`. Beta2 remains 0.999. Beta1 cycles inversely with the
learning rate from max momentum 0.95 to base momentum 0.85 and back to 0.95.

The production update indices are 0 through 7459. The warmup endpoint is index
372, where LR is 0.001 and beta1 is 0.85. The final used index 7459 reaches LR
0.0000004 and beta1 0.95. The training loop advances the scheduler after each
of its 7460 updates, so the resulting index 7460 is also checked as the
post-last, unused scheduler state. PyTorch allows that one state; it evaluates
the cosine phase just beyond the endpoint. The JAX-side scalar port preserves
this behavior without clamping. Further scheduler steps are outside this
contract.

`src/wnn_jax/onecycle.py` is standard-library-only. The bounded reference
checker uses one scalar AdamW parameter with zero gradients and compares LR and
both betas to the actual repository scheduler at all 7461 indices. It does not
load a model, data, JAX, or an accelerator. The runner draft caps it at one CPU,
2 GiB RAM, and 180 seconds. Root must run and review that reference check, and
retain all other E465 admission gates, before fitting.
