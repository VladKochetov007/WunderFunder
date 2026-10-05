# Optimizer approximation and activation-shape interpretation

Primary abstract sources checked 2026-10-05; these are theoretical context,
not evidence that a particular change improves Connectome.

[Beyond the Ideal: Analyzing the Inexact Muon Update](https://arxiv.org/abs/2510.19933),
Shulgin et al., 2025, analyzes approximate orthogonalization using an additive
oracle-error model. Its theory and NanoGPT experiments connect approximation
accuracy with learning rate and momentum. Arithmetic BF16 versus FP32 is not
identical to the paper's oracle precision or Newton–Schulz iteration count.
The paper motivates measuring the optimizer approximation rather than
assuming the same nominal LR produces comparable updates.

[On MUON optimization: From non-convergence to an error analysis](https://arxiv.org/abs/2608.04607),
Do et al., 2026, studies generalized Newton–Schulz polynomials, gives
nonconvergence examples for certain stochastic problems, and derives bounds
depending on steps and batch size. It does not show our GRU diverges, nor
justify rejecting Muon as a family. Fixed-step behavior and approximation
error require empirical diagnosis in this task.

[Mining Generalizable Activation Functions](https://arxiv.org/abs/2602.05688),
Vitvitskyi et al., 2026, searches activation functions using synthetic tasks
and out-of-distribution fitness to target inductive biases. Our exposed
financial panel is a different setting. E473's matched linear control and
E481's matched-central-slope cubic control are necessary for distinguishing
periodicity from changed local derivatives; the existing results establish
neither periodic prediction nor unseen-market generalization.

The E485 native/custom update peak discrepancy is measured evidence. A BF16
rounding explanation remains an inference. An appropriate new diagnostic
holds raw gradients, clipping, matrix layout, momentum and NS coefficients
fixed and varies only orthogonalizer arithmetic. It must separately compare
native versus custom gradients under each precision, and identical gradients
through independently implemented optimizer formulas. Report every matrix
and seed; do not widen the failed E485 gates after seeing the discrepancy.
