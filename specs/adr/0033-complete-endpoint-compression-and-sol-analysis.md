# ADR0033: Compare complete compressed blends; Sol analysis exception

Status: Accepted. Date: 2026-10-11.

The latest user makes final submission score the decision target: a larger blend
may justify quantization loss if its complete endpoint is better. The user also
explicitly requests marimo-pair, creative research, and Sol 6.1/xhigh analysis
subagents with managed resources, while local and Kaggle work continues.

Compare fixed complete endpoints against the strongest feasible Q8 alternative
on the same official metric and qualified evaluation membership. A precision
loss from a different pair or an in-sample serving panel cannot close all lower
precision combinations. Separate disk representation from native inference
precision. In particular, current Q8VNNI ZIPs store float32 NPZ tensors; reducing
that transport representation may allow a larger Q8 ensemble. Any change needs
actual producer/consumer compatibility, matched output/quality measurements,
size, deterministic reset, ABI and one-CPU runtime evidence. No native INT4,
SOTA compression, exact equivalence or competition runtime claim is implied.

Freeze a bounded endpoint set before grading; record selection exposure and
conditional gains. Keep full-data serving fidelity separate from held-out
recipe evidence. No local-delta-to-public arithmetic or numerical public promise.

This supersedes ADR0029's Luna-only rule **for the explicitly requested analysis
roles in this task**: use gpt-6.1-sol/xhigh, isolated worktrees, no recursive
delegation, at most three subordinate roles total. Root plans and integrates;
implementation and independent artifact review remain Luna/max. Analysis roles
have no fit, provider or spending authority; CPU support is explicitly bounded
inside the existing five-core, 18/20 GB, no-swap campaign slice.

ADR0032's package-now instruction remains effective. Human uploads, independent
weights, permitted data, official clipping/magnitude weights, source-qualified
ancestry and causality remain binding. No paid resource or budget is added;
Vast B remains stopped with zero credit. Local and Kaggle successors require
their existing actual-device, quota and source-bound admission checks.
