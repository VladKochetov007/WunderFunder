# E483: isolate recurrent momentum from activation shape

Register before implementation or scoring. The user's activation-by-optimizer
question remains open: bounded positive candidate shape improved Muon at three
initializations, while its AdamW effect varied. Investigate whether the shape
effect depends on recurrent gradient-history smoothing. This is a controlled
momentum comparison, not a promised forecasting or public-score improvement.

Use fresh initialization seeds271831/271832/271833, both fixed baseline `u`
and fixed positive sine `u + 0.1 sin(pi u)` (`u=tanh(v)`), and recurrent Muon
momentum0.8/0.9/0.95: **18 full fits**. Preserve all cells regardless of outcomes.
Do not reuse seeds selected for large gains. Evaluation remains the same16
previously exposed components; new seeds are not new market validation.

Reuse the frozen E481 model/initialization semantics and E470 fit-only loader,
with exactly H64/two layers/61058 parameters/42 causal inputs,24 fit components,
B4×256 and474 updates, one epoch, constantLR0.001 andWD0.01. Gradient clipping,
order, telemetry steps0/46/236/473 and full-sequence state handling remain fixed.
Bind all parent code hashes. Generate a private initial state for each of the
three explicit new seeds; do not alter E481 global seed/arm lists or monkeypatch
its optimizer factory. Save honest E483 receipts for the18-cell matrix.

**Isolate the factor:** only groups marked `muon=True` receive the specified
first beta. Every AdamW group remains `(0.9,0.999)`, including input/head/bias
parameters. Keep Muon Newton–Schulz iterations5, scaling0.2 and unchanged CUDA
BF16 orthogonalization. LR scheduling and Adam momentum must not vary with
recurrent momentum. Record resolved groups, parameter names, gate blocks and
effective betas, and verify them with independently computed update checks.

Predeclared contrasts, separately per seed:

- Sine-minus-baseline at each momentum: three.
- Momentum0.8-minus0.9 and0.95-minus0.9 for each shape: four.
- Joint shape-by-momentum interactions relative to0.9: two.

Thus27 global contrasts in total. Use2000 numeric-sorted component resamples
jointly shared across arms within each seed; report both target scores and all
global effects/intervals without pooling seeds or selecting a champion on this
panel. Count clipping over all474 updates from the actual pre-clip norm (or an
equivalent detached device-side counter); four sampled telemetry points cannot
estimate whole-run clipping frequency. Inspect this count and fixed-step
recurrent/non-recurrent gradient/update RMS descriptively; they do not diagnose
a causal mechanism. Any local update helper must reproduce the frozen clipped
step in CPU/CUDA checks, without altering the immutable parent helper.

Independent blind design critique, source review and bounded fit-only CPU
checks precede an actual B4×256 GPU probe. Require finite and changed steps for
all18 cells, alpha0/native reset-after output/state/gradient parity and clipped
Muon-step parity, and synthetic checks proving non-Muon beta invariance. Probe
failure blocks full fits. Score only after complete runner/artifact/fit/checkpoint
receipts validate, using an independently reconstructed model/score path.

One local GPU worker only;2CPU/6GiB host worker,2GiB allocator/3GB observedVRAM
stop, actual aggregate12CPU/30decimalGB RAM/no swap. Reserve at most7200 GPU
seconds from the existing24-hour local block and80-device-hour campaign ceiling;
no new budget. Timeout invalidates a complete-matrix claim; keep partial artifacts
as partial. Start only after E481 teardown and verified resource/budget headroom.
Stop new admissions07:00UTC/10:00Kyiv; already admitted jobs drain gracefully.
Every result/failure/receipt enters research notes and JSONL; figures go in this
study's `eda/` directory and the running marimo notebook.
