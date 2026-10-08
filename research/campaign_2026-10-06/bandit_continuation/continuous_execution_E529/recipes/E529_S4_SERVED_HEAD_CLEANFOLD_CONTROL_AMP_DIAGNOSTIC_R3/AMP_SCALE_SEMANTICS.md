# AMP skip semantics reviewed for R3

The frozen trainer creates `torch.amp.GradScaler(device.type,
enabled=args.precision == "fp16")` in `scripts/train_rnn.py` and calls
`unscale_`, the existing skip counter, `scaler.step(opt)`, then
`scaler.update()` for each attempted profile update. The profile-only early
return prints `GRAD_STATS`/`PROFILE` before `EpochUpdateCounter.finish()`, so
R1's partial profile receipt contains gradient nonfinite statistics but no
final `amp_skipped_updates` aggregate.

The existing `src/wnn/run_identity.py::EpochUpdateCounter.observe_unscaled`
reads the scaler's already-recorded `_found_inf_per_device(optimizer)` directly
after the trainer's existing `unscale_`, sums the flags on device, and counts
nonzero as a skipped optimizer update. It performs no gradient or optimizer
changes.

R3 pins the installed PyTorch `2.14.0+cu130` Python sources separately by
their runtime roles:

- `torch/amp/grad_scaler.py`:
  `97c411da028daaf6a6ed15d06b9b20c017404846db68203be1a586e276e44039`
- `torch/optim/adamw.py` (the `AdamW` class/configuration module):
  `39c5eaa04f83632285b114fd093eeb0006866e4adf59ad50cc28de12e63a8459`
- `torch/optim/adam.py` (the unwrapped inherited `AdamW.step` implementation):
  `6f4a96e7bccd5e233a66f846c39f7a9da0157cf657c2275d2a269a78a6b90921`

R2's runtime version was correct. Its verifier used
`inspect.getsourcefile(torch.optim.AdamW)`, which selects `adamw.py`, but paired
that path with the `adam.py` digest. The R3 verifier checks both the class
module and `inspect.unwrap(torch.optim.AdamW.step)`, which resolves to the
inherited update implementation in `adam.py`. The test fixture distinguishes
the `AdamW` class module, the `adam.py` implementation, and the `optimizer.py`
decorator wrapper, and rejects a changed update implementation.

In the pinned `grad_scaler.py`, `_maybe_opt_step` calls the optimizer only when
the recorded found-inf sum is zero. `GradScaler.step()` documents that it skips
an optimizer update for inf/NaN gradients. Its fused-optimizer branch marks the
optimizer AMP-aware and passes `found_inf` into `optimizer.step()`.
`GradScaler.update()` applies backoff on skipped updates. In the pinned
`adam.py`, Adam enables this AMP-aware path; fused AdamW forwards `found_inf`
to `torch._fused_adamw_`, then corrects the incremented state-step counters.
Thus a scale reduction after the original update is evidence of an overflow/
skip event for that iteration, while a stable scale indicates no skip until a
growth interval. The 60-step window is below the default 2,000-step growth
interval.

The installed wheel does not include the CUDA kernel implementation source,
so this audit binds the Python caller/contract and exact runtime version/source
hashes; it does not claim an independent audit of compiled kernel internals.
R3 logs scale transitions and retains the trainer's existing `GRAD_STATS`
line. A single scale decrease should coincide with the one nonfinite step seen
in R1; the R3 result remains diagnostic and requires root review.
