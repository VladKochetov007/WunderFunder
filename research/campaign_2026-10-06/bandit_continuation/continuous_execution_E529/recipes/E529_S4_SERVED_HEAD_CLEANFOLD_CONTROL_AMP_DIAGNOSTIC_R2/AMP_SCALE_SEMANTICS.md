# AMP skip semantics reviewed for R2

The frozen trainer creates `torch.amp.GradScaler(device.type, enabled=args.precision == "fp16")` in `scripts/train_rnn.py` and calls `unscale_`, the existing skip counter, `scaler.step(opt)`, then `scaler.update()` for each attempted profile update. The profile-only early return prints `GRAD_STATS`/`PROFILE` before `EpochUpdateCounter.finish()`, which is why R1's partial profile receipt contains gradient nonfinite statistics but no final `amp_skipped_updates` aggregate.

The existing `src/wnn/run_identity.py::EpochUpdateCounter.observe_unscaled` reads the scaler's already-recorded `_found_inf_per_device(optimizer)` immediately after the trainer's existing `unscale_`, sums the flags on device, and counts nonzero as a skipped optimizer update. It performs no gradient or optimizer changes. This is an independently implemented run diagnostic already present in the frozen source.

For the installed profile runtime (PyTorch `2.14.0+cu130`, source git revision `08187d9e0fba026dc8217405802ab5381dc88d90`), the Python source pins are:

- `torch/amp/grad_scaler.py`: `97c411da028daaf6a6ed15d06b9b20c017404846db68203be1a586e276e44039`
- `torch/optim/adam.py`: `6f4a96e7bccd5e233a66f846c39f7a9da0157cf657c2275d2a269a78a6b90921`

In the pinned `grad_scaler.py`, `_maybe_opt_step` (lines 363–372) calls the optimizer only when the recorded found-inf sum is zero. `GradScaler.step()` documents that it skips an optimizer update for inf/NaN gradients (lines 375–396). Its fused-optimizer branch marks the optimizer AMP-aware and passes `found_inf` into `optimizer.step()` (lines 418–475). `GradScaler.update()` documents that skipped iterations multiply the scale by the backoff factor and supplies the combined found-inf tensor to `_amp_update_scale_` (lines 489–553). In the pinned `torch/optim/adam.py`, Adam enables this AMP-aware path (line 107); fused AdamW forwards `found_inf` to `torch._fused_adamw_` (lines 877–895), then subtracts found-inf from the incremented state-step counters (lines 896–899). Thus a scale reduction after one original update is direct evidence of an overflow/skip event for that iteration, while a stable scale indicates no skip until a growth interval is reached. The 60-step window is below the default 2,000-step growth interval.

The installed wheel does not include the CUDA kernel implementation source, so this audit binds the Python caller/contract and exact runtime version/source hashes; it does not claim an independent audit of compiled kernel internals. R2 logs scale transitions and retains the trainer's existing `GRAD_STATS` line. A single scale decrease should coincide with the one nonfinite step seen in R1; the R2 result itself remains diagnostic and requires root review.
