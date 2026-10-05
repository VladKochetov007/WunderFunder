# E477 CUDA probe mode incident and v2 review

## Preserved v1 incident

The controller reports that sealed E477 probe v1 failed before its first optimizer update with `RuntimeError: cudnn RNN backward can only be called in training mode`. It had entered `.eval()` for both the candidate GELU/p=0 model and its E474 reference before running the baseline gradient-parity backwards. PyTorch's cuDNN RNN backward requires a training-mode forward to create its reserve space. The zero-dropout baseline has identical forward behavior in `.train()` and `.eval()`, so using a fresh `.train()` forward for gradient parity is correct. The attempt exited 1, tore down, and had no OOM. The executed v1 source remains preserved at SHA-256 `1275b980942943ef9982992d965587ffdd3dacc21d593e41d02fdf9bf282cdc4`.

## v2 source review

The new `gpu_probe_v2.py` uses fresh train-mode candidate and reference forwards for the alpha-free GELU baseline gradient comparison. Both have p=0 dropout, so this changes cuDNN's backward-reserve behavior without changing the evaluated function. It preserves the v1 output/state tolerance (`atol=1e-6`, `rtol=1e-5`) and gradient maximum threshold (`1e-5`), does not widen any threshold, and additionally fails closed on nonfinite parity gradients. It then runs every one of the fixed 12 seed/activation/dropout cells in train mode and requires a changed parameter plus finite updated parameters after the actual B4×256 update. Precision context, fit-only cohort, parent source checks, and peak-memory collection are preserved. It writes `probe.json`; the root-sealed artifact must name that file.

The v2 result's source-hash map explicitly includes `fit_activation_head_mixture.py`, `tests.py`, preserved `gpu_probe.py`, and the executed `gpu_probe_v2.py`. The numerical trainer and CPU test source remain unchanged from the original review.

I ran only bounded CPU checks, no CUDA command:

- `gpu_probe_v2.py --help` imports successfully from the repository working directory.
- `python -m py_compile gpu_probe_v2.py` succeeds.
- Ruff format and lint pass for the new probe.
- The unchanged E477 CPU invariant suite passes under one CPU and a 2 GiB memory cap.

Exact identities at review time:

- Preserved v1 probe: `1275b980942943ef9982992d965587ffdd3dacc21d593e41d02fdf9bf282cdc4`
- New v2 probe: `20cb61632ceb43bbe0b3b858d98a30ef97c0f9267d35fbb424d467f416a57594`
- Unchanged fit source: `2d4b1b103dd31912fea7ab24a63e6ce176e532dca8d0dfa2cdb32aebd9e567d7`
- Unchanged tests: `876bd7bb397ef2f948fe0fe8f4bcd75b41bc6f67223892b0060ffe5b960cb778`

**Decision: GO for a newly sealed v2 actual-data probe only.** This review does not admit full fits. The v2 probe has not been run by this reviewer; the root controller must seal the new source and exact `probe.json` artifact before execution.
