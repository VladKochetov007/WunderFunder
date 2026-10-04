# Independent source critique: finite-aware E425 diagnostics

Reviewed read-only on 2026-10-04. No imports, tests, training, or jobs were run.

## Result

No blocking correctness issue found in the finite-aware accounting patch. The active module differs from the preserved passed-v2 module only in its diagnostic docstring and gradient accounting: it tracks finite/nonfinite attempts per expert, safely replaces nonfinite gradient entries before norm arithmetic, excludes an expert's whole attempt from the finite norm mean when any of its gradients are nonfinite, and emits `null` when its finite denominator is zero. It does not mutate `.grad` or alter a loss, optimizer, model forward, or update.

The measurement scope is documented accurately. In `train_rnn.py`, the diagnostics observe gradients after `scaler.unscale_(opt)` and after `dynamics.perturb_and_clip(step_count)`, immediately before `scaler.step(opt)`. This is the complete objective's gradient on expert GRU/head parameters; it is not an isolated main-loss gradient. The diagnostic counter records optimizer attempts itself; the trainer's `EpochUpdateCounter` separately owns AMP skips and completed optimizer updates. The post-call parameter snapshot delta is a direct movement measure, so a skipped step with unchanged parameters correctly contributes zero movement.

The new test distinguishes the original failure: it injects an infinite gradient into one expert while keeping route probabilities finite. The old source would produce an infinite mean and count that expert as nonzero; the repaired source instead produces zero finite samples, one nonfinite sample, and `None` for its finite mean, while other experts retain finite means. The unchanged parameters also exercise the zero-delta result.

## Nonblocking test caveat

The test manually invokes `after_optimizer_step` without calling an optimizer. That is a useful test of measured parameter deltas but does not exercise `GradScaler.step` or prove trainer skip-counter integration. The accompanying documentation makes this boundary explicit. For a stronger later regression, add one assertion that each expert's `finite_steps + nonfinite_steps == optimizer_attempts`, and use a small fake scaler/optimizer integration only if validating the trainer-owned AMP skip counter becomes part of this patch's acceptance criteria. This is a coverage improvement, not a blocker for the current accounting logic.

## Reviewed identities

- Preserved passed-v2 diagnostics source: `1337b008ef1e352be9232a023e099a25b5d0017709611d7d2ae89d0d5c0d8822`
- Active finite-aware diagnostics source: `61119c2a00a91ec43bc6db39cff28c1bad81e117e7c78b9058b78fcd5c271d2d`
- New finite-accounting test: `5ac470a3e7ba98e2a1199eaf27b2d58e20f62bc3a589adb0f23cf6cb6573e0c8`
- Accounting note: `fd72b75eb0cad9f206b5a36bee2fc19df699797facc3cd1606f3289fdfcabe7d`
- Trainer source at review: `6f53181c9db9642c214701a632a85b0c5686541e397efa32ee3f0c90112e7f80`
- Specialist model source at review: `33ecf61be096bcf0ddd018a9581970bf4786e2225d75d9154a09fcd70cae925f`

The E425 finite-accounting SHA list includes only the diagnostics module, its dedicated test, and the accounting note. The pre-existing trainer and model sources were inspected for call-site semantics; no trainer or model edit is part of the finite-aware patch.
