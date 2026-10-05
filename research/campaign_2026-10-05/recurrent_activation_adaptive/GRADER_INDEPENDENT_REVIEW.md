# E479 grader independent review

**Decision: GO for the controller-owned grade, bound to the exact scorer hash below.** This is a source and synthetic-test review only. I did not open evaluation labels, validation parquet rows, prediction checkpoints, or run a grade.

## Frozen identity reviewed

- Scorer: `score_recurrent_adaptive.py`, SHA-256 `83ec9a2a64b34b1d6532e1785ff2cb3d95d5bfec686f20e13b84558a45018346`.
- Synthetic tests: `test_score_recurrent_adaptive.py`, SHA-256 `26e5abf173e7332ba3b7de66fda4fc73ab240fd16eeb9e3f7e2e4462e11607eb`.
- Sealed E479 v2 fit spec: `e479_sixteen_fit_v2.sealed.json`, SHA-256 `18ad9aa867f25424d80f3e1e5e8e81c2a6ae67e62e07ad037f7d9ace00494d42`.
- CPU preflight manifest: SHA-256 `92fcb33ea416527dc742911605da9059eb1d7c728a4a594ed9e5a65dae5bbb66`.

## Review evidence

The scorer reconstructs the reset-after, two-layer GRU recurrence independently, including per-layer `u + alpha * sin(pi*u)`, external fit-only feature normalization, identity model normalization buffers, and `[2, batch, 64]` carried state. Receipt checks validate the canonical sealed spec and its digest, exact preflight/source identities, successful runner and systemd result, all expected arms, update counts and telemetry, artifact hashes, and exact normalizer values independently recomputed from the fit groups. The export path refuses existing result/NPZ outputs; raw predictions are checked for finiteness before metric clipping. Contrast scoring uses the registered 28 contrasts and a shared, numerically sorted component-draw mapping within each seed.

The author-added scorer self-check now verifies both initialization correspondences: fixed-zero versus learned-zero and fixed-0.1 versus learned-0.1. For the positive learned case, `raw_alpha = atanh(0.5)` produces the intended effective coefficient 0.1. In both cases it loads common model weights and compares outputs and final carried states. I independently compared the scorer reconstruction with the fit implementation on synthetic CPU tensors for all four conditions (`fixed0`, `fixed01`, `learned0`, `learned01`), using identical seed-271828 parent weights, input shape `[2, 37, 42]`, and nonzero initial state shape `[2, 2, 64]`; output and state maximum absolute differences were exactly zero for every condition.

The bounded synthetic test command passed: `scripts/limited .venv/bin/python -m pytest -q research/campaign_2026-10-05/recurrent_activation_adaptive/test_score_recurrent_adaptive.py` (4 passed). No metric result or experimental winner is endorsed by this review. Any scorer edit changes the reviewed identity and requires a new check.
