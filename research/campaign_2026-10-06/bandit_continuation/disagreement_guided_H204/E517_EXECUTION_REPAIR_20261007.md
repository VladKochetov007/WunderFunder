# E-517 execution-session repair — 2026-10-07

The original E-517 Marimo editor scope `run-p724888-i754379.scope` reached its 45-minute runtime lease at 09:08:58 Europe/Kyiv. Its terminal receipt is [E517_SESSION_LEASE_TIMEOUT_20261007.json](E517_SESSION_LEASE_TIMEOUT_20261007.json). The exclusive attempt marker was absent, the runner was never entered, no score or prediction output was produced, and the campaign OOM counters did not change.

To avoid reopening the broader E-515/E-516 analysis notebook and rewriting its frozen outputs, the same click-gated E-517 action now lives in the minimal Marimo notebook [E517_single_action_eda.py](E517_single_action_eda.py). It has four cells: imports/path setup, the exact manifest-checking E-514 loader, the visible inactive run button, and the single registered action. The action body was extracted from the code-mode-verified original cell, keeps its exclusive marker and guard, and has exactly one call to `run_registered_e517`. No E-515 or E-516 analysis cells are present in the new executor.

The old [H204_complementarity_eda.py](H204_complementarity_eda.py) editor had already terminated, and port 2720 was confirmed closed before its two E-517 cells were removed. This eliminates a second executable action path; the original E-515/E-516 cells and artifacts remain unchanged. This is an execution-wrapper repair only: candidate, target, fold partition, threshold, fit objective, controls, bootstrap, official metric and pass rule are unchanged. The new executor is to be independently reviewed before its input loader or action is run.

Source hashes at this repair:

- Original notebook before dormant-cell removal: `8ea80256a4bee554ae3a1c92d5b16725afb4f385629fb4d8b264e2ecd36aa5c8`
- Original notebook after removal: `3481d49aac97cd345fbc7ab63e852f03677b9459fcffc9deda087737bf364d63`
- Minimal E-517 executor: `44cb079681272a8f2bffbd3e5127a9da6f113d74c32644d19802b7eb8b4465a7`
- E-517 runner implementation: `b0d43d462a604d0ec9f42413d2a410343d6557d0e2570a9c528c6f77bdd169dc`
- E-514 input loader: `3e64d854fcc3a0df080f7ef9741cf6cd10b07664084be7a2eee952e003c4042d`

Recorded at `2026-10-07T06:14:49.557728+00:00`.

## Final UI and independent review addendum

The inactive action now displays explanatory text only; the exported button is shown once in its own trigger cell. The final minimal executor SHA-256 is `ed15fc20e6553faeb5968faa7359522b152d517070842d99f39ef65a3071b8cf`. Independent static review of this exact file returned PASS; see [E517_INDEPENDENT_PRE_SCORE_REVIEW_20261007.md](E517_INDEPENDENT_PRE_SCORE_REVIEW_20261007.md).
