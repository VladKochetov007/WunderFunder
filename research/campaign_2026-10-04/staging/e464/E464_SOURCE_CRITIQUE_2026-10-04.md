# E464 staged-source critique

Read-only review of `scripts/run_head_fit.py`, `scripts/predict_e464_head.py`, `scripts/score_e464_all.py`, the draft builders/specs, the synthetic scope tests, and the frozen trainer/scorer contracts. No imports, tests, fits, or exports were run.

## Initial findings and resolution status

1. **Blocking: the post-fit batch-count assertion was off by one. RESOLVED in the staged wrapper.** The frozen trainer builds a `DataLoader(..., batch_size=16, drop_last=True)` from 1,505 fit row groups (`scripts/train_rnn.py:1420-1424, 1592-1594`), giving 94 batches and 940 optimizer attempts at 2,000 rows/update over a 20,000-row sequence. `run_head_fit.py` now derives both expected values from the verified membership and recorded recipe, then requires the update counts to reconcile.

2. **Blocking for the claim “all 16 preregistered contrasts”: exact-comparison validation was missing. RESOLVED in the staged scorer.** It now requires the exact ordered pair list from the generated E464 panel, all nine expected arms, and the fixed combo mixture, rejecting duplicate or omitted pairs before scoring.

3. **Medium: prediction shape/dtype/output-width enforcement was missing. RESOLVED in the staged scorer.** It now memory-maps each child cache and requires FP32 `[1873, 20000, 2]`, and validates `outputs_exported=2`, `model_outputs=23`, fixed checkpoint/architecture/objective provenance, and parent-child fit membership.

## Scope checks that pass by inspection

The named-parameter scopes are correct for the current three factories: specialists freeze `inp`, `gru`, `experts`, `prefix_proj`, and `router` while leaving `universal_head.*` and `expert_heads.*`; adapters freeze `inp`, `gru`, `adapter_*`, and `prefix_prior` while leaving `head.*`; GRU controls freeze `inp` and `gru` while leaving `head.*` (`run_head_fit.py:24-40`). These align with the actual registered module names in `src/wnn/specialist.py`. Before/after state fingerprints cover every state-dict tensor, including buffers; key-set changes, non-head byte changes, and head layout changes fail closed (`run_head_fit.py:243-276, 340-355, 386-390`).

The generated one-epoch recipe follows `continuation_recipes.head_args`: seed 3, AdamW, lr `5e-4`, weight decay `0.01`, batch 16, chunk 2000, `gcorr`, mask weighting, inherited preprocessing and auxiliary arguments, fixed epoch 0, and no in-fit evaluation. The wrapper checks the output epoch, selected epoch, args, auxiliary settings, and unchanged fit-group set (`run_head_fit.py:358-416`). Full-VALID export fixes epoch 0 and FP32, derives output width from the loaded state, requires 23 outputs, uses the canonical target-cache group order, and carries child/parent/audit/source-closure identities into the sidecar. Prediction caches are memory-mapped; the frozen scorer reads stored target arrays by memory map and scores one sequence at a time, making the 2 GiB score spec plausible by source inspection. No memory bound has been measured here.

Parent/head pairing and provenance are substantively bound: each screen candidate is a fixed 0.5/0.5 mixture; the wrapper verifies parent and child checkpoint hashes, audit status and hash, parent sealed source spec, all-VALID row order, and common predictor source closure before scoring. The historical 368-row holdout is explicitly treated as exposed. The current screen builder enumerates the intended 16 contrasts; finding 2 is a gap in enforcement, not an error in the draft list.
