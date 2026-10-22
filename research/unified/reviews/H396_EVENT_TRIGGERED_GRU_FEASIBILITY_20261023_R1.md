---
kind: review
status: complete
created_utc: 2026-10-22T22:39:48Z
updated_utc: 2026-10-22T22:39:48Z
hypothesis_ids: [H-396]
experiment_ids: []
evidence_paths:
  - research/unified/ideas/FRESH_REFRAME_20261023_R1.md
  - research/experiment_drafts/online-relationships-20261022/PANEL_ADMISSION_R1.json
---

# H-396 event-triggered GRU update feasibility

## Question or claim

Can H-396 skip recurrent updates on causal novelty while keeping this predictor's per-row feature path and endpoint output intact, using existing APIs and without adding a Python row loop?

## Mechanism and method

**Source-only feasibility review.** No data or model payloads were read; no tests, builds, fits, replays or profiles were run. I inspected the source tree at root HEAD `2cb6a7082859b7d5f5bbc56030369e06d58f1c44`. The admitted online-relationships panel remains owned by root: `PANEL_ADMISSION_R1.json` pins source revision `17470a770f2a6aa25be232c017bff45b5d56ad97`, recipe SHA-256 `edc873c813de77f55859d1a13180d907004e5f18fcc537e64eda7af1308e2508`, and its grader files. This review did not touch that panel or runtime source.

The 211-wide input is 112 raw columns plus 99 causal event features (`native/src/stream.rs:8-12`). `StreamFeatures::update` must still run per row: it sanitizes/advances previous-row state, flags, held values, three event-age counters and `step` (`native/src/stream.rs:148-209`). Python's `StreamingGraph` likewise rebuilds its input and calls the model for every row (`src/wnn/streaming.py:108-160`). All 211 model-input features still need to be built every row even if the GRU state is held, because the current input projection and readout use the current row. In particular, `since = log1p(step-last)` advances during quiet raw spans. H-396 says “standardized `x_t - x_{t-1}`” but does not specify whether `x` is the raw112 row or engineered211 input. If it is engineered211, the novelty score includes moving clock features; unchanged raw rows still do not imply that all 211 features are unchanged. The trigger representation must be frozen explicitly.

The current model APIs do not expose a recurrent skip. `RNNModel.forward` always encodes current inputs, invokes fused `nn.GRU`, then reads out concatenated recurrent output and current encoding (`src/wnn/nn.py:199-206`). Native `FastRnn::step` always dispatches to a full step (`native/src/rnn_fast.rs:1505-1545`); both F32 implementations compute current input projection `z`, all recurrent input/hidden projections and GRU updates, then the head (`native/src/rnn_fast.rs:1668-1741`, `2312-2405`). `PyPredictor.build_input` must keep sanitization, feature construction and configured online normalization/representation updates (`native/src/python.rs:1599-1641`), and its `step` currently invokes `FastRnn.step` unconditionally (`1740-1757`). Its `reset` already resets features, optional online state and the model (`1701-1709`). The scalar `RnnModel.step` is also unconditional (`native/src/rnn.rs:153-205`).

Under the no-added-Python-loop requirement, the direct route is a compiled branch in the Rust predictor: after building the current 211 input and `z`, branch around every GRU layer's `w_ih`/`w_hh` projections and state update, retain the old hidden state on a skipped row, and still compute the current head/readout from `[old_h, z_t]`. The branch must cover the actual F32 dispatch path for the pinned parent shapes (`step_general_f32` or `step_impl`). This can run inside the existing Rust `PyPredictor.step` call without adding a Python-side gate loop. The current `scripts/predict_native.py` caller already loops over rows in Python (`30-38`), so this would not remove that boundary cost; it must not add another loop. The current APIs do not implement this branch. A custom compiled or PyTorch implementation could also be built, but is unimplemented and unprofiled here; this source review does not rule it out.

The H-396 card specifies “standardized” difference and a threshold ladder but does not define the scalar reduction, normalization identity, exact thresholds or tie rule. Those must be frozen before a skip mask is reproducible. A zero threshold must explicitly mean **always update**; a rule such as `novelty > threshold` could otherwise skip exact-zero rows and would not be a parity control. The first row must always update. Reset must clear any added previous-input/gate state along with feature clocks and recurrent state. Same-binary, strict-FP32 zero-mode output and hidden-state parity should be exact, row by row and after repeated resets; deterministic replay should reproduce the same skip mask and predictions.

## Evidence and interpretation

The two exact reusable parent endpoints are already pinned in the online recipe and are the matched fold references, with zero new GRU fits required. Values 1001 and 1002 identify the current readout-panel fold cells; they are not labels for new parent training seeds:

- F0 / readout-panel cell seed 1001: `clean_t3_f0_control_s3_a9_t3r5_r2_p3_envr1@ep5`, checkpoint SHA-256 `e30577ffb39d061bc40043f68a90285965bd2cf790f6ba5947ba78c60b776eda`; mean/std identities `bb85c72b87656979f97c1786400aec6bc9e43acd337078000eb482c09c5a2da8` / `b730600851e52e246fc638230844d6e4f2828e4972b45db1f0128cce4e4454c5`.
- F1 / readout-panel cell seed 1002: `oldz4_h230_parent_f1_s3_r4@ep3`, checkpoint SHA-256 `be2273cc96027295c19133504354a262cd1a2066fa5ae3df7d6be64ed0a1ad8e`; mean/std identities `bae7ab6474b4bb6d5ab35f796a140382b5d3aa39532b3d75f6e7d5fffc74d2ea` / `e0e1b2a5c89941f5da0c9138bb5c1163d0aafdf7f963ebd55cde404917f176b6`.

`load_parent` verifies these identities, loads strict state, sets inference FP32 and records the backend (`src/wnn/training/online_relationship_sources.py:24-83`). The recipe records historical training precision as fp16 and fp16_amp respectively, so these are strict-FP32 **inference** endpoints, not independent FP32 training replications; F1's initial ancestor is unknown. `evaluation.predict` builds inputs, uses the strict-FP32 inference flag and applies a stored outcome readout when present (`src/wnn/training/evaluation.py:21-46`). `PyPredictor` returns the native head directly and has no outcome-readout API. Before using the PyTorch parent prediction caches as native endpoint references, resolve whether each pinned parent's readout is present and verify/export an identical mapping; the safer matched runtime control for H-396 is the same checkpoint through the same native F32 path with gating disabled.

## Limits and validity

No runtime savings, skip frequency, quality effect or endpoint parity has been measured. Zero new fits is not zero execution cost: a full replay still reads each sequence, builds the 211 inputs and position state, calculates novelty, input projection and readout, and grades predictions. Branch overhead can erase recurrent savings. Parent prediction caches contain parent outputs, not a reusable 211-column stream, so a gate replay still needs raw sequences and event-feature construction. The existing frozen 12-cell grader is not an H-396 grader and must not be edited to admit this arm. All folds are exposed; any resulting score is discovery/robustness evidence, not fresh confirmation.

## Decision and next step

Feasible through a new inference-only implementation, with the existing Rust predictor as the direct path under the no-added-Python-loop constraint; current APIs alone do not expose the gate, and no quality/runtime claim is supported. After the live panel completes, any separate run requires root admission. Minimal future contract: (1) freeze the novelty representation (raw112 or engineered211), reduction, normalization, thresholds, comparator and reset semantics; (2) use an exact same-runtime ungated control; (3) pass a no-data strict-FP32 fixture for bitwise zero-mode output/hidden-state parity, first-row update, reset isolation and repeat determinism; (4) replay the two pinned fold parents and a predeclared threshold ladder on identical complete sequences, scoring with unchanged `global_wp_clip2_v1` and recording per-fold full endpoint, skip counts, weighted skipped-row coverage and quiet-run diagnostics; (5) measure warmed end-to-end inference on the same sequences and hardware, reporting observed wall time and gate overhead. Include any saved outcome-readout mapping identically. Do not infer runtime benefit from a skip count. Root owns admission, grader design and unified README indexing.

## Source records

- [H-396 proposal](../ideas/FRESH_REFRAME_20261023_R1.md), especially lines 54-61.
- [Panel admission](../../experiment_drafts/online-relationships-20261022/PANEL_ADMISSION_R1.json) and [frozen execution recipe](../../experiment_drafts/online-relationships-20261022/EXECUTION_RECIPE_R1.json).
- Provider/model record: native collaboration subagent, assigned/requested `gpt-6-luna`, reasoning `max`; runtime exposes no separately verifiable provider build or model-version string.
