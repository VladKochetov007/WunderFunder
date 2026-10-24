---
kind: review
status: complete
created_utc: 2026-10-24T12:08:18Z
updated_utc: 2026-10-24T12:08:18Z
hypothesis_ids: [H-ONLINE-RELATIONSHIPS-20261022-R1]
experiment_ids: [E-20261024-online-f0-exploratory-r2, E-20261024-online-f0-correction-shrink-r3]
evidence_paths: [../../../experiment_drafts/online-relationships-20261022/EXECUTION_RECIPE_R1.json]
---

# Online relationship control interpretation R1

This source-only note clarifies what the saved A, B and C readout arms compare.
It does not add a fit or reinterpret any measured score.

## Actual input and state

The canonical feature code sets `BASE_INPUT_WIDTH = N_FEATURES +
N_EVENT_FEATURES = 112 + 99 = 211`. It appends four 52-channel groups—matured
i0 movement, forecast, error and slope—for 208 additional features, so all
three heads receive a 419-wide input. `_sequence_features()` concatenates the
parent-normalized raw/event base with the relationship state. The 99 event
features include causal event changes and recency/history fields.

For each matured 16-row i0 movement, A still emits the 52 matured values, 52
decayed-mean forecasts and 52 errors from that evolving mean. Its `_slope()` is
hard-zeroed, but `_should_update()` remains true and `_update_pair()` continues
updating the online state, including `my`, with the fixed exponential decay.
Thus A is a slope-zero, mean-only relation estimator; it is not a raw-only,
no-history, or no-adaptation arm. The current raw/event features also remain in
its input.

The recipe describes B as updating slope/intercept through zero-based row 98
then freezing the estimator, and C as continuing those updates. The source
implements this with B’s `_should_update()` condition `self.rows <= 98`; A and C
continue updating. The update is considered only after the 16-row movement has
matured.

Each arm uses the same 419-to-64-to-2 residual head. Its output layer starts at
zero and prediction is `parent + 0.1 * residual`. The parent prediction is
therefore initially preserved in every trained arm; any measured loss relative
to parent comes from the learned correction.

## What the current comparisons identify

| Endpoint/control | What it contains | What its contrast can show |
|---|---|---|
| Untouched parent | The frozen parent prediction, with no readout head applied | Total effect of a fitted A/B/C correction versus the parent endpoint |
| Hypothetical 419-wide zero-stat head | Same residual-head width, but all 208 appended relationship channels forced to zero | Would isolate added head capacity/training from relationship statistics; this arm was not fitted or evaluated |
| Actual A | 211 raw/event channels plus matured movement, decayed-mean forecast and mean-relative error; slope is zero | Tests a mean-only relation correction, including the readout capacity and ongoing mean adaptation |
| Actual B | Same input width, with slope/intercept updates only through row 98 | Tests a prefix-fitted, then frozen relationship estimator |
| Actual C | Same input width, with continuing slope/intercept updates | Tests a continually updated slope/intercept estimator |

A-versus-parent does not isolate relationship features from readout capacity,
and A is not a “no-history” control. A-versus-B/C compares a slope-zero mean
estimator with estimators that expose nonzero slope/intercept behavior and use
different update schedules. The absent 419-wide zero-stat head leaves the
capacity-only alternative unresolved. These distinctions apply to the F0
results and to the saved F1 A endpoint under review.

## Sources

- [`online_relationship.py`](../../../../src/wnn/online_relationship.py): input dimensions, matured 16-row state, zero slope for A, update schedule and feature groups. SHA256 `e6cf32e4b260b60a7fab0efc32beec4ec74c750d79634de8aad3be82471dcb59`.
- [`features.py`](../../../../src/wnn/features.py): the 99 causal event features. SHA256 `c2b317d1f148d0cad9632726a1436b94f4ae76016458deaa70aa36fdca21ecb2`.
- [`online_relationship_fit.py`](../../../../src/wnn/training/online_relationship_fit.py): 419-wide feature assembly and strict-FP32 fit path. SHA256 `cbc2c1f349f61a829ba91fea779e61df6d242e6c58666391aaa37d5d040fa774`.
- [`online_relationship.py` residual head](../../../../src/wnn/training/online_relationship.py): zero-initialized parent-plus-residual head. SHA256 `8dcbc404d93ee7fa28dd4e18576dc3faab7b1f58548c3abc03322ee524304772`.
- [`EXECUTION_RECIPE_R1.json`](../../../experiment_drafts/online-relationships-20261022/EXECUTION_RECIPE_R1.json): frozen A/B/C descriptions and saved run identities. SHA256 `edc873c813de77f55859d1a13180d907004e5f18fcc537e64eda7af1308e2508`.

Only source and recipe metadata were read for this note. No arrays, labels,
models or training runs were accessed.
