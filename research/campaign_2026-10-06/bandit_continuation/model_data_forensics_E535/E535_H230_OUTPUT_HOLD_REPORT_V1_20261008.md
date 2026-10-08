# H230 paired-event output-hold diagnostic

The predeclared output hold substantially reduced the exposed H230 candidate's official score. On the frozen 468-group panel, the exact graded baseline reproduced at WP **0.6817195017** (808,863 scored rows). Holding each prediction at the most recent row where **both** `i0_p0` and `i0_p11` changed produced WP **0.5581720934**, a delta of **−0.1235474083**. The two target deltas were −0.1225335 (`t0`) and −0.1245613 (`t1`).

| Scored partition | Rows | Baseline WP | Held WP | Target-mass share, approximately |
|---|---:|---:|---:|---:|
| Paired change event | 156,106 (19.3%) | 0.83163 | 0.83163 | 16.7% |
| Between paired events | 652,757 (80.7%) | 0.65158 | 0.49595 | 83.3% |

Event rows are unchanged by construction. The held output differs from baseline on 99.9994% of scored between-event rows. The baseline identity check matches the root's bound `global_wp_clip2_v1` grade exactly, including the selected-row count; the transformed event subset also matches baseline exactly.

The transform is deliberately narrow: for each validation group, row 0 emits the original row-0 prediction, and each later row emits the baseline prediction at the most recent simultaneous `i0_p0`/`i0_p11` change. No target or future feature enters the transformation; all hidden model state remains as emitted by the frozen model. Evaluation uses the unchanged official clip, clipped-label-magnitude weights, and scored-row mask. The exact 468 groups were read in ascending row-group order, one complete group at a time, with five projected columns. The full Parquet and full prediction array were not materialized. Prediction storage was float32; the bound prediction sidecar records **fp16 inference**.

This is negative evidence for this one output-cache heuristic on one previously exposed H230 endpoint. It suggests the continuously varying emissions between these paired price-code events contain useful correlation on this panel, but does not identify why they help. It is not a fresh forecast, causal proof, or rejection of hidden-state event gating, elapsed-time decay, other event definitions, or state-memory architectures. The earlier incumbent output-hold screen is a separate historical result; neither result generalizes to all output or state interventions.

The next discriminating test, if this direction is reopened, is a new independently trained, matched own-weight pair: compare continuous row updates against paired-event state gating/decay and a matched-rate random-gate control, pinning data membership, cache, target treatment, optimizer, update budget, reset behavior, and endpoint aggregation. Grade the complete endpoint on an appropriate untouched outcome when available. Do not spend a fit merely to repeat this exposed output-cache transform.

The numerical prior remains **null**. The diagnostic and all group-level hashes are in [the result receipt](E535_H230_OUTPUT_HOLD_RESULT_V1_20261008.json); execution, input bindings, and resource accounting are in [the execution receipt](E535_H230_OUTPUT_HOLD_EXECUTION_RECEIPT_V1_20261008.json). The figure is [here](plots/E535_H230_seed3_ep1_event_output_hold_V1.png) with an SVG companion.
