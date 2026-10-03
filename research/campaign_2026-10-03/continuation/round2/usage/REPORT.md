# Strong-GRU input-history usage map

The fold-matched GRU256 responds to both instruments' current and past
streams under frozen causal donor interventions. Identity and planted
history controls pass. This is an input-space usage map, not evidence
that a new representation supplies missing predictive information.
No representation-changing training proposal is justified.

## Recovery, control and sampling

The own statphys report found local ridge history gains but unresolved
event-versus-row smoothing. The user reports the later strong-model gate
failed. That negative does not prove saturation; this pass instruments
the existing model rather than repeating a ridge fit.

Control: c20261001_d256_s3_f0@ep3; fixed epoch3, hidden256, two GRU layers.
Checkpoint SHA256: f92504fbbbcf956a44e70aa3f7a08825d903b9c6a6cacce7bd9f89b6875fcad8
Split SHA256: 09263ad4fe2b18e2b2f3b79f4562678dce801b067b868134ddc79279f9316a3d
Prediction SHA256: 58f522606400c203ebdc2ba39a3934ee313e5690ffb4dc01f518e4fd675ae553
All inspected reference source hashes remained unchanged during the pass.
Actual run fit groups (10535 train/1405 valid) passed component-exclusion
checks against the frozen manifest and feature-only graph. Historical
teacher lineage was not independently reconstructed in this pass.

PREREG.json froze six interventions before sample/response inspection.
Fold0 sequence IDs ranked by SHA256(round2-usage-v1: + seq_ix); first six
distinct components. Targets, masks, classes, features and predictions
never entered selection. All validation is already campaign-exposed;
all sample targets were in the own census. None match reconstructed old
own feature groups or the own32-sequence CPU pass. Unknown CLI exposure
remains unknown, and no fresh campaign independence is claimed.

| Row group | seq_ix | Component | Scored fixed queries |
|---:|---:|---:|---:|
| 1395 | 1602371461 | 3235 | 6 |
| 1200 | 1388271624 | 2929 | 0 |
| 164 | 169421375 | 2920 | 1 |
| 1691 | 1949747365 | 3272 | 4 |
| 1653 | 1895548128 | 3049 | 2 |
| 1062 | 1227566939 | 2860 | 1 |

## Prefix replay and identity

Propagate original rows0..4127, including all99 warmup rows. Query511..4095
every64 rows:57 per sequence,342 total. State never resets at a window
boundary. Carry saved full two-layer states before t-256, t-32 or t.
Complete window predictions and original prefix-boundary states are saved
in sequence_00..05.npz, with raw/packed inputs and masks/targets.

The211 channels partition into i0=125, i1=78, auxiliary=8. Current inputs
include causal deltas, flags, recency and held-event features, not just
raw current state. Auxiliary channels stay original in every arm.

Six arms: identity; i0_past256; i1_past256; i0_current; i1_current;
i0_past32. Past arms replace the instrument's channels on[t-H,t) with
observed donors64 rows earlier; complete query-row inputs remain original.
Current arms substitute only rowt using t-64. Future inputs resume original
rows through t+32, while the changed hidden state propagates.

All donors precede affected rows: prefix compatible, no future injection.
These are observed channel bundles in model-input space, not physically
reconstructed market paths. Cross-stream, auxiliary, engineered-clock and
hidden-history consistency can break. OOD movement is not signal importance.

Synthetic planted80-row i0 memory responds to i0_past256 (RMS .40418),
not short32 or i1 history. Current-only sham has zero history/identity
response; current i1 is positive (RMS .27827). Future canaries pass.
The toy validates intervention discrimination, not real identifiability.

Independent actual reference: separately loaded original model with
train_rnn.predict(chunk4000, CPU). Instrument: segmented carried-state replay.
CPU float32 bound is atol2e-5 + rtol2e-5, fixed before measurements.
Reference max error=8.34e-07;
identity replay max error=8.34e-07.
Cached GPU/CPU max=0.009242,
max RMS=0.001535, correlations>.999.
Cached AMP predictions satisfy separately frozen cross-backend bounds;
not bitwise CPU/GPU equality. IDs, row order, raw prefix targets, masks
and warmup align exactly. Original CUDA autocast disables on CPU with a
preserved warning. No GPU executed.

## Measured usage and propagation

RMS pools both outputs; per-target values are retained in results.json.
Positive lag means original inputs resumed. Identity query RMS is1.79e-7;
all five real arms exceed the frozen .001 usage threshold.

| Arm | Query RMS | Lag4 RMS | Lag32 RMS | Scored-query weighted-MSE damage |
|---|---:|---:|---:|---:|
| identity | 0.000000 | 0.000000 | 0.000000 | +0.000000 |
| i0_past256 | 0.336241 | 0.226428 | 0.091567 | -0.697071 |
| i1_past256 | 0.170195 | 0.129993 | 0.058910 | +0.042640 |
| i0_current | 0.319871 | 0.060097 | 0.014499 | -0.858476 |
| i1_current | 0.084292 | 0.041539 | 0.009735 | +0.083083 |
| i0_past32 | 0.315725 | 0.193571 | 0.070036 | -0.682562 |

Long256-minus-short32 i0 query RMS=0.086382
raw, 0.086382 clipped.
All six component contrasts are nonzero (clipped RMS .0242–.1189).
Earlier changes still affect outputs after32 identical altered rows.
This resolves a tested older-history contribution, not an exact horizon
or missing usable history. Current-only substitutions recover faster;
both GRU layers retain history-induced changes after32 resumed rows.
One-row versus256-row and125-channel versus78-channel amplitudes cannot
rank instrument importance or identify a skip/head route.

## Quiet age and predictive-value limits

Primary age is elapsed rows since ANY of22 i0 price identifiers changed,
not seconds, best-quote ordering, or inferred target class. The saved field
organizer_age means the CODE'S ev_i0px counter (p0 AND p11 change), not an
official event clock. It disagrees with primary age at47 queries.

| Age | Queries | Components | i0 current RMS | i0 past256 RMS | i1 past256 RMS |
|---|---:|---:|---:|---:|---:|
| 0 | 82 | 6 | 0.35703 | 0.40527 | 0.14350 |
| 1–3 | 55 | 6 | 0.39488 | 0.30929 | 0.15290 |
| 4–15 | 75 | 6 | 0.34870 | 0.34091 | 0.21788 |
| 16–63 | 79 | 6 | 0.27939 | 0.31847 | 0.16740 |
| 64+ | 51 | 6 | 0.11522 | 0.25079 | 0.15068 |

Quiet age64+ does not erase history sensitivity. But donor displacement
varies with age (current i0 standardized RMS about1.65→.63 from age0 to64+).
Plots expose that confound; intrinsic age-dependent kinetics remain unknown.

Only14 queries on five components are scored. Apparent i0 error improvements
have component-bootstrap intervals spanning zero. i1-history corruption
increases error (.04264; descriptive95% .00828–.11059), suggesting local
useful-history sensitivity. Tiny OOD tests cannot establish conditional
mutual information, a generalizable correction, or competition WP gain.
No residual head or WP improvement was fitted. Big movement is not unused
signal evidence; a posthoc null closes its specific cue/head, not saturation.

## Ranked follow-ups

1. Amplitude/state-matched past donors using only observed feature distances.
   Match displacement, channels and age; kill age-kinetics claims if the
   contrast disappears. No targets in matching.
2. Compare raw-stream replay with causal features recomputed against packed
   substitutions, holding query raw state fixed. Kill physical causal
   interpretations if only inconsistent packed histories respond.
3. Freeze a larger metadata-only component panel and later prefixes with
   fixed cadence before testing targets. Kill apparent i0 improvement if
   it fails transport; use component rather than row uncertainty.
4. Cross-fit a fixed history-cue residual test against a flexible frozen-state
   control and equal-budget shams. A null rejects that cue/head, not every
   memory mechanism. Independent critique precedes integration.

Representation-changing training proposals: NONE justified. Existing usage
is supported; missing predictive information is unknown. No GPU launch.

## Cost and retained artifacts

Six inference script bodies total 51.04s;
longest 9.08s;
peak RSS 943.2MiB, CPU one thread.
All computation used run_cpu.sh round2-usage under900s per command.
Measured body time excludes module imports and CPU-slot admission.
No fullvalid forward, fitting, installs, live source/model changes or
writes outside usage/. No current-pass model/resource failure occurred.
Runnable scripts, COMMANDS.txt, PREREG, STATE, results, synthetic controls
and replay/state/window arrays are retained with hashes and sample exposure.
usage_map.png and usage_controls.png were visually inspected. No active
blocker; next step is independent critique/reproduction and matched donors.
