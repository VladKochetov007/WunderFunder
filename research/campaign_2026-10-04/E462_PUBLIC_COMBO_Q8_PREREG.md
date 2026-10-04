# E462: fixed q8 conversions of the public combo

**Status: draft, not admitted, not executed.** This preregistration authorizes no conversion,
build, prediction, score, or timing run. It fixes the candidates and gates for root admission.

## Question and candidates

Can zero-update q8 conversion reduce the deployment weight stream of the already public E412
predictor while preserving its own represented-model predictions closely enough for a fixed
deployment alternative? This is a deployment-compression test, not a training experiment and not
a claim that quantization improves the model.

The sole source is
`submissions/c20261004_combo_s3_i16/submission.zip`, SHA-256
`21f9b7368eddd64ad8ae19ae0702990a57a1806531a85148b657e5aed719dafb`. Its manifest has one
member, `weights_0.npz`, coefficient 1, source
`c20261004_combo_s3_final_r1@ep3|c20261004_combo_head_s3_final@ep0`. The fixed candidates are:

1. that exact original i16 archive (control);
2. `q8r`: the exact member converted with quantizer preset `recurrent`;
3. `q8x`: the exact member converted with preset `recurrent_head`.

There are no fitted scales, calibration rows, labels, or updates. q8r/q8x are different represented
predictors: all four GRU matrices are converted; q8x also converts the merged first head. Input
projection, biases, final output layer, and all other unselected tensors remain FP32. Do not
requantize at inference or select a candidate from the holdout.

## Export and artifact identity

The source member is already the exact two-head fusion produced by
`scripts/package_submission.py:merged_heads`; do not rebuild the source from an approximate or
later checkpoint. Before converting, record the source ZIP hash, `members.json`, the byte hash of
`weights_0.npz`, every packaged `solution.py`/native-library hash, and the E457 source checkpoint
hashes. E457 records parent `model_ep3.pt` SHA-256
`64c2b28651b414778d6af3c33bb9d54f653a46fb0be087de2e3c44ca88c31166`, head `model_ep0.pt` SHA-256
`f1cb88f38b6d5b5b67859346b439e69d610928bc836baac8867e84e7b1c95d85`, and bitwise-equal non-head
trunks. The archived member's float tensors must be checked against the deterministic fusion of
those exact checkpoints; the checkpoint links establish lineage, while the archived member bytes
establish the actual conversion input.

Use `src/wnn/quantization.py:quantize_model` and
`src/wnn/quantization_export.py:write_q8_export` with the two fixed presets. Preserve the emitted
int8 codes and float32 scales. On reload, require `read_q8_export` to reproduce the metadata,
codes/scales, and represented selected matrices exactly. The package-local loader must retain the
code/scales dtypes, restore represented selected tensors for shape/model construction, and pass the
one member's q8 code map to `wnn_native.EnsemblePredictor`; passing all NPZ arrays through the i16
loader would cast away the q8 representation. Map `recurrent -> q8r` and
`recurrent_head -> q8x`. Do not modify root native code or the source i16 archive. Each output ZIP
gets a new immutable path and complete member/provenance manifest.

Packaging is eligible only if exact source-member verification, archive integrity, constructor
smoke, finite-output check, warmup behavior, and package-size checks pass. Any failed export/build/
wrapper attempt is retained in the E462 output directory with command, stdout/stderr, exit code,
and artifact hashes; never overwrite the source or erase a failed candidate.

## Prediction parity gates

The independent reference for each q8 candidate is **that candidate's own represented weights**,
not the original FP32 model and not the i16 package. Reconstruct the exact represented matrices
from q8 codes/scales, then evaluate them with a separately implemented Torch FP32 dense recurrent
reference. This measures native q8 arithmetic against the mathematical model represented by its
saved q8 tensors. The original i16 predictor remains the deployment control and may differ from
both q8 references.

Run the real packaged callback and independent reference on the 20 E457 full streams, each exactly
20,000 rows (`valid_subset20.parquet`, 400,000 rows total, dataset SHA-256
`cd6a349b3655bb6f98314e5db391e69b1e1ea7f12ab18f1d15c61cefd28daa37`). Freeze and hash the same
raw 112-column scorer inputs for both paths. The independent reference should follow E416's
`probe_q8_model.py:torch_reference`: derive the 99 event features with the pinned
`wnn.features.event_features`, append them to the raw inputs, and run the represented weights
through Torch's dense recurrent implementation. Hash both raw inputs and derived event features;
the native callback gets the raw inputs and computes its causal event state internally. Compare all
required prediction rows and targets in the same order. Predeclared gates, per q8 preset:

- maximum absolute native-vs-own-reference prediction error `<= 3e-4`;
- absolute global-WP delta `<= 5e-6` against that own reference;
- all required predictions finite; callback returns no prediction during warmup;
- portable, v2, and v3 q8 predictions bitwise identical;
- A,B,A sequence replay gives bitwise-identical A outputs, and changing a future input suffix
  cannot change earlier outputs;
- the reference's restored code/scales and selected effective matrices exactly match the archived
  q8 export.

Report q8-vs-i16 prediction error and score differences descriptively across all 20 streams; do not
apply the own-reference parity thresholds to i16-vs-q8. A failure of a parity gate rejects that
package for scoring/timing, but the failed outputs/logs remain archived.

## Scoring design and provenance limit

Only after both q8 presets pass parity, compare q8r and q8x separately with the fixed E412 i16
control on the same 368 historically excluded validation sequences, targets, masks, and row order.
This is an exposed, repeatedly inspected holdout, so all differences are descriptive and cannot
select between q8r/q8x or tune any parameter. Report pooled weighted Pearson, mean per-sequence
weighted Pearson, t0/t1, all market-class slices, exact selected-row/sequence counts, and paired
2,000-draw component-bootstrap intervals for each candidate-minus-i16 difference. Keep the score
contract identifier and metric code hash with the report; until the official aggregation is
resolved, present both pooled and per-sequence forms.

`scripts/screen_score.py` is not directly valid for this converted artifact. Its holdout path
requires a prediction sidecar naming one run/checkpoint and follows one `args.init` chain. The E412
member is a deterministic fusion of two distinct checkpoints, and the converted member is derived
from that fused archive; assigning either single run as its identity would misstate its source.
Before scoring, implement or use a narrowly scoped evaluator that loads the frozen
`research/campaign_2026-09-30/audit/campaign_score.py` arithmetic and explicitly validates a
composite lineage record: both exact checkpoint hashes, fixed epoch identities, no holdout inclusion
in either training ancestry, bitwise-equal recurrent trunk, exact fusion-to-source-member equality,
source ZIP/member hashes, q8 export/code/scales hashes, callback prediction hashes, validation
target/mask/row-order hashes, and holdout component mapping. Store that lineage beside every
prediction array. Do not fabricate a single-run sidecar or weaken `screen_score.py` checks. If the
composite proof cannot be established, stop before the holdout score and report the blocker.

The historical holdout score is not the public score. E412's user-reported public 0.6655 belongs to
the original i16 archive only. Do not estimate a q8 public score by adding its local holdout delta
to 0.6655, and do not upload any candidate as part of E462.

## Runtime and resource plan

Timing is gated on successful package parity and the paired score report. Use the exact complete
`ScorerStepByStep` callback with native causal feature updates, fresh predictor reset for every
trial, fixed CPU15 affinity, and the same first three complete E457 streams. Do not time only the
native recurrent kernel. Compare i16-vs-q8r and i16-vs-q8x in five alternating pairs each (ten
60,000-row callback trials per comparison), reversing pair order on alternating repeats. Save every
trial's row count, elapsed time, cycles, affinity, load average, package/library hashes, and command;
report all values and the median, not a best-of result. The timing sample is three 20k sequences per
callback (60,000 rows), not the 400,000-row, 20-stream parity set and not the full 37.46-million-row
validation replay.

Measured E457 evidence for the fixed i16 combo is three callback trials over 60,000 rows, with
`timing_0_equal_mixture.log` 101.319 us/row, `timing_1_equal_mixture.log` 103.308 us/row, and
`timing_2_equal_mixture.log` 100.471 us/row (median 101.319 us/row, about 6.08 seconds per
60,000-row trial). These are three-sequence callback measurements. The separate E457 parity run
covered 20 sequences/400,000 rows. The user-reported portal runtime of 2,267 seconds is the full
test-set runtime and must not be treated as the time for a 20k callback.

E416 W384 native diagnostics measured q8r at 162.719 us/row and q8x at 111.247 us/row, versus
q16x at 107.765 us/row; that W384 model had one ordinary head and is only a planning analogue for
the current merged-head package. Using those rates, the planned 20 trial calls (each 60k rows)
would consume roughly 4 minutes of callback time; budget 8 minutes wall time for the entire timing
phase including startup and snapshots. Replace the estimate with E462 observed values. This estimate
is diagnostic only: E416 found no wall-time q8 improvement, and q8x may differ on the merged head.

All CPU build, parity, scoring, and timing work must be separately admitted and run with one CPU,
2 GiB RAM, no swap, no GPU, and one BLAS/PyTorch thread. Timing command template, after an actual
E462 harness exists and root admits that phase:

```bash
systemd-run --user --scope --slice=wnn-campaign.slice \
  -p CPUQuota=100% -p MemoryMax=2G -p MemorySwapMax=0 \
  env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  taskset -c 15 .venv/bin/python scripts/e462_q8_public_combo.py timing \
  --control submissions/c20261004_combo_s3_i16/submission.zip \
  --q8r <immutable-q8r-zip> --q8x <immutable-q8x-zip> \
  --data cache/valid_subset20.parquet --sequences 3 --paired-repeats 5
```

Set bounded phase timeout/RSS limits in a sealed runner spec before any command is executed. The
timing command above is a proposed interface, not an existing runnable script or authorization to
launch.

## Interpretation

q8r/q8x remain optional deployment packages only if represented-reference parity and package
validity pass. A lower byte count is a storage result, not an accuracy or speed result. q8x is the
primary compression candidate because E416's W384 analogue reduced streamed weight storage more;
q8r is retained as a predeclared comparator. Report both regardless of outcome. Preserve all
failures. No candidate promotion, score-driven tuning, runtime-conversion claim, or public upload is
part of E462.
