# E457: fixed parent/head public-transfer candidates

The already trained full-data combo has a fixed epoch3 parent and a fixed
epoch0 head-only fine-tune with exactly the same recurrent trunk. E412 packages
their equal prediction mixture through one shared trunk. This experiment
prepares two additional deterministic candidates: the parent alone and the
fine-tuned head alone. Neither checkpoint nor mixture weight is selected from
the historical holdout. Both endpoints are prescribed before callback checks.

This tests whether the final global-correlation head procedure transports to
public evaluation. Local head transfer has evidence in E410; this experiment
does not assume that its public gain has the same size or sign. The user has uploaded the existing equal-mixture ZIP: E412 scored0.6655 in
37m47s at625MB (submission10tq248b). This supplies a measured complete-predictor
reference. The endpoint ordering still has no defensible narrow numerical
public forecast before uploads; do not infer endpoint gains from one mixture result.

Sources are `c20261004_combo_s3_final_r1/model_ep3.pt` and
`c20261004_combo_head_s3_final/model_ep0.pt`. Both retain the existing training
pool, historical holdout exclusion, native engine, i16 arithmetic, fixed model
membership and offline callback. Preserve checkpoint and ZIP hashes.

For each new ZIP, run the actual callback on20 full20k-row validation streams
in i16/v3, i16/v2 and f32/v3. Require finite required-row outputs, correct
warmup behavior, exact v2/v3 output identity, <=3e-4 maximum i16/f32 error and
<=5e-6 WP difference. Then replay one full stream under sequence IDs A,B,A to
check reset identity. Measure complete-callback kcycles against the existing
verified E412 ZIP on the same CPU and explicitly label concurrent-load timing
if GPU fitting is active. A timed projection is not an official runtime.
No competition upload is performed by this script.

This needs no GPU fits. One admitted CPU job, oneCPU,2GiB RAM/no swap,
1200-second timeout, during a verified CPU-only interlude between local E424 fits and with
enough time remaining before20:00 Kyiv. Existing packages are never overwritten.
Save readiness JSON, callback arrays, stdout and timing under
`research/campaign_2026-10-04/E457_head_transfer/`; outputs are
`submissions/c20261004_combo_parent_s3_i16/submission.zip` and
`submissions/c20261004_combo_head_s3_i16/submission.zip`.
