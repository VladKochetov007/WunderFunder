---
kind: result
status: complete
created_utc: 2026-10-15T18:00:12Z
updated_utc: 2026-10-15T18:00:12Z
hypothesis_ids: [H-CPU-POLICY-20261014]
experiment_ids: [E-CPU-POLICY-PUBLIC-20261015-R1]
evidence_paths: [research/campaign_2026-10-14/cpu_student_frontier_r1/inference_policy_r1/PUBLIC_CPU_POLICY_RECEIPTS_20261015_R1.json]
---

# Competition CPU policies: fusion leads, 256-bit path times out

## Question or claim

Which engine serves unchanged xteam_h4 fastest on the competition host? The user supplied three new portal outcomes plus historical controls. These are reported submission outcomes, distinct from laptop benchmarks.

## Mechanism and method

A/B/C/D preserve original weights, preprocessing, Q8 representation and fixed25% coefficients. A is the original512-bit engine; B uses256-bit EVEX VNNI; C changes packing/register scheduling; D fuses compatible GRU projections with separate scales and correct old-state use. Prior independent checks establish bitwise callback/reset parity on two complete20k sequences. No fitting or blend selection follows from these public results.

## Evidence and interpretation

|Policy|Portal ID|Duration|Memory|Displayed score|Outcome|
|---|---|---|---|---|---|
|A original|1hl88h25|50m16s|650MB|0.6681|Completed October12|
|B256-bit|0emo1mvc|61m57s|650MB|—|TIMEOUT October15|
|C blocked|2nk0qaui|57m27s|656MB|0.6681|Completed October15|
|D fused|182a1r2g|47m27s|651MB|0.6681|Completed October15|

D saves169seconds versus historical A:5.60% less elapsed time,1.0594× throughput. C adds431seconds (+14.29% duration). B's timeout is censored; reported elapsed time is not a completed full-data runtime and may include termination/other overhead. Do not assign B a full-row throughput.

At39.4million rows D amortizes to72.26µs/row including overhead, with12m33s to60minutes and7m33s to55minutes. Linear whole-callback scaling gives1.159× current work at55minutes or1.264× at60minutes. These are planning ratios, not certified model capacities or deadline guarantees; larger shapes and shared overhead may scale differently.

The historical `pair_lowlr.zip` result is0.6680/45m7s/666MB (00olgj1c). All eight supplied rows, including QAT0.6680/54m26s, are retained in the structured receipt. Different weights prevent treating these historical durations as engine-only comparisons. D matches the original's displayed quality with shorter observed duration; no new quality best is established.

## Limits and validity

User-transcribed portal evidence; no authenticated portal read or uploaded-payload hash verification. Filenames match frozen handoff files; receipt hashes are expected local hashes. Preserve displayed timestamps verbatim with unspecified timezone; the QAT row differs in chronology from an earlier report.

One run per policy; A is three days earlier. Host frequency/cotenancy and variance are unknown, so D's advantage is provisional. Rounded score equality cannot prove exact predictions; prior local sequence parity provides stronger numerical evidence. These outcomes select an engine provisionally, not architectures or blend weights.

## Decision and next step

Prefer D provisionally for serving; retain A fallback. Deprioritize these exact B/C implementations on this endpoint, without rejecting256-bit kernels or blocking on other shapes. After explicit resume, contemporaneous A–D–A is the useful timing check. No new experiment, upload, rental or fit is admitted; research remains paused.

## Source records

- [All supplied receipts and arithmetic](PUBLIC_CPU_POLICY_RECEIPTS_20261015_R1.json).
- [Package manifest](PACKAGE_MANIFEST_R1.json).
- [Independent implementation/parity/ABI review](../../../unified/reviews/CPU_POLICY_IMPLEMENTATION_REVIEW_20261014_R1.md).
- [Submission handoff](../../../../submissions/10-14-cpu-policy-r1/README.md).
- [Pause and bounded drain](../../../unified/operations/RESEARCH_PAUSE_20261015_R1.md).
- Append-only ledger experiment `E-CPU-POLICY-PUBLIC-20261015-R1`.
