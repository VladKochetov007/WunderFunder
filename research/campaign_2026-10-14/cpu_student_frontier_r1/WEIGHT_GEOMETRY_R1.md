---
kind: result
status: complete
created_utc: 2026-10-14T01:43:29.946786+00:00
updated_utc: 2026-10-14T01:45:22.773646+00:00
hypothesis_ids: [H-STUDENT-LADDER-20261014]
experiment_ids: [E-STUDENT-WEIGHT-GEOMETRY-20261014-R1]
evidence_paths: [WEIGHT_COMPRESSION_GEOMETRY_R1.json, WEIGHT_COMPRESSION_GEOMETRY_R2.json]
---

# Static recurrent spectra do not support aggressive post-hoc truncation

## Question and method

How much matrix energy would a cheap low-rank approximation or row-wise quantizer discard? The [bounded diagnostic](../../../scripts/diagnose_student_compression.py) loads two hash-pinned historical clean-F0 checkpoints: H256×2 control CP5 and H384×2 Muon teacher seed3 CP2. It computes double-precision singular spectra and the existing deployment row quantizers for all learned matrices. No training examples, labels, GPU or optimizer are used. The first receipt covers full stacked GRU matrices; gate decomposition is a separately owned successor.

## Evidence and interpretation

[Full numeric receipt](WEIGHT_COMPRESSION_GEOMETRY_R1.json): exit0,0.41s diagnostic time,560MB maximum RSS, one CPU under a1.8GB cap. The parent-plus-unused-worker-plus-support bound was16.68GB against17GB; the healthy student fit continued.

| Checkpoint / recurrent layer | Minimum rank for95% squared singular-value energy | Minimum rank for99% | Relative Frobenius error at half rank | W8 error | W4 error |
|---|---:|---:|---:|---:|---:|
| H256 control /0 |207|243|0.471|0.00751|0.13648|
| H256 control /1 |205|242|0.457|0.00767|0.13915|
| H384 teacher /0 |310|364|0.467|0.00773|0.14019|
| H384 teacher /1 |301|361|0.438|0.00869|0.15747|

For a stacked3H×H recurrent matrix, a two-factor rank-r implementation has4Hr matrix elements versus3H² dense elements. Arithmetic savings require r<0.75H: below192 forH256 and below288 forH384. Every95%-energy rank above exceeds that threshold. Half-rank truncation would discard a substantial component in this unweighted matrix norm. This supports training/adapting a structured low-rank model before comparing it, rather than treating a high-energy post-hoc truncation as a free serving optimization.

## Limits and decision

Matrix energy is not prediction sensitivity. Inputs and recurrent states occupy a nonuniform, correlated distribution; gate saturation and downstream projections can suppress or amplify these errors. A large weight-norm error need not imply a large score loss, and a small error cannot certify a long recurrent sequence. W4 here is PTQ weight reconstruction only, without activation quantization, retraining, gate/state dynamics or native callback timing. The two checkpoints differ in width, recipe and training exposure; their comparison does not isolate Muon. No low-rank or W4 research arm is rejected.

Continue per-gate diagnostics, measured factorized serving and bounded low-rank fitting after the matched architecture screen. Use official end-to-end grades to allocate precision. [Earlier theory and cost formula](THEORY_AND_DISCRIMINATING_TESTS_R1.md) and [precision/state plan](PLAN.md) remain the governing comparison.

## Per-gate successor

[Gate-decomposed receipt R2](WEIGHT_COMPRESSION_GEOMETRY_R2.json) corrects the diagnostic selector from `rnn.weight_*` to the actual checkpoint keys `gru.weight_*` and asserts all seven whole matrices plus twelve gate blocks per checkpoint. R1 whole-matrix results remain valid and unchanged; its script revision is `9c73c640`. R2 exits0 in0.66s,621MB RSS under an800MB cap. For six H256 recurrent gates,95%-energy ranks are148–154; for six H384 gates they are212–229. Independent H×H factorizations save arithmetic only belowH/2 (128/192), so this representation also requires adaptation for aggressive compression. Half-rank errors are0.292–0.314 and0.266–0.310 respectively. Per-gate W4 matrix error varies from0.131 to0.147 inH256 and0.136 to0.174 inH384, but this does not rank end-to-end gate sensitivity.
