# H230 seed3 public submission handoff

The user requests public scoring despite the lower exposed local grade. Human-upload handoff; no portal upload or public result is claimed.

- ZIP: `submissions/h230_hetero_s3_ep1_i16/h230_hetero_s3_ep1_i16.zip`
- SHA256: `446836bd5fcf88f1d84e83513aca87a4928727755517b5bed404760642fc6048`
- Bytes: 4,674,489.
- Candidate: independently trained H230 heterogeneous/family student, seed3, fixed epoch1, single member, i16 serving. This is an independent analogue, not the teammate's final all-data two-student submission.
- Package checks: static members/CRC/weight identity, ABI, offline callback/reset/determinism and CPU FP32 parity have separate preserved receipts in this directory. Official portal acceptance remains untested.
- Existing exposed local grade: candidate 0.6817195 versus frozen incumbent FP32 source reconstruction 0.6898345; local evaluator is `global_wp_clip2_v1`. This comparison is not an exact i16 incumbent replay. Our public incumbent is 0.6660. No forecast is inferred by adding local deltas.

Public result will test this exact ZIP. If it contradicts local ranking, investigate serving endpoint parity, fitting/evaluation membership, component distribution and selection exposure. Preserve the original local grade and validation; introduce identified successor comparisons rather than rewriting outcomes or selecting a favorable new split after seeing the score. One public observation can justify a validation audit but does not establish a universally better protocol.

Status: delivered for human upload; awaiting submission ID and public score.
