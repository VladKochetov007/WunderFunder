> Extracted from the final October 12 handoff. Its original SHA-256 was `e033f8d52f4d4444e5f0c5fa006c2b4f56b6e82f7187e2ed9fb1b74b20e6f704`. See [archive index](../README.md) and [split manifest](../SPLIT_MANIFEST.json). Local links were rebased for this page.

## Fixed endpoint-weight optimization

The small numerical primitive implements centered prediction covariance,
regularized linear initialization, exact real-arithmetic clip-2 weighted
Pearson objective evaluation, and a local SLSQP search with bounded starts.
Because official scoring casts synthesized predictions to float32 before
clipping, the proxy gradient is not the exact computational derivative of
the float32 scorer. Candidate values are re-scored through the unchanged
official metric; the method makes no global-optimum claim.

J1 compared three own H384 CP2 teacher-seed outputs over 48 previously
exposed F0 connected components, split before optimization into 32 calibration
components / 110 groups and 16 evaluation components / 52 groups. Calibration
selected approximately [0.39956, 0.24662, 0.35382], scoring 0.6783535252
versus equal blend 0.6783214526 on calibration. Those weights were frozen
before evaluation access. On evaluation they scored 0.7662406664, below the
frozen equal mixture 0.7663608706 by 0.0001202042; calibration lift did not
carry to that evaluation split. This is one exposed component partition, not a
fresh fold, global optimum, or public predictor. There is no post-hoc selection
of a different pair. No model fit or prediction generation occurred. Receipts:
[J1 execution handoff](../../../../campaign_2026-10-12/joint_endpoint_optimization_r1/J1_EXECUTION_HANDOFF_R1.md),
[root admission](../../../../campaign_2026-10-12/joint_endpoint_optimization_r1/ROOT_J1_ADMISSION_R1.json),
[fixed component manifest](../../../../campaign_2026-10-12/joint_endpoint_optimization_r1/J1_COMPONENT_PANEL_MANIFEST_R2.json).
