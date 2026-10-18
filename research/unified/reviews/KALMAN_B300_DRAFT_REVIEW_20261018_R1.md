# Kalman B300 teacher draft source review, R1

## Scope

Independent source-only review of draft PR #17, `experiment/kalman-teacher-gru-distill-b300`. The review inspected the strict checkpoint loader, adapted prediction/cache path, teacher source identity, profile recipe, uncertainty recipe and launch readiness. It ran no tests, synthetic smoke, data read, profile, fit or GPU task.

## Findings and correction

The first review found that the prediction sidecar copied `model_source_sha256` from checkpoint provenance and the cache builder compared it back to the same value. That did not verify the currently installed adapter source. It also found the identity omitted `src/wnn/nn.py`, whose `RNNModel` is used by the adapter.

The draft was corrected to hash both `src/wnn/kalman_local_trend.py` and `src/wnn/nn.py` as one deterministic source identity. Training provenance records the file map and aggregate digest; prediction export checks current files against the checkpoint before inference and writes the current identities to its sidecar; cache construction independently checks current source, checkpoint and sidecar identity and preserves these fields in the cache record. The files also enter the broader trainer implementation identity.

The follow-up review cleared this source path and the predeclared bootstrap recipe. It found no remaining source defect in the reviewed files. The PR is still operationally blocked from task fitting by the deliberately null per-cell teacher registry and missing actual B300 full112 profile; it remains `launch_ready: false`. Those are explicit prerequisites, not a fit authorization or scientific negative.

## Evidence limits

This review validates code/document consistency only. It does not establish PyTorch adapter behavior, numerical parity, B300 throughput, teacher quality, distillation value, package quality or leaderboard transfer. The local R2 Kalman score belongs to a separate prediction-only fixed-GRU diagnostic and is not evidence for the proposed trained teacher/student endpoint.

## Provenance

- Draft PR: [#17](https://github.com/VladKochetov007/WunderFunder/pull/17)
- Branch: `experiment/kalman-teacher-gru-distill-b300`
- Reviews: independent B300/source reviewer, bounded source-only passes
- Related hypothesis/experiment: `H-KALMAN-LOCAL-TREND-20261018` / `E-KALMAN-LOCAL-TREND-B300-20261018`
