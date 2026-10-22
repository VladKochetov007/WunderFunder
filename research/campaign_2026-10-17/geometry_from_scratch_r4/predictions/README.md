# R4 evaluation predictions

This folder stores fixed epoch-2 FP32 predictions for the exact component-held
out evaluation groups from each frozen CampaignSplit fold. Each array is paired
with a sidecar containing run and checkpoint identity, fold membership, row
group order, split-manifest hash and prediction content hash.

The matrix runner checks shape `(N, 20000, 2)`, dtype, finiteness and exact group
membership in bounded chunks before scoring. Targets and masks are read from
the hash-bound `cache/valid_targets.npz`; they are not duplicated into each
prediction archive. These exposed-fold predictions are development evidence,
not pristine confirmation or public-transfer evidence.
