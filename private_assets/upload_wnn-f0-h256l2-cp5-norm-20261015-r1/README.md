# F0 H256×2 CP5 normalization-only asset stage R1

This physical stage contains one clean F0 seed-3 H256×2 CP5 checkpoint and its provenance, both content-addressed under the standard private Kaggle asset filename convention. The profile trainer copies only the pinned 211-element input mean/std buffers from this checkpoint. It does not load the donor's learned GRU or head weights.

The [asset manifest](assets-manifest.json), [private dataset metadata](dataset-metadata.json) and [stage receipt](stage-receipt.json) are committed as small provenance records. The two `.wnn` payloads remain in local physical storage and are excluded from Git. The stage receipt says `provider_action_performed: false`; root authorization allows creating this private dataset only after the independent asset review and exact owner/readiness check.
