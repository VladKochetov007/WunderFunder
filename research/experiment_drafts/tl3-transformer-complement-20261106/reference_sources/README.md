# Published clean-TL3 source identities

These metadata mirrors were retrieved on 6 November 2026. No checkpoints were
downloaded, deserialized or executed for this code preparation.

- `pr49-checkpoints.manifest.json`: the original sixteen-role
  [PR49 manifest](https://github.com/VladKochetov007/WunderFunder/blob/c42baf7b95946994216801b4eb9909422ea7effa/research/experiment_drafts/checkpoint-publication-20261029/tabdata-pr49-checkpoints-20261029-r1.manifest.json).
  This mirror is the API-decoded text, not an asserted byte-identical release asset.
- `tabdata-heterogeneous-checkpoints.manifest.json`: actual manifest downloaded
  from the [PR52 checkpoint release](https://github.com/VladKochetov007/WunderFunder/releases/tag/tabdata-pr52-heterogeneous-checkpoints-20261030-r1).
  Only its restored `tl3c_w18final_distilled_s5` fixed endpoint is used here.

`src/wnn/training/sequence_reference_pins.py` freezes the seventeen
`fixed_endpoint` checkpoint hashes, exact original run-provenance hashes and
zero-based epochs from these sources. It deliberately uses `model_epN.pt` hashes:
serialized `model.pt` bytes can differ even when they describe the same endpoint.

The consumer validates every original provenance file and fixed checkpoint on
the receiver, the exact `0c4de3...` train-only split, all direct fitting/evaluation
memberships and recorded training-source inventories. This pins the actual
published reconstructed TL3; it does not reconstruct historical consumed-cache
bytes that were not retained. That limitation is included in every reference
identity. Fitting-reference predictions are in-sample and not OOF.

The replay producer runs `scripts/predict_run.py` from clean PR49 commit
`c42baf7b95946994216801b4eb9909422ea7effa`, including its actual causal-feature
and chunk behavior. Per-role training-source digests are preserved separately;
this replay code is not relabeled as the original trainer of every ancestor.
