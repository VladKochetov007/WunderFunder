# Checkpoint artifact publication, 29 October2026

The user explicitly requested all best model checkpoints attached to their
corresponding GitHub experiments.24 existing best checkpoints were found on the
L4 node and matched their recorded SHA256 values. The untrained W18 node has no
checkpoint. No model was fitted, stopped, quantized or rescored during export.

| Experiment | Models | Download | Original scientific source |
|---|---:|---|---|
| PR46, plain GRU + GALAM panel |8|[Release](https://github.com/VladKochetov007/WunderFunder/releases/tag/tabdata-pr46-checkpoints-20261029-r1)|50053909|
| PR49, TL3 dependencies/students |16|[Release](https://github.com/VladKochetov007/WunderFunder/releases/tag/tabdata-pr49-checkpoints-20261029-r1)|61f0a00b; first teacher ec3f1a666|

Both releases are research prereleases in the existing private repository,
published as **SemyonEpanov**. They do not replace the repository's latest release.
Each contains a ZIP, a readable JSON manifest and SHA256 sidecar. The ZIPs are
19,573,488 and80,858,660 bytes. Actual API downloads of all6 assets were hashed
again and matched the local originals. Draft temporary URLs were replaced with
the published tag URLs in both PR comments:
[PR46 comment](https://github.com/VladKochetov007/WunderFunder/pull/46#issuecomment-6496427704),
[PR49 comment](https://github.com/VladKochetov007/WunderFunder/pull/49#issuecomment-6496445486).

## Checkpoint roles and contents

- **best_observed:** strict maximum full-valid WP in the saved producer history
  or best pointer; this is exposed validation selection.
- **fixed_endpoint:** the original predeclared TL3 epoch, preserving the17-fit
  experiment and teacher ancestry.
- **selected_runtime:** original model.pt bytes referenced by prediction export
  receipts. Different serialization hashes remain distinguishable.

All8 PR46 models have a best checkpoint. All16 trained TL3 models have best,
fixed and runtime roles, sharing archive entries when their source path agrees.
Six best epochs differ from their fixed endpoints: ffull_s3,fin256ft_s4,
fmu128_s3,frawW_s3ft,frawW_s4ft,orgbase. The archives contain46 checkpoint files
for24 models, not46 independent trained models. W18 is explicitly listed absent.

Original checkpoint bytes retain the complete model state, **mean/std buffers**,
configuration and embedded provenance. Captured history/pointers, configurations,
source identities and the official primary split accompany the weights. Raw
datasets, prediction arrays and optimizer/resume payloads were outside this
checkpoint-only export. All best/fixed/runtime checkpoint payloads loaded on CPU;
their model tensors were finite. Original run files and sources remain unchanged.

## Verification and implementation

[Inventory](CHECKPOINT_INVENTORY.json) binds original paths, best scores/epochs,
producer hashes and fixed endpoint roles. [Export receipt](EXPORT_RECEIPT.json)
records the receiving archives; [download receipt](DOWNLOAD_RECEIPT.json) records
the local byte copies. [Publication receipt](PUBLICATION_RECEIPT.json) records
the actual release/comment URLs, authenticated author, asset IDs, sizes, hashes
and successful API download verification.

[PR46 manifest](tabdata-pr46-checkpoints-20261029-r1.manifest.json) and
[PR49 manifest](tabdata-pr49-checkpoints-20261029-r1.manifest.json) are readable
indexes of every run, epoch, score, role and archive member. Sidecar metadata was
captured once so validation and archived bytes use the same snapshot. Independent
review by launch_data_manifest checked all24 models' roles, source-qualified
memberships, fit-only normalization, split, payload hashes and safe archive paths.

The initial packaging review found insufficient run-name and score/epoch guards;
the unuploaded R1 staging copy is preserved on the node. R2 added safe unique
panel IDs, pinned run roots and exact producer-based score/epoch/path/hash checks.
Both R2 archives passed the independent review. Publication additionally requires
the private fixed repository, SemyonEpanov account, matching PR/model counts and
the official split. Cross-host download redirects strip the API credential.

Implementation: [exporter](../../../scripts/export_tabdata_checkpoints.py),
[selection contract](../../../scripts/checkpoint_export_contract.py), and
[release publisher](../../../scripts/publish_checkpoint_releases.py). Ruff
format/check passed. Receiver CPU loads, complete ZIP member SHA/size checks,
independent role/ancestry review and actual GitHub byte roundtrips establish
artifact integrity; no new scientific score or training replication is claimed.
