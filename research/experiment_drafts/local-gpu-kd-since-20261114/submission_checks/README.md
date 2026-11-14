# November 14 submission checks

- `SELECTION_R1.json`: fixed endpoints/coefficients, exposed-validation policy, and four public results supplied by the user.
- `FULL_VALID_R1.json`: official full-valid comparison of train-only s3, s4, their fixed mean and alpha 0.9; reference scorer agreement asserted.
- `PAIR_PRELIMINARY_R1.json`: first fixed-pair score, retained as intermediate evidence.
- `NATIVE_PARITY_R1.json`: 64-sequence native INT8 versus FP32 parity; final-data model scores are fitted-data diagnostics only.
- `*_REPLAY_RECEIPT.json`: exact ZIP and prediction hashes, ABI, CRC, finite-output and A-B-A reset checks under Python 3.11, measured local timings.
- `MANIFEST_R1.json`, `SHA256SUMS`: delivered ZIPs, exact source checkpoints and training provenance; weights stay git-ignored under submissions/local_20261114.
- `*.export.log`: serialized GPU prediction export logs; successful exporters are silent, completion is established by prediction sidecars and checksums.

[Handoff](../SUBMISSIONS_R1.md); [review status and limits](../SUBMISSION_AUDIT_R1.md). No new fitting or competition upload. Full native evaluation on all 1,873 sequences was not repeated; native parity uses the frozen 64-sequence panel.
