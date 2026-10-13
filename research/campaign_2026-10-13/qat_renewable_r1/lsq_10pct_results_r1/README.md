# Learned row scales: first native comparison

The same-source fixed and learned row-scale uniform-W4 fits each completed1,440 updates from the same clean-T3 CP5 F0 seed3 parent, identical1152-sequence fitting subset and sampler/normalization identities. Every selected matrix scale is learned in the treatment, including the Q8 first-head blocks; the control recomputes row absmax scales after updates. Both export the fixed raw epoch-0 checkpoint with exact learned/fixed codes and scales.

| Native endpoint | WP | Target0 | Target1 |
|---|---:|---:|---:|
| Fixed row-scale QAT |0.681298282|0.674579514|0.688017050|
| Learned row-scale QAT |0.681492421|0.675033309|0.687951533|

The preliminary paired point difference is +0.000194139. Independent component-bootstrap grading is pending. These are full468-group exposed-F0 trajectories with141 components and808,863 official scored rows, one fitting seed, and full fitting-set parent/teacher ancestry. This does not establish a best package or public-transfer gain. The10% floating continuation's native-Q8 score remains higher at0.681818498.

- [Fixed result](fixed_result.json) and [learned result](learned_result.json) preserve native scores, callback/reset information and complete prediction hashes.
- [Fixed manifest](fixed_manifest.json) and [learned manifest](learned_manifest.json) bind separate checkpoint, exact archive and source identities. [Result index](RESULT_INDEX_R1.json) names approved physical prediction storage; no large arrays are copied into research notes.
- [Paired fit completion](../LSQ_PAIRED_FIT_COMPLETION_R1.json) pins1440 updates, identical sampler/normalization, and152.9/164.9-second local fits.
- [Fixed archive reproduction](../LSQ_FIXED_ARCHIVE_REPRODUCTION_R1.json) shows byte-identical packed output to the earlier fixed control; this is engineering reproducibility, not an independent training replication.
- [Export recovery](../LSQ_EXPORT_RECOVERY_R2.json) preserves the missing-file preparation failure and its separate complete-source successor. [Native admission recovery](../NATIVE_LSQ_LEARNED_RECOVERY_R2.json) preserves a pre-start17GB refusal; the exact same learned endpoint completed once the other replay released capacity.

H-QAT-LSQ-20261013 / E-QAT-LSQ-10PCT-F0S3-20261013-R1. This is LSQ-inspired scale learning with the prior identity latent-weight STE retained, not a canonical LSQ reproduction. Recurrent activation rounding remains outside training.
