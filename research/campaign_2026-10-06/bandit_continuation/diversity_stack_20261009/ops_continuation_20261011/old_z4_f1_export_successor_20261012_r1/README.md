# Old Z4 fold-1 export successor R1

This packet is an export-only contract correction for the frozen R4 fold-1 seed-3 parent/student pipeline. The R4 independent recipe audit found that both frozen R4 export contracts still expected R3 fit attempts. The successor export contracts point to the actual R4 parent-fit and student-fit producer attempts, and use new export attempt IDs to keep the corrected contracts distinct.

The original R4 contracts, fit recipes, argv, source snapshots, input membership, checkpoint epoch, and export output paths are preserved. For each export JSON, the only field changes from its R4 source contract are `attempt` and `expected_fit_attempt`. The new expected producer is the matching R4 fit attempt.

This packet does not run a fit or export. At preparation, the R4 parent CP3 and student CP1 output files were absent; the student fit attempt record was also absent. Their checkpoint and provenance SHA-256 values are therefore deliberately null in the producer manifest. Before either export, root must inspect the exact terminal R4 producer attempt, verify its successful exit and fixed epoch, validate finite checkpoint/provenance/training-state consistency, and bind the actual CP and provenance hashes. Root must also freshly verify that the target prediction NPY/sidecar outputs are absent before dispatch. No output is overwritten by this packet.

The fold-1 evaluation is exposed robustness evidence, not a fresh holdout. Fit membership is 10,533 train groups plus 1,416 eligible validation groups; the fixed 457 outer evaluation groups and 73 excluded train groups remain out of fitting. This is unchanged from R4.
