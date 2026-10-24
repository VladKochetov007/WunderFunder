# Frozen F0 R1 source payloads

This folder restores the two R1 files whose working copies were amended for
R2's post-score target archive digest check. Both reconstructed files match the
SHA-256 values recorded in `SOURCE_MANIFEST_R1.json` exactly:

| File | R1 SHA-256 |
|---|---|
| [f0_grade_scoring.py](f0_grade_scoring.py) | `198cfa756dc8f7a5be58d7db35974e4e9dd71207e8a70219a33276bc7211ea26` |
| [test_f0_grade_synthetic.py](test_f0_grade_synthetic.py) | `bba9f179de64bc46afe041c6827bf81cc2e608df490cdc50088ffbd3638a2dd9` |

All other source payloads in the R1 manifest remain at their original paths
with the same bytes. R2's exact scoring and test sources remain at the active
working paths and are pinned by `SOURCE_MANIFEST_R2.json`. R1 was reviewed but
never graded; the later R2 sources add the after-score target hash check.
