# Post-run source edit reconciled by exact content identity

The precision diagnostic finished before its working source was modified. The executed seal binds SHA 5a41d096c4c0290346c7afbeba7e5930ba70aaee6baee2f7f56246384eeaf067 (13,255 bytes). The later working file SHA c35f1b7ff65d4405c3056b2d6eb8af79f7f8afd4d1d577646b692fde5fa19b8a was preserved as `versions/precision_policy_probe_postrun_c35f1b7f.py`.

Root reconstructed the exact executed source by reverting four identifiable post-run changes: the flags type annotation gained `| str`, an explicit float32-matmul-precision setter was added, and two policy dictionaries gained `float32_matmul_precision: highest` entries. The reconstructed bytes hash exactly to the original sealed SHA and size, so this is verified source recovery rather than an inferred equivalent program. The executed copy is retained as `versions/precision_policy_probe_executed_5a41d096.py`; the working source is restored to these exact bytes. No output, tolerance, numerical result or failed attempt was changed.

Future announced/sealed sources must be frozen before handing off to the parent. Proposed improvements require a new version and new execution identity. The later source was not run in this campaign and must not be credited for the existing diagnostic.
