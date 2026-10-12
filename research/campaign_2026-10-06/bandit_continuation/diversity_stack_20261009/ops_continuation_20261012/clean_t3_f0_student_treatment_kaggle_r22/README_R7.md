# Treatment kernel R7 identity correction

R7 is a title-length correction to the frozen R6 packet. Kaggle rejected the
R6 save request because its title exceeded the provider's 50-character limit.
R7 uses the title-derived target
`vladkochetov07/wnn-clean-t3-f0-student-20261012-r24`, whose title is
`wnn-clean-t3-f0-student-20261012-r24` (36 characters).

The only runtime-source change from R6 is `KERNEL_ID`; the metadata changes
only `id` and `title`. The source and asset dataset slugs, source and asset
hashes, paired science plan, worker arguments, profile checks, two-device
10,000-second wall limit, 20,000 GPU-second cohort ceiling, and R6 deadline
cleanup behavior are unchanged. The root-release consumer now requires the
new R7 kernel ID through that constant. The intended source dataset version
is 2 under a new root release; the treatment asset dataset remains at version
1. R6 and its provider rejection evidence remain preserved.

`identity_preflight_r7.py` checks the exact R6-to-R7 source and metadata diff,
title length and title-derived slug, runtime/release target agreement, and
rejection of a release targeted to R6. Its synthetic release is a consumer
fixture only; it is not root authority and does not authorize profile, fit,
upload, or dispatch. This packet remains unuploaded pending independent review,
root-issued R7 authority, exact source-version READY verification, and fresh
quota/session checks.
