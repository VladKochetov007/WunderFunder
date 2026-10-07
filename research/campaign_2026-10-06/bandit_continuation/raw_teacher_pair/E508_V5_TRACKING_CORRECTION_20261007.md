# E-508 v5 bookkeeping correction

After the successful Kaggle push, the first tracking append used the wrong campaign folder prefix (`campaign_2026-10-07`) in experiment-log evidence paths. The actual artifacts live under `campaign_2026-10-06`, where the active campaign directory was created. This was a path-only bookkeeping error: the frozen packet, pushed kernel, preregistration, quota snapshot and source hashes are unchanged.

The append-only `research/experiments.jsonl` rows `E-508-V4-TERMINAL` and `E-508-V5` preserve the original mistaken paths. The correction row `E-508-V5-BOOKKEEPING-CORRECTION` records the canonical paths. The lifecycle receipt and tree artifacts use the canonical `campaign_2026-10-06` paths.

Correct paths:

- V4 terminal evidence: `research/campaign_2026-10-06/cloud_preflight/e508_profile_terminal_v4.json`
- V5 preregistration/quota gate: `research/campaign_2026-10-06/cloud_preflight/e508_profile_prereg_v5.json`
- V5 push receipt: `research/campaign_2026-10-06/cloud_preflight/e508_profile_push_attempt_v5.json`
- V5 independent packet review: `research/campaign_2026-10-06/bandit_continuation/raw_teacher_pair/E508_V5_INDEPENDENT_PACKET_REVIEW_20261007.md`

No fit, score, scientific result or public-gain claim was recorded by this correction.
