---
kind: experiment
status: complete
created_utc: 2026-10-13T12:40:00Z
updated_utc: 2026-10-13T12:55:00.486000+00:00
hypothesis_ids: [H-TEAM-NATIVE-20261013]
experiment_ids: [E-TEAM-XTEAM-S6-RELEASE-20261013-R1]
evidence_paths: [research/campaign_2026-10-13/team_models_r1/EXPLORATORY_SUBMISSION_AUTHORITY_R1.json]
---

# Exploratory cross-team seed-6 submission

## Question or claim

Can a new fixed four-member cross-team blend transfer better to public scoring? The user explicitly requests this public probe even without a local-best result and confirms a new four-model blend. This supersedes the earlier release timing requirement for this candidate. It does not change the scientific interpretation of validation scores.

## Mechanism and method

Use equal 25% prediction coefficients for teammate `distilled_s6@ep1`, teammate `fttall_s6@ep3`, our merged Z4 seed99 model, and our H390 homogeneous seed3 model. Preserve their exact NPZ bytes and reuse the established `xteam_h4` Q8 runtime. The two teammate seed-5 members in `xteam_h4` are replaced by their seed-6 counterparts from `pair_ftt2`; our two members remain the same.

This fixes one comparison before any score for this endpoint is available. There is no coefficient search or extra training. All four trunks are distinct. The model shapes and inference engine match the previous four-member cross-team package, providing a practical runtime reference without certifying this exact package on the host CPU.

## Evidence and interpretation

The user reports public scores of 0.6682 for `pair_ftt2`, 0.6681 for `xteam_h4`, and 0.6675 for own E. These are complete-ensemble observations; they do not rank the individual seed-6 members or predict the new blend's score. The requested package is an exploratory endpoint. Technical checks and final artifact identity will be recorded in its submission directory.

## Limits and validity

The final archive weights have fitted or unknown overlap with supplied validation. Current member exports and native archive replays remain diagnostic. Current FTT training source and source-qualified CV counterparts are unavailable. No clean-CV improvement, new public score, or exact-host runtime measurement is claimed. The official metric clips the complete blend to ±2; the serving recipe does not introduce per-member clipping or a wall-clock membership governor.

## Decision and next step

The [16.32MB ZIP and release receipt](../../../submissions/10-13-team-r1/README.md) are ready in repository artifact storage. Independent recipe/byte review, ELF ABI, isolated Python3.11 import and two complete sequences repeated twice under both native V2 and V3 pass. V2/V3 outputs are bitwise equal on that panel. All runtime files match our optimized E submission exactly. This completes packaging, not public evaluation; the user uploads the artifact. The already admitted comparison jobs continue.

## Source records

Subsequent public feedback: the user reports **0.6679**,46m51s/650MB, portal 0mbzevhv. This is +0.0004 over E and below the original xteam_h4 and pair_ftt2. The [latest exact portal receipt and interpretation](PUBLIC_LB_UPDATE_20261013_R2.md) preserve all five supplied rows; the initial release recipe and validation receipts remain unchanged.

- [Explicit release authority and fixed member hashes](EXPLORATORY_SUBMISSION_AUTHORITY_R1.json)
- [Archive and tensor inventory](INVENTORY.json)
- [User-transcribed portal observations](public_results.json)
- [Original comparison contract](CONTRACT.json), preserved unchanged
- [Runtime interpretation](EPYC_RUNTIME_REFLECTION_R1.md)
- [Team-sharing decision](../../../specs/adr/0035-official-team-code-and-weight-sharing.md)
