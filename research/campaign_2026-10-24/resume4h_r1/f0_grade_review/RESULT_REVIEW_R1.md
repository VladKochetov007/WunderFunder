# Independent result review: exploratory F0 online grade

## Verdict

**The grade is internally consistent as an exploratory comparison of seven
fixed F0/seed1001 prediction endpoints.** It does not support a mechanism-wide
negative conclusion or a top-tier promotion. All six online-head endpoints
score below their fixed parent on this exposed fold. The small primary
between-arm deltas have 95% component-bootstrap intervals that cross zero.

The old F0/F1 panel remains incomplete. This result adds no fits, separates
neither fold nor fitting-seed variation, and says nothing about public
transfer. I inspected the scalar result, label-free preflight JSON and resource
receipt only; I did not reopen target, prediction or sequence-cache payloads.

## Identity, source pins and execution

- Grade R2 SHA-256:
  `bfc06c9820f48b5214bd32d13783a268a3e35da8ffc8c61f65b91912352e2899`.
- Execution receipt SHA-256:
  `9edbdb0cea33955e1c5c7d60dfc8a87ba3654e561f513bc4c29c50c67f37ceea`.
- Frozen recipe R2 SHA-256:
  `0d6cf458a895c322f94dd25ec1d228c5b3bfc5320f0da10c55027f58f84fd059`.
- Source manifest R2 SHA-256:
  `55b6ded533703d3430ecfbdc60d28ccd62d6812b249029028cea549c602d9553`.
- The grade records the matching recipe, source manifest, root admission and
  label-free preflight R3 identities. Its target archive expected, pre-score
  and post-score SHA-256 values all match:
  `dbb7a1a48dd0af9fa524a80c2694008a56cb91d5dae80e8d0d4984c0c5622955`.
- Execution returned 0 in 32.71 seconds. It records zero new fits, an
  authorized grade, 468 evaluation groups, 141 components and 808,863 scored
  rows for every endpoint.
- Historical terminal pins report 27 runtime and 15 grader sources. Recursive
  ancestry covers 10 runs, 27 edges and three distillation caches. The pinned
  membership check reports 1,405 valid fit groups and zero source-qualified
  valid-fit/evaluation overlap; train and valid remain separate namespaces.

The metric and mask match the official contract: `global_wp_clip2_v1`, the
official mask from row 99 onward, ±2 clipping, absolute clipped-target weights,
and the five reported row bands. The result has four primary contrasts
(`C−B` and `B−A` within each lag) and all 11 original descriptive contrasts.
Each pair has matching selected rows and all five time-band deltas. The
component bootstrap uses 2,000 draws with seed 271828 and is conditional on
the fixed endpoints.

## Fixed-endpoint scores

| Endpoint | WP | Delta vs parent |
|---|---:|---:|
| Parent | 0.6811541494 | — |
| A, lag 1 / lag 16 | 0.6805967662 | −0.0005573832 |
| B, lag 1 | 0.6805564865 | −0.0005976629 |
| C, lag 1 | 0.6806097840 | −0.0005443654 |
| B, lag 16 | 0.6805295732 | −0.0006245762 |
| C, lag 16 | 0.6805285842 | −0.0006255652 |

`C, lag 1` is the best saved candidate, still 0.0005444 WP below the parent.
The A lag 1 and lag 16 prediction hashes are identical in the pinned
preflight, consistent with their exactly equal scores and zero lag contrast.
All four primary contrasts have intervals containing zero:

| Primary contrast | WP delta | 95% conditional component interval |
|---|---:|---:|
| C−B, lag 1 | +0.00005330 | [−0.00003841, +0.00015966] |
| B−A, lag 1 | −0.00004028 | [−0.00012523, +0.00003790] |
| C−B, lag 16 | −0.00000099 | [−0.00014716, +0.00014316] |
| B−A, lag 16 | −0.00006719 | [−0.00017714, +0.00001743] |

The secondary candidate-versus-parent intervals also cross zero. These
intervals resample the 141 fixed evaluation components; they do not estimate
training variation. The recorded `p_le_0` values are empirical bootstrap tail
fractions, not a substitute for a prospective significance test.

## Resource review and limits

The execution receipt records a 399,998,976-byte cgroup memory peak under the
400,000,000-byte worker ceiling, zero swap, and zero OOM/OOM-kill events. The
peak is only 1,024 bytes below the cap. `memory.events` records 6,001 `max`
events, indicating substantial memory-boundary pressure/reclaim despite
successful completion. Treat this run as compliant but effectively at the
limit; do not assume spare memory for a larger local grade. The cgroup CPU
usage was about 31.9 seconds against 32.7 seconds wall time.

Scientific interpretation remains narrow. These are already exposed F0
predictions from one fitting seed. The bootstrap conditions on those files and
cannot tell whether a fresh fit would reproduce the ranking. The result
is a degradation from a learned correction on a parent-preserving
initialization: both `OnlineRelationshipResidualHead` and `AgeResidualHead`
zero-initialize their output layers and return `parent + 0.1 × residual`
([online_relationship.py](https://github.com/VladKochetov007/WunderFunder/blob/c3e208b387ead0ec967f9abc4b14275eb8ba94c5/src/wnn/training/online_relationship.py),
[age_readout_model.py](https://github.com/VladKochetov007/WunderFunder/blob/c3e208b387ead0ec967f9abc4b14275eb8ba94c5/src/wnn/training/age_readout_model.py)).
The grade therefore does not support an explanation based on dropping the
parent or failing to initialize at the parent. Fit-side in-sample
parent-residual mismatch and objective/regularization mismatch remain
prospective explanations; neither is established by this grade.

This result suggests the learned relationship corrections do not improve the
fixed predictions on this F0 cell; it does not rule out the direction or
descendants that change training, seed, fold or combination. Ordinary refit
cost remains the relevant follow-up comparison. No independent
inference-parity check or public score was obtained.
