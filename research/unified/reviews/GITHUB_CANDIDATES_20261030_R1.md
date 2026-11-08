---
kind: review
status: complete
created_utc: 2026-10-30T17:35:53Z
hypothesis_ids: [H-TL3C-SUBMISSION-20261029, H-GALAM-BLOCK-20261029, H-GALAM-BN-20261029]
experiment_ids: [E-TL3C-SUBMISSION-20261029, E-GALAM-BLOCK-20261029-R1, E-GALAM-BN-20261029-R1]
evidence_paths: [receipts/GITHUB_CANDIDATE_REFRESH_20261030_R1.json]
---

# GitHub candidates after checkpoint publication

## Question or claim

The user asks whether GitHub contains useful work to download and compute next.
The authenticated REST snapshot contains49 pull requests, including closed ones;
the highest number is51. This is a portfolio/source review, not execution or
scientific acceptance. The GitHub connector cannot access the private repository;
the existing Git credential supplied read-only API access without exposing it.

## Mechanism and method

Inspect all49 PR titles/statuses, selected PR bodies and the changed-file lists
and comments for50/51/47/48/39/38. Download71 selected source/config/documentation
files (304132bytes) at immutable heads for50/51/47/48. PR50 is compared against
its PR46 dependency, avoiding interpreting the stacked inherited work as new.
Source inspection is not a complete runtime audit of these branches.

| Priority | Existing packet | Useful next output | New scientific fits | Remaining acceptance |
|---|---|---|---:|---|
| 1 | [PR51](https://github.com/VladKochetov007/WunderFunder/pull/51), `677e40c48924680ba6201651056d064f58c864c0` | Served singleton versus fixed fbc/tiedlead/tiedlead blend; FP32 versus dynamic INT8 | 0 | Native runtime asset, receiver ABI/reset/parity, full-valid score and CPU cost |
| 2 | [PR50](https://github.com/VladKochetov007/WunderFunder/pull/50), `d3d38cb0c2e9d4faa0086805a1e02b9a581be570` | GRU72 block numerical acceptance/update throughput; separately frozen train-only BN diagnostic | 0 | Separate clean checkout, one free L4/B300, real numerical/profile checks and immutable epoch/index binding |
| 3 | [PR47](https://github.com/VladKochetov007/WunderFunder/pull/47), `f0d8da40fb545a630e5f042bfcc3b3360784177e` | Two source-clean Transformer teachers plus paired labels/KD GRU students | 6 | Complete source-qualified data/split, two-GPU B300 profiles per fit and lineage/cache checks |
| 4 | [PR48](https://github.com/VladKochetov007/WunderFunder/pull/48), `7c9e1db91219` (full SHA in receipt) | Width versus past-context teacher comparison, two seeds each | 6 | B300 profiles, full source split and substantial new fitting; downstream student benefit remains untested |

## Evidence and interpretation

PR51 initially lacked the fresh weights on its preparation machine. Our published
PR49 archive now provides all7 required endpoint/ancestor metadata records. A
stdlib-only comparison finds7/7 consistent producer source inventories, fixed
endpoint epochs, admitted split hash,10325 train/zero valid fitting members and
1873 evaluation sequences. No model was imported or evaluated. This is a useful
asset compatibility check, not full ancestry or native execution approval.

The proposed PR51 mixture is34%fbc1k +33%tiedlead seed5 +33%tiedlead seed6.
Its mixture score is unknown. The individual Torch scores are respectively
0.691355,0.691415,0.690949. This replaces the blocked W18 member as a distinct
candidate; it does not complete the canonical TL3 recipe. Native streaming can
differ from Torch's4000-row feature resets. Compare actual native outputs against
the pinned reference; preserve a mismatch instead of loosening tolerance. INT8
also enables approximate activations, so its quality delta includes both changes.

PR50 retains per-time training BatchNorm rather than flattening time into batch,
implements block GRU/attention/convolutions, and declares step fallback for sampled
diagnostic observers. Its3x speedup is a target with no measurement. Recalibration
is separate:256 predeclared fitting sequences, features only, layerwise pooled
eval-input statistics, changed BN buffers only, paired full-valid evaluation.
This can hurt under shift. Full step-backend validation may remain expensive;
zero new fits does not imply negligible wall time.

PR47 explicitly uses fresh train-only teachers and matched labels/KD students;
its six-fit budget includes both teachers. PR48 tests768x16/window1024 against
1024x16/window1024 and768x16/window4096. Both are prepared B300 studies, not
measured ready-to-run L4 recipes. Their nominal windows and six-fit counts do not
establish duration or memory acceptance. The two panels overlap scientifically;
do not independently count duplicate baseline teachers as reusable without the
explicit source/config/ancestry adapter.

## Limits and validity

ADR0044 remains mandatory. Primary valid stays out of normalizers, initializers,
teachers, BN recalibration and learned mixture weights. Use the official boundary
with whole sequences; shuffled seq_ix supplies no new chronological ordering.
All primary-valid scores remain development-exposed. The metadata check does not
bind every teacher prediction byte consumed during the legacy child fits.

Older PR12 is a plan helper, not a completed cohort trainer. PR39's frozen
train+valid component-fold panel and PR37's old fold proposals cannot be launched
unchanged under the current chronological policy. Reuse their mechanisms in
explicit successors if selected; preserve historical recipes/results.

## Decision and next step

Prefer reuse of the fresh checkpoints through PR51 and PR50 before another broad
training grid. For a new model-training comparison, PR47 answers teacher-to-student
transfer more directly than PR48's standalone scaling screen. Diagnose W18's
preserved cache-stage failure separately; do not rerun the16 completed TL3 fits.
The heterogeneous LSTM/TCN teacher proposal remains a distinct unadmitted successor.

This turn starts zero fits, profiles, inference jobs or SSH sessions; no resource
or completion-time claim is made. The next execution requires the actual receiver
inventory and a bounded measured profile. No PR is merged or submission uploaded.

## Source records

- [Compact refresh and metadata-check receipt](receipts/GITHUB_CANDIDATE_REFRESH_20261030_R1.json).
- [Published checkpoint manifests and download verification](../../experiment_drafts/checkpoint-publication-20261029/README.md).
- [TL3 terminal report](https://github.com/VladKochetov007/WunderFunder/pull/49#issuecomment-6484849146).
- [Heterogeneous teacher proposal](../ideas/HETEROGENEOUS_TEACHERS_20261029_R1.md).
- [Chronological policy](../../../specs/adr/0044-chronological-primary-validation.md).
