---
kind: review
status: complete
created_utc: 2026-11-12T12:49:12Z
updated_utc: 2026-11-12T12:49:12Z
hypothesis_ids: []
experiment_ids: []
evidence_paths:
  - SUBMISSION_STATUS_20261112_R1.json
  - DEDUP.json
  - PR53_FIXED_MIXTURES_REPORT.json
---

# Submission review, 12 November 2026

## Question or claim

Which available, previously unsubmitted endpoint should receive the next bounded
CPU check? **Keep PR76's frozen 90% TL3 + 10% LSTM ZIP as the next receiver
candidate.** It is already packaged and its source checkpoints are available.
This is a preparation decision, not a claim that the ZIP beats the public best.

The [current observation](SUBMISSION_STATUS_20261112_R1.json) supplements the
frozen November10 [dedup record](DEDUP.json); it does not rewrite it. No new model
execution, coefficient variant or upload occurred during this review.

## Evidence and interpretation

Signed-in [portal history](https://wundernn.io/connectome/submit) still shows
108 submissions, last November4, and 0/5 daily attempts used. The PR76 filename
is absent. The [leaderboard](https://wundernn.io/connectome/leaderboard) shows
kotiki at rank53 with **0.66832**, versus leader Gargantua **0.67668**. The latest
full-corpus W18 singleton scored0.66453. These are public scores, separate from
the development results below.

| Endpoint | Measured evidence | Current disposition |
|---|---|---|
| PR76 fixed 90/10 TL3/LSTM | Research WP0.69250206 vs matched TL3 0.69237742; +0.00012464 on all1873 exposed official-valid sequences | Frozen14.01MB ZIP available; CPU parity, quality and runtime unmeasured |
| [PR89 since32](https://github.com/VladKochetov007/WunderFunder/pull/89#issuecomment-6818869466) | L4 fixedE6:0.686306013/0.685903598 vs0.683019498/0.682629372; mean paired +0.003280371 | Retrieve exact weights/lineage first; same-source zero-weight control still pending |
| [PR87 depth1](https://github.com/VladKochetov007/WunderFunder/pull/87#issuecomment-6818527358) | FixedE6:0.683348193/0.683038532, small positive paired differences | Research candidate; no ready CPU ZIP |
| [PR88 numeric PLE](https://github.com/VladKochetov007/WunderFunder/pull/81#issuecomment-6818841548) | E3 seed204101:0.682872728 vs0.682569317 | Partial result, plannedE4 missing; not a ready submission |

The controls and recipes differ across these rows. The table does not rank their
absolute scores as if they were one matched comparison, nor add any development
delta to a public score. PR85 reports no new student fit; PR90 has code and
configs but zero executions by this agent. Old dashboard rows are not proof of
live training.

The private [HF archive](https://huggingface.co/hft-kagglers/wnn-challenge2-model-archive/tree/main)
was inspected in authenticated Chrome with the user's permission. Its main HEAD
remains `167452c72a8e994b280446d2fca93f7c08e16f66`, the recorded October12 revision,
154 commits. No newer weights are present there. No new checkpoint download was
found in the inspected latest PRs or GitHub releases. This says nothing about
unpublished files remaining on the producers' machines.

## Frozen comparison and limits

The PR53 coefficient panel selected90/10 from six exposed-validation comparisons.
Its +0.000125 is small and not evidence of statistical significance. The5%LSTM
variant is already measured and weaker; do not create another ZIP just to obtain
a different filename or coefficient.

The receiver measures the actual CPU callback against its **same-pass served
train-only TL3 component**. This isolates the LSTM addition within this serving
implementation. It does **not** replay the historical public TL3 archive
`d80e69d3cb52d8ee45158fc8d0f3a8af232b029b07ca5b313c61cffef029e569`, whose
member/export identities differ. A positive receiver delta alone does not prove
superiority over that public incumbent. Chunk/continuous feature semantics,
INT8/activation arithmetic and incomplete historical teacher-cache ancestry are
recorded in [README.md](README.md).

## Decision and next step

Use the existing **R2** handoff, preserving its immutable bytes. R1 delivery and
build-r2/build-r3 are historical artifacts, not extra submission candidates.

| Artifact | Identity |
|---|---|
| `kotiki-tl3-w18-lstm10-cpu-20261110.zip` | 14,011,657bytes; SHA256 `766bab1f3205697819af994acbb54cfe60682651977f393d76bec69608a75e3e` |
| `cpu-receiver-handoff-20261110-r2.tar.gz` | 14,093,920bytes; SHA256 `faa37344f199aff0f447cbb341f3727e136f0843e6a99f03d2415084982fe983` |

Local artifact root:
`/Users/alfa/Documents/WunderFunder_submission_ready/tl3-lstm-complement-20261110`.
The ZIP is in `build-r4/`, the portable tar in `delivery-r2/`.
The outer/inner handoff checksums and four embedded NPZ hashes were inspected;
this is byte-level evidence, not model testing.

[RUNBOOK.md](RUNBOOK.md) is the executable handoff: transfer the R2 tar, bind the
canonical valid.parquet and an available CPU, build/pin the Linux image, then
`bash launch_server.sh profile`. Review that receipt before
`bash launch_server.sh full`. Profile has a30-minute bound; full validation a
two-hour bound, both1CPU/16GB, no swap/GPU/network during model execution. These
are maximums to admit with a fresh server budget, not elapsed-time measurements.

Before either launch, record the named operator, host, data path, CPU assignment,
budget and output directory alongside the receipts. No usable server/data path
or fresh execution budget has been supplied in this session. No command was run.
The full run preserves paired predictions and scores all1873 sequences; it does
not replace validation with the two-sequence profile. The hidden test row count
remains unknown, so the whole-test time projection cannot yet be cleared.

If parity passes and the served increment/resource measurements support release,
refresh portal history and review this exact ZIP for a later explicit upload.
Otherwise preserve the negative result. Do not retune against portal feedback.

For the next distinct candidate, the [since32 artifact handoff](SINCE32_ARTIFACT_HANDOFF_20261112_R1.md)
records exact producer paths, terminal checkpoint hashes and required provenance.
Canonical R4 readout acceptance is not attached at the pinned PR89 head after
the R3 data-path failure; the reported research scores do not resolve that gap.
Collect both fixedE6 since32 checkpoints,
their exact source/configs, normalizers, fitting/split/teacher ancestry, terminal
logs and checkpoint/grade hashes from PR89's producer. Preserve both seeds and
the pending same-source control; do not select a best epoch after viewing scores.
An additional CPU implementation/comparison is still required. Existing training
jobs, shared pipeline and optimization settings remain untouched.

## Source records

- [Frozen recipe](recipe.json), [source/weight identities](INPUTS.json),
  [original dedup audit](DEDUP.json), [PR53 fixed-mixture results](PR53_FIXED_MIXTURES_REPORT.json).
- [Receiver R2](RECEIVER_R2.md), [server commands](RUNBOOK.md),
  [current observation and evidence boundaries](SUBMISSION_STATUS_20261112_R1.json).
- [PR76](https://github.com/VladKochetov007/WunderFunder/pull/76),
  [PR85](https://github.com/VladKochetov007/WunderFunder/pull/85),
  [PR90](https://github.com/VladKochetov007/WunderFunder/pull/90).
