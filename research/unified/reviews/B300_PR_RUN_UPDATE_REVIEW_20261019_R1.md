---
kind: review
status: complete
created_utc: 2026-10-19T16:07:07Z
hypothesis_ids:
  - H-B300-R01
  - H-CAUSAL-GEOMETRY-DISCRETE-B300-20261018-R1
experiment_ids:
  - E-B300-R01
  - E-CAUSAL-GEOMETRY-DISCRETE-B300-20261018-R1
  - E-B300-PR4-AGE-SUCCESSOR-V1
  - E-B300-PR14-INNOVATION-SUCCESSOR-V1
evidence_paths:
  - https://github.com/VladKochetov007/WunderFunder/pull/15
  - https://github.com/VladKochetov007/WunderFunder/pull/25
  - https://github.com/VladKochetov007/WunderFunder/pull/26
  - https://github.com/VladKochetov007/WunderFunder/pull/27
  - https://github.com/VladKochetov007/WunderFunder/blob/967f2f342e919e69f4f6cee39373b3b01b25c16a/research/campaign_2026-10-18/b300_execution_r1/stop_20261019_r1/README.md
---

# October 19 B300 pull-request results and stop review

## Finding

The latest useful B300 evidence is **partial**. A user-requested stop was
confirmed on October 19 at 14:33:55 UTC. The proof reports no active project
units, processes, or CUDA clients; it preserves files and leaves foreign vLLM
processes untouched. Interrupted and queued cells are not completed fits.

The most informative quality comparison is PR25's 12 verified width/depth
pairs. Its equal-fold point estimate modestly favors the wide H512×1 model over
the H384×2 model. PR15's partial geometry comparison also favors its control.
Neither is complete CP30 evidence or a promotion. PR26 and PR27 have no scored
paired cells, so their runs supply no paired quality result.

## Pull and source snapshot

The review used GitHub PR metadata/comments and the preserved stop packet on the
PR25 head `967f2f342e919e69f4f6cee39373b3b01b25c16a`. The stop packet's
`STOP_CONFIRMED.json` has Git blob `74311b78e20839d5e60b38d7ee4c00f22f9681a0`.
Its verifier reports all twelve existing R01 grade receipts passed, with zero
rejections; it checked command/output and fit/export receipt hashes, split,
fixed epoch 5, official metric and finite paired scores. I independently
recomputed the displayed fold and equal-fold arithmetic from those twelve JSON
rows. I did not load large model/prediction payloads or repeat the sealed grade
verification.

All four PRs below remain open drafts targeting `research-team` at the checked
snapshot. The stop proof is linked directly from
[PR25's result/stop comment](https://github.com/VladKochetov007/WunderFunder/pull/25#issuecomment-6268923635).

## B300 results

| PR | Partial result at stop | Assessment |
|---|---|---|
| [#25: width vs depth](https://github.com/VladKochetov007/WunderFunder/pull/25) | 12/30 verified matched fold/seed pairs; wide `0.68409111`, deep `0.68331379`, deep−wide `−0.00077732` equal-fold WP | Useful directional evidence for this labels-only six-epoch, fixed-epoch-5, near-parameter recipe. Wide won 9 of 12 pairs; deep won 3. Deep-minus-wide by fold was −0.002236, −0.000794, −0.000306, then +0.000227. The 12 pairs span four previously exposed folds and three seeds; component bootstraps condition on each fixed pair and do not estimate training-seed uncertainty. Not a completed CP30 or public forecast. |
| [#15: geometry features](https://github.com/VladKochetov007/WunderFunder/pull/15) | 16/30 paired cells per arm. Baseline `0.67824425`; moments `0.67809108` (−0.00015317); dictionary `0.67756978` (−0.00067447); combined `0.67702747` (−0.00121678) | The tested partial bundle does not support spending next on these exact geometry combinations. All folds were exposed; there is no complete matrix or significance claim. Do not use this to close distinct geometry or feature descendants. |
| [#26: field age](https://github.com/VladKochetov007/WunderFunder/pull/26) | 0/30 scored matched pairs; last report showed four active paired cells before the stop | No paired quality evidence. This count is not a count of completed individual fits. |
| [#27: causal innovations](https://github.com/VladKochetov007/WunderFunder/pull/27) | 0/30 scored matched pairs; last report showed four active paired cells before the stop | No paired quality evidence. This count is not a count of completed individual fits. |

For PR15 the stop record gives equal-fold means over its partial matched cells;
it does not supply a full-panel uncertainty estimate. These are development
folds that have already been inspected. Do not convert their means into a public
score estimate or add their deltas to another model.

## Other relevant research-team update

The separate full-GRU tweak study is not a B300 result. Its newest indexed
teacher record is an RTX A4000 startup snapshot at 2026-10-19 15:40:38 UTC:
seed-3 H384×2 was active at epoch 0, batch 20/373, with no terminal checkpoint
or teacher/student score. The configured 22,380 updates are expected work, not
completed work. Its [current continuation](https://github.com/VladKochetov007/WunderFunder/blob/135c403ac1d929580ab783fcddfa19b78db98bde/research/campaign_2026-10-19/full_gru_tweaks_r1/CONTINUATION.md)
still records this as a startup state; later commits on the fetched branch did
not update that teacher receipt.

A different, useful but much smaller CPU screen is the one-fit sparse+EMA/NLL
result: pooled WP `0.5044141`, versus stored sparse-WMSE `0.4824412` and
dense-WMSE `0.4798731`. This is on an exposed small panel and is not comparable
to the B300 GRU scores; the missing dense+NLL arm prevents an interaction claim.
It is a candidate for a matched follow-up, not evidence of GRU transfer.
[Its source report](https://github.com/VladKochetov007/WunderFunder/blob/135c403ac1d929580ab783fcddfa19b78db98bde/research/campaign_2026-10-19/idea_combo_r1/RESULTS.md)
retains the full limitations and receipt identities.

## Decision

Use PR25 as a caution against assuming more recurrent depth helps at this
capacity and recipe. Treat PR15 as a partial negative for the tested geometry
bundle. Keep age/innovation unresolved. Collect the existing F3 width/depth
study separately before allocating another architecture matrix, as the
[October 20 plan](../../../specs/plans/2026-10-20-plateau-review-local-kaggle.md)
already requires. The stopped B300 work needs explicit new authority before any
resumption; this review does not request or launch jobs.
