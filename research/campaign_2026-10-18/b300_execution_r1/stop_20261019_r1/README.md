# tabdata stopped and preserved results, 19 October 2026

**Remote stop is CONFIRMED at 14:33:55 UTC / 17:33:55 Moscow.** The successful user-requested SSH session authenticated as `epanov_s` on `ml-app100-lp1.hq.bc`. At 14:31:34 UTC it executed `systemctl --user stop --no-block 'tabdata-*'`. The four reporting timers/paths were subsequently disabled. No new training, export, grading or scientific retries were launched.

[STOP_CONFIRMED.json](STOP_CONFIRMED.json) contains the live host evidence, foreign process identities, existing R01 grade receipts and verifier hashes. Its SHA256 is `e0f86ca64259d54f2398b929c21739ebf6a5da9f962d15652669506802a0fe1e`. The checksummed transfer was independently decompressed and verified locally. The original evidence remains at `/data/epanov_s/tabdata/state/stop-20261019-final/STOP.json`.

## Verified stopped state

- Zero active, activating or deactivating `tabdata` units; zero project processes.
- No project CUDA clients on any of the eight GPUs. The three original foreign vLLM clients remain with identical PID/start ticks: 4060404/207469348, 4086591/208123403 and 4061037/207474461.
- Both reporting timers and both completion paths report `disabled`.
- Checkpoints, predictions, logs, partial fits and earlier failure records remain under `/data/epanov_s/tabdata`. No artifacts were deleted. Large weights remain on the node; this packet transfers stop/result metadata, not every checkpoint.

The campaign must remain stopped. Interrupted work and queued cells are not completed experiments. An interrupted epoch may have unsaved progress; the stop does not reconstruct it. Earlier failed-SSH receipts and unconfirmed-stop comments in historical snapshots remain dated history, superseded by this proof.

## Preserved scientific evidence

The [API snapshot](GITHUB_SNAPSHOT.json) preserves PR metadata/comments. Automated status comments last updated around 14:30 UTC before the successful stop; they describe the earlier running state and are not current resource observations.

| Experiment | Preserved evaluation stage | Interpretation |
|---|---|---|
| [PR15](https://github.com/VladKochetov007/WunderFunder/pull/15#issuecomment-6256107437) | 64/120 scored fits; 16/30 scored cells per arm | Partial geometry comparison below |
| [R01/PR25](https://github.com/VladKochetov007/WunderFunder/pull/25) | 12/30 existing paired official grades verified; 24 scored fits | Partial width/depth comparison below |
| [Age/PR26](https://github.com/VladKochetov007/WunderFunder/pull/26#issuecomment-6256781766) | Last reporter snapshot: 0/30 scored pairs | No reported paired quality comparison; not a count of completed individual fits |
| [Innovations/PR27](https://github.com/VladKochetov007/WunderFunder/pull/27#issuecomment-6256782166) | Last reporter snapshot: 0/30 scored pairs | No reported paired quality comparison; not a count of completed individual fits |

PR15 equal-fold means over available matched cells:

| Arm | Mean WP | Paired delta versus baseline | Scored cells |
|---|---:|---:|---:|
| baseline | 0.67824425 | - | 16/30 |
| moments | 0.67809108 | -0.00015317 | 16/30 |
| dictionary | 0.67756978 | -0.00067447 | 16/30 |
| combined | 0.67702747 | -0.00121678 | 16/30 |

## R01 existing official grades recovered without recomputation

The sealed recovery publisher looked for `grade.stdout.txt`/`grade.stderr.txt`; the delivery helper had written `stdout.txt`/`stderr.txt`. Its twelve rejected publication chains therefore did not establish failed training or absent grades. Reading the existing files through the correct sealed `verify_grade_receipt` succeeded for all twelve, with zero rejections. No frozen runtime was patched, and no fit, export or grade was rerun.

The verifier and its dependencies match the frozen delivery watcher: manifest SHA256 `4538090865237d80d163edd5cf8bc95d91067aefaa0446fc27e74400f8114c22`; receipt verifier SHA256 `3ab366ba85b722fb4f1804c0f5e73ed769909f5fbbb609f3b7067aa2d2b44167`; core SHA256 `ccbe2da929ed09df83c5750a42d44dd03ade41385c0d015c7e681c81cb3b4a09`. The local frozen copies have the same hashes. Verification checked original command, output/log hashes, bound fit/export receipt hashes, split identity, fixed epoch 5, metric ID and finite matching scores. This closeout does not independently rehash every large model/prediction payload or rerun the full provenance audit.

Each fold has the same three fitting seeds, 1003, 1004 and 1005:

| Fold | Pairs | Wide WP | Deep WP |
|---|---:|---:|---:|
| 0 | 3 | 0.67462892 | 0.67239269 |
| 1 | 3 | 0.66133103 | 0.66053688 |
| 2 | 3 | 0.69694938 | 0.69664307 |
| 3 | 3 | 0.70345510 | 0.70368251 |
| Equal-fold mean | 12 total | 0.68409111 | 0.68331379 |

Deep minus wide: **-0.00077732**. This is the six-epoch strict-FP32 labels-only fixed-epoch-5, near-parameter R01 comparison, not PR2 matched FLOPs. The per-pair original score receipts, grade paths/hashes and component-bootstrap summaries are preserved inside [STOP_CONFIRMED.json](STOP_CONFIRMED.json).

All comparisons use exposed development folds. Partial means are not an aggregate confidence interval, a significance result or a public-score forecast. No model is promoted, and neither incomplete panel is relabelled as complete. Preserve negative and inconclusive evidence; no resumption is authorized by this closeout.
