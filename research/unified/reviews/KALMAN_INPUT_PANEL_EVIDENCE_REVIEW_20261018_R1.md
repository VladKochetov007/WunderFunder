# Independent evidence review: expanded Kalman input panel

## Verdict

**Pass for the narrow fixed-endpoint discovery claim.** The replay, score, and receipts have no confirmed blocker. This is not evidence of learned-feature quality, student transfer, online model updates, clean-fold performance, or public transfer.

## Reviewed material

The independent reviewer checked the frozen [recipe](../../campaign_2026-10-18/online_features_r1/kalman_input_panel_r2/recipe_r1.json), [replay implementation](../../campaign_2026-10-18/online_features_r1/kalman_input_panel_r2/replay_panel.py), preflights, [execution receipt](../../campaign_2026-10-18/online_features_r1/kalman_input_panel_r2/execution_r1.json), [grader](../../campaign_2026-10-18/online_features_r1/kalman_input_panel_r2/grade_panel.py), [result](../../campaign_2026-10-18/online_features_r1/kalman_input_panel_r2/result_r1.json), and official metric/bootstrap implementation. The reviewer did not open label or prediction-array payloads and did not rerun or modify the experiment.

## Findings

- The current recipe/source hashes agree across the final preflight, execution receipt, grading preflight, and result. Independent hashing confirmed the pinned replay, grader, Kalman, metric, and evaluation sources.
- Identity-only checks verify every selected 20,000-row sequence and exact `step_in_seq` order before inference; the grader repeats that check before loading labels or masks. It verifies `need_prediction == (step >= 99)` and scores `need_prediction AND is_scored` with `global_wp_clip2_v1`.
- The paired component bootstrap matches the frozen recipe: 10,000 replicates, seed `20261020`, resampling 82 source-qualified components jointly across the two fixed arms.
- The replay finished in 455.17 seconds with 2.21 GB peak worker RAM and 350 MB peak observed VRAM, below the 5 GB reservation/cap. It used no swap; attempt and parent OOM/memory-cap event deltas were zero.

## Evidence and limits

The raw endpoint scored `0.72188269 WP`; the local-trend replacement scored `0.72199901 WP`, a paired change of `+0.00011632 WP`. The conditional component-bootstrap 95% interval is `[-0.00007495, +0.00028789]`.

The 141-group prefix was expanded after the first 16-group panel produced a small positive point. Both panels use the same repeatedly exposed F0 fold and the same raw-trained checkpoint. Four input channels were replaced only at inference, creating a train/inference representation mismatch. The wider point is smaller and uncertain; it is discovery evidence, not independent confirmation. The post-hoc temporal bins do not establish a long-horizon or online-adaptation effect.

## Decision

Keep the broader retained-raw causal-state teacher/student proposal in [B300 draft PR #17](https://github.com/VladKochetov007/WunderFunder/pull/17) as a separate, unrun hypothesis. This replay does not establish a trainable-feature gain and does not justify a new B300 draft or submission.
