# Expanded causal Kalman input replay

## Question and status

This follow-up tests whether the small positive local-trend input-replacement point on the original 16-group panel persists across a wider portion of the same exposed F0. The frozen panel uses the first 141 source-qualified fold-0 evaluation groups, covering 82 components. This is an exposure-aware discovery continuation, not clean validation or independent confirmation. The replay is prediction-only: no parameters are fitted, and it compares the raw parent with the same fixed local-trend posterior replacement used in the first screen.

The recipe is frozen in [recipe_r1.json](recipe_r1.json). It pins the representative completed GRU replay profile and its source recipe by SHA-256; the preflight verifies the profile is successful, below the declared RAM/VRAM caps, and uses the same frozen inference path before the controller reserves resources. The replay and independent evidence review are complete. Status: **inconclusive positive, discovery only**.

## Why this follow-up

On the first 16 groups, the local-trend replacement scored `+0.000338 WP` over raw, with a paired component-bootstrap interval of `[-0.000019, +0.000716]`. The interval is uncertain and the panel is small. This wider pass gives the same fixed endpoint 82 source components before allocating a learned teacher/student experiment. It does not convert the previously exposed fold into a clean holdout.

## Expanded-panel result

On 141 groups / 82 source components and 228,976 official scored rows, the raw parent scored `0.72188269 WP`; replacing four channels with local-trend posterior levels scored `0.72199901 WP`, a paired change of `+0.00011632 WP`. The 10,000-replicate paired component bootstrap (seed `20261020`) gives a 95% interval of `[-0.00007495, +0.00028789]`; the bootstrap fraction at or below zero is `0.1221`, not a formal p-value. Independent review found no replay, cache, mask, metric, or receipt blocker.

The expanded estimate is smaller than the first-16 point and still uncertain. Post-hoc position bins are mixed: the 5,000–10,000 window is negative while the other three are positive. This is one fixed checkpoint trained on raw inputs, with four input replacements at inference; the representation mismatch remains. It is not a clean-fold, learned-feature, student, online-feedback, training-seed, or public-transfer result. Keep the trainable retained-raw local-trend teacher/student experiment in draft PR [#17](https://github.com/VladKochetov007/WunderFunder/pull/17); this result does not establish a large new direction or justify a separate handoff.

## Method and scoring

- The original GRU weights, FP32 inference, normalization buffers, event features and recurrent state behavior are preserved.
- Only columns `i0_p0`, `i0_p3`, `i1_p0`, `i1_p3` are replaced by causal local-trend posterior levels. The filter resets per sequence, reads current/past inputs, and has fixed process/observation noise from the first recipe.
- Prediction generation calls `SequenceFile.read(targets=False, mask=False)`. Labels and masks are loaded only by the separate grader, after prediction-cache hashes and the grading preflight are written.
- Before inference, the preflight reads only `seq_ix` and `step_in_seq` for the 141 selected groups. It checks all 20,000 rows per group against frozen sequence identity and exact step order, and writes a digest. It does not read target or score-mask columns. The grader independently repeats this check before loading labels.
- The preflight also hashes the bounded Parquet footer and trailer against the source identity in the frozen split manifest; it checks file size/mtime, schema, row-group count and selected sequence identity.
- The grader uses `need_prediction AND is_scored`, verifies `step_in_seq` is exactly 0–19,999 in row order, and checks that `need_prediction` matches steps 99–19,999 for every group. This also audits the equivalent row-99 mask used by the original 16-group replay.
- Before it reads target or mask values, the grader verifies the execution receipt, each cache sidecar, the sidecar-to-execution equality, the cache hashes, checkpoint and normalization ancestry, source hashes, exact arm names, dtype and dimensions.
- The official centered `global_wp_clip2_v1` implementation and a paired bootstrap over source components are used. The bins by sequence position are descriptive and post hoc.

## Artifacts

- [Replay implementation](replay_panel.py): metadata-only preflight plus a bounded raw/local-trend replay that reuses the pinned canonical GRU inference path.
- [Independent grader](grade_panel.py): verifies source, cache, mask and group identities before scoring.
- [Initial metadata preflight](preflight_initial_superseded.json): preserved for provenance, then superseded before inference when independent review found the grader also needed to verify exact `step_in_seq` ordering.
- [Second preflight](preflight_metadata_stale_superseded.json): the identity-only source check and execution receipt binding were added after this earlier receipt was written; it was superseded before inference.
- [Third preflight](preflight_identity_pre_footer_superseded.json): the bounded source-footer digest check was added after this identity/order receipt; it was superseded before inference.
- [Fourth preflight](preflight_pre_controller_command_superseded.json): the durable one-GPU admission command was frozen after this receipt, changing the recipe hash; it was superseded before inference.
- [Fifth preflight](preflight_pre_profile_pin_superseded.json): the durable controller requires a resource profile; this receipt predates the addition of a SHA-256 pin and semantic checks for the profile and its recipe, so it was superseded before inference.
- [Sixth preflight](preflight_pre_scope_sync_superseded.json): it used a unique limited-scope name that did not match the command frozen in the recipe; this command-fidelity correction was made before inference.
- [Final preflight](preflight_r1.json), [execution receipt](execution_r1.json), [grading preflight](grade_preflight_r1.json), and [result](result_r1.json) bind the completed stages.
- [Independent evidence review](../../../unified/reviews/KALMAN_INPUT_PANEL_EVIDENCE_REVIEW_20261018_R1.md) passes the source/hash, row-order, metric/mask, bootstrap, exposure, and resource-receipt audit; it supports only the fixed-endpoint discovery claim above.
- The durable [runtime attempt receipt](../runtime/attempts/kalman-input-panel-f0-141-20261018-r1.json) records a 455.17-second successful replay, 2.21 GB peak worker RAM, 350 MB peak observed VRAM, no swap, and zero attempt/parent OOM or memory-cap event deltas.
- Prediction arrays are kept under `/home/vlad/development/wunder/research_tmp/online_features_r1/kalman_input_panel_r2/`; the repository contains only small provenance receipts and summaries.

## Resource bounds

One local GPU worker through the durable campaign controller and its GPU lock, one worker CPU core within the two-core aggregate, a 5 GB reservation backed by a SHA-256-pinned completed representative replay and enforced as the hard RAM cap, no swap, a 20-minute worker timeout, and physical scratch at `/home/vlad/development/wunder/research_tmp`. Preflight checks the prior profile's measured peak RAM/VRAM against the declared caps; the controller also admits against live GPU free memory using its safety reserve. No paid compute is used.

## Frozen commands

Run from the repository root. Exact commands and resource bounds are also recorded in the recipe:

```sh
WNN_SLICE=wnn-campaign.slice WNN_CPU_QUOTA=50% WNN_THREADS=1 WNN_MEM_MAX=500000000 WNN_RUNTIME_MAX=120s WNN_SCOPE_UNIT=wnn-kalman-panel-preflight-20261018-r9.scope scripts/limited .venv/bin/python research/campaign_2026-10-18/online_features_r1/kalman_input_panel_r2/replay_panel.py --mode preflight

WNN_SLICE=wnn-campaign.slice WNN_CPU_QUOTA=20% WNN_THREADS=1 WNN_MEM_MAX=1500000000 WNN_RUNTIME_MAX=120s WNN_SCOPE_UNIT=wnn-kalman-panel-launch-20261018-r2.scope scripts/limited .venv/bin/python scripts/bandit_resources.py --root . --state research/campaign_2026-10-18/online_features_r1/runtime launch --attempt kalman-input-panel-f0-141-20261018-r1 --kind gpu --cpus 1 --reserve-bytes 5000000000 --profile research/campaign_2026-10-18/online_features_r1/runtime/attempts/kalman-gru-replay-20261018-r3.json --contract research/campaign_2026-10-18/online_features_r1/kalman_input_panel_r2/recipe_r1.json --admission-timeout 900 -- /usr/bin/timeout --signal=TERM --kill-after=10s 1200s /usr/bin/env PYTHONDONTWRITEBYTECODE=1 WNN_THREADS=1 WNN_LOADER_WORKERS=0 ARROW_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 /home/vlad/development/wunder/wnn_connectome_starterpack/.venv/bin/python /home/vlad/development/wunder/wnn_connectome_starterpack/research/campaign_2026-10-18/online_features_r1/kalman_input_panel_r2/replay_panel.py --mode execute

WNN_SLICE=wnn-campaign.slice WNN_CPU_QUOTA=50% WNN_THREADS=1 WNN_MEM_MAX=1500000000 WNN_RUNTIME_MAX=900s WNN_SCOPE_UNIT=wnn-kalman-panel-grade-20261018-r1.scope scripts/limited .venv/bin/python research/campaign_2026-10-18/online_features_r1/kalman_input_panel_r2/grade_panel.py
```

## B300 handoff

This frozen-input replacement is not by itself a large-scale training direction. The more useful trainable path retains raw inputs and adds learned causal level/slope/innovation/uncertainty features for teacher training and GRU distillation; it is already covered by draft PR [#17](https://github.com/VladKochetov007/WunderFunder/pull/17). The PR description now includes this wider, uncertain fixed-filter result and its exposure/representation limits for the B300 team. No separate draft was opened from this screen, which cannot establish a trained-feature gain.
