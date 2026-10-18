# Idea screens R2: fixed-alpha selection and EMA parts

**Executed:** [measured results](RESULTS.md), [independent result audit](RESULT_AUDIT.json), [actual raw evidence](records/attempt_v2/terminal_capture/README.md). V1 failed before data loading because the initial staging list below omitted transitive R1 imports; use the [seven-file correction](STAGING_CORRECTION_REVIEW.json). V2 completed all six views, exit0/captured/stopped.

The preparation manifest and initial independent review bind the prepared README/source revision at Git commit `758db928` (also retained byte-for-byte in the remote bundle and raw launch source snapshot). This execution annotation was added later; all six analysis Python source hashes remain unchanged. No prepared receipt or raw output was rewritten.

R2 is a bounded CPU analysis packet for the fixed question in [NEXT_WAVE](../idea_screens_r1/NEXT_WAVE.md). It compares six predeclared representations, selects one ridge alpha per view using only three component-held-out folds inside the 14 fit components, refits on all stride-sampled fit rows, and scores the already exposed six-component outer panel once. This is exploratory regularization evidence on the R1 sample, not a fresh validation result, full-recipe comparison, or expected public gain.

## Fixed method

The views are `raw` (112), `i0_only` (52), `raw_lags_1_16_128` (448), `raw_signed_delta_ema_hl32` (224), `raw_absolute_innovation_ema_hl32` (224), and `raw_both_ema_parts_hl32` (336). Feature construction reuses `idea_screens_r1.feature_views.build_view`. EMA halves are sliced from the exact R1 half-life-32 output: signed delta is `delta = current - previous`, and absolute innovation is `abs(delta - previous_signed_ema)`, before updating that EMA. State resets for every sequence.

All causal histories are built over complete sequences before the frozen stride40 fit-row selection. Inner folds are a deterministic shuffle of sorted fit component IDs with seed19020, distributed round-robin into five, five and four held components. Every fit sequence belonging to a held component is scored at full length on its original mask. The official `GlobalAccumulator` pools all held components for each alpha; selection maximizes the pooled mean of target WP scores, with exact ties choosing the larger alpha. Each fold computes its own unweighted mean/std from only its sampled training rows. The same alpha serves both target heads. Outer labels do not enter normalization or alpha selection.

The alpha grid is fixed at `[1, 100, 10000]`. The packet reports each view's complete pooled and per-fold inner table, selected-alpha outer result, alpha1 full-fit diagnostic baseline, coefficient norms, six component raw-moment tables, and fixed-prediction drop-one-component scores. It also reports actual fit/eval row counts, per-view CPU time, process CPU/RSS, the exact source/sample hashes, and any interrupted progress. The total is54 inner model fits plus selected and alpha1 full-fit refits (deduplicated when alpha1 is selected); both target heads are fitted in each model fit.

`i0_only` is a joint ablation: it retains raw columns0:52 and omits both the i1 block and the eight tail channels104:112. Its contrast with raw112 cannot isolate i1's incremental effect or establish that i1 is harmful. R2 addresses a regularization/data-size explanation for R1's feature comparisons and separates the two R1 EMA summaries; it does not establish gradient conflict, sparse-factor identification, or recurrent-state staleness.

## Source and sample pins

Use the exact R1 sample at `/home/grinchenkoeg/computer-vision-homework-20261018/research_tmp/nf19-sample-v1`. Its `SAMPLE.json` SHA256 is `0b03a4e2a6aab4eca36bdbb721d30e94b675c3425c0e0a7deca5ae562c2bbaa5`. R1's `load_sample` checks every selected `.npy` byte length and SHA256 before exposing read-only memmaps; the runtime `SOURCE_MANIFEST.json` records every role/source/group/component/sequence identity and each exact feature/target/mask hash.

Imported source pins are checked before fitting:

| Source | SHA256 |
|---|---|
| `idea_screens_r1/sample_data.py` | `13a4acfeb62ace285184c8fe0f88ff7ba65a3969843eaa0e0f6175189325abc7` |
| `idea_screens_r1/feature_views.py` | `60d42c3c961001d9177c7d7d300fcd3472662cc3d024deaac74263e37fe75550` |
| `source-a266b036/src/wnn/data.py` | `a93da840137adbd7afdb79a22687fb73150e77107f91ca2adc5211b854bc7e00` |
| `source-a266b036/src/wnn/gmetric.py` | `2cf10e740a831379f62b7d1269456053f5c1f9efa2dbab03156e366972bed5aa` |

The runtime manifest also hashes all R2 Python sources and records canonical metric ID `global_wp_clip2_v1`. No source parquet scan or model inference occurs in R2. The run uses float64 normal equations for transparent CPU analysis, not a strict-FP32 campaign-training claim.

## Checks and remote command

From the repository root, with the already provisioned Python/NumPy/canonical source available:

```sh
PYTHONPATH=src:research/campaign_2026-10-19 python -c 'from idea_screens_r2 import test_ridge_cv as t; t.test_fit_row_collection_ignores_outer_partition_labels(); t.test_seeded_component_folds_are_a_partition(); t.test_signed_and_absolute_ema_views_are_exact_causal_r1_parts(); t.test_outer_score_matches_canonical_global_scorer_and_drop_one(); print("4 R2 invariant checks passed")'
```

On B, stage sibling `idea_screens_r1` and `idea_screens_r2` folders under `B/support-source/eda19-r2-v1`; include the exact R1 `sample_data.py`, `feature_views.py`, and `__init__.py` bytes. Then run from that bundle root under the approved serialized 50%-CPU, 1.5GB/no-swap, CUDA-hidden, one-BLAS-thread, physical-scratch, 600-second cgroup:

```sh
PYTHONPATH=/home/grinchenkoeg/computer-vision-homework-20261018/source-a266b036/src:$PWD \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  CUDA_VISIBLE_DEVICES= \
  TMPDIR=/home/grinchenkoeg/computer-vision-homework-20261018/research_tmp \
  python -B idea_screens_r2/run_analysis.py \
    --sample /home/grinchenkoeg/computer-vision-homework-20261018/research_tmp/nf19-sample-v1 \
    --output /home/grinchenkoeg/computer-vision-homework-20261018/research_tmp/nf19-r2-v1
```

The CLI refuses any other sample manifest or imported dependency hash. It creates a fresh output directory with `SOURCE_MANIFEST.json`, `START.json`, `RESULT.json`, `DONE.json`, and fsynced `incremental_progress.jsonl`. A failed/partial output is retained; do not retry over it or silently shrink the panel, feature set, alpha grid, fold scheme, or resource envelope. Root owns independent review and any later execution/admission. No install, remote operation, sample-array execution, or model fitting was performed while preparing this packet.
