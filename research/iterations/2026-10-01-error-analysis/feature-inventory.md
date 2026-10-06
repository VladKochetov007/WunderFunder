# Current student4: input inventory and one-pass error-regime extraction

Source/header inspection only, 2026-10-01. No predictions, targets, raw feature values, new scores, inference, or training were computed during this inventory. Remote NPY/NPZ headers and the small chain map were read with one CPU thread and CUDA hidden. Existing continuation artifacts remain unchanged.

## Exact available inputs

Remote root: `/home/grinchenkoeg/WunderFunder-creative`. Let `C=research/iterations/2026-10-01-continuation`, `P=research/iterations/2026-10-01-overnight`; folds are `f=0,1`.

| Artifact under root | Verified shape/type | Role |
|---|---|---|
| `C/cache/pred_cont_sh256_s3h2_f{f}_valid.npy` | `(1873,20000,2)` float32 | Shallow h256×1 seed3 H2 member |
| `P/cache/pred_on_ctl128s3_f{f}_valid.npy` | same | h128 seed3 H2 member |
| `P/cache/pred_on_ctl128s4_f{f}_valid.npy` | same | h128 seed4 H2 member |
| `P/cache/pred_on_ctlc500_f{f}_valid.npy` | same | h128 chunk500 H2 member |
| `C/student/native-quality-01/cache/pred_student_quant_f{f}_valid.npy` | same | Exact current four-member native INT8 ensemble output |
| `cache/valid_targets.npz` | `y:(1873,20000,2)` float32; `mask:(1873,20000)` bool | Existing target/mask source |
| same NPZ | `seq_ix,row_group,market_class:(1873,)` int64 | Prediction-cache order and retrospective class |
| `cache/chain_folds.npz` | `split,row_group,chain,fold,pure:(12479,)`; split U5, indices int64, pure bool | Joined by `(split='valid',row_group)` |
| `research/artifacts/market-classes-v1.npz` | `train:(10607,)`, `valid:(1873,)` int64 | Historical class metadata; not an inference input |
| `datasets/valid.parquet` | 1873 row groups; each 20000×112 raw features | One-sequence causal extraction |

All ten prediction NPY headers exist and report **299,680,128 bytes/file**, C order. Cache position is not automatically parquet row group: join through `valid_targets['row_group']`. Individual GPU caches have companion `.inputs.json`; native composites have `.json`. These already bind exact endpoints, fold rows, chain map and prediction hashes. Selected fold rows are finite and other rows NaN according to their completed producers/checkers; the new inventory did not rescan bodies.

**Native members were not saved separately.** The native-quality cache contains exactly the two composite NPYs and two receipts. GPU/AMP member disagreement is available immediately, but must not be called native-member disagreement. Native composite minus the literal GPU four-member mean is a separate numerical-deployment drift diagnostic, not a fifth independent model.

Use only the same exposed **860 pure sequences /250 chains** in folds0/1 for primary errors. Each model uses its own held fold. Original valid file has1873 sequences; do not silently score non-pure/other-fold predictions. Chains are resampling clusters, never state-continuity groups. Every sequence resets, including consecutive members of the same chain.

## Existing source worth reusing

- `src/wnn/data.py`: `SequenceFile.read(row_group, targets=False, mask=False)` reads exactly one row group, single-threaded Arrow, returns contiguous float32 raw array and `seq_ix`.
- `src/wnn/features.py`: `event_features(raw)` gives99 causal channels with row0 deltas/flags zero. `StreamingEventFeatures` is the reset/prefix reference. Current model input is raw112+these99=211.
- `C/student/evaluate_student.py` and `C/student-seed4/evaluate_seed4_v2.py`: validated fold/rowmap/cache joins and literal ordered float32 averaging; reuse I/O and provenance, not their fixed comparison CLI.
- `scripts/fold_eval.py`: existing six-sum metric implementation. The new shared error-analysis metric design must explicitly decide pooling and error definitions; `docs/data_overview.md` describes per-sequence averaging whereas the existing research evaluator pools sums. Do not use that prose as an implementation spec or quietly change the historical score.

Raw ordering is fixed: i0 prices0:22, volumes22:44, trade prices44:48, trade volumes48:52; i1 counterparts52:74,74:96,96:100,100:104; extras104:112. These are anonymized component codes, **not ordered depth levels or known physical prices**.

## Minimal causal regime proposal, frozen definitions before extraction

No new fitted thresholds or nuisance models are needed for extraction. All rolling/history operators use rows0…t, including warm-up and unscored rows. Windows are truncated at sequence start, with explicit denominator `min(t+1,W)`. Suggested fixed windows32 and256; do not select them from subsequent error results.

| Family | Exact candidate columns | Source/meaning |
|---|---|---|
| Current event/activity | 8 event flags | Existing features66:74: i0 joint selected bid/ask price-code change; i0 bid/ask volume changes; i0 trade-price/trade-volume changes; i1 price/volume/trade changes |
| Recent activity | i0-price and i0-trade-volume event rates over32/256 rows (4 floats) | Trailing inclusive sums of flags0 and4; current row is observable |
| Memory/age | 3 capped log ages | Existing features74:77, since i0 price/trade-volume and i1 price event; cap10000 as in actual model |
| Jump/code transition | i0/i1 changed-price-coordinate counts; RMS of current22-coordinate raw code deltas; RMS of held last i0 price-event delta (5 values) | Existing delta slices0:22,44:66 and held77:99. Call these code-transition magnitude, not physical return/mid-price jump |
| Sequence/training position | step index, step modulo2000, step modulo500 (3 integers) | Sequence context and known truncated-training boundary diagnostics; seq_ix itself is an identifier only |
| Prediction state | literal four-member mean for each target; member standard deviation and member range per target; sign disagreement per target | Available GPU member outputs. Compute dispersion before clipping; preserve raw mean for shared metric clipping policy. Sign disagreement excludes zeros explicitly or records zero separately; freeze choice before run |
| Numerical drift | native composite minus GPU mean for t0/t1 | Checks whether an apparent regime issue follows the deployment approximation; does not require native member reruns |

The raw current row may additionally carry volume-code norms/extras for later interpretation, but do not grow a112-channel error cache before the first pass. The above small set covers activity, abrupt observed changes, retained-history age, position, confidence/disagreement and deployment drift.

**Retrospective labels kept separate:** target magnitude/sign/zero, is_scored, market class, fold and chain may stratify error reporting but cannot be claimed as causal serving features. `wnn.data.market_class()` derives class from target quanta over the labelled sequence. Sequence-wide means/medians or future event distances similarly cannot define a deployable regime. Avoid reusing `scripts/micro_features.py::trade_rows`, which uses the whole-sequence median, as an online event detector.

## One pass, small memory, no model execution

1. Validate the existing input receipts/order hashes once; load only tiny rowmap/fold/chain metadata. Open the ten prediction files read-only with mmap. Decode cached y/mask once (~337MB combined arrays), or agree an equivalent raw-parquet target source explicitly; do not mix sources silently.
2. For each selected pure sequence, read its raw112 array once (~8.96MB), compute the99 existing features transiently (~7.92MB), and derive the small regime columns above. Reset all history at its first row. Check returned seq_ix against metadata.
3. Slice that sequence from its four GPU member caches and its native composite cache. Keep the exact member order `[sh256s3,ctl128s3,ctl128s4,ctlc500]` and literal float32 sum/divide4. This produces both current deployment predictions and interpretable member disagreement without inference.
4. Feed `(fold,chain,cache_position,row_group,step,causal_regime_columns,retrospective_labels,y,GPUmean,native_pred)` to the agreed shared error reducer. Apply `mask AND step>=99` only in that reducer, after feature histories were computed. If unscored/scored comparison is wanted, retain the same causal histories and separate report strata.
5. Prefer per-chain/per-regime aggregate sufficient statistics plus a small deterministic set of illustrative row windows. If row-level output is necessary, stream Parquet groups per sequence and only the agreed fields/rows; do not cache raw112 or full99 again. Full860×20000×32float32 alone would add2.20GB, so an unbounded feature dump is unnecessary.

Only860 row groups are decoded (17.2M input rows; ~7.71GB of raw float32 values before compression). Working memory should stay below1GiB with existing target cache plus one sequence, rather than materializing all features; compressed-Parquet decoding time remains unmeasured. No GPU, model reload, new checkpoint or teacher cache is required. This is ready for a compact CPU extractor once the shared metric/regime/report design is fixed; no extractor or metric was executed here.
