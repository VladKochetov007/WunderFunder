# Exact conditional-feature variants

These separately versioned scripts leave the E415 proxy cache and scorer unchanged.
They use the same component-excluded backbone and exactly its prespecified sampled
rows. No backbone states from different fitted models are pooled. A and B exchange
residual-fit/evaluation roles; only predictions and metric moments may be pooled.

`feature_v2_cache.py router` fits a four-class balanced ridge classifier using only
TRAIN market-class labels and each sequence's raw rows 0–98. It selects at most
1024 campaign-eligible TRAIN groups by fixed label-independent hash. The target-cache
row_group mapping handles the corrupt TRAIN group hole; sequence IDs must match.
No VALID labels or VALID-derived class priors enter router fitting or normalization.
`histories` then adds causal histories to existing frozen-state caches and binds
every sequence, row, component, router hash and base-cache hash.

| Family | Candidate | Prespecified controls |
| --- | --- | --- |
| H082 regime heads | Four warmup-routed 32-coordinate state heads | Four hash-routed heads and one universal 128-coordinate state head |
| H083 regime affine | Four regime intercepts and four prediction slopes per target | Hash routing, equally sized state head, global affine |
| H084 target clocks | t0 row EMA; t1 event EMA, alpha 2/33 | Swapped assignment, common mixed clock, row both, event both, event lag4096 |
| H085 activity interactions | Row-history × EMA event rate/price-change variance at32/256 | Additive map, stale128 gate, fixed shuffled fit-row gate, history alone |
| H086 i1 innovations | Conditional raw i1 residual, lag8 | Current-row i1, lag8 own-i0 residual, i1 lag136 |
| H087 event age | Row-history × log age of latest i0 price update | Additive map, stale128 gate, shuffled fit-row gate, history alone |
| H088 transitions | Current-minus-previous raw state and coordinatewise state products | Stale128 transition, current raw input |
| H089 cycle smoothing | Trailing eight-row average followed by EMA32/256 | Cycle7 + EMA32.5/256.5; row EMA35.5/259.5; original row EMA32/256; phase8 and phase7 |

Every family also reports the untouched backbone and a two-coefficient global
affine calibration per target. Except the deliberately smaller global affine
control, mechanism and capacity controls have equal learned head width per target.
H082 uses131 coefficients; H083 uses8; remaining families use99. Random feature
maps, normalization, ridge0.01 and seeds are fixed before evaluation. The conditional
i1 predictor fits raw-input reconstruction only on residual-fit components. Its
normalizer and parameters are frozen before evaluating the opposite components.
The shuffled-gate control permutes only fitting rows with seed271831; evaluation
uses the original causal gate. It never reads future evaluation activity.

Cycle mean-lag controls are explicit: trailing8 adds3.5 rows, trailing7 adds3;
the EMA mean lag is horizon−1. The phase filters instead match exponential row
decay and are separately reported. None is relabelled as another mechanism.

`feature_v2_screen.py` saves every arm's predictions, coefficients, transforms and
official Global WP, plus paired connected-component bootstrap intervals against
every candidate control. Sequence counts are retained separately from component
counts. A relative win against a damaged capacity control is not a model gain:
the candidate must improve the untouched predictor and simple affine control,
then beat the mechanism-specific placebos on independent confirmation. Existing
fold0 historical exposure remains a limitation; this is screening, not a final
claim. E415's degraded residual heads do not veto these new variants.

`feature_v2_recipes.py` writes explicit executable CPU/GPU/TPU drafts and optionally
seals all code/input bytes with `--seal`. Missing artifacts prevent sealing. Router
and history preparation are CPU-only, one core,4GiB and3600 seconds. Screens use
one core,4GiB and900 seconds; GPU solver reservations are1GiB operating/2GiB stop.
TPU stays a draft until hardware compatibility and matched FP32 numerics pass.
CPU hides CUDA. No command launches itself or bypasses controller admission.

Example screen generation after both history caches exist:

```bash
python scripts/feature_v2_recipes.py --root . --operation screen --backend cpu \
  --name exact_cycle8_direction0 --family cycle8 \
  --fit path/to/A.npz --evaluation path/to/B.npz \
  --fit-histories path/to/A_exact.npz --eval-histories path/to/B_exact.npz \
  --seal --output research/campaign_2026-10-04/specs/exact_cycle8_direction0.json
```

The bounded synthetic tests exercise all eight executable families, causal prefix
invariance, exact clocks/cycles, TRAIN cache index holes, no evaluation-fit leakage,
sample identity, bootstrap sequence counts, and CPU/GPU sealing. No real-data
scientific fits are part of this implementation milestone. H095 mature sequence
gain and H099 reset-lag probes remain separate work.
