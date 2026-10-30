# Heterogeneous teachers against fresh TL3

The user authorizes the proposed six new fits and a comparative ensemble table.
`ЕД3` is interpreted as `TL3` entered in the Russian keyboard layout. The primary
baseline is the freshly reconstructed canonical fixed mixture:34%W18 +33%tiedlead
seed5 +33%tiedlead seed6. W18's previously blocked cache/fit is a separate recovery
dependency; never substitute fbc1k and label that result canonical TL3. Historical
`tl3.zip` has validation ancestry and is not a clean primary baseline.

## Actual receiving progress

[Initial full-valid table](results/R1/COMPARISON.md) completes5 of12 predeclared
rows. Canonical fresh TL3 scores0.6923774245652725; its fixed90/10 GALAM mixture
scores0.692496447 (+0.000119022), while90/10 TCN gives0.692343740 (−0.000033684).
These are exposed fixed-prediction comparisons, not a completed six-fit screen
or a public-score result. Missing rows remain explicitly unavailable. See the
[result index](results/README.md) for source/metric/coverage evidence.

The two new teachers and the original W18 baseline completion are fitting in
detached, wall-bounded owned workers. LSTM/TCN passed80 finite real-data updates
each and actual CUDA prefix/chunk-carry checks; W18 passed40. The identified
W18 repair selects the actual fixed `model.pt` named by its prediction sidecars,
retaining explicit epoch/ancestry checks and0.8125 coefficients. It stores only
the consumed source/family targets, preserving mathematical targets while saving
unused cache variants. The first disposable profiles were rejected before updates
because continuation saving is incompatible with profile mode; corrected profiles
remove that flag only from engineering jobs.

The first new pool-cache admission rejected legitimate legacy membership ordering.
Source R3 reuses the canonical typed/duplicate-safe membership-set validator;
prediction alignment remains a separate ordered group-key join. The failed R2
attempt is preserved and produced no output. R3's GRU-only cache is complete;
[30 actual scalar coordinates](receiving/GRU_ONLY_ACTUAL_ARITHMETIC_R1.json)
agree with an independent weighted-sum calculation within FP32 rounding bounds.
See the [receiving index](receiving/README.md) and
[independent source review](INDEPENDENT_REVIEW_R1.json). Actual final quality
comparisons remain pending; no uncompleted mixture score is inferred.

## Frozen fitting panel

| Role | Run | Shape | Seed | New fits |
|---|---|---|---:|---:|
| LSTM teacher | `het6_lstm_s1201` | event211, H256×2 |1201|1|
| TCN teacher | `het6_tcn_s1202` | event211, H128×12, RF4096, mix0 |1202|1|
| GRU-only students | `het6_control_s1301`, `het6_control_s1302` | event211, GRU256×2 |1301/1302|2|
| Heterogeneous students | `het6_heterogeneous_s1301`, `het6_heterogeneous_s1302` | same GRU |1301/1302|2|

All six use4epochs, fixed zero-based endpoint3, batch32, chunk1000, AdamW LR.001,
WD.01/beta2.999, OneCycle5%warmup, clip1, strict FP32, dropout0, EMA consistency1,
EMA decay.999 and evaluated EMA, auxiliary next/structural weights.3/.3,
aux-weight.5, since-event32/floor.25. Each fit has322 complete batches per epoch,
20updates per batch:25760updates; total154560updates/24configured epochs.
Twenty-one fitting members are omitted by drop-last in each shuffled epoch.
The declared fitting pool remains10325 members; the reused legacy normalizer
fits its moments from the first two seeded fitting batches, entirely within train.

There is no initialization, learned propensity, validation-mask weighting,
correlation fine-tuning, sweep, bootstrap target or automatic extra seed. The
four students differ within each seed only by run name and teacher cache.
Teacher architectures/compute are intentionally unmatched: this tests the useful
pool and its transfer, not a pure architecture effect at equal compute.

## Data and targets

Use official earlier train only,10325 members after282 connected exclusions;
evaluate all1873 official-valid sequences/3528273 scored rows. Split SHA256 is
`0c4de3a253d23b3a5d5ca73b7aeb7fcae2b974673e4f9ec51e94e593c59d114c`.
No normalizer, teacher, initializer or cached fitting target learns from valid.
Keep whole sequences; shuffled seq_ix does not supply chronological ordering.
Primary valid is already development-exposed, not new confirmation evidence.

Control teacher target:50%fin256_s3 +50%fin256_s4, both fixed legacy epoch3.
Heterogeneous target:35%fin256_s3 +35%fin256_s4 +10%GALAM-LSTM128 human epoch3
(zero-based2) +10%new LSTM epoch3 +10%new TCN epoch3. This fixes70%total GRU
family weight. No mixture coefficients are fitted on primary valid.
Sum raw predictions in declared order using FP32; do not preclip components or
renormalize them. The canonical student then clips the pooled teacher and mixes
50%teacher/50%label, bootstrap0. Cached future teacher values are not substituted.
EMA consistency and auxiliary objectives are retained identically across students.

## Receiving commands

Use a separate immutable reviewed source and existing provisioned environment.
Parent owns one persistent SSH connection, original W18 repair and node allocation.
No foreign process is stopped or new provider/billing created. Numeric runtime and
memory estimates require actual representative profiles; unmeasured values stay
unknown. Full validation, train-eval diagnostics, LR, preclip gradient norms,
finite/attempted/successful updates, GPU/CPU/RAM/disk and checkpoint status must be
recorded. Fit workers must survive SSH disconnect and have explicit wall bounds.

```bash
export PYTHONPATH="$SOURCE/src:$SOURCE"
python "$SOURCE/scripts/heterogeneous_tl3_recipes.py" \
  --split-manifest "$SPLIT" --run-root "$RUN_ROOT" --output "$OUT/plan.json"
```

This materializes a JSON plan only. Invoke the existing canonical
`scripts/train_rnn.py` with one node's exact `args` after a qualified profile.
The controller/launch receipts separately bind source, data and resources.
Profiles may omit the incompatible profile-only `--save-training-state` flag;
scientific fits retain it. No scientific batch/precision/horizon fallback is implicit.

## New pool-cache interface

The old W18 cache has different coefficient/family/epoch contracts and must remain
unchanged. `scripts/build_heterogeneous_pool_cache.py` writes format
`wnn.heterogeneous-teacher-pool.v1`. Its input JSON has this top-level structure:

```json
{
  "format": "wnn.heterogeneous-teacher-pool-inputs.v1",
  "profile": "gru-only",
  "sources": [
    {
      "run": "tl3c_fin256_s3", "family": "gru", "epoch": 3, "coefficient": 0.5,
      "producer": "legacy_rnn_chronological",
      "prediction": {"path": "/absolute/prediction.npy", "sha256": "actual SHA256"},
      "sidecar": {"path": "/absolute/prediction.json", "sha256": "actual SHA256"},
      "checkpoint": {"path": "/absolute/model.pt", "sha256": "actual SHA256"},
      "provenance": {"path": "/absolute/provenance.json", "sha256": "actual SHA256"},
      "qualification": {"path": "/absolute/qualification.json", "sha256": "actual SHA256"}
    }
  ]
}
```

The example is a field template, not a complete source list. `gru-only` requires
both frozen GRUs at.5 each; `heterogeneous` requires the five declared sources in
the exact order above. New LSTM/TCN exports use the legacy producer schema.
GALAM uses `producer="galam_feature_export"`, `family="galam"`, run
`g2-galam-lstm128`, epoch2, coefficient.1; replace `provenance` with
`epoch_index:{path,sha256}` and add `epoch_index_canonical_sha256` equal to the
captured index's canonical `index_sha256`. All `sha256` file pins hash raw bytes.

Qualification JSON (`schema="wnn.teacher-source-qualification.v1"`) must bind
`run`, `producer`, `epoch`, `checkpoint_sha256`, `prediction_sha256`,
`sidecar_sha256`, `split_manifest_sha256`, actual `fit_groups` (`train`:the complete
10325 integers, `valid`:empty), complete `evaluation_valid_groups`, `source_sha256`
and `normalization_fit_groups_sha256` (SHA256 of ordered fitting train row-group
IDs encoded as little-endian int64). Qualification is not a trusted summary:
the consumer separately checks actual pinned raw provenance/epoch-index evidence.

Every export sidecar binds `run`, `epoch`, `checkpoint_sha256`,
`prediction_sha256`, complete unique `row_groups`, `feature_only=true`,
`split="train"`, `inference_precision="fp32"`, `split_manifest_sha256`.
Legacy sidecars additionally bind `run_provenance_sha256`; their
`campaign_split_manifest_sha256` alias is accepted. GALAM sidecars bind
`epoch_index_sha256` to the raw index-file byte hash. Preserve native producer
schemas: no fabricated legacy GALAM provenance. Legacy exports may cover10607
physical train row groups, GALAM exactly10325 fitting groups; all inputs are
joined by actual row-group IDs, never positional truncation.

```bash
python "$SOURCE/scripts/build_heterogeneous_pool_cache.py" \
  --inputs "$OUT/gru-inputs.json" --data-root "$RUN_ROOT" \
  --split-manifest "$SPLIT" --output "$RUN_ROOT/gru-only-targets" \
  --chunk-sequences 8 --min-free-gib 12
```

Repeat with heterogeneous input and output `heterogeneous-targets`. Each output
contains only `main_source_train.npy` (10325×20000×2 FP32 in filtered target-index
order), empty `main_source_valid.npy`, and bounded `metadata.json`. No valid labels
are read during construction. Incomplete attempts retain `.partial`; outputs are
create-only. Original producer bytes, memberships, coefficients, row order and
cache hashes are checked at construction and consumption. The new-format-only
student gate rejects bootstrap, family loss, scaling and drift from this recipe.
New consumer modules are included in source identity; old frozen sources stay intact.

## Comparison and interpretation

Root's separately frozen evaluator supplies singleton and ensemble t0/t1/WP,
delta versus canonical fresh TL3, weights, checkpoint/prediction hashes and full
coverage. Report the fixed GRU-only and heterogeneous teacher pools, paired
students/seed means, and predeclared contributions to TL3. A fallback mixture
using fbc1k instead of W18 must have a different name and cannot fill the TL3 row.
Two matched student seeds do not establish significance or teacher-seed variance.
A useful working transfer threshold is+.0005WP; mixed outcomes remain uncertain.
Retain every negative/failure and avoid interpreting low correlation alone as gain.

## Evidence status

Sixteen focused recipe/cache cases pass locally, including shuffled/excluded-row
joining, preclip arithmetic, raw-producer/fit-membership qualification, new-format
consumer dispatch, create-only outputs and byte tampering. The synthetic cache
integration uses reduced membership with only the production cardinality/source
file-loading guard mocked; it does not establish actual data or model validity.
Including applicable old distillation arithmetic gives26passes/1Windows legacy
case deselected. Ruff checks all9changed code files; the canonical structure
checker reports124files/zero errors. These checks start no model forward, fit or
GPU job; tiny synthetic cache files are engineering fixtures.

A broader archival-cache run has29passes/15failures:14 missing historical
sparse-checkout snapshots/receipts and one Windows `resource` import. Representative
missing-snapshot and `resource` failures reproduce in the unchanged prior TL3
worktree. No historical fixtures, cache guards or platform code are relaxed here.
Actual source/profile/launch/grade receipts are owned and appended by the parent.

## Bounded continuation after the first teachers start

`scripts/run_heterogeneous_six.py` and its two focused support/evidence modules
continue this exact six-fit graph. Stage all three together outside the immutable
model source (for example under `tools-r1`); the controller pins its own code and
verifies the actual `source-r3.identity.json` inventory separately. It adopts the
two already-running teachers through their original bound request/status records.
It also adopts explicitly issued control profiles/full fits instead of launching
duplicates. The failed R2 cache attempt stays preserved; the explicit successful
control-cache key is `cache-gru-only-r2` from the corrected immutable R3 source.

Before student admission it validates actual cache bytes and emits
`status/{profile}-arithmetic-fixture.json`:15 fixed train-group/timestep examples,
independent Python float64 sums, the stored FP32 results and an explicit FP32
roundoff bound. Source prediction banks are joined by original group IDs. This
diagnostic contains no validation targets and no quality claim. Control students
use GPU5/6; heterogeneous students use GPU0/1 after original teachers/exports
release those cards. GPU2/3 baseline/export work and the GPU4 service are untouched.

The fixed profile is80 finite updates (`profile-steps3`, `profile-warmup1`), with
the profile-only save-training-state conflict removed and scientific fits still
retaining resumable state. Owned per-command workers have900s profile,1800s
cache/export and at most21600s fit bounds; the overall controller closes admission
at21600s and drains within an additional2000s. It never retries failed fits,
changes a recipe or signals a foreign process. `status/controller-six-r1.json`
records actual requests, source identities, PID birth identities, phases and
failures. Incomplete graphs remain explicitly partial. Four focused continuation
metadata/arithmetic tests pass locally; actual launch is a separate root action.

The identified R2 controller repair adds `--attempt r2`, preserving the original
R1 status/claim/stop receipt and adopting its surviving owned export. It accepts
either the exact verified source-binding object or its exact aggregate SHA string
when the unchanged request and status use the same encoding. Request byte hashes,
exact command/role/GPU/source and PID birth checks remain binding. It does not
restart any fit or export. Six focused continuation checks pass, including a
no-spawn adoption test that preserves the original receipt bytes.

See [recipe generator](../../../scripts/heterogeneous_tl3_recipes.py),
[cache builder](../../../scripts/build_heterogeneous_pool_cache.py),
[chronological policy](../../../specs/adr/0044-chronological-primary-validation.md)
and [original pool proposal](../../unified/ideas/HETEROGENEOUS_TEACHERS_20261029_R1.md).
