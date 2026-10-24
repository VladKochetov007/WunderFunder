# Saved-artifact comparison feasibility

## Completed action

The feasibility comparison recommended grading the six saved F0
online-relationship heads against their matching F0 parent as a **new
exploratory F0-only amendment**. Root subsequently admitted and executed that
grade using the separately reviewed R2 recipe. Its endpoint scores and
contrasts are recorded in
[F0_EXPLORATORY_GRADE_R2.json](F0_EXPLORATORY_GRADE_R2.json); root owns its
scientific interpretation.
The original 12-fit recipe, admission and grader remain unchanged and
incomplete. This is the closest available path to a new official-metric
result: six fitted endpoints already existed, and their F0 parent prediction
and target archive were available locally. This fixed-cell grade does not
estimate a fold or training-seed effect, establish generalization, or complete
the old panel.

### Exact frozen question and resource cost

- **Endpoints:** F0/seed1001 parent plus six heads: A/B/C at lag 1 and lag 16.
- **Primary comparisons:** within each lag, C−B and B−A; retain the original
  frozen recipe's remaining A/B/C-to-parent, C−A, and lag16−lag1 contrasts as
  descriptive secondary outputs. Keep all comparisons fold-local.
- **Metric:** unchanged `global_wp_clip2_v1`, original scored-row mask and
  label-magnitude weights, clipping at ±2. Report paired dependency-component
  bootstrap intervals as conditional fixed-endpoint uncertainty only.
- **New fits:** zero. The seven-endpoint exploratory grade completed.
- **Exposure:** F0 has 468 evaluation groups and 141 dependency components.
  Every local fold is exposed; this is exploratory evidence, not fresh
  confirmation. Fit seed, parent ancestry and fitting subset are fixed to the
  original F0/1001 cell.
- **Artifact sizes verified with `stat` only:** six head predictions are each
  74,880,128 bytes; the parent F0 prediction is 74,880,128 bytes; and
  `cache/valid_targets.npz` is 337,186,198 bytes. Prediction payload values
  were not inspected. The six heads plus parent total 524,160,896 bytes before
  sidecars/checkpoints/provenance.
- **Cost:** zero training. The historical full two-fold grade reserved 600 s
  at two CPU cores and 4 GB; that is a conservative planning ceiling, not a
  measured F0-only cost. The separate adapter extracts `y.npy` and `mask.npy`
  in 1 MiB blocks to physical scratch and scores one group at a time through
  read-only memory maps. It caps the grade worker at one CPU core,
  400,000,000 bytes and zero swap; this leaves headroom for root's notebook
  within the 1 GiB aggregate RAM limit. The actual grade took 32.71 seconds,
  peaked at 399,998,976 bytes, and recorded no OOM or swap events in its
  [execution receipt](F0_GRADE_EXECUTION_R2.json).
  Kaggle execution is unnecessary and costs zero here: every required
  prediction, parent output and target archive is already on approved local
  storage, and no artifact has been staged remotely.

### Files, provenance, and gaps

- Frozen source packet:
  [online-relationships README](../../../experiment_drafts/online-relationships-20261022/README.md),
  [execution recipe](https://github.com/VladKochetov007/WunderFunder/blob/c3e208b387ead0ec967f9abc4b14275eb8ba94c5/research/experiment_drafts/online-relationships-20261022/EXECUTION_RECIPE_R1.json),
  [terminal inventory](https://github.com/VladKochetov007/WunderFunder/blob/c3e208b387ead0ec967f9abc4b14275eb8ba94c5/research/experiment_drafts/online-relationships-20261022/TERMINAL_COLLECTION_R1.json),
  [terminal receipt](https://github.com/VladKochetov007/WunderFunder/blob/c3e208b387ead0ec967f9abc4b14275eb8ba94c5/research/experiment_drafts/online-relationships-20261022/PANEL_TERMINAL_R1.json),
  [grade specification](https://github.com/VladKochetov007/WunderFunder/blob/c3e208b387ead0ec967f9abc4b14275eb8ba94c5/research/experiment_drafts/online-relationships-20261022/GRADE_SPEC_R1.json),
  and the existing `grade_online_relationships.py` plus
  `online_relationship_grade_*.py` modules.
- Existing local F0 predictions are under
  `/home/vlad/development/wunder/research_tmp/online-relationships-20261022/runs/`:
  `online_rel_f0_s1001_lag{1,16}_{a,b,c}_r1/prediction.npy`, with a colocated
  `prediction.json`, `provenance.json`, and `model_ep2.pt` for each completed
  head. Parent prediction metadata and bytes are under
  `/home/vlad/development/wunder/research_tmp/online-relationships-20261022/parents/f0/`;
  the prediction path is `evaluation_predictions.npy` and the metadata path is
  `evaluation_predictions.json`.
- The official target archive is `cache/valid_targets.npz`; the frozen recipe
  records SHA-256
  `dbb7a1a48dd0af9fa524a80c2694008a56cb91d5dae80e8d0d4984c0c5622955`.
- The terminal source audit found no runtime/grader pin mismatches (27 runtime
  and 15 grader pins). The separate preflight reads those maps from
  `TERMINAL_COLLECTION_R1.json`, checks the terminal summary's counts and
  mismatches, and hashes every pinned runtime/grader source. It independently
  checked all seven prediction NPY headers and payload hashes, exact
  source-qualified row refs, and the recursive parent/resume/initializer,
  distillation-teacher and propensity ancestry. All ten provenance nodes
  carry the same 468 evaluation groups and 1,405 valid fit groups, with zero
  valid-fit/evaluation overlap. The old F0 cell's parent run is
  `clean_t3_f0_control_s3_a9_t3r5_r2_p3_envr1`; its checkpoint, provenance,
  split-manifest and parent-cache hashes are in the old frozen recipe and
  `parents/f0/evaluation_predictions.json`.
- The recursive lineage check follows the three parent continuation runs to
  `c20261003_combo_s3_f0@ep3`, both initializer distillation teachers, the
  parent's three strong-teacher cache sources, and `c20260929_prop_f0`.
  It hashes each exact checkpoint and provenance file, verifies the split and
  fit/evaluation membership, and checks the 323 KB distillation-cache metadata
  hash and row-group join records. It deliberately does not scan the large
  feature/teacher prediction arrays: their hashes and source memberships are
  pinned in the verified metadata/provenance. The seven grade prediction
  payloads are hashed in full.
- The existing grader cannot consume a shortened recipe: its contract requires
  `admitted_new_fits == 12`, exactly the F0/F1 × lag1/lag16 × A/B/C grid, and
  its scoring module hardcodes 13 contrasts per fold. Keep those frozen files
  unchanged. The F0 amendment needs its own recipe, adapter and output receipt.
  The reviewed [R1 proposal](F0_EXPLORATORY_RECIPE_R1.json) is preserved. The
  R2 [recipe](F0_EXPLORATORY_RECIPE_R2.json) and
  [source manifest](SOURCE_MANIFEST_R2.json) add the required post-score target
  hash check without replacing the reviewed R1 files.
- Do not infer prediction parity from the sidecars: the original grader checks
  exported bytes and provenance but does not rerun all head checkpoints.

### Intended command

This is the bounded, label-free preflight command that passed. It does not
open target labels or arrays:

```bash
WNN_MEM_MAX=400000000 WNN_CPU_QUOTA=100% WNN_THREADS=1 \
WNN_RUNTIME_MAX=900 CUDA_VISIBLE_DEVICES='' PYTHONPATH=src \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
NUMBA_NUM_THREADS=1 POLARS_MAX_THREADS=1 ARROW_NUM_THREADS=1 \
scripts/limited .venv/bin/python \
research/campaign_2026-10-24/resume4h_r1/feasibility_scout/grade_online_f0.py \
--recipe research/campaign_2026-10-24/resume4h_r1/feasibility_scout/F0_EXPLORATORY_RECIPE_R2.json \
--recipe-sha256 0d6cf458a895c322f94dd25ec1d228c5b3bfc5320f0da10c55027f58f84fd059 \
--preflight-only \
--output /home/vlad/development/wunder/research_tmp/resume4h-r1/f0-online-preflight-r3.json
```

Observed preflight receipt SHA-256:
`f8fd01dab2226fdc46660e72a1cc25d1da3457429f404d0d0f036e18f01d7692`.
It reports 27 runtime and 15 grader source pins, seven exact `<f4`
`(468, 20000, 2)` prediction arrays, ten ancestry nodes and 27 hashed lineage
edges. The target is not read by preflight. The grade path checks its digest
before the first label read and again after scoring; both must match the
frozen target hash, and the final receipt records both digests.

Root admitted the R2 recipe after the independent review. The exact executed
command and resource receipt are preserved in
[F0_GRADE_EXECUTION_R2.json](F0_GRADE_EXECUTION_R2.json); the grade output is
[F0_EXPLORATORY_GRADE_R2.json](F0_EXPLORATORY_GRADE_R2.json). The frozen
recipe's pre-score and post-score target checks both passed. Result
interpretation is owned by root.

The grade path is create-only. It must not revise the old 12-panel result or
use label-informed choices to retune variants.

## Alternatives

### 1. Fixed causal response-speed transforms on cached i1 B outputs

The bounded proposal in [Astra 10-A](../../../unified/reviews/astra_ultra_audit_20261023_r1/10_unrestricted_red_team.md)
uses the existing B predictions in F0/1001 and F2/1002 and freezes five
transforms of `z = clip(p, -2, 2)`, `d_t = z_t - z_(t-1)` with `d_0 = 0`:
`z`, `z + 0.125d`, `z - 0.125d`, `1.125z`, and previous `z`. No fit cost.
The new script should score every transform on the official metric, compare
each to unmodified B, and report both fold-local deltas and component intervals.
Its folds are already exposed and fold/seed are confounded. The fixed coefficient
must not be tuned on these labels.

Inputs are present at `cache/pred_i1_f0_s1001_b_r1_valid.npy`,
`cache/pred_i1_f2_s1002_b_r1_valid.npy`, and `cache/valid_targets.npz`.
Their sizes are 74,880,128; 70,880,128; and 337,186,198 bytes respectively.
The completed [i1 grade](https://github.com/VladKochetov007/WunderFunder/blob/c3e208b387ead0ec967f9abc4b14275eb8ba94c5/research/campaign_2026-10-21/next_cycle_r1/i1/runtime/GRADE_R1.json)
reports 60.61 s for its full two-fold six-fit panel under the historical
environment; this is only a cost anchor. No response-speed evaluator or frozen
recipe exists yet, so this is less immediately runnable than the F0 online
grade. Expected cost is zero fits plus one two-fold grade, conservatively under
600 s only if streaming/memory bounds are verified first.

### 2. Repair the remote age R3 CPU profile

The R2 Kaggle profile failed at the missing transitive import `wnn.i1_history`
after 36.874 s, with 352,714,752-byte peak process-tree RSS, zero swap, zero
fits, no scoring, and the target cache unopened. The R3 local source overlay
adds 38 audited source paths to the provenance allowlist, changing only
`src/wnn/training/age_readout_provenance.py`; its 368,640-byte package was
prepared in 0.084 s. R3 remote validation and the profile were never run.
The previous profile allowance was 600 s, two CPU cores and 4 GB on Kaggle.

R3 could validate the age runner import closure and measured profile cost, but
it would not produce a quality grade. The exact remote R2 invocation is
preserved in `KAGGLE_PROFILE_TERMINAL_R2.json`; it is stale because it pins the
R2 source manifest. Repackage/upload the R3 overlay, freeze a matching new
source manifest and admission, then invoke `scripts/age_readout_screen.py
profile --fold 0 --device cpu` with those R3 pins. Kaggle CPU capacity must be
confirmed by root. This is useful only as a path to a later age fit/grade, not
as a substitute for one.

## Review state

The label-free preflight hashed the seven saved predictions and exact lineage
files; it did not open the target archive's label or mask members. Four
synthetic streaming and target-identity invariants passed under the 400 MB/one-core limit, and
Ruff format/check passed for the adapter. Root then admitted and completed the
new exploratory F0-only grade, with the execution facts recorded above. No
fit, profile, feature-cache scan, provider call, or remote write was run. The
historical 12-fit panel's incomplete state is preserved. Root owns scientific
interpretation and shared-index updates.
