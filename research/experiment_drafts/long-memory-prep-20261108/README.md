# Long-memory preparation and TL3 role scoring

Published as [draft PR73](https://github.com/VladKochetov007/WunderFunder/pull/73)
by SemyonEpanov. Initial evidence/preparation commit: `9cf7ab550`.

[Same-row residual diagnostics](RESIDUAL_DIAGNOSTICS_R1.md) are a separate
read-only successor. They are completed alongside the live B300 fits, rather
than claimed as completed before fit admission; no live training recipe changes.

Status: R4 score-only replay is complete; long-memory work remains preparation
only. No model was loaded for inference and no fit or profile was run during
this packet update. See [the R4 results](RESULTS_R4.md) and
[the 12-epoch architecture re-audit](SCREEN_REVIEW_R1.md). The historical
validation is exposed development evidence. ROOT owns any future fit admission
and native execution.

## TL3 role artifacts found

The native project root identified by ROOT is:

`/data/epanov_s/tabdata/runtime/architecture-execution-20261107-r1/project`

It contains all three standalone validation prediction arrays, their JSON
sidecars, and their fixed role checkpoints with `provenance.json`. The exact
native availability receipt is
`C:/Users/admin/Documents/Codex/artifacts/transformer-optimizer-horizon-20261108/b300-rpc/results/004-tl3-individual-availability.json`.
Each array has 299,680,128 bytes, 1,873 ordered validation sequences, FP32
predictions, and `feature_only=true`. The transferred mixture cache remains at
`C:/Users/admin/Documents/Codex/artifacts/generalization-fix-20261108/reference-transfer-r1/reference-evaluation.tar`;
it is not used as a substitute for the individual arrays.

| Role run | Epoch | Available checkpoint | Checkpoint SHA256 | Prediction array | Prediction SHA256 |
| --- | ---: | --- | --- | --- | --- |
| `tl3c_w18final_distilled_s5` | 1 | `project/artifacts/runs/tl3c_w18final_distilled_s5/model_ep1.pt` | `a1934af24bd1520b78be874c294532b1e2d62acdd8bd1dc3d6b5c220b2dad107` | `project/cache/pred_tl3ref_tl3c_w18final_distilled_s5_valid.npy` | `e9a664be18feec38563be730c8a1f64dbc2a9907ed0d4d4873309d674f78eeb6` |
| `tl3c_fttall_tlead_s5` | 3 | `project/artifacts/runs/tl3c_fttall_tlead_s5/model_ep3.pt` | `cf944dffac66d674894e4c589d839c407b3d28cb01d35a26f58ff08c7743f65f` | `project/cache/pred_tl3ref_tl3c_fttall_tlead_s5_valid.npy` | `3c124a7e89110472b6b8fb878e09087fddf00a2b7f4983b8115c8acbd53c7352` |
| `tl3c_fttall_tlead_s6` | 3 | `project/artifacts/runs/tl3c_fttall_tlead_s6/model_ep3.pt` | `24a565b03e35dc36679bf809f63f605f24f693404bd0cfda58bfc82177a33598` | `project/cache/pred_tl3ref_tl3c_fttall_tlead_s6_valid.npy` | `67d3c7804d571b1accf096d14b10b7c768d27cb87849654a36d04ed72b0588e6` |

Paths are explicit in the scorer: arrays and sidecars are
`project/cache/pred_tl3ref_<run>_valid.npy` and `.json`; checkpoints are
`project/artifacts/runs/<run>/model_ep1.pt` or `model_ep3.pt`, accompanied by
`provenance.json`. No alternate cache prefix is inferred. The local
`scripts/score_tl3_chronological.py` expects a different `pred_<run>_valid.npy`
layout, so the new scorer adapts the same score accumulation to the exact native
paths above without changing any cache.

## Bounded score-only recipe

`grade_tl3_roles_r4.py` reads the pinned arrays and checkpoint hashes, verifies
the sidecar and run-provenance hash/membership/fixed-epoch fields, loads the
exact chronological split, then streams `valid.parquet` with Arrow threading disabled.
It computes three standalone role scores and the fixed FP32 left-to-right
0.34/0.33/0.33 blend. It does not deserialize a checkpoint or run a model. It
requires a new output path and never writes to prediction arrays, sidecars,
checkpoints, provenance or datasets.

The first native attempt used the B-source `CampaignSplit` API incorrectly by
expecting `evaluation_source` and `evaluation_mask` attributes. It failed before
reading prediction arrays or validation rows; ROOT confirmed no data or model
updates occurred. The frozen R1 script and remote staged R1 file are preserved.
R2 uses the actual compact split interface (`schema`, `sha256`, row-group
memberships and source-qualified `components`), confirms that valid components
cover the full evaluation membership, and explicitly reads `is_scored` from
`valid.parquet`. The B `CampaignSplit` type does not expose an evaluation-mask
attribute; the R2 scorer does not add an alias.

ROOT staged and launched the R2 bytes before a local cosmetic receipt-field
wording adjustment. In that frozen R2 output, `inputs_unchanged_during_score`
means the role sidecars, provenance files, prediction arrays and checkpoints
were rehashed after scoring. The split/source identities are validated at the
start by the loader, but the parquet and split manifest are not rehashed at the
end. Do not interpret that field as a post-score recheck of those two inputs;
the native R2 run is preserved without a cosmetic rerun.

The R2 API smoke test constructs the native B-source `CampaignSplit` type and
validates it directly. It passed using
`C:/Users/admin/Documents/Codex/reviews/WunderFunder-training-runtime-fixes-20261108/src`
as the first `PYTHONPATH` entry: `1 passed`. Ruff 0.17.3 also passes on the
R2 script's lint check. Its formatter check panicked with an internal source-map
range error before reporting formatting. These checks use CPU only and do not
read native data or load models.

ROOT runs it through the existing native CPU stream with one thread and its
already identified split path:

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
PYTHONPATH=<native-source-root>/src python <staged-packet>/grade_tl3_roles_r2.py \
  --project-root /data/epanov_s/tabdata/runtime/architecture-execution-20261107-r1/project \
  --split-manifest <frozen-chronological-primary-manifest> \
  --output <new-output-path>/tl3-roles-r2.json
```

The script fails closed unless the manifest SHA is
`0c4de3a253d23b3a5d5ca73b7aeb7fcae2b974673e4f9ec51e94e593c59d114c`, membership
is 10,325 official train sequences and all 1,873 official valid sequences, and
the full `is_scored AND row_index>=99` mask selects 3,528,273 rows. The output
labels official valid as exposed development evidence and makes no independent
or public score claim.

The run-level raw `provenance.json` hash is checked against each prediction
sidecar; the run JSON's canonical hash is reported separately. The transferred
mixture manifest records raw provenance-file hash pins: all three equal the
corresponding run-file byte hash and differ from the canonical JSON hash. The mixture
manifest records `training_predictions: in-sample` and
`published-legacy-consumed-cache-byte-ancestry-not-reconstructed`; the score
pass cannot reconstruct those consumed training-cache bytes. This historical
ancestry limit remains attached to the reference.

## Long-memory screen status after historical-run audit

The two `gru_local` runs were already completed for 12 epochs on seeds 101001
and 101002. Therefore GRU-local is an existing, tested architecture, not a new
representation. The 12-epoch runs peak near E5–E6 and then lose official-valid
WP; a new 6-epoch schedule is a conditional schedule retest only. Its E6
checkpoint cannot be equated with the endpoint of a fresh 6-epoch cosine
schedule. The paired GRU-local-vs-GRU delta at old E6 is mixed across seeds and
averages −.000399. See [SCREEN_REVIEW_R1.md](SCREEN_REVIEW_R1.md) for every
epoch curve, exact old/new config differences, and interpretation.

The four JSONs in `configs/` are retained only as conditional short-schedule
retest recipes for GRU vs. the existing GRU-local package on two provisional
seed pairs. They are not admitted, are not an architecture discovery, and do
not authorize fits. Their six-epoch B32/chunk1000 policy differs from the old
12-epoch B16/chunk2000 runs and the local config also reduces
`ff_multiplier` from 4 to 1.

The more distinct 4-fit idea is a new Mamba2-memory + causal-local-attention +
direct-head hybrid versus a freshly trained Mamba2 control on two paired seeds.
Standalone Mamba2 has been tested; this combination has not. It remains a
proposal only: no adapter/config, causal implementation tests, static parameter
count, or L4 profile exists for the combined architecture. ROOT must approve
any follow-up and own all GPU admission. No fits, profiles, serving claims, or
public-gain claims are made here.

## R2 native failure and corrected R3 grader

The staged R2 scorer failed before scoring because `_validate_split` incorrectly
required fitting groups to equal `range(10_325)`. Native receipt
`017-inspect-actual-split-provenance.json` confirms that the pinned manifest has
10,325 fitting train groups while the official train source has 10,607 groups;
the qualified fitting membership is noncontiguous and ends with groups 10,599
through 10,606. The manifest schema and SHA match the frozen primary split. No
prediction scoring occurred in R2.

`grade_tl3_roles_r3.py` is a new source file; R1 and R2 remain unchanged. R3
removes only the invented contiguous-range requirement and explicitly rejects
duplicate train members. It retains the exact split SHA/schema, train/evaluation
counts, empty valid-fitting membership, full valid-source component coverage,
`assert_fit_groups`, and exact run-provenance equality with the manifest's train
membership. Its focused tests accept a noncontiguous 10,325-member split and
reject duplicate train membership, missing valid components, and any run
provenance membership that differs from the split. Three CPU tests passed;
Ruff and `py_compile` passed. The synthetic fixture validates the actual
`CampaignSplit` API and cardinality, not the unavailable complete native member
list.

ROOT should stage R3 as a new script and preserve the failed R1/R2 receipts by
using a new output path:

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
PYTHONPATH=<native-source-root>/src python <staged-packet>/grade_tl3_roles_r3.py \
  --project-root /data/epanov_s/tabdata/runtime/architecture-execution-20261107-r1/project \
  --split-manifest <frozen-chronological-primary-manifest> \
  --output <new-output-path>/tl3-roles-r3.json
```

No R3 native scoring result is claimed until ROOT reports its receipt. As in
R2, `inputs_unchanged_during_score` rechecks pinned role sidecars, provenance,
prediction arrays and checkpoints after scoring; parquet and split manifest are
validated at load time but not rehashed at the end.

## R3 native provenance-order failure and R4 correction

ROOT's native R3 attempt failed before scoring at the combined provenance guard
for the first role; see `023-first-full-fit-progress-and-tl3.json`. The
condition-by-condition native inspection in
`025-diagnose-tl3-provenance-conditions.json` found that all three roles have
the exact train member set (10,325 entries, no duplicates, no missing or extra
groups), but the run provenance records that set in seed-shuffled order. The
evaluation member set also matches exactly, with a different list order. The
split SHA, empty valid-fitting membership, role name, fixed epoch, evaluation
flag, train source and args split SHA all match. This is an ordering mismatch,
not evidence of a fit-membership or ancestry violation; R3 produced no scores.

`grade_tl3_roles_r4.py` is a separate corrected scorer. Its provenance adapter
requires each train, fit-valid and evaluation list to be an integer list of the
expected length, contain no duplicates, and have exactly the manifest's member
set. It accepts order permutations only and records separate observed-order
and manifest-order SHA256 values for each list. Prediction alignment remains
strict: each cache sidecar's `row_groups` must equal the official validation
source order exactly, and scoring retains that per-group mapping. Checkpoint,
prediction, sidecar, split, role-argument, official-mask and scoring checks
remain in place; R4 does not relax teacher or cache ancestry qualification.

The R4 focused tests passed 10 cases, including shuffled provenance order,
official prediction-row order, and rejection of missing, extra or duplicate
train/evaluation members. Ruff lint and `py_compile` passed. ROOT then ran the
frozen R4 scorer as a score-only replay on the native CPU stream; it succeeded
on all three pinned arrays and the fixed blend. The four scores and input hashes
are in [RESULTS_R4.md](RESULTS_R4.md), sourced from native receipts
`027-live-training-and-tl3-r4.json` and
`028-fetch-tl3-result-and-first-evaluations.json`. The fixed blend exactly
reproduced the recorded baseline WP; this is not a new gain. R1, R2 and R3
remain grader implementation failures before scoring, not negative model
results.

The frozen R4 score-only command was:

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
PYTHONPATH=<native-source-root>/src python <staged-packet>/grade_tl3_roles_r4.py \
  --project-root /data/epanov_s/tabdata/runtime/architecture-execution-20261107-r1/project \
  --split-manifest <frozen-chronological-primary-manifest> \
  --output <new-output-path>/tl3-roles-r4.json
```

The manifest membership and prediction-row alignment remain distinct checks:
provenance member lists may be seed-shuffled but must have the exact unique
manifest set, while cached predictions must stay in official validation source
order. The scorer neither rewrites nor reorders any cache. The available role
checkpoint and prediction-array paths plus their SHA256 values are recorded in
the [R4 result](RESULTS_R4.md).
