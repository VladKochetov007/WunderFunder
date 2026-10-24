# B300 high-compute campaign — BF16/BF16x9 R4 continuation

[Stage B receiving-host handoff](stage_b/README.md) publishes the six full LR
configs and the preparation/execution commands for the next B300 allocation.
It requires completion and GPU release of the active four W2 fits before any
receiver checks or profiles. No automatic successor or job modification is installed.

This implements stages B–H of the [expanded plan](../HIGH_COMPUTE_PLAN_20261024.md)
in [PR #32](https://github.com/VladKochetov007/WunderFunder/pull/32). Each stage
materializes complete trainer configs and explicit commands. It does not launch
a queue. **No tests, training, inference or profiles have been run locally. B300
acceptance and all scientific results are pending.** No hardware has been booked.

**24 October correction:** [Precision amendment](../PRECISION_AMENDMENT_20261024.md).
New materializations explicitly use `bf16_bf16x9`. Old materializations and
receiver receipts require a new campaign root at this source revision; they are
not resumable under the amended configuration. The team’s active four W2 fits
remain pinned to PR40 source and are not changed by this package.

The four [R3 teachers](../large_teacher_r3_muon30/README.md) remain stage A, with
their historical configs unchanged. R4 adds up to **30 teacher and six student
fits**, conditional on the choices below: 40 fits including A, or 34 if the final
candidate equals the reference and six duplicate G fits are eliminated. These
are fit counts, not GPU-hour estimates. Every teacher and student trains for
30 epochs. Profiles, exports, failed attempts and recovery cost extra.

## The implemented comparisons

| Stage | Variant keys | New fits | Evaluation |
|---|---|---:|---|
| B | `lr_1e-4`, `lr_3e-4`, `lr_1e-3` | 6 teachers | Inner component holdout |
| C | `warmup_0.01`, `warmup_0.05`, `warmup_0.10` | 4; reuse two B controls | Same inner holdout |
| D | `context_1024`, `context_2048`, `context_4096` | 4; reuse two selected controls | Same inner holdout |
| E | `chunk512_acc8`, `chunk2048_acc2` | 2; reuse two controls | Same inner holdout |
| F | `teacher_1.54B`, `teacher_3.15B` | 2; reuse two controls | Same inner holdout |
| G | `reference`, `candidate`; seeds 301001/301002/301003 × F0/F2 | 12, or 6 unique fits if identical | Exposed outer F0/F2 |
| H | `labels`, `single_kd`, `mean3_kd` | 6 students; reuse G teachers | Exposed outer F0/F2 |

B–F use seeds 201001 on F0 and 201002 on F2. The inner manifests reserve entire
source-qualified components from each outer fitting pool; every remaining
fitting sequence is used. Normalization is recomputed only from that inner
fitting membership. Outer evaluation components never enter either inner pool.
This is full inner training, not a 10%/30% proxy. G/H return to complete outer
fitting membership. Inner and outer scores are not interchangeable controls.

All teachers use causal raw112 inputs, BF16 autocast with cuBLAS BF16x9,
efficient exact-window attention, compiled blocks and no activation recomputation.
Master weights, gradients, loss, optimizer and carried state remain FP32. Muon
uses auxiliary AdamW; its FP32 tensor operations may use cuBLAS emulation.
Every fit runs 30 epochs and uses final checkpoint index 29. Start with the R3 1792×40, 28-head,
FF×4 model, context1024, chunk512, accumulation8, batch2/rank and eight ranks.
The larger F teacher is 2560×40 with 40 heads (~3.15B parameters). Muon uses
momentum .95, five FP32 Newton–Schulz iterations, epsilon1e-7 and independent Q/K/V
updates in the RMS-matched LR convention. This is not IEEE-FP32 equivalence.
Auxiliary AdamW beta=(.9,.95), eps1e-8;
matrix decay .01, vector/bias decay0, clip1, dropout0. B starts at warmup5%; every
schedule ends at .1×peak. See [R3 parameters](../large_teacher_r3_muon30/PARAMETERS.md)
for exact scheduler indexing and optimizer groups. These are candidates to
measure, not empirically optimal settings.

E changes chunk shape, backward horizon and accumulation together. It keeps
4096 new rows per sequence per full update and the selected forward context;
do not attribute its result to backward horizon alone. F transfers the selected
LR to a larger model; it does not establish that size's optimal LR.

H freezes a raw112 GRU512×2 student with BF16/BF16x9 compute and FP32 carried
state, without Transformer compilation or attention flags: AdamW LR3e-4, beta=(.9,.95), eps1e-8,
decay.01, clip1, dropout0, 5% warmup/cosine floor.1, chunk512/accumulation8,
batch2×8 ranks, 30 epochs, all-fit normalization, all post-warmup label rows,
two outputs and no hints. Seeds are 401001/F0 and 401002/F2. Labels use KD weight0;
both KD arms use weight1. Single KD always uses G candidate seed301001; mean KD
uses an equal FP32 mean of all three candidate seeds of that fold, without
clipping each teacher before averaging. Teacher choice is frozen before reading
G scores. No student architecture or seed is chosen from H outcomes.

## Receiving-host setup and stage preparation

Use a clean checkout of this PR's reviewed commit on the allocated Linux B300
host. Run `uv sync --frozen` there. Do not change the locked Torch/precision to
make a profile pass. Inputs and run artifacts live outside the checkout on real
disk, not RAM-backed scratch. Paths below are placeholders for existing data and
canonical F0/F2 manifests; the preparation commands validate their ancestry.

```bash
export WNN_EXPECTED_COMMIT='<reviewed full commit containing this R4 handoff>'
export WNN_DATA_ROOT='/actual/data/root'
export WNN_SPLIT_F0='/actual/canonical/f0.json'
export WNN_SPLIT_F2='/actual/canonical/f2.json'
export WNN_CAMPAIGN_ROOT='/actual/nvme/b300-high-compute-r4'
export CUDA_VISIBLE_DEVICES='<exact eight allocated GPU IDs or UUIDs>'
export WNN_HOST_RAM_BUDGET_GIB='<host RAM allocated to this eight-rank fit>'
export WNN_REVIEWER='<receiving operator>'
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUBLAS_EMULATE_SINGLE_PRECISION=1
PACKET=research/experiment_drafts/b300-transformer-distillation-runtime/high_compute_r4
INPUTS='/actual/nvme/b300-high-compute-inputs'
mkdir -p "$INPUTS"

uv run --no-sync python scripts/prepare_sequence_inner_split.py \
  --parent-manifest "$WNN_SPLIT_F0" --data-root "$WNN_DATA_ROOT" \
  --seed 41001 --evaluation-fraction 0.2 --output "$INPUTS/inner-f0.json"
uv run --no-sync python scripts/prepare_sequence_inner_split.py \
  --parent-manifest "$WNN_SPLIT_F2" --data-root "$WNN_DATA_ROOT" \
  --seed 41002 --evaluation-fraction 0.2 --output "$INPUTS/inner-f2.json"

uv run --no-sync python "$PACKET/campaign.py" init \
  --campaign-root "$WNN_CAMPAIGN_ROOT" --data-root "$WNN_DATA_ROOT" \
  --outer-f0 "$WNN_SPLIT_F0" --outer-f2 "$WNN_SPLIT_F2" \
  --inner-f0 "$INPUTS/inner-f0.json" --inner-f2 "$INPUTS/inner-f2.json"
uv run --no-sync python "$PACKET/campaign.py" prepare \
  --campaign-root "$WNN_CAMPAIGN_ROOT" --stage B

# Once for this exact source: CPU contract tests, actual CUDA recovery,
# two-rank objective/Muon parity plus BF16 streaming/backend checks.
# Legacy checks disable emulation in separate subprocesses. Runs on B300 only.
bash "$PACKET/run.sh" check --campaign-root "$WNN_CAMPAIGN_ROOT"
```

`campaign.json` binds runtime, packet, original R3 configs, data root and both
split layers. Each stage records every new/reused cell and hashes its complete
production, one-rank and peak-LR configs. Changes to those inputs fail closed.
Nothing creates training weights during `init` or `prepare`.

## One cell at a time

```bash
STAGE=B
bash "$PACKET/run.sh" list --campaign-root "$WNN_CAMPAIGN_ROOT" --stage "$STAGE"
CELL='<id of a cell with reused=false from list>'
bash "$PACKET/run.sh" profile-one --campaign-root "$WNN_CAMPAIGN_ROOT" --stage "$STAGE" --cell "$CELL"
bash "$PACKET/run.sh" profile-eight --campaign-root "$WNN_CAMPAIGN_ROOT" --stage "$STAGE" --cell "$CELL"
bash "$PACKET/run.sh" profile-peak --campaign-root "$WNN_CAMPAIGN_ROOT" --stage "$STAGE" --cell "$CELL"
bash "$PACKET/run.sh" admit --campaign-root "$WNN_CAMPAIGN_ROOT" --stage "$STAGE" --cell "$CELL"
bash "$PACKET/run.sh" fit --campaign-root "$WNN_CAMPAIGN_ROOT" --stage "$STAGE" --cell "$CELL"
bash "$PACKET/run.sh" evaluate --campaign-root "$WNN_CAMPAIGN_ROOT" --stage "$STAGE" --cell "$CELL"
bash "$PACKET/run.sh" bundle --campaign-root "$WNN_CAMPAIGN_ROOT" --stage "$STAGE" --cell "$CELL"
```

Each profile performs twelve real optimizer updates. One-rank differs only by
world size; peak stress differs only by warmup0. Admission reuses R3 checks for
exact source/data/runtime, finite gradients, near-peak LR, 20% GPU headroom,
conservative host resume memory and disk capacity. It does not prove full-run
stability. Repeat this sequence for each **new** allocated cell, including H. These W8 batch2 settings are unprofiled starting points; PR40’s
W2 batches96/64 and measured memory headroom do not transfer to these shapes.
Reused controls retain their original paths and cannot be relaunched as new fits.
Disjoint eight-GPU groups can run independent cells concurrently with per-fit
host budgets; do not change the rank count or global batch to fill idle devices.

For a genuine interrupted fit use `resume` with the same arguments. It restores
the last completed epoch and replays a partial epoch under the existing strict
source/config/device contract. Do not delete failure evidence, overwrite a
profile/export directory or splice checkpoints from different cells. Failed
receiver checks can use a new `--check-id attempt2` before an acceptance pointer
has been published. Failed profiles/exports need a separately retained attempt
root and explicit reallocation; no automatic retry is hidden here.

## Finish and freeze each stage

After every B–F arm has a fixed-final score or a recorded failure, explicitly
select a completed variant. Both of its fold cells must succeed. Selection
records the reason and all comparison evidence; later stages revalidate it.
Instability is retained as a failure, not replaced with an earlier good epoch.

```bash
# Example syntax only: choose from the observed B results, not this example.
uv run --no-sync python "$PACKET/campaign.py" select \
  --campaign-root "$WNN_CAMPAIGN_ROOT" --stage B --variant lr_3e-4 \
  --reason '<decision from all B scores, variation and compute cost>'
uv run --no-sync python "$PACKET/campaign.py" prepare \
  --campaign-root "$WNN_CAMPAIGN_ROOT" --stage C

# A failed arm needs a written reason and actual preserved evidence.
uv run --no-sync python "$PACKET/campaign.py" record-failure \
  --campaign-root "$WNN_CAMPAIGN_ROOT" --stage B --cell '<failed cell id>' \
  --phase fit \
  --reason '<numerical failure or infrastructure diagnosis; do not conflate them>' \
  --evidence '<absolute path to preserved terminal status or command log>'
```

Continue `select C → prepare D`, then D→E, E→F, F→G. Inspect each stage before
allocating its new fits. G freezes both recipes before any new outer scores and
does not require another choice. `compare --stage G` reports paired results
without selecting a seed or a checkpoint. H uses the already frozen G candidate,
even if the reference scores better; deciding not to spend on H is an operator
decision, not an automatic rewrite of teacher ancestry.

```bash
uv run --no-sync python "$PACKET/campaign.py" compare \
  --campaign-root "$WNN_CAMPAIGN_ROOT" --stage G
uv run --no-sync python "$PACKET/campaign.py" prepare \
  --campaign-root "$WNN_CAMPAIGN_ROOT" --stage H
```

## Export the frozen teachers, then distill

For each of the six G candidate endpoints (three seeds per fold), use its
original **new** cell ID from `list --stage G`. If candidate/reference recipes
were identical, use the new reference cell behind the candidate alias.

```bash
CELL='<original new G candidate cell id>'
bash "$PACKET/run.sh" cache --campaign-root "$WNN_CAMPAIGN_ROOT" --stage G --cell "$CELL"
bash "$PACKET/run.sh" cache-parity --campaign-root "$WNN_CAMPAIGN_ROOT" --stage G --cell "$CELL"

# Once per fold, after all three G parents pass export parity:
CELL='<H mean3_kd cell id for this fold>'
bash "$PACKET/run.sh" merge-cache --campaign-root "$WNN_CAMPAIGN_ROOT" --stage H --cell "$CELL"
```

`cache-parity` exports the complete fitting membership on one GPU, checks it
against the eight-GPU export with rtol1e-5/atol1e-6 and publishes a hash-bound
receipt. All arrays must be finite FP32, 20,000×2, with identical membership and
producer. Keep both exports and receipts while this campaign depends on them.
The merge streams one sequence at a time, requires three distinct seeds and
checkpoints of the same recipe/split/source, preserves full parent provenance,
and publishes only a completed directory. It does not invent a checkpoint hash
for an arithmetic ensemble. Before H profiles/fits, dispatch verifies exactly
the selected parents, their completed endpoints, parity receipts and merged
composition. Cache manifests and payload hashes remain checked by the trainer.

Run the profile/admit/fit/evaluate/bundle commands above for all six H cells.
Finally use `campaign.py compare --campaign-root "$WNN_CAMPAIGN_ROOT" --stage H`.
The result compares each KD student against its matched label-only student,
not teacher quality as a proxy for student improvement.

## Evidence and limits

Return all stage manifests, selections/failures, canonical score files,
predictions, paired component intervals, profile/check/admission receipts and
logs. Preserve every attempted arm. `bundle` separates fit evidence and weights;
retain sibling profile/check/cache directories separately in private storage.
Training source is identical across staged cells; a source fix requires new
checks/profiles and a newly declared campaign, not silent reuse of old controls.

F0/F2 are exposed. Inner search is development; G's six paired fold/seed cells
and H's two paired cells are robustness evidence. Neither correlated repeats nor
a component bootstrap establishes independent statistical significance or a
public score. A mean WP gain of .001 with no fold regression is a useful
predeclared signal, considered with uncertainty and cost; an interval crossing
zero does not close a model family. Sequential tuning can miss LR–warmup and
other interactions; those require separately allocated follow-ups.

Large teachers are research assets. The raw112 GRU student still needs measured
Python3.11/offline/one-core CPU inference, 16GB RAM, 60-minute runtime and 20MB ZIP
acceptance before submission. Native211/TL3 is a separate adapter. Quantization
and submission packaging are separate from these mixed-precision experiments.

G reference controls inherit R3 architecture/optimization settings with the same
BF16/backend/compiler amendment as candidate cells; R3 FP32 endpoints are not
matched controls. All new source/config identities require fresh admission.

The [recipe](recipe.json), [scientific contract](PR.md) and
[static validation record](VALIDATION.json) define the current implementation
and the checks that remain unexecuted.

Implementation map: `campaign.py` exposes preparation and selection;
`campaign_io.py`, `source_bindings.py` and `split_inputs.py` bind immutable inputs;
`catalog.py` and `templates/student_gru512.json` define complete trainer recipes;
`cell_materialize.py`, `stage_inner.py`, `stage_outer.py`, `stage_support.py` and
`materialize.py` write and revalidate stages. `endpoints.py`, `failure_receipts.py`,
`results.py`, `comparisons.py` and `paired_metrics.py` retain fixed outcomes and
reasoned choices. `run.sh`/`dispatch.py` execute explicit receiver actions;
`runner_checks.py`, `admit.py` and `cache_parity.py` enforce their prerequisites.
`parent_cache_config.py` binds single and ensemble KD inputs to the declared G
producer configs. The shared inner-split and cache-merge entrypoints live in `scripts/` and reuse
`wnn.splits` and the existing sequence trainer's teacher-cache interface.
