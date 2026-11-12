# Train a Transformer for its fixed contribution to clean TL3

Status: **implemented handoff; receiver acceptance pending; zero new fits executed**.
Hardware availability is user-reported; this packet has not profiled the receiving B300s.
H-B300-TL3-TRANSFORMER-COMPLEMENT-20261106 / E-B300-TL3-TRANSFORMER-COMPLEMENT-20261106.
Published: [draft PR #67](https://github.com/VladKochetov007/WunderFunder/pull/67) targeting `research-team`; implementation commit
`7f660553a1df3066f2e735e8975b39c225ec9870` ([publication receipt](PUBLICATION.json)).
Owner: team kotiki / receiving experiment operator.
Base: `7e4074314a62d0907c9608f9220cf72958708a2d`.
See the [eleven-section PR](PR.md), [machine recipe](recipe.json) and
[6 November submission audit](SUBMISSION_AUDIT.md).

## Question and fixed endpoints

Does training a causal Transformer for **90% frozen clean TL3 + 10% Transformer**
beat training the same Transformer on labels and only then making that same blend?
Both arms are evaluated as `p = 0.9 b + 0.1 z`; standalone Transformer quality is
not the primary selection endpoint. This extends the frozen-parent correction
family in PR38 and fixed-mixture evidence in PR52/53. It does not claim a new
boosting algorithm or repeat their original fits.

The [portal audit](SUBMISSION_AUDIT.md) found team best TL3 at public WP **0.66832**,
rank 39, versus leader **0.67596**. The clean reconstructed development TL3 is a
different artifact/protocol: official-valid WP **0.6923774245652725**. These scales
must not be mixed. The latest all-labels W18 public result, **0.66453**, and tiny
exposed mixture gains motivate testing complementarity; they do not prove this
Transformer objective will improve a submission.

For original target `y`, let `c = clip(y, -2, 2)`, `w = abs(c)` and `m` be the
unchanged training mask. Each target uses the canonical globally reduced weighted
denominator across ranks and the full accumulation window.

- Control: `sum(m*w*(z-c)^2) / sum(m*w)`.
- Treatment: `sum(m*w*(0.9*b + 0.1*z-c)^2) / (0.1^2 * sum(m*w))`.

The treatment is algebraically a weighted residual-target regression against
`(c-0.9*b)/0.1`. We keep weights from the **original clipped labels**, never from
the residual target, and do not clip `b` or `z` before composition. The fixed
reference is detached. Dividing by alpha squared matches curvature with respect
to `z`; it does **not** match gradients, clipping incidence or AdamW/Muon
trajectories. Inspect those diagnostics before assigning a mechanistic explanation.
The official evaluator applies its normal clipping only to the composed endpoint.

`blend_objective_alpha` is absent in controls. Omitted/zero treatment preserves
the old config identity and objective arithmetic; existing KD stays separate.
Treatment requires a distinct `--reference-cache`, forbids KD/hints and is
Transformer-only. A raw treatment head cannot be exported as an ordinary endpoint:
explicit evaluation composition is required and binds its fitting reference.
Composed fit-cache/KD export and compact-student training are **not implemented**.

## Frozen four-fit panel

| Cell | Fitting seed | Objective | Primary evaluated endpoint |
|---|---:|---|---|
| `labels_s101001` | 101001 | Original-label WMSE on z | 0.9 TL3 + 0.1 z |
| `complement_s101001` | 101001 | Curvature-scaled composite WMSE | 0.9 TL3 + 0.1 z |
| `labels_s101002` | 101002 | Original-label WMSE on z | 0.9 TL3 + 0.1 z |
| `complement_s101002` | 101002 | Curvature-scaled composite WMSE | 0.9 TL3 + 0.1 z |

This is **two new fits per arm, four total**, 30 epochs each / 120 model-epochs.
All use the same exact PR49 source split, not F0/F2 cross-validation:
`expected_fold=0` is this protocol's identifier. Fit membership is 10,325 train
sequences and zero valid sequences; evaluation is all 1,873 official valid
sequences after component exclusions. Preserve manifest bytes with SHA256
`0c4de3a253d23b3a5d5ca73b7aeb7fcae2b974673e4f9ec51e94e593c59d114c`.

| Setting | Frozen value |
|---|---|
| Model | Causal raw112 Transformer; width 768, 24 layers, 12 heads, FF multiplier 4 |
| Temporal processing | Context 1024, chunk 256; detached chunk history; sequence resets |
| Attention | Efficient SDPA, compiled blocks; no activation checkpointing |
| Precision | `bf16_bf16x9`; BF16 autocast plus process-wide `CUBLAS_EMULATE_SINGLE_PRECISION=1` |
| Optimizer | Canonical Muon/AdamW; peak LR 0.0003, weight decay 0.01 |
| Muon | Momentum 0.95, 5 Newton–Schulz steps, epsilon 1e-7 |
| AdamW groups | Betas 0.9/0.95, epsilon 1e-8 |
| Schedule | `warmup_cosine_v2`; first 5% warmup; minimum LR ratio 0.1 |
| Work per rank | Batch 1, four accumulated chunks; eight ranks |
| Gradient clipping | Global norm 1.0; record preclip norms and clipping incidence |
| Preprocessing | `normalization_scope=all_fit`; raw112, fitting-only moments |
| Training mask | `all_after_warmup`; original absolute clipped-label weights |
| Other regularization | Dropout 0; KD/hint weights 0, hint dimension 0 |
| Checkpoint | Final epoch 30 only; resumable training state; no best-valid-epoch selection |

The precision follows the [user's BF16 amendment](../b300-transformer-distillation-runtime/PRECISION_AMENDMENT_20261024.md), superseding older strict-FP32 draft defaults.
These are frozen starting settings, not measured optima. Eight B300s are required
per fit; the panel comparator requires the **same physical eight GPUs and software
identity** for all four cells. Sequential fits reuse that allocation. It does not
accept silently mixing one/eight ranks or different hardware receipts.

Extra compute buys Transformer capacity trained for the final blend. It does not
buy a larger competition serving limit. GPU-hours, peak RAM/VRAM and wall time
remain unmeasured. The 17 historical checkpoint roles are reused, not 17 new fits;
six reference exports, checks, twelve short profiles and all evaluation/archive
work are additional engineering/inference costs, not statistical replications.
The current sampler yields 1,291 per-rank sequence batches and 20 accumulated
chunk windows per batch: **25,820 optimizer updates/epoch, 774,600/fit and
3,098,400 across the panel**; warmup is 38,730 updates/fit. These are static
budget counts, not measured throughput or wall-time estimates.

## Receiver prerequisites and immutable reference

All commands below are for the **receiving Linux B300 host**. They were not run
locally. Use a new clean checkout at the reviewed PR commit, persistent output
storage, and a provisioned environment from the committed `uv.lock`. Do not
alter running jobs or their output directories. The wrapper rejects tmpfs/ramfs
outputs and places TMPDIR, TorchInductor and Triton caches under the persistent
run root; size that disk for reference banks, compiled caches and checkpoints.
After source/config changes,
new checks/profiles are required; do not relabel old receipts.

Set actual absolute receiver paths and the full reviewed commit SHA:

```bash
cd /absolute/path/to/reviewed/WunderFunder
export WNN_EXPECTED_COMMIT='<reviewed-full-40-character-commit-SHA>'
export WNN_DATA_ROOT='/absolute/path/to/preserved-canonical-PR49-data-root'
export WNN_SPLIT='/absolute/path/to/original-PR49-primary-split.json'
export WNN_RUN_ROOT='/persistent/disk/tl3-complement-20261106-r1'
export WNN_REFERENCE_ROOT='/persistent/disk/tl3-complement-reference-r1'
export WNN_PR49_SOURCE='/absolute/path/to/clean-PR49-c42-source'
export WNN_ASSET_ROOT='/persistent/disk/published-tl3-assets'
export WNN_PR49_ARCHIVE="$WNN_ASSET_ROOT/tabdata-pr49-checkpoints-20261029-r1.zip"
export WNN_PR52_ARCHIVE="$WNN_ASSET_ROOT/tabdata-heterogeneous-checkpoints.zip"
export PACKET='research/experiment_drafts/tl3-transformer-complement-20261106'
export PYTHONPATH="$PWD/src"
uv sync --frozen
```

`WNN_DATA_ROOT` must retain canonical `datasets/train.parquet`,
`datasets/valid.parquet`, `cache/source_audit.json`,
`cache/chronological_chains.npz`, `cache/{train,valid}_index.npz` and their
`*.identity.json` sidecars. Preserve `cache/train_targets.npz` as well: the native
full-train feature-only export uses its `row_group` index. The original split
binds sizes, modification times,
footer/index hashes and component identities. Recreating indexes or casually
copying files without their metadata breaks this identity. **Do not rebuild or
edit the split to make a mismatched receiver pass.** Restore the preserved source
assets and metadata or record a new protocol needing separate approval.

Download the two exact published assets on the receiver, retaining originals:

```bash
mkdir -p -- "$WNN_ASSET_ROOT"
gh release download tabdata-pr49-checkpoints-20261029-r1 \
  --repo VladKochetov007/WunderFunder \
  --pattern tabdata-pr49-checkpoints-20261029-r1.zip --dir "$WNN_ASSET_ROOT"
gh release download tabdata-pr52-heterogeneous-checkpoints-20261030-r1 \
  --repo VladKochetov007/WunderFunder \
  --pattern tabdata-heterogeneous-checkpoints.zip --dir "$WNN_ASSET_ROOT"
```

Use a separate clean legacy checkout at the original inference commit:

```bash
git fetch --no-tags origin experiment/tl3-chronological-20261028
git worktree add --detach "$WNN_PR49_SOURCE" c42baf7b95946994216801b4eb9909422ea7effa
uv run --no-sync python "$PACKET/reference_stage.py" \
  --pr49-archive "$WNN_PR49_ARCHIVE" \
  --pr52-archive "$WNN_PR52_ARCHIVE" --data-root "$WNN_DATA_ROOT"
bash "$PACKET/run.sh" data-check
```

The worktree command is only needed if that separate checkout does not exist.
Obtain the two published archives from the [source receipts](reference_sources/README.md);
`reference_stage.py` restores only the 17 pinned `model_epN.pt`/original
`provenance.json` pairs into `artifacts/runs/<run>/`. It verifies each archived
entry's hash without model deserialization and refuses to replace different
existing bytes. `model.pt` is not a substitute for the pinned `model_epN.pt`.

The three reference members are fixed:

| Export key | Historical endpoint | Weight |
|---|---|---:|
| `w18` | `tl3c_w18final_distilled_s5/model_ep1.pt` | 0.34 |
| `tiedlead_s5` | `tl3c_fttall_tlead_s5/model_ep3.pt` | 0.33 |
| `tiedlead_s6` | `tl3c_fttall_tlead_s6/model_ep3.pt` | 0.33 |

Export each of the six banks with **one explicitly allocated GPU**. The wrapper
runs the actual pinned `scripts/predict_run.py` with FP32 inference, original
causal-feature behavior and full-row source-qualified membership. It unsets
BF16x9 for these legacy replays. Existing banks are create-only and never reused
merely because their filenames look right.

```bash
export CUDA_VISIBLE_DEVICES=0  # replace with one actually allocated GPU ID
bash "$PACKET/run.sh" reference-export w18 train
bash "$PACKET/run.sh" reference-export w18 valid
bash "$PACKET/run.sh" reference-export tiedlead_s5 train
bash "$PACKET/run.sh" reference-export tiedlead_s5 valid
bash "$PACKET/run.sh" reference-export tiedlead_s6 train
bash "$PACKET/run.sh" reference-export tiedlead_s6 valid
bash "$PACKET/run.sh" reference-import
bash "$PACKET/run.sh" baseline-score
```

The importer joins explicit row-group IDs and constructs the raw FP32 mixture
left to right, 0.34/0.33/0.33, without component preclipping. It emits separate
`fit/` and `evaluation/` caches in the `sequence-reference-cache-v1` schema,
with identical stable reference ancestry hashes and distinct array manifests.
It does not disguise the reference as a generic teacher cache. Every ancestor,
array and export sidecar is bound. Baseline replay must reproduce
**0.6923774245652725 within absolute 2e-6** before expensive fits. A failure means
investigate source/data/numerics; do not widen the tolerance after seeing results.

## Acceptance, profiles, fitting and evaluation

Bind the actual eight allocated B300 IDs. Keep that same ordered device set for
all production fits and comparisons; example IDs are not a reservation.

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
bash "$PACKET/run.sh" check
```

This records topology and hardware, runs receiver-only causality/reset/mask,
reference/split, objective/default, recovery and optimizer checks, two-rank
weighted-loss checks for both default and alpha=0.1 (including zero padding),
and a two-rank Muon check. Compiled finite-output/gradient parity checks for
`bf16` and `bf16_bf16x9` run in **separate processes**; skipped required GPU cases
cannot produce successful acceptance. These are implementation checks, not
quality evidence or full-run memory measurements.

For each of the four declared cell names, execute these stages explicitly:

```bash
CELL=labels_s101001  # then complement_s101001, labels_s101002, complement_s101002
bash "$PACKET/run.sh" profile-one "$CELL"
bash "$PACKET/run.sh" profile-eight "$CELL"
bash "$PACKET/run.sh" profile-peak "$CELL"
```

Each profile runs 12 optimizer updates on real shapes without writing a
scientific checkpoint. `profile-one` changes world size to one and uses the
first bound GPU; `profile-eight` uses the exact production config; `profile-peak`
uses eight ranks and zero warmup solely to expose peak-LR behavior. Check all
rank memory/headroom, compile and data costs, finite losses, clipped-gradient
frequency and treatment composite/scaled WMSE. A successful short profile does
not guarantee 30-epoch stability. If it is unsafe, preserve the failed evidence
and amend the packet before any new scientific attempt.

A receiving reviewer must inspect all three profiles, copy
`$WNN_RUN_ROOT/$CELL/profile-eight/profile_receipt.json` to
`$WNN_RUN_ROOT/$CELL/approved_profile.json`, set `approved` to `true` in the copy,
and add a reviewer/date/reason note. Preserve all original identity, measurement
and source fields and all original receipts. The wrapper does not auto-approve.
The one-rank and peak-LR diagnostic receipts cannot replace the production
receipt. This is the existing trainer's admission contract, not a request for
permission to publish this code.

```bash
bash "$PACKET/run.sh" fit "$CELL"
bash "$PACKET/run.sh" evaluate "$CELL"
bash "$PACKET/run.sh" bundle "$CELL"
```

For an interrupted nonterminal fit with an intact exact-source resume state:

```bash
bash "$PACKET/run.sh" resume "$CELL"
```

`fit/model.pt` is the fixed final epoch-30 endpoint; `fit/training_state.pt` is
resumable state. The wrapper passes the fitting reference only to treatment.
`evaluate` always explicitly composes both arms with the same evaluation
reference and alpha=0.1 before canonical scoring. Retained-best-epoch search,
blend-weight search and raw treatment-head scoring are forbidden here.
After all four complete:

```bash
bash "$PACKET/run.sh" compare
```

`comparison.json` rejects different source/config/data/normalization/runtime,
reference manifest, optimizer or completed budgets. It reports every seed's
label-blend versus TL3, complement versus TL3 and complement versus label-blend;
no best seed is selected. Resume/profile/export failures are retained, not
silently converted into negative scientific scores. Existing live training is
outside this packet and must not be changed.

## Interpretation and follow-up gate

Primary metric is canonical pooled `global_wp_clip2_v1` over official scored
rows after warmup. Do not average rank correlations. The frozen comparison uses
paired component bootstrap, 2,000 draws, seed 271828; intervals condition on the
four fitted endpoints and the fixed reference. Two fitting seeds on one exposed
source split do not establish training variance reliably, independent replication,
statistical significance or private/public transfer.

A **follow-up-budget trigger only** requires all three:

1. Mean complement-minus-TL3 WP across the two seeds is at least **+0.0005**.
2. Complement-minus-TL3 is nonnegative for **each** seed.
3. Complement-minus-label-blend is nonnegative for **each** matched seed.

Passing supports proposing a separately frozen confirmation/serving study. It
launches no further fits, authorizes no submission and guarantees no public gain.
A small positive or mixed result is inconclusive; inspect the fixed contrasts
and optimization diagnostics. Consistent negative results reject this recipe,
not all Transformers. An interval crossing zero alone does not close the family.
No result may be reported by adding a development delta to the public 0.66832.

The split preserves the supplied train/valid boundary and removes linked
components from fitting. Inputs remain causal; normalization uses only fitting
members; no evaluation targets enter the training objective. **However:**

- Fitting references are **in-sample**, not cross-fitted. The reference may hide
  hard training errors and make residual fitting optimistic. This design cannot
  resolve that effect without a separately costed OOF-reference study.
- The 17 original checkpoint/provenance pins and source inventories verify the
  published reconstructed lineage. Historical consumed training-cache bytes
  were not all retained and are **not independently reconstructed** here.
- Official valid has already driven team development and mixture inspection;
  it is exposed. Calling it chronological or rerunning seeds does not create
  an untouched holdout. Calendar timestamps are unavailable, limiting claims
  about future-market representativeness.

This packet prepares a large GPU research endpoint. It does **not** provide a
CPU-ready competition ZIP. The single-vCPU/16-GB/offline/one-hour/20-MB serving
contract still needs a compact student and actual delivered-artifact checks.
Distilling the composed teacher is a possible next experiment with its own
implementation, clean caches, fitting budget and CPU measurement; it is not
silently included in these four fits.

## Files and evidence to return

| Files | Responsibility |
|---|---|
| `README.md`, `PR.md`, `recipe.json` | Frozen scientific contract, eleven-section review and machine inventory |
| `SUBMISSION_AUDIT.md` | Dated portal, team-result and non-duplication evidence |
| `VALIDATION.json` | Static source-review/check receipt; explicitly records unexecuted receiver work |
| `configs/*.json` | Four canonical production trainer configurations |
| `run.sh` | Explicit receiver stages; no queue, allocation or auto-approval |
| `receiver_data.py` | Exact primary-split/source/component check |
| `reference_stage.py`, `reference_assets.py` | Restore pinned archive members; audit 17 ancestors and legacy source |
| `reference_export.py`, `reference_import.py` | Actual PR49 native replay and checked source-qualified cache conversion |
| `reference_score.py` | Canonical TL3 baseline replay and fixed tolerance gate |
| `reference_sources/` | Published metadata mirrors, source links and archival limitations |
| `profile_config.py`, `acceptance_receipt.py` | Explicit diagnostic variants and source-bound receiver acceptance |
| `compare_panel.py` | Four-cell identity validation and fixed paired decision report |
| `src/wnn/training/sequence_blend*.py` | Canonical objective, fail-closed composition and provenance validation |
| `src/wnn/training/sequence_reference*.py` | Native reference cache contract and published pins |
| `tests/test_sequence_blend_*.py`, `tests/test_sequence_reference_cache.py` | Receiver-only algebra, DDP-related invariants, GPU parity and lineage checks |

Each cell's `bundle/` separates evidence from `model.pt`/resume weights and
records file/archive hashes. Preserve rank0–7 logs, metrics, terminal status,
source/config/runtime/provenance, profile receipts, official score and prediction
manifest. Keep companion root `data-check.json`, storage receipts, `acceptance/`,
`baseline-score.json`, `comparison.json`, reference manifests/export sidecars,
array/asset hashes and access locations alongside all four bundles. The cell
bundle alone does not contain every shared reference array or root acceptance
file; archive the companion evidence explicitly. Datasets stay in their approved
storage, never Git or automatically attached archives.

Implementation review and static lint/AST/shell checks are preparation evidence.
No local tests, model imports, training, inference or GPU profiles were run.
All four results, measured resources and receiver archive locations remain pending
until actual execution. Publication is recorded above; publication is not GPU acceptance.
