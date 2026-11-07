# PR67 receiving amendment: four parallel two-B300 fits

Actual ROOT execution is now recorded in [EXECUTION_R1.md](EXECUTION_R1.md).
All four fresh fits started18:36UTC after native acceptance, twelve profiles,
independent review and clean-TL3 baseline replay. Measured full-schedule cost
requires the explicit finite9-day receiving amendment; see its unchanged exact
receipt in receiving-evidence. No final improvement is claimed. Preparation-only
statements below describe the earlier builder stage and are historical.

The later [bounded receiving controller](receiving/README.md) stages outside
frozen source-r3, reuses ROOT's known ownership helpers, enforces the mandatory
short run alias, and advances four independently approved cells through finite
profiles/fits/evaluation/bundles. Its local CPU/static checks do not launch work;
ROOT must retain the native lifecycle proof before GPU use.

Status: prepared receiving source; no models, profiles or native inference were
executed by this preparation agent. The root operator owns all SSH/GPU work.
The November 7 user explicitly authorized configuration and launch on the
existing eight B300s and eight L4s. This packet owns exactly four fresh 30-epoch
B300 fits, two arms times seeds 101001/101002. It adds no student fit or submission.

The [frozen PR67 packet](../tl3-transformer-complement-20261106/README.md),
recipe and original eight-rank configs at
`f0f48926c0dd19457ab4a58bf2ebd7d751f3b085` remain historical and unchanged.
New production configs are [configs/](configs/). This is a separate runtime
identity, with fresh acceptance and profiles; old receipts cannot admit it.

## Scientific identity and allocation

The primary protocol is the original official train/valid boundary, preserving
complete sequences and component exclusions. Fit 10,325 train sequences and zero
valid sequences; report all 1,873 official valid sequences and 3,528,273 scored
rows after warmup. The exact split SHA256 is
`0c4de3a253d23b3a5d5ca73b7aeb7fcae2b974673e4f9ec51e94e593c59d114c`.
Do not regenerate it, infer chronology from `seq_ix`, or fit preprocessing on
validation. Calendar timestamps are unavailable and this validation is exposed.

Each final endpoint is `0.9 * frozen clean TL3 + 0.1 * Transformer`, composed in
FP32 without clipping either component first. The labels control fits original
label-weighted WMSE; complement fits the unchanged curvature-scaled composite
WMSE. All objective, model, normalization, optimizer and precision settings remain
as PR67 except the explicit allocation and monitoring additions below.

| Setting | New receiving identity |
|---|---|
| Model | T768/L24, 12 heads, context 1024, chunk 256, raw112 |
| Numerical arm | BF16 autocast/BF16x9; FP32 master/state/loss/grad/optimizer |
| Optimizer | Canonical Muon/AdamW; unchanged schedule, LR and clipping |
| Per fit | Two B300 ranks, batch 4/rank, accumulate 4 chunks |
| Effective chunk mass | `2 * 4 * 4 = 32`, matching frozen eight-rank source |
| Static budget | 25,820 updates/epoch, 774,600/fit, 3,098,400/panel |
| Checkpoint | Fixed final epoch 30; resumable safe-epoch state |
| Monitoring | 32 fixed fitting sequences; full official valid every epoch |

The root operator binds disjoint pairs for four parallel jobs, recording actual
UUIDs and topology. Corresponding software, capability, memory, model, batch,
update counts and data must agree. Different physical UUID pairs introduce an
explicit exploratory hardware confound. The amended comparator records UUIDs
but permits different devices with otherwise identical runtime specifications;
the original comparator default still requires the exact physical allocation.
This is not bitwise or optimization-trajectory equivalence to eight ranks.

The native 12-update profile, first full epoch and 30-epoch wall time are pending.
No duration is claimed. Return measured compile/load/update/evaluation/checkpoint
times, all-rank memory/headroom, and then a clearly labelled extrapolation for
the remaining fixed 30-epoch budget. A 12-update profile is not a completed fit.

## Dependency graph and commands

Use the provisioned receiver Python, recording actual Python/Torch/CUDA/package
versions. Do not replace it or run a package upgrade as an implicit preparation
step. The observed environment supplied to this agent is Python 3.12.14 and
Torch 2.14/cu130; this preparation has not independently measured it.

1. Preserve canonical data/cache identity, including `train_targets.npz` and its
   original row-group index; run the source/split check.
2. Restore the 17 pinned model/provenance pairs from exact PR49/52 archives.
3. Verify the exact c42 PR49 inference source capsule and export six member/source
   banks with one explicitly bound GPU per process, legacy FP32/BF16x9 unset.
4. Import immutable fit/evaluation references, then reproduce baseline WP
   **0.6923774245652725 within absolute 2e-6**. A failure closes fitting admission.
5. Complete fresh implementation/DDP/Muon/compiled-precision checks.
6. Run all three profiles for each cell on its actual pair. An independent
   receiving reviewer approves a copy of its production two-rank receipt.
7. Launch the four matched fixed-budget fits, then evaluate, bundle and compare.

The [receiver wrapper](run.sh) reuses the PR67 scripts and canonical trainer. It
requires no Git or uv executable on the node. Both source capsules are bound by
sender-produced manifests, each with an independently supplied SHA256:
`{"schema":"wnn.git-source-capsule.v1","commit":"<full SHA>","files":{"path":"sha256"}}`.
Every inventoried file is hashed before a stage; extra Python under src/scripts
is rejected. This is an explicit source transfer, not a simulated Git checkout.

```bash
cd /absolute/path/to/new-reviewed-receiving-source
export WNN_PYTHON=/absolute/path/to/provisioned/venv/bin/python
export WNN_EXPECTED_COMMIT='<new reviewed receiving commit>'
export WNN_SOURCE_MANIFEST=/absolute/path/to/receiving-source-manifest.json
export WNN_SOURCE_MANIFEST_SHA256='<trusted sender-side SHA256>'
export WNN_DATA_ROOT=/data/epanov_s/tabdata/runtime/multistage-optuna-20261031-r1/data
export WNN_SPLIT="$WNN_DATA_ROOT/chronological_primary.json"
export WNN_RUN_ROOT=/data/epanov_s/tabdata/runtime/pr67-execution-20261107-r1
export WNN_REFERENCE_ROOT=/data/epanov_s/tabdata/runtime/pr67-reference-20261107-r1
export WNN_PR49_SOURCE=/absolute/path/to/exact-c42-source-capsule
export WNN_PR49_SOURCE_MANIFEST=/absolute/path/to/c42-source-manifest.json
export WNN_PR49_SOURCE_MANIFEST_SHA256='<trusted sender-side SHA256>'
export PACKET=research/experiment_drafts/tl3-transformer-complement-execution-20261107
export OLD=research/experiment_drafts/tl3-transformer-complement-20261106
export PYTHONPATH="$PWD/src"

"$WNN_PYTHON" "$OLD/reference_stage.py" \
  --pr49-archive /absolute/path/to/pr49models.zip \
  --pr52-archive /absolute/path/to/pr52models.zip --data-root "$WNN_DATA_ROOT"
bash "$PACKET/run.sh" data-check
```

The legacy capsule commit is exactly
`c42baf7b95946994216801b4eb9909422ea7effa`; include original `src/`, `scripts/`,
`utils.py`, `pyproject.toml` and `uv.lock` when sending it. Export each combination
of `w18`, `tiedlead_s5`, `tiedlead_s6` and `train`, `valid` explicitly. The following
is one of six exports, not a queue or automatic allocation:

```bash
CUDA_VISIBLE_DEVICES=0 bash "$PACKET/run.sh" reference-export w18 train
bash "$PACKET/run.sh" reference-import
bash "$PACKET/run.sh" baseline-score
CUDA_VISIBLE_DEVICES=0,1 bash "$PACKET/run.sh" check
```

Before the last three commands, all six exports must have completed successfully.
For each declared cell, bind its actual pair consistently for profiles and fit:

```bash
CELL=labels_s101001
export CUDA_VISIBLE_DEVICES=0,1
bash "$PACKET/run.sh" profile-one "$CELL"
bash "$PACKET/run.sh" profile-two "$CELL"
bash "$PACKET/run.sh" profile-peak "$CELL"
# Independent reviewer copies profile-two/profile_receipt.json to
# approved_profile.json, preserves all identity/measurements and records
# approved=true plus reviewer/date/reason. No auto-approval.
bash "$PACKET/run.sh" fit "$CELL"
bash "$PACKET/run.sh" evaluate "$CELL"
bash "$PACKET/run.sh" bundle "$CELL"
```

The other cells are `complement_s101001`, `labels_s101002` and
`complement_s101002`. `resume` uses only the same exact-source safe-epoch state.
After all four fixed-final endpoints complete, run `bash "$PACKET/run.sh" compare`.
Retain baseline/data/acceptance/source manifests alongside four separate cell
bundles. The wrapper creates no allocation, queue or receiving approval.

## Implementation and review

- `sequence_source_capsule` and explicit legacy export manifest arguments add a
  fail-closed Git-free receiver path; the original Git path remains available.
- `sequence_complement_monitor` reuses the canonical November monitor scorer,
  event journal and identity helpers. Evaluation uses the unwrapped master model
  only on rank zero, restores mode/RNG/BatchNorm state, and never changes fitting
  membership, normalizers, loss, schedule, stopping or endpoint selection.
- Monitor memberships/reference manifest hashes enter run provenance on every
  rank. Both arms use the same reporting references; only complement receives a
  fitting reference. Safe-epoch state binds journal offsets and preserves orphan
  report tails during resume. Final primary selection remains epoch 30.
- Final composed export now returns an explicit success sentinel from its
  reference-integrity callback. Previously that callback returned `None`, which
  the canonical collective helper rejects even after a successful integrity check.
- [Focused tests](../../../tests/test_pr67_receiving.py) cover capsule mutation,
  the export sentinel, composition plus RNG restoration, and report-log recovery.

Local Ruff, Bash syntax and training structure checks pass. Independent CPU
review passed 53 tests; the subsequent observer revision passed 59 focused CPU
checks in 8.48 seconds. All native receiver/CUDA/profile/full-data checks remain
pending. This preparation establishes
no numerical parity, throughput, model score or public/private transfer. Independent
review is owned by the compact-model builder; findings are resolved separately.
The frozen PR67 two-seed contrasts, bootstrap and follow-up-only trigger remain
the scientific decision rule. Validation monitoring adds exposure, not a holdout.

## Explicit source-r3 observational extension

The previous source-r2 commit `f754f0be4f48d99c4cff9f284f6e0fa684495267`
remains frozen. At root's request, source-r3 adds the independently authored
stdlib observer/progress helpers from architecture execution source
`b080d3f0874d45a6112f409603a0a10f6ec48420`, plus narrow integration hooks.
This is a new source identity requiring fresh native checks and profiles.
The [independent r2 review](INDEPENDENT_REVIEW_R1.md) and
[reviewed source digests](REVIEWED_SOURCE_R1.json) remain scoped to r2.

The wrapper now observes each profile/fit process and descendants independently,
recording all-device UUID/resource telemetry in `monitor_observer.jsonl`.
It enables a five-second observational heartbeat in `monitor_progress.json`.
Training, evaluation, checkpoint and actual terminal completion are distinct
phases; only a successful finite optimizer update advances update progress.
A heartbeat alone is not successful optimizer work. The observer never signals
processes, adopts foreign GPU PIDs or changes model/update state. Its default
observation bound is 604,800 seconds; root can set `WNN_OBSERVER_MAX_SECONDS`
within the helper's declared bounds. That bound stops observation only, so the
durable worker owner must retain its own execution/collection deadline.
Do not add a second observer writing the same output journal.

No observer is required on legacy FP32 reference exports or model-independent
scoring. Both observer artifacts are included in each production evidence bundle;
resuming preserves the independent observation history while the scientific
report streams restore their checkpointed offsets.
