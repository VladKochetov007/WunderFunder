# E426 GPU smoke and three-arm run drafts

**Draft only; not a launch approval.** This packet prepares a bounded real-data
GPU smoke and three matched four-epoch fold-0 fits for the staged E426 event
EMA trainer. It records source, inputs, runtime budgets, and open risks. The
controller must seal each immutable runner spec and separately decide whether
to launch it. No GPU work was launched while preparing this packet.

## Evidence reviewed

The real fold-0 event-count audit completed with exit 0 and verified teardown.
It counted only input column `a6`, group by group, for 10,535 TRAIN plus 1,405
fit-VALID sequences. The pinned split-manifest SHA-256 is
`09263ad4fe2b18e2b2f3b79f4562678dce801b067b868134ddc79279f9316a3d`; TRAIN
group 528 was excluded and independently rejected before count-source reads,
and eligible TRAIN group 529 was included. The seed-3 epoch-0, batch-32,
drop-last sampler retained 11,936 sequences and dropped four. It yielded 7,460
updates and 31,685,622 retained endpoint events. The complete sequence-count
matrix hash is
`230fe4ce00c437b9f582a56b7905b4f5dcaf6729c41b5ccc90da2c2ac5cf73ae`; the
ordered fit source hash is
`54d88ce9a4d534e39de9c12678fd14e6f3070736f197bcc280badefb6eec1b12`. All
three first-epoch schedules preserve the same decay product, approximately
`0.0005735115463124`. Audit record:
`runner/attempts/e426-real-event-count-audit-v1.json`; output:
`checks/e426_event_count_audit_f0_s3_v1.json`.

The audit used the local runner's requested one CPU / 2 GiB limit, ran in
39.83 seconds, peaked at 1,441,067,008 bytes of host memory, and observed no
GPU use. It checks the event pre-read and schedule construction, not model
training or EMA model memory.

Recent E424 GPU fits provide a workload reference, not a validation of E426's
separate EMA topology. The four-epoch H093 learned and H094 conditional runs
completed in 2,113 and 1,786 seconds. Their observed VRAM peaks were 4.456 and
4.459 GB, below the 4.8-GB stop but above the configured 4.2-GB allocator
budget. The E424 attempt records show cgroup memory peaks near their 24-GiB
worker limit, but this is cgroup usage, not anonymous RSS or committed model
memory. Root supplied a live cgroup snapshot with `memory.current` at the
24-GiB cap, `anon=4,773,466,112`, `file=20,718,895,104`, `shmem=2,550,464,512`
(included in `file`), and `inactive_file=17,543,159,808`; cgroup `max` events
were 28,231 while `oom=0` and `oom_kill=0`. Most current usage was file cache,
including a large inactive-file component, so it should not be described as a
10-MB malloc/RSS margin. This root-supplied snapshot is input-only evidence and
is pending persistence in the attempt record. The extra E426 consistency-
teacher EMA copy is estimated at about 6 MB per model, but its actual memory
effect and the split-EMA GPU path have not been measured. Resource risk remains
unresolved for the full fits.

The staged README describes the implementation and synthetic CPU coverage.
The real count audit now covers the production fold membership and raw count
source. The audit does not exercise the GPU trainer's split-clock path. The
bounded smoke below deliberately uses the shared EMA mode; it cannot establish
that split clocks fit within the worker limits.

## Runner interface and common closure

Use `scripts/run_campaign.py` (`seal`, then `launch`) and the active campaign
runner state. Do not call the staged trainer in an unconstrained shell. Give
each attempt a fresh ID and output path. A sealed spec must list the staged
entrypoint and trainer, staged WNN overlay files, every root `src/wnn/*.py`
file included by the entrypoint's source-closure hash, and all direct runtime
dependencies. The campaign runner separately pins its own runtime closure.
Hash these files at seal time; do not reuse hashes from the older E426 smoke
draft if any source has changed.

Four unsealed JSON drafts are in `staging/e426/specs/`. Each has 51 code paths
(50 files required by the staged runtime source closure plus the combo recipe
source) and 28 input paths. They contain path lists only; no hashes, seal,
admission record, or launch was produced here. The controller owns those steps.

Pin the following inputs in each spec, using the hashes in
`E426_READY_FOR_SMOKE.md` and re-sealing current bytes at launch:

- Fold-0 manifest and portable input manifest.
- `datasets/train.parquet`, `datasets/valid.parquet`, `cache/chains.npz`, and
  `cache/chain_folds.npz`.
- `cache/train_targets.npz` and `cache/valid_targets.npz`.
- Both fold-0 teacher checkpoints, prediction arrays, and prediction sidecars:
  `c20260929_since_s3_f0` and `c20260929_since_s4_f0` for TRAIN and VALID.
- Fold-0 propensity checkpoint, TRAIN/VALID caches, and their metadata.
- `history.json` or provenance parents only when the frozen validator requires
  them; include every path the sealed trainer actually reads.

Before launch, the spec and staged trainer must agree on the same manifest,
seed-3 ordered TRAIN + fit-VALID source list, all source sidecars and cache row
orders, propensity absolute-index cache (TRAIN 528 remains the hole), and
teacher prediction identity. The already completed audit pins the input event
source list and count matrix, but the training attempt must still record its
own code/input closure and per-epoch schedule hashes.

## Bounded shared-EMA GPU smoke

This smoke is only a two-batch real-data path and resource check. It uses the
20,000-row sequences, 32-row batch, 1,000-row recurrent chunk, combo losses,
Muon's RNN scope, fold-0 caches, and actual training/checkpoint code. The
normalization pass consumes two batches; training then processes exactly two
batches and makes 40 optimizer update attempts in the no-skip case.
`--teacher-ema-clock shared` avoids the
event-clock guard for `--max-batches`; this smoke does not test E426's separate
consistency and evaluation EMAs or the split-clock optimizer path.

Draft argv (the campaign worker substitutes its pinned Python executable):

```text
{python} research/campaign_2026-10-04/staging/e426/scripts/run_event_ema_trial.py
  --name e426_smoke_shared_f0_s3_2b_v2
  --device cuda --train-source train+valid --train-seqs 10607 --valid-repeat 1
  --valid-train-fraction 1 --full-valid
  --split-manifest research/campaign_2026-09-29/artifacts/splits/fold0.json
  --loss wmse --aux-struct 0.3 --aux-next 0.3
  --propensity c20260929_prop_f0 --prop-power 2 --prop-eps 0.03 --prop-cache
  --valid-true-mask 0.03 --hidden 256 --layers 2 --dropout 0
  --batch 32 --chunk 1000 --epochs 1 --max-batches 2
  --lr 0.001 --wd 0.01 --seed 3 --optim muon --muon-scope rnn
  --ema-teacher 1 --ema-decay 0.999 --eval-ema
  --since-weight 32 --since-floor 0.25
  --distill c20260929_since_s3_f0,c20260929_since_s4_f0
  --distill-alpha 0.5 --distill-boot-h 8 --fast --precision fp16
  --batch-order seeded --teacher-ema-clock shared
  --checkpoint-policy fixed --checkpoint-epoch 0 --no-eval-during-fit
```

Worker limits: one GPU, 8 CPUs, 24 GiB host memory, zero swap, 4,200,000,000
allocator bytes, 4,800,000,000 observed-VRAM stop bytes, and a 600-second hard
timeout. Use `WNN_THREADS=8`, `WNN_LOADER_WORKERS=2`,
`WNN_LOADER_RAM_GB=3`, and one thread each for OpenMP, MKL, OpenBLAS, Numba,
Arrow, and Polars. Let the runner set reserved resource environment variables.
There must be no competing GPU worker.

Accept only if teardown is verified, the run exits 0, exactly two training
batches and 40 optimizer update **attempts** are recorded, and the fixed
noncandidate epoch-0 checkpoint is written. Record separately the completed
optimizer updates, AMP-skipped updates, and teacher/evaluation EMA update
counts; require `optimizer_completed_updates + amp_skipped_updates == 40` and
do not assume all attempts completed. No validation or score may be produced;
there must be no cgroup OOM, host memory must stay within 24 GiB, observed
VRAM within 4.8 GB, and runtime within 600 seconds. Keep the checkpoint out of
selection and prediction. A pass is bounded shared-mode resource evidence
only.

## Matched scientific fits

These three arms preserve the registered fold-0, seed-3 combo data, model,
teachers, propensity weighting, and 23-output `wmse + aux-struct 0.3 +
aux-next 0.3` objective. They do not initialize from a prior model checkpoint;
each is a fresh combo fit under the same seeded sampler. The fixed export is
epoch 3. `--no-eval-during-fit` prevents any validation score from selecting a
checkpoint. The separated evaluation EMA updates at decay 0.999 each optimizer
step and supplies inference weights; only its epoch-3 state is exported for
later scoring. The intervention is the consistency-teacher EMA clock.

Common production argv, with `<RUN>` and `<CLOCK>` substituted from the table:

```text
{python} research/campaign_2026-10-04/staging/e426/scripts/run_event_ema_trial.py
  --name <RUN>
  --device cuda --train-source train+valid --train-seqs 10607 --valid-repeat 1
  --valid-train-fraction 1 --full-valid
  --split-manifest research/campaign_2026-09-29/artifacts/splits/fold0.json
  --loss wmse --aux-struct 0.3 --aux-next 0.3
  --propensity c20260929_prop_f0 --prop-power 2 --prop-eps 0.03 --prop-cache
  --valid-true-mask 0.03 --hidden 256 --layers 2 --dropout 0
  --batch 32 --chunk 1000 --epochs 4
  --lr 0.001 --wd 0.01 --seed 3 --optim muon --muon-scope rnn
  --ema-teacher 1 --ema-decay 0.999 --eval-ema
  --since-weight 32 --since-floor 0.25
  --distill c20260929_since_s3_f0,c20260929_since_s4_f0
  --distill-alpha 0.5 --distill-boot-h 8 --fast --precision fp16
  --batch-order seeded --teacher-ema-clock <CLOCK>
  --teacher-ema-shuffle-seed 86173
  --checkpoint-policy fixed --checkpoint-epoch 3 --save-training-state
  --no-eval-during-fit
```

| Arm | Unique run name | Teacher clock | Effective consistency-teacher decay |
| --- | --- | --- | --- |
| Fresh combo control | `e426_f0_constant_s3_4ep_v2` | `constant` | `0.999` on every update |
| Event-aligned | `e426_f0_aligned_s3_4ep_v2` | `aligned` | `0.999 ** (events_t * 7460 / total_events_epoch)` |
| Order-shuffled control | `e426_f0_shuffled_s3_4ep_v2` | `shuffled` | Same formula after seed-86173 within-epoch permutation of that epoch's count vector |

All arms use epoch-specific event counts from the actual fit rows and actual
seeded epoch sampler; the permutation changes only the order applied to the
shuffled arm. Each epoch independently normalizes the product to the constant
arm's `0.999 ** 7460`. Do not use `--max-batches` in any scientific arm. The
count audit establishes the first-epoch 7,460-update schedule; the trainer
must recompute and record all four actual epoch schedules, including each
epoch's dropped sampler tail. Each arm therefore attempts 29,840 update
windows over four epochs; record completed optimizer steps, AMP skips, and
both EMA update counts independently rather than treating attempts as
completed optimizer steps.

Run the arms sequentially, one worker at a time, with limits per fit of 8 CPUs,
24 GiB host memory, zero swap, 4,200,000,000 allocator bytes,
4,800,000,000 observed-VRAM stop bytes, and 12,000 seconds. Keep the same
thread and loader settings as the smoke and no competing GPU worker. Preserve
failed attempts under their unique identities; do not relaunch under an
existing name. Required fit artifacts are fixed epoch-3 model, provenance,
history, and complete training state. Record EMA-state identities, actual
batch/update counts, four event schedule hashes, resource telemetry, code/input
hashes, and verified teardown.

Scoring is not part of these fit specs. After fixed checkpoints pass integrity
checks, seal a separate inference/evaluation spec that uses the epoch-3
evaluation-EMA weights for every manifest evaluation VALID group, the same
official prediction mask/scorer for all arms, and no target-derived fitting or
selection. Do not score an intermediate, teacher-EMA, or student checkpoint.

## Open issues before any launch

- Previous E424 cgroup usage reached the 24-GiB limit with about 4.46 GB
  observed VRAM, while the root-supplied snapshot attributes most current
  usage to file cache rather than anonymous memory. E426's extra EMA state and
  count arrays have not been measured in the GPU trainer. The shared-mode
  smoke cannot measure the split-EMA overhead. These drafts retain the
  requested ceilings but do not establish the split-clock path's resource
  envelope; the controller should treat this as unresolved.
- The smoke uses `--max-batches` and shared EMA, so it does not validate the
  E426 split-clock code path. That path has synthetic CPU resume tests plus the
  real input-count audit, but no production-length GPU evidence.
- Confirm at seal time that the current staged trainer still accepts this
  complete argv, exports epoch 3 under `--no-eval-during-fit`, and writes both
  independent EMA states. Verify each teacher sidecar/cache row order against
  its target-cache `row_group` array and revalidate all portable input hashes.
- The fixed-epoch-3 models have not been predicted or scored here. No predictive
  result or improvement is implied by these run drafts.
