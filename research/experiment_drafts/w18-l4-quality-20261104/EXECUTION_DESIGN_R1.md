# W18 L4 execution design R1

Status: local reuse audit complete and corrected to the canonical W18 trainer.
Node-side source/data/parent receipts and frozen candidate recipes are pending
root. This design performs no SSH connection, fit, GPU action, data copy, or
cleanup. Root owns node admission and the existing SSH session.

## Scope and governing contract

Use [CONTRACT_R1.json](CONTRACT_R1.json): eight fresh scientific fits, four
configurations by two matched seeds, official earlier train fitting, complete
official valid evaluation, at most 16 engineering profiles, at most 12 hours
from first profile with 45 minutes reserved for collection. The submitted
full-corpus W18/teacher weights are ineligible as a validation baseline. The
clean train-only W18/TL3 candidate must pass its ancestry audit; otherwise the
control must be explicitly qualified as a fresh clean GRU256x2 reference.

The official train/valid file boundary and whole-sequence membership remain
binding under [ADR0044](../../../specs/adr/0044-chronological-primary-validation.md).
No chronology may be inferred from `seq_ix`. The official valid contains all
1,873 sequences in the historical split build; recheck exact current source and
membership hashes before freezing the split. Repeated valid inspection is
development exposure, not fresh confirmation.

## Reusable implementation paths

W18's canonical numerical path is
[`scripts/train_rnn.py`](../../../scripts/train_rnn.py), its compatibility
facade, through [`wnn.training.runner`](../../../src/wnn/training/runner.py) and
the reusable training components. Keep this established recipe and numerical
path; do not migrate the GRU to the unrelated `train_sequence.py`/GALAM runtime.
The canonical two-epoch W18 reference is strict FP32, full official valid,
train-only input ancestry, fixed checkpoint epoch 1, and the already measured
L4 runtime was about 1,283 seconds for two epochs. New candidate recipes remain
unfrozen until their hypothesis, baseline/parent ancestry, and exact args are
reviewed.

The existing
[`scripts/run_w18_submit_recovery_20261103.py`](../../../scripts/run_w18_submit_recovery_20261103.py)
is the closest lifecycle precedent: it invokes the same `train_rnn.py`, imports
[`run_owned_command`](../../../src/wnn/multistage_search_owned_command.py),
[`base_environment`](../../../src/wnn/multistage_search_runtime.py),
`hold_gpu_slot`, and `process_identity`, captures a per-phase process receipt,
and removes `--save-training-state` only from its engineering profile. Reuse
those helpers and the current `source-a266`/`control-r2` source paths after root
binds them. Do not clone the old one-fit recovery script into an eight-fit
controller.

The source-CLI contracts are:

```bash
# Discard-only real-source profile: use the frozen W18 CLI recipe, except for
# a unique profile run name, removing --save-training-state and appending:
python scripts/train_rnn.py [FROZEN_RECIPE_ARGS] \
  --profile-warmup WARMUP_BATCHES --profile-steps TIMED_BATCHES

# Full fit: original frozen recipe arguments, unchanged.
python scripts/train_rnn.py [FROZEN_RECIPE_ARGS]

# Full official-valid predictions at the fixed endpoint, after all fits.
python scripts/predict_run.py RUN_NAME valid --epoch 1 \
  --campaign-split-manifest PRIMARY.json --campaign-evaluation-only \
  --device cuda --inference-precision fp32
```

The profile is a separate disposable run with no full evaluation or checkpoint;
the 20261103 precedent used `--profile-warmup 1` and derived a small number of
timed batches from the recipe's chunk size. Freeze per-cell batch counts, bounds
and outputs after the actual CLI/source audit. Profile evidence is not a quality
result and cannot approve itself. Full fits retain the frozen recipe exactly.

The current [`scripts/galam_l4_quality8/`](../../../scripts/galam_l4_quality8/)
launcher and `train_sequence.py` are GALAM-specific and hardcode a different
model/runtime contract; do not use them for the W18 lineage. The old
`run_heterogeneous_six.py` invokes `train_rnn.py`, but carries its own teacher,
cache, six-fit and continuation workflow; reuse only proven common command
helpers rather than its controller loop.

The historical `heterogeneous` controller and source are useful evidence for
receipt layout and bounded detached execution, but
[`run_heterogeneous_six.py`](../../../scripts/run_heterogeneous_six.py) invokes
the legacy `train_rnn.py` path and carries teacher/cache phases specific to its
old six-fit design. It is not the default runner for this eight-fit wave.

## Receiver data and cache path

Root reports the existing raw Parquets at
`/data2/lc-epanov_s/tabdata`, totaling about 34.9 GB, plus the completed
index/chronological-chain and `sequence_normalization` cache. These inputs are
already on the node and must be reused in place. Do not recopy the Parquets or
rebuild caches over existing paths. Confirm before admission that cache/source
hashes, group order, target index, split, train-only normalization ancestry, and
read-only permission all match the pinned recipe. The normalization cache may
be shared only if its manifest proves it was fitted on exactly the admitted
training members and the same source/config contract.

Freeze the primary split from the official train/valid files and existing
source-qualified component exclusions. The previous L4 packet reported 10,325
fit sequences after 282 connected exclusions, 1,873 full-valid sequences and
3,528,273 scored rows; those values and its split SHA are historical checks, not
substitutes for validating the current node data and exact membership.

## Profile, fit and lifecycle

Use one separately receipted profile for every exact frozen config/seed before
fitting unless the existing controller's identity contract explicitly supports
shared profile receipts. The inspected trainer binds profile approval to the
complete config identity, including seed, so plan for eight profiles. Each is
discard-only, has no official-valid evaluation, exercises the configured
diagnostic cadence, and records updates, finite loss/optimizer state, step
throughput, peak allocated/reserved VRAM, host RSS and exact device UUID. A
profile is not a quality result and cannot approve itself.

The detached cohort should run one independent fit per admitted GPU, never more
than one fit per physical GPU. Record each child's argv, environment, CUDA UUID,
PID, process-group ID, `/proc` start ticks, start time, log path, config hash,
profile receipt hash and source identity before declaring it running. Use a
single bounded cohort deadline from the contract and expose per-phase state.
On failure/deadline, stop only a process whose PID, start ticks, process group
and recorded launch command still match this cohort. Leave GPU4's 1,148 MiB
service and all other foreign processes alone. Do not reconnect to SSH or start a
replacement fit from a partial checkpoint on this assignment.

The profile ceiling is 16 attempts total, including failed pre-CUDA attempts.
Preserve failed logs and receipts. Root sets a fit-specific timeout only after
actual profile throughput and endpoint costs are measured; the whole wave retains
the original 12-hour limit and 45-minute collection reserve.

## Metrics, live graphs and standalone artifacts

The canonical W18 trainer emits flushed `VALID {json}` records after each
completed epoch and writes `history.json`, `provenance.json`, `model_ep*.pt`,
`model.pt` when selected, and the continuation state required by the recipe.
Each `VALID`/history row carries full-valid WMSE/WP by target plus the fixed
train-eval diagnostic. `GRAD_STATS` records epoch gradient summaries; online
progress records loss, LR and sequences/second. The detached owner adds per-phase
stdout/resource logs and process/resource receipts. Preserve all of these raw
artifacts. The previous full-corpus recovery's
[`plot_w18_recovery_20261103.py`](../../../scripts/plot_w18_recovery_20261103.py)
is a training/resource plot precedent, but labels all-data W18 honestly and does
not plot a clean full-valid series.

The train Parquet has no `is_scored` column. Its 32-sequence `train_eval` uses
`all_after_warmup`; official valid uses the original `is_scored` mask and must
cover all 1,873 sequences / 3,528,273 scored rows. These metrics are not
identically masked. Plot online objective, train WP proxy/WMSE and full-valid
WMSE/WP separately with their cohort and mask printed in labels. Never describe
their absolute gap as an overfit measure. Preserve the fixed epoch-1 primary
endpoint separately from best-observed valid diagnostics; exposed valid must not
select ancestry, coefficients or fitting members.

For live viewing, first reuse the actual dashboard/bridge already staged under
node `delivery-r1/r2/r3` or `control-r2`; root must provide its exact input/output
contract because the generic `src/wnn/dashboard` consumes event JSONL, while the
legacy W18 trainer writes flushed text records and a final `history.json`. Do not
claim those logs are already connected to the dashboard. Root owns dashboard
launch and any SSH forwarding. Retain freshness/error labels; the dashboard is
a live view, not the authoritative artifact store.

After each epoch and at terminal completion, export actual-data PNG/SVG plots
from the flushed `VALID`, `GRAD_STATS`, online-progress and resource receipts.
Bind every plot manifest to raw-log hashes, config/args hash, fixed checkpoint
SHA and source/split identity. Plot full-valid WMSE/WP and target-0/target-1,
train-eval WMSE/WP proxy, online objective, LR, preclip gradient/clipping,
nonfinite/skipped updates, throughput and CPU/RAM/VRAM telemetry. Reuse a
compatible plotter from `control-r2`/`delivery-r*` if present; otherwise add a
small plot-only adapter after recipes freeze. It must not launch a fit or repeat
model evaluation. The 20261103 recovery plotter is not reused unchanged because
it has fixed paths and appropriately states that its all-data fit has no
independent score.

## Source binding and receipts

Before any profile, bind the actual deployed tree (commit plus implementation
file manifest and hashes), Python/Torch/CUDA/driver, trainer and controller
entrypoints, every config, data Parquets and indices, train/valid and excluded
membership manifests, normalization recipe and cache, baseline and all parent
checkpoints. Store byte hashes, not path names alone. Profiles, fit outputs,
validation predictions/scores and plots must all link back to this same binding.
The current contract's base commit `6493556e8` is not by itself proof that the
receiving `source-a266` bytes match; record the actual node identity.

Keep a durable receipt for every phase: receive inventory; data/split/cache
verification; baseline ancestry qualification; source admission; each engineering
profile and independent approval; each fit launch and terminal state; each full
valid prediction/score export; each plot export; final collection and hashes.
Receipts record actual timestamps, process/resource identities, command/status,
input/output hashes and failures. Keep raw logs/checkpoints/predictions; summaries
must not replace them. No external publication is part of this execution packet.

## Resource and output limits

The receiver inventory reports eight L4 devices at 23,034 MiB each, 192 CPUs and
582 GiB available RAM. The aggregate caps are 64 CPUs and 400 GiB RAM. Reserve
at most eight CPU threads per fit when all eight GPUs are active, and verify
actual CPU affinity/quota, cgroup RAM, device UUID and available VRAM at each
launch. GPU4's unrelated 1,148 MiB allocation remains untouched.

The prior quality8 profiles are useful historical headroom bounds only: the
largest allocated peak was 16.849 GiB/device, largest reserved peak 21.016 GiB
of 22.045 GiB, and summed host peak RSS about 136.2 GiB across eight profiles.
New recipes/source/data must be profiled again; these numbers do not admit them.
The new experiment may write at most 8,000,000,000 bytes and must keep at least
3,000,000,000 bytes free on `/data2` throughout. Node free space was about
12,899,225,600 bytes at inventory, leaving about 1.90 GB slack beyond the output
cap. Do not duplicate input caches. Include checkpoints, optimizer/resume state,
predictions, logs, plots, receipts and temporary exports in the same output
accounting; stop admission before either byte cap or free-space floor is crossed.

## Open receiver checks before implementation

1. Root supplies read-only CLI/source receipts for `control-r2`, `source-a266`,
   `launch-20261028`, the reusable dashboard, and the existing delivery scripts.
2. Baseline auditor returns an exact eligible checkpoint/config/source/split/
   normalization/ancestry binding or records why the fresh GRU control is needed.
3. Root freezes the four hypotheses, paired seeds, endpoint, data/cache hashes,
   configs and profile/fit resource bounds. No candidate outcome may change them.
4. The receiver confirms current disk-free floor, aggregate memory/cpu controls,
   all eight physical UUIDs, GPU4 foreign-service boundary, and module/runtime
   compatibility. No profile or fit runs before these are receipt-bound.
5. After freeze, implement only the smallest panel/config adapter that feeds the
   existing trainer/controller, plus meaningful CPU contract checks. An
   independent reviewer signs off; the builder does not approve its own code or
   scientific result.
