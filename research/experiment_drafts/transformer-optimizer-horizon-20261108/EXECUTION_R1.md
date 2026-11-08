# Execution and receiving R1

ROOT owns the B300 RPC stream, actual source/data/device verification, approved
profile copies, detached controller and all fit admission. The command blocks are
the exact packet interface; they do not authorize another actor to connect or
launch. Six exact-config50-update profiles are engineering qualification, never
scientific fits. Current ROOT state/receipts and actual clock timestamps remain
the authoritative execution record.

## Frozen native roots

```bash
R=/data/epanov_s/tabdata/runtime/transformer-optimizer-horizon-20261108-r1
S=$R/source-r2
Q=$R/receiver-r2
OLD=/data/epanov_s/tabdata/runtime/architecture-execution-20261107-r1/receiver-b300-r1
SUPPORT=/data/epanov_s/tabdata/runtime/multistage-optuna-20261031-r1/operator-r26/src
PY=/data/epanov_s/tabdata/.venv/bin/python
DATA=/data/epanov_s/tabdata/runtime/architecture-execution-20261107-r1/project
SPLIT=/data/epanov_s/tabdata/runtime/multistage-optuna-20261031-r1/data/chronological_primary.json
```

Use source commit `5d73bc2b10ae20e607e0d4b4533da85e76e27bca` and manifest
`R/source-r2-manifest.json` SHA256
`a1e16d24ab53032643807edb5471b8bc65b06bdd96da516944bcf36736339937`. Config
files are staged unchanged under `R/configs/`. The data root and split are
already qualified; do not download or copy dataset rows.

## Exact profile launcher

The six profiles use this fixed nonoverlapping wave schedule: first C0A on GPUs
0,1; C1A on2,3; C2A on4,5; C0B on6,7. After that wave exits, C1B uses2,3 and
C2B uses4,5. Each profile is owned by the existing process/reaper helper, with a
900-second hard bound and no automatic retry. An observer-only timeout is not a
worker limit. The optional B256/world1 engineering profile is not in this plan.

Before launch ROOT verifies/creates that `/data/epanov_s/tdh8` is a symlink to
`R/scratch`; the helper refuses any different target. ROOT writes the first
profile epoch and the fixed8h authority without resetting it. Example detached
invocation from the staged packet:

```bash
export PYTHONPATH="$OLD:$SUPPORT"
"$PY" "$Q/receiving/profile_runner.py" \
  --plan "$Q/profile-plan.json" --runtime "$R" --python "$PY" \
  --accepted-receiver "$OLD" --support-src "$SUPPORT" --detach
```

The helper rechecks the profile-plan/source-manifest/split/config identities,
imports ROOT's already accepted `Controller`, uses the accepted `owned_phase.py`
and `run_owned_command` with900s per profile, captures process-descendant
receipts, and requires50 updates, finite/no-error completion and both-rank strict
model/optimizer/scheduler/RNG reload. Every fit requires its own seed/config
profile receipt. ROOT copies the exact passing receipt with `approved=true` and
`reviewer=ROOT` to `R/approvals/<cell>.json`; raw receipts stay under
`R/profiles-r2/<cell>/profile_receipt.json`.

## Build the immutable receiving plan

After all six profiles pass, ROOT writes a source-bound
`R/software/COMPLETE.json` qualification. It binds the exact source commit,
source manifest and official split, runtime versions/device/NCCL status, the
focused CPU-test JUnit SHA, and both raw-profile and ROOT-approval hashes for
all six cells. The receiver independently rechecks each exact profile's source,
config, split,50 updates, no-checkpoint status and two-rank strict reload; no
historical correlation-oracle or old-fit retirement receipt is required.
`R/authority-r2.json` binds the original first-profile clock; build a one-shot
plan. The authority contains `fit_cap=6`, `profile_cap=6`, fit deadline at
first-profile+7h, collection deadline at+8h and a1h reserve.

```bash
export PYTHONPATH="$OLD:$SUPPORT"
"$PY" "$Q/receiving/build_plan.py" \
  --profile-plan "$Q/profile-plan.json" \
  --runtime "$R" --python "$PY" \
  --source-manifest "$R/source-r2-manifest.json" \
  --source-manifest-sha256 a1e16d24ab53032643807edb5471b8bc65b06bdd96da516944bcf36736339937 \
  --authority "$R/authority-r2.json" \
  --profile-root "$R/profiles-r2" \
  --output "$Q/plan.json"
PLAN_SHA=$(sha256sum "$Q/plan.json" | cut -d' ' -f1)
```

`build_plan.py` freezes byte/config identity, source identity, normalized config
hash, profile receipt path, authority hash and exact6 cell list. It refuses an
existing plan path rather than replacing admitted work.

## Detached finite receiver

ROOT launches only after the source-bound six-profile qualification and r2
authority are present. It executes source/software acceptance, a CPU-only plot readiness
check, then the predeclared4+2 fit waves. Same-pair cells from different waves
cannot overlap. The per-cell fit phase executes the source-r2 trainer directly
with its approved exact profile receipt, wrapped by the existing read-only
`scripts/observe_sequence.py` to populate `monitor_observer.jsonl`. The outer
accepted owned-session lifecycle still owns signals and the absolute deadline;
the observer's `25200`-second cap only bounds observation. Fit uses no reference,
teacher, cache, blend or monitor-reference flags. Fit admission closes at+7h; full-valid
prediction/official scoring, plots, bundle and collection finish by+8h.

```bash
export PYTHONPATH="$OLD:$SUPPORT"
"$PY" "$Q/receiving/receiver.py" \
  --plan "$Q/plan.json" --plan-sha256 "$PLAN_SHA" \
  --python "$PY" --accepted-receiver "$OLD" --support-src "$SUPPORT" --detach
```

The receiver consumes approvals, never writes them. It has one fit attempt per
cell and no automatic resume/retry. On failure it preserves all outputs and
receipts and reports the exact cell; it never signals unrelated or historical
processes. The final primary prediction manifest is graded by the unchanged
official scorer on every valid sequence; all-epoch plots and monitors remain
report-only. ROOT returns resource, time, profile, fit, evaluation, terminal and
archive receipts under the runtime's controlled path.
