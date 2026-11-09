# Receiving B300 commands

These commands are prepared and unexecuted. Use an isolated clean checkout and
the team's provisioned Linux B300 environment. See `PUBLICATION.json` for the
implementation commit; pin that full SHA as `WNN_EXPECTED_COMMIT`. Do not change
any live checkout or occupy GPUs belonging to another run.

From the isolated repository root, replace the path examples with the preserved
canonical data, original split file and a fresh physical-disk output directory:

```bash
export WNN_EXPECTED_COMMIT=67c351ae70cacef11f4ad485b5b98192f0d5e81c
export WNN_DATA_ROOT='/absolute/path/to/canonical/data'
export WNN_SPLIT='/absolute/path/to/original/chronological-split.json'
export WNN_RUN_ROOT='/absolute/path/on/persistent-disk/qk-norm-r1'
export CUDA_VISIBLE_DEVICES='0,1'
PACKET=research/experiment_drafts/transformer-qk-normalization-20261109
bash "$PACKET/run.sh" data-check
bash "$PACKET/run.sh" check
```

The GPU IDs are an example, not a reservation. Keep the same pair and order for
both arms of a seed. A second verified pair may process the other seed;
all four cells must share the canonical data, software and source identity.
The wrapper requires exactly two GPUs per invocation. It does not select free
devices, create an environment, download data or start an observer/queue.

For each cell, execute both engineering phases before any fit:

```bash
CELL=plain_s203301
bash "$PACKET/run.sh" profile "$CELL"
bash "$PACKET/run.sh" profile-peak "$CELL"
```

Review both receipts under `$WNN_RUN_ROOT/runs/$CELL`, including errors, peak
allocated/reserved memory, wall time and both-rank model/optimizer/scheduler/RNG
reload. Preserve them. Make a reviewed copy of the production receipt at
`runs/$CELL/approved_profile.json`, changing only `approved` to `true` and adding
nonempty `reviewer`, `reviewed_at` and optional `approval_note`. No script here
approves a profile. The fit additionally checks the exact live runtime against
the approved production receipt.

After that cell is admitted:

```bash
bash "$PACKET/run.sh" fit "$CELL"
bash "$PACKET/run.sh" evaluate "$CELL"
bash "$PACKET/run.sh" bundle "$CELL"
```

Process `plain_s203301`, `rms_s203301` in that order on one pair;
then `plain_s203302`, `rms_s203302` on its assigned pair. The two
seed streams may be concurrent; do not run two cells on the same pair at once.
Each cell needs its own two profiles and explicit approval. The fixed endpoint
is E4/3,280 updates. Preserve failed outputs; the wrapper never automatically
retries or resumes. `resume CELL` is an explicit continuation from that cell's
safe `fit/training_state.pt`, only after failure and ownership review.

After all four fixed endpoint grades exist:

```bash
bash "$PACKET/run.sh" compare
bash "$PACKET/run.sh" panel-bundle
```

`comparison.json` contains all four scores, every paired contrast and checkpoint,
score and approved-profile hashes. It does not select epochs or launch a follow-up.
The optional secondary route sets `WNN_REFERENCE_ROOT` to the exact frozen PR49
TL3 cache root, runs `evaluate-blend CELL` for **all four** endpoints, then runs
`compare-blend`. It never fits or tunes the fixed 90/10 mixture coefficient.

Preserve the complete run root. The standard bundle includes final weights and
resume state; all-epoch weights and index snapshots need a separate companion
archive if they are transported. These Transformer weights are a research
teacher artifact, not a CPU submission ZIP.

`panel-bundle` preserves the aggregate comparison, acceptance and per-cell bundle
manifests in a separate hashed evidence archive. Transfer it together with all
cell bundles; a cell-only archive omits the paired comparison.
