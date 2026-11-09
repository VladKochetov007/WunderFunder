# Conditional continuation preparation for the fixed PR67 fits

Status: preparation only. ROOT reports all four fits admitted and initializing
on 7 November after twelve successful native profiles and a finite nine-day
authority amendment. This note changes no live source, controller, environment,
approval or process. It does not authorize replaying historical stopped fits or
starting a new scientific fit/seed.

## Actual frozen behavior

Source-r3 is `6ab2f679e70928f913b97d5a29f0d1e09ceacaa3`. Its wrapper calls
`scripts/observe_sequence.py` around fitting with a default 604,800-second
observation limit, and that observer rejects larger limits. On expiry it warns
and returns 124 without signalling its worker. Bash exits under `set -e`; the
external receipt-owned relay then cleans its PPID/birth-proven separate rank
groups and leader group. Thus expiry becomes a failed owned command rather
than indefinitely unobserved fitting. Native ownership/proof/log evidence must
confirm the actual result; do not infer exit solely from an old heartbeat.

The canonical `sequence_fit_modes.run_full_fit` writes `training_state.pt`
initially and after **every complete safe epoch**, irrespective of
`checkpoint_retention=final_only`. The `all_epochs` option adds immutable
inference snapshots; it is not required for resumability. Safe state contains
model, optimizer, scheduler, next epoch, global update count, fitting history,
rank RNG, rank log offsets, aggregate metrics offset and monitor offsets.

`run.sh resume <cell>` uses that same cell's `fit/training_state.pt`, original
config/source/reference, profile approval and physical GPU pair. Canonical
loading validates exact identity, restores all fitting state and truncates
post-checkpoint scientific log tails while preserving orphan evidence. Replayed
partial-epoch work begins from the last complete checkpoint, retaining the
original logical fit, seed, schedule and final epoch-30 endpoint.

## Prefer no continuation when actual timing supports that

The measured warm production interval is about 0.71–0.73 seconds/update. The
fixed 774,600 updates imply about 6.4 days of core fitting before validation,
I/O and checkpoints. This is a profile extrapolation, not measured completion.
Compare actual complete epoch-to-epoch evaluation/checkpoint intervals after
epochs land. If the final epoch is expected before seven days with useful
headroom, keep the healthy fits running unchanged and take no continuation
action. No live source/controller repair is requested now.

## Only if actual seven-day observation expiry occurs

ROOT must review a concrete continuation receipt before executing anything:

1. Bind the original cell/logical fit, source-r3 manifest, original exact config,
   full data/reference/normalization identity, production approval and physical
   GPU UUID order. Keep the original scientific fit attempt; reserve no new fit
   or seed. The current pairs are 0,1 labels101001; 2,3 complement101001;
   4,5 labels101002; 6,7 complement101002.
2. Preserve and hash the original native owned-command receipt and fit log.
   Require the actual observer-expiry warning, approximately 604,800 seconds of
   observed fitting, terminal nonzero owned exit and matching PID/birth/PGID.
   Distinguish expiry from a numerical failure, user stop, resource breach,
   driver failure or arbitrary old paused run. Those other cases require their
   own diagnosis and are outside this preparation.
3. Require the original worker and all recorded rank/descendant identities to be
   absent or zombie, with no active owned compute on that cell's UUID pair.
   The relay sidecar is written before its final leader-group cleanup; a final
   `remaining_non_zombie` field can precede that last kill. Verify actual
   post-exit process identities rather than trusting that pre-kill field alone.
4. Hash and preserve the existing checkpoint before launch. Inspect its
   `wnn.sequence-resume.v1` schema and exact contract against the approved native
   profile. Require `next_epoch` in 1..29, history reaching that whole epoch,
   `global_step = next_epoch * 25820`, two rank RNG/log states and valid monitor
   offsets. A completed epoch-30 terminal needs collection, never continuation.
   Missing/corrupt/mismatched state is not admitted by this note.
5. Bind the already accepted nine-day authority amendment, unchanged four-fit
   cap and final collection reserve. Check the measured remaining work fits the
   remaining pre-reserve time. If it does not, stop admission for further ROOT
   review; neither shorten epochs nor silently extend the deadline.

The external continuation command, after that review, is the existing wrapper:

```bash
bash "$S3/research/experiment_drafts/tl3-transformer-complement-execution-20261107/run.sh" resume "$CELL"
```

Use the original exact receiving environment and mandatory short run alias:
`WNN_RUN_ROOT=/data/epanov_s/tdn7run`, frozen manifest/commit/split, unchanged
data/reference roots, original CUDA pair and provisioned Python. Preserve
`CC=/usr/bin/gcc`, `CXX=/usr/bin/g++` and compile threads 4. Set
`WNN_OBSERVER_MAX_SECONDS` to no more than the remaining pre-reserve seconds
and no more than 604,800. At day seven the nine-day envelope leaves less than
two days, so this stays inside the frozen observer's accepted range.

Run through the same known exact-owned helper and relay, with **new external
continuation ownership/log/proof paths**, outside active receiver-r1. Include
`continuation_of_owned_receipt_sha256`, checkpoint SHA256, source/config/approval
SHA256 and unchanged logical cell/fit ID in that receipt. Do not modify the
active Q controller, its old receipts or S3. This plan contains no automatic
launcher and is not an approval file.

After native fixed-final completion, ROOT can execute the existing `evaluate`
and `bundle` stages with bounded collection time; compare only after all four
fixed-final endpoints are complete. The original controller may have marked an
expired cell failed, so an external continuation/collection receipt must explain
the recovery without rewriting that historical status. Retain both attempts as
execution history of one scientific fit.
