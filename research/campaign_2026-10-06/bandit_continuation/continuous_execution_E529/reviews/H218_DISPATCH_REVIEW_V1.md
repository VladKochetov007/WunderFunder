# Independent H218 and research-dispatch review

Review scope: read-only inspection of `scripts/research_dispatch.py`, its tests,
`src/wnn/block_memory_attention.py`, and its tests on 2026-10-07. No root source,
test, runtime, queue, or ledger files were changed.

## Dispatch findings

Reviewed dispatcher SHA-256: `ff5244409ffd1c55b510ec3694f23fb6ff55b8eae14ee13408306c8fb4bba96b`.
Reviewed test SHA-256: `2d8e43b2ba0dab52bd96f102d50fb3815f475063196ef17ba7b6340821329e61`.

The dispatcher hashes the complete job, including its literal argument vector.
Fresh root admission binds that digest and source hashes, then the controller
holds an `flock` while it inspects and writes state. It durably records the
submit intent before calling `subprocess.run` with no shell and `cwd=root`.
Timeouts and nonzero submit results remain reserved and are not retried. Old
unfinished attempts reserve their lane even when a later queue omits them.

Completion requires a receipt bound to attempt and job digest, the exact
expected artifact-name set, each artifact's current SHA-256, and verified
teardown for Vast. Failed-attempt reconciliation requires a root-owned receipt,
provider-terminal confirmation, hashed evidence, and Vast teardown when
applicable. The only process action is the admitted literal adapter command;
the dispatcher has no process-wide kill or unrelated-process cleanup path.

Three state-boundary gaps remain for root to decide and repair:

1. `inspect_queue` falls back to the durable attempt record when current
   completion verification fails. If that record says `complete`, the row is
   reported complete and does not hold its lane even when its receipt is now
   missing or invalid. Normal promotion first verifies the receipt and every
   artifact, but the record does not preserve their manifest hash for later
   revalidation. Store and verify the completion receipt/artifact manifest
   identity, or treat missing current evidence as needing review while keeping
   the lane reserved.
2. The CLI confines the resolved runtime directory, but its child
   `runtime/attempts` path is followed without a containment check. A symlink
   there could redirect attempt reads/writes outside the project. Reject
   symlinked runtime state paths or resolve and confine every state path.
3. `--reconcile-terminal` resolves the receipt path but does not require that
   file itself to be inside the project root. Its referenced evidence files
   are root-confined and hashed. Confine the receipt path too if project-local
   ownership is the intended boundary.

The current tests demonstrate sequential restart idempotence and lane
reservation, not a concurrent two-process race. The implementation's `flock`
serializes controller operations across CLI processes; add a concurrent test
when changing that path. `subprocess.run` owns and times out the direct adapter
process only. The adapter is explicitly responsible for durable provider
lifecycle; the dispatcher does not search for or terminate descendants.

## H218 prototype findings

Reviewed source SHA-256:
`737d97e41b24eec956221d5cbf1657a27fe3029925987dbb820e8abba3888453`.
Reviewed test SHA-256:
`2a595201aaa23a347619478c2e7c9a41b1fb92fa99c3766537efb2ec98dc7666`.

The four CPU tests cover uneven chunking (`[3, 6, 4]`) and single-row streaming
against full-sequence outputs, future-prefix isolation, reset with `states=None`,
nonzero gradients through completed-block memory updates, explicit state detach,
and full-versus-chunked parameter-gradient parity without detachment. These are
small checks: input width 5, hidden width 8, two layers, block size 4, and at
most 13 rows. They do not demonstrate a trained teacher, long-horizon benefit,
production integration, or GPU performance.

## Executed verification

The combined dispatch and H218 test run passed **14 tests in 1.68 seconds** on
one pinned CPU. A process-tree RSS watchdog capped observed use at 1,000,000,000
bytes; sampled peak was **648,032,256 bytes** and the guard did not trigger. No
GPU, fit, runtime state, or repository source mutation was involved. The test
run used the repository Python 3.13 environment, `PYTHONPATH=src`, and pytest
with its cache provider disabled.
