# Continuation admission: implementation awaiting independent review

The optional policy is bound by path and SHA256 in the new READY record.
Default initialization without a policy retains the prior admission behavior.
The policy checks local UTC start time both at reservation and immediately
before the scientific child starts, after input hashing and notification.
At the exact cutoff, CPU and GPU starts and retries are refused; admitted
children can finish. All worker commands use `Restart=no`.

Host admission requires current MemAvailable minus the full hard RAM limits
of the candidate and outstanding workers to leave at least 8,000,000,000 bytes.
Counting full outstanding limits is conservative; currently resident allocations
are not subtracted without trusted measurements. The four-GiB support reserve
is applied only to aggregate parent reservation capacity, avoiding a duplicate
charge against live MemAvailable. A new 2-GiB CPU worker can pass with 13 decimal
GB available; a 6-GiB helper cannot. Backpressure records `not_started`, charges
actual monitor time, and does not globally pause healthy work. It does not retry.

The continuation verifies parent MemoryMax <=40,000,000,000,
MemoryHigh <=36,000,000,000 and swap=0. Monitoring stops the affected worker
if aggregate charged parent RAM reaches the soft threshold. Existing worker
CPU/RAM/VRAM bounds are unchanged. The prior state's physical GPU lock is reused.

## Controller migration after independent review

1. Reconcile the old state and confirm no live/uncertain attempts or unmanaged
   services. Root alone changes the shared slice's hard/soft limits if necessary.
2. Keep the old state frozen and retain all attempts. Initialize a new state with
   the reviewed policy and explicit old-state reference:

```bash
env WNN_SLICE=wnn-campaign.slice WNN_CPU_QUOTA=100% WNN_THREADS=1 WNN_MEM_MAX=2G \
  scripts/limited .venv/bin/python scripts/run_campaign.py \
  --state research/campaign_2026-10-05/continuation_night/runner init \
  --contract research/campaign_2026-10-05/continuation_night/CONTRACT.md \
  --block local001 \
  --admission-policy research/campaign_2026-10-05/continuation_night/admission/policy.json \
  --carryover-state research/campaign_2026-10-04/runner
```

Initialization creates an exclusive `CONTINUATION.json` authority claim under
the old state's controller lock before publishing the new controller. The claim
binds the consumer path, source hashes, spent allowance, contract and runtime.
Another sibling consumer is rejected, and old-state admission is closed. Raw
attempts remain unchanged. If initialization crashes after the claim, admission
stays closed; root must diagnose/recover that exact consumer, never delete the
claim to obtain another allowance. Copying the new READY file to another state
does not transfer authority.

3. Inspect `status` before reservations: the initial carried used allowance was
   48,511.247415846 seconds, leaving 37,888.752584154 seconds from the same 24-hour
   block. Initialization recomputes from source evidence instead of trusting
   these illustrative numbers. Changes/new attempts in the source state invalidate
   the carryover and block GPU admission. The continuation cannot open a fresh
   block through the `block` command.
4. Archive the final reviewed runtime hashes from READY: campaign.py,
   campaign_spec.py, campaign_local.py, resources.py, and run_campaign.py;
   also retain contract and policy hashes, source block/READY/attempt hashes,
   test output and independent review. Initialization binds the exact source;
   subsequent code changes invalidate readiness.

No state initialization, systemd property mutation, scientific child, provider
job or retry was performed by this implementation actor.

## Verification

Executed under the specified one-CPU/two-GiB scope:

```bash
env WNN_SLICE=wnn-campaign.slice WNN_CPU_QUOTA=100% WNN_THREADS=1 WNN_MEM_MAX=2G \
  scripts/limited .venv/bin/pytest -q tests/test_campaign_runner.py tests/test_campaign_allocator.py
```

First result: **105 passed in 0.38 seconds**. Cases include old behavior, cutoff
before/at/after reservation, retry refusal, cutoff during hashing/notification,
post-cutoff drain, changing host headroom, decimal parent limits, soft-threshold
monitoring, malformed policies, exact MemAvailable boundaries, outstanding
reservations, preserved spent time, source mutation and unreconciled migration.
After the independent reviewer identified duplicate sibling consumption, the
exclusive-claim repair passed the same suite: **105 passed in 0.33 seconds**.
Independent review owns the adversarial sibling/controller-copy tests.
