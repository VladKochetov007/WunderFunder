# Independent continuation runner review

Initial verdict: HOLD. Reviewer authored neither runner nor runner tests.
No initialization, provider call or scientific child was launched for review.

Blocking finding R1: two independent destination states can initialize from
the same source state, inherit the same 37,888.752584154 remaining seconds,
and spend separately. The shared fcntl GPU lock prevents concurrency but
does not prevent sequential double spending. Required repair: a durable,
single-consumer migration claim under the source authority lock; source must
refuse new admission after the claim, including another continuation. Preserve
original attempts and budget evidence. Crash during migration must fail closed.

Independent reproduction command, bounded to one CPU/two GiB/no GPU:

```bash
env WNN_CPU_QUOTA=100% WNN_THREADS=1 WNN_MEM_MAX=2G \
 WNN_SLICE=wnn-campaign.slice WNN_RUNTIME_MAX=120 CUDA_VISIBLE_DEVICES= \
 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 scripts/limited .venv/bin/python -m pytest \
 research/campaign_2026-10-05/continuation_night/activation_transfer/test_runner_independent.py \
 tests/test_campaign_runner.py tests/test_campaign_allocator.py -q
```

Initial result: 1 failed, 112 passed in 0.25s. The independent sibling-consumer
fixture fails because the second initialize does not raise. Other independent
checks passed: actual OS flock contention across independently opened handles,
malformed/missing/duplicate host-memory fields fail closed, exact cutoff rejects
admission and missing/unparseable parent soft cap is rejected.

Source inspection confirms deadline checks at reservation, pre-service launch
and immediately before scientific Popen after input hashing/notify; admitted
children can drain. Live MemAvailable checks conservatively reserve full worker
limits and leave eight decimal GB. Parent properties verify <=40e9 hard RAM,
<=36e9 soft RAM, <=12 CPUs, zero swap. Worker monitoring rereads live cgroup
memory and parent limits every 0.5s; missing/malformed reads fail closed. No
cached host telemetry is trusted. GPU usage is independently observed.

Budget source hashes and exact charged time are carried; policy mode refuses
fresh block renewal. These safeguards do not resolve R1 by themselves.
Final GO requires the author repair plus rerunning the unchanged independent
fixture and source review of the handoff race/crash paths.

## Repair verification and final verdict

GO for the reviewed continuation runner, subject to ordinary per-job resource,
scientific source, budget and deadline admission. This is not scientific approval
of any candidate and does not authorize an unmeasured full-data allocation.

The author added a durable source CONTINUATION.json claim under the source
controller lock before publishing destination readiness. A second consumer is
rejected; source readiness/admission is closed. Destination readiness binds its
state path, and budget binds the source claim's consumer, used allowance and
hash. No raw attempt evidence was rewritten. A crash after claiming closes both
automatic reuse paths; recovery requires explicit reconciliation.

Reran the unchanged sibling-consumer fixture plus new independently authored
source-readiness, copied-consumer, and crash-between-claim-and-ready fixtures.
Result: **115 passed in 0.24s**, under the same 1CPU/2GiB/noGPU scope. Reviewed
source lock order and claim-before-publication path. No unresolved blocker found.

Reviewed SHA256 identities:

| File | SHA256 |
|---|---|
| src/wnn/campaign.py | 35dd701fdb33c50494a3c781a315731212da22f41cae331f2f85073af0637605 |
| src/wnn/campaign_local.py | 0213ff7a809ce898be5d3180c15e31d228ad6f3f03f1f08fd968f9186e91353d |
| src/wnn/campaign_spec.py | 5136703299585e81a9463a253133ba7f12da2c64ea863776c005d1fc5d3e8d7b |
| src/wnn/resources.py | 06a624ad963dfc2b0c75ce717278de5f1bb85f48518a99e519b697a371d4cf09 |
| scripts/run_campaign.py | ebe3ebc5c648e7e8a49e8f4ee69c0194a51c65a098fe619632fffc700d7db52b |
| admission/policy.json | 35a0c948fcf23689da6024649914eeaa5a69730be0dc3e71223b8bc06e986950 |
