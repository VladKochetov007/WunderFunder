# E-495 cadence profiles under the cap-3G contract

Status: separate B16/12 GB resource-admission capsule. It does not edit or
replace the October 5 cadence drafts. No profile, fit, prediction or score has
run here. The existing cadence source gate is imported read-only before draft
materialization and its exact bytes are source-hashed.

The original E-495 design compares `update1000` (update/detach 1000/1000),
`pooled2000` (2000/1000) and `connected2000` (2000/2000), with T1000 microchunks
and matched EMA decay per processed row. This capsule preserves those flags and
the original H256, two-layer GRU, loss, optimizer, schedule, teacher caches,
fixed checkpoint and complete parent/head endpoint.

## Common batch change

The old cadence drafts use parent batch 32. This new resource profile changes
the parent batch to **16 for all three arms and every seed/fold block**. It is
a newly registered common baseline because changing batch changes optimizer
steps per epoch and gradient noise. Effects from this capsule apply only to
matched contrasts under the B16 recipe. The final head retains the existing
B16/T2000 recipe.

The screen remains seed 3/fold 0, with seed 4/fold 0 and seed 3/fold 1 reserved
for confirmation. Parent and complete endpoints are both frozen; the complete
predictor uses a fixed equal prediction mixture of parent epoch 3 and head epoch
0. It reports all three registered pairwise comparisons and 2,000 paired
component-bootstrap draws.

## Resource gate

Each seed-3/fold-0 arm has its own production-loader profile with the registered
update/detach flags. Profiles warm up one step and measure two. Requests are
capped at 2 CPUs, 12,000,000,000 host bytes, 2,500,000,000 allocator bytes,
2,800,000,000 observed device bytes and 600 seconds. The 12 GB host cap is the
current live-headroom admission variant; the October 5 18 GB drafts remain
unchanged. It is a proposed cap, not measured usage or root admission.

`connected2000` can retain the larger recurrent graph. Its B16 profile must pass
before any B16 cadence fit. If that profile exceeds either accelerator limit,
do not fit a partial B16 panel or mix batch sizes. Reprepare a B8 common panel
for all three arms, heads and endpoints under a new source-hashed identity, then
profile all three arms again. Root owns resource sealing, live quota/headroom,
reservation and launch.

The source gate reads the current frozen trainer, training-control and cadence
materializer implementation. No trainer, campaign runner, treatment flags or
historical capsule is edited by this capsule.
