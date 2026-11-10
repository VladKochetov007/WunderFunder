# Independent actual B300 profile admission review

Disposition: no recipe, identity, finite-work or measured memory blocker in all
twelve source-r3 profiles. Supports ROOT's finite nine-day authority amendment
and admission of the unchanged four fixed 30-epoch fits. No SSH, launch, model
inference or source/controller edit by this reviewer.

Evidence: local `b300-rpc/results/043-profile-progress.json`,
`044-warm-step-metrics.json` and complete native JSON receipts captured in
`045-admission-evidence.json`, under the architecture execution artifact root.
The reviewer independently parsed all twelve receipts and recomputed normalized
config digests from the committed cell configs plus the declared one-rank and
zero-warmup peak variants.

All twelve pass complete/twelve-update/error-zero/no-scientific-checkpoint,
config SHA256, canonical data digest, 10,325 train-only fitting members, all
1,873 evaluation members, fitting-only all-fit normalization and no KD teacher.
All share source SHA256
`bdad606dcff66781b904ae4774e7435b8ee21c21de2911cd0320eb2636c725d7`, one
normalization moment digest and one reporting reference. Each model has
170,198,786 parameters. Production UUID pairs match the recorded 0/1, 2/3,
4/5 and 6/7 mapping. Peak reserved memory is at most about 6.096 GB against
287.429 GB/device. This is short-profile headroom, not a full-run stability claim.

Exact schedule arithmetic is 5,163 padded sequences/rank, 1,291 B4 batches,
79 chunks per sequence and 20 A4 windows per batch: **25,820 updates/epoch and
774,600/fit**. The actual complete warm windows forward 8,192 real rows,
matching `world2 * B4 * A4 * 256`. One epoch forwards 206.5 million actual rows;
one 30-epoch fit processes 6.195 billion rows. The count is not an accidental
fourfold duplication introduced by the receiving packing.

Production warm windows mostly take 0.71–0.73 seconds, including about 0.16
seconds of optimizer time. At 0.71, the frozen schedule needs about **6.37 days
of core fitting**, before validation/checkpoints/I/O. The original 24-hour
envelope cannot complete it. Nine days is a justified finite planning allowance,
not a measured wall-time guarantee; actual per-epoch timing must refine it.

B16/A1 is not equivalent: each update uses 32 sequences at one temporal chunk
instead of eight sequences over four temporal chunks, with different parameter
update timing against carried KV history. Its 765,510 steps also retain about
1.42 days of measured optimizer-only cost at 0.16 seconds, even with zero model
compute. It cannot be silently called a faster equivalent 24-hour recipe.
Larger-batch variants require separate identities/profiles and are not a reason
to keep the already supported unchanged fits idle.

The source observer's seven-day cap remains unchanged. See the
[conditional same-fit continuation preparation](../operations/PR67_OBSERVER_EXPIRY_CONTINUATION_PREPARATION_20261107.md)
if actual cap expiry occurs before fixed-final completion; no automatic replay
or current source/controller edit is proposed.
