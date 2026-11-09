# Causal raw-input stem for the B300 Transformer

## Status and scope

This is a local experiment draft based on PR72, not an execution result or a
published PR. It proposes six fresh fixed-four-epoch fits on one repeatedly
exposed official fold: plain input, a kernel-1 stem, and a kernel-16 stem, each
at seeds 203201 and 203202. No fit, profile, inference, local test, receiver
gate, or remote job has been run for this packet. All gains remain unknown.

The run-source baseline is PR72 head
4021de8ec5130143c207319c73a102a453eb911e. The intended target is research-team
at 7e4074314a62d0907c9608f9220cf72958708a2d. The eventual runtime source SHA
is unset until the root-owned implementation is committed and reviewed;
run.sh requires an exact clean full commit before it can proceed.

This branch is stacked on PR72. A future PR targeting research-team includes
the inherited PR72 runtime changes; review the incremental stem diff against
4021de8ec5130143c207319c73a102a453eb911e and review the inherited PR72 diff
separately. This packet does not authorize publishing or merging PR72.

## Mechanism

The input is the 112 standardized raw channels, u=(raw112-fitmean)/fitstd.
The candidate computes v=u+depthwiseConv1d(u), with 112 groups, no bias, and
zero-initialized convolution weights. Kernel 16 uses only the current row and
the preceding 15 standardized rows, carries those rows across chunk boundaries,
left-pads only at sequence start, and resets at each sequence. Kernel 1 is a
per-channel instantaneous-factorization control. Plain omits
model.input_stem_kernel, preserving the legacy no-stem path.

The primary contrast is K16 minus K1. Secondary contrasts are K16 minus plain
and K1 minus plain. The extra stem parameters are assigned to the AdamW-matrix
group, outside the Muon block; K1 adds 112 parameters and K16 adds 1,792.
Each seed gets a fresh backbone with the same initialization across its three
arms; stem weights start at zero. Parameter counts are not matched across arms.

This is not a new claim for convolutions, numeric bins, or local dynamics.
The older [GRU Conv16 pilot (E-168)](../../unified/archive/2026-10-12/ledger-snapshot/experiments.asof-20261012.jsonl#L168)
was positive on one exposed seed/fold and reset history at each chunk; its
result did not establish a Transformer input stem. The [H-264 equal
attention/TCN endpoint mixture](../../unified/archive/2026-10-12/ledger-snapshot/experiments.asof-20261012.jsonl#L620)
lost 0.000319 WP at its 10% TCN setting. The [large-dynamics
proposal](../../unified/ideas/LARGE_DYNAMICS_TRANSFORMER_B300_20261019_R2.md)
already discusses causal convolution and numeric embeddings, and
[PR35](https://github.com/VladKochetov007/WunderFunder/pull/35) already
implemented typed numeric/event features. This packet tests only the declared
residual raw-row stem in the current Transformer recipe.

## Frozen panel and cost

All six configs are structural copies of
transformer-optimizer-horizon-20261108/configs/C0_muon256_a4_s203101.json.
Every setting is identical except seed and model.input_stem_kernel; the plain
config omits that field. The panel retains the PR72 C0 recipe: Transformer
768x24, 12 heads, context 1,024; per-rank sequence batch 128, world size 2,
global batch 256; chunk and gradient horizon 256; accumulation 4; Muon 256;
4 epochs, 820 updates per epoch, 3,280 updates per fit, and 19,680 total fit
updates. There are no reused controls, teacher fits, student fits, reference
caches, or automatic follow-up fits.

The receiver plan has 12 engineering profiles: one 50-update profile and one
50-update zero-warmup peak-memory diagnostic per cell, or 600 profile updates
total. Each profile must bind the exact source, config, and split; no old
profile receipts can be reused. These are gates and diagnostics, not scientific
fits. There are six 2-GPU fit allocations if admitted. A possible mapping is
GPUs 0/1 for seed 203201 and 2/3 for seed 203202; ROOT must verify actual
allocation. Within each seed, all three arms must use the same physical pair,
runtime, and order (plain, K1, K16). No device reservation is asserted; other
GPUs remain available to existing work.

Fit wall time and disk demand have not been measured. Before fitting, the
receiver must record the persistent physical output mount and capacity. The
run root cannot be /tmp, /dev/shm, tmpfs, or ramfs. Do not infer an ETA or a
checkpoint/archive size from update counts.

## Validation and decision

The primary metric is global_wp_clip2_v1 on all 1,873 official validation
sequences and 3,528,273 scored rows. The split is the PR49 chronological fold-0
manifest, SHA-256
0c4de3a253d23b3a5d5ca73b7aeb7fcae2b974673e4f9ec51e94e593c59d114c.
All 10,325 training sequences are fitting membership; valid fitting membership
is zero. Validation is exposed development evidence, not a fresh holdout.

The primary endpoint is the configured final epoch E4. All four epoch
checkpoints may be retained for diagnosis; they do not replace or select the
E4 endpoint. Report both seed-specific paired deltas. A descriptive two-seed
mean is not a significance test or reliable estimate of training variance.
Any component bootstrap is conditional on the trained fits, with 2,000 draws
and RNG seed 271828; it describes evaluation-unit sensitivity only.

A separately reviewed follow-up may be proposed only if mean K16-minus-K1 WP
is at least +0.0005, both seed deltas are positive, and K16-minus-plain is
nonnegative for both seeds. This is an allocation rule for another review, not
an automatic launch. A failed or incomplete cell is preserved and diagnosed;
no automatic retry is implied.

## Current evidence context

The following linked GitHub PR comments are context for the baseline and nearby work,
not evidence for this stem:

- [PR69, comment 6714306711](https://github.com/VladKochetov007/WunderFunder/pull/69#issuecomment-6714306711):
  the 9/30-epoch label and mixture endpoints were retired partials. Their
  fixed 90/10 mixture scores were 0.69257549/0.69189968 for label-trained
  candidates and 0.63652906/0.63753590 for complement-trained candidates.
- [PR70, comment 6716525560](https://github.com/VladKochetov007/WunderFunder/pull/70#issuecomment-6716525560):
  four fixed-4E fits completed. Label-trained candidates scored
  0.681669/0.681607 standalone and 0.692531/0.692454 for the fixed 90/10
  mixture. The unscaled-mixture-WMSE-plus-correlation candidates scored
  0.604447/0.612139 standalone and 0.690990/0.690883 for their mixture.
- [PR71, comment 6716252348](https://github.com/VladKochetov007/WunderFunder/pull/71#issuecomment-6716252348):
  six 6E GRU fits produced 24 grades; SWA mixtures ranged 0.692209–0.692304,
  below the 0.6923774248 TL3 reference.
- [PR72, comment 6731395515](https://github.com/VladKochetov007/WunderFunder/pull/72#issuecomment-6731395515):
  six fixed-4E C0/C1/C2 grades were 0.680881/0.682194,
  0.670532/0.672148, and 0.680625/0.681024, respectively. Producer grades
  were complete, while the checkpoint byte audit was still 0/6 pending in
  that comment. This new screen trains its plain controls afresh.
- [PR73, comment 6727640776](https://github.com/VladKochetov007/WunderFunder/pull/73#issuecomment-6727640776):
  scoring-only roles were 0.6914165040, 0.6914152791, and 0.6909489195;
  the fixed blend was 0.6923774248. Its Mamba2/local-attention/direct-head
  proposal was not implemented.

The earlier [PR68 compact 12E report, comment
6714381491](https://github.com/VladKochetov007/WunderFunder/pull/68#issuecomment-6714381491)
is not the fixed-E4 primary endpoint here; its peaks were at E5–E6.

## Packet map

- PR.md: the eleven-section experiment contract and receiver handoff.
- [RUNBOOK.md](RUNBOOK.md): environment variables and exact staged GPU commands.
- [INDEPENDENT_REVIEW.md](INDEPENDENT_REVIEW.md): independent static findings
  and fixes; receiver correctness remains untested.
- recipe.json: machine-readable identities, panel, costs, gates, and config hashes.
- VALIDATION.json: ROOT-owned source-only/static validation record; it is not
  receiver acceptance or model-quality evidence.
- configs/: six frozen JSON copies of the PR72 C0 recipe with only the declared
  seed and stem-kernel differences.
- REPO_UPDATE.md: branch/base, publication, and static-preparation status.
- run.sh: ROOT-owned receiver lifecycle; use bash because the file is not
  required to be executable.
- receiver_data.py and profile_config.py: ROOT-owned source/split admission and
  peak-profile recipe preparation.
- acceptance.py and check_stem_ddp.py: ROOT-owned source/profile admission and
  DDP gates.
- compare_panel.py: ROOT-owned paired comparison and conditional bootstrap.

The primary fit uses the cache-free standalone-label objective. Optional
evaluate-blend/compare-blend actions require the exact PR49 reference cache and
all six blend grades; they are outside the primary screen. No CPU-ready ZIP,
KD result, public score, or quality improvement is claimed.

All four epoch weights are retained under each receiver run root for local
diagnosis. The standard bundle includes the configured final model and resume
state; retaining or transferring intermediate epoch weights requires preserving
the host run directory or a companion archive.
