# Fold0 matched spread-label screen: independent specification review

Preregistration SHA256 `267bea071f57ca4954cd2aaccf0f3b2760488864a1665eb14e6a3066a5e95695`.
Selection SHA256 `b9864062e3ab706fea03a5c889f3b658afd1610fc35a3672300070c171b3aec6`.
Conditional approval of this bounded label-only comparison after clean controls;
no current result establishes better forecast supervision. Original proxy can
carry useful future-tail information. No live trainer patch or candidate score.

## Frozen intervention

Both arms initialize the exact clean seed3 fold0 parent epoch3 checkpoint,
inherit preprocessing/output dimensions, and run one fixed continuation epoch0
with the same fresh optimizer/OneCycle/EMA, batch32/chunk2000, order/seed3,
loss weights, propensity and successful updates. Candidate replaces only spread
labels with the accepted current-code12-bin label. Control retains the original
proxy label. Both multiply only the spread CE numerator by the exact same
known-code mask; denominator remains original `sw.sum().clamp_min(1e-8)`.
Branch/class CE and main/next losses preserve labels, masks and denominators.
Unknown cached255 labels must be replaced by a safe0..11 placeholder before CE,
then masked; raw255 cannot be passed to cross_entropy. Patch clamps to11 first.
No new inference feature or parameter enters this screen.

The unapplied patch follows this formula. Cached tensors are appended after
propensity, extracted before that tensor, moved by either standard transfer or
the existing all-tensor CUDA prefetcher, and never added to model inputs. Default
mode preserves original supervision. Inherited label options are deliberately
absent from checkpoint continuation options, so two-stage GCLP FT uses original
proxy supervision in BOTH arms. Root must freeze exact stage commands/source
hashes, stage1 and stage2 epoch0 outputs, parent links and FT seed3 before fitting.
The compared replacement pairs are0.5 unchanged seed4FT+0.5 new seed3FT each;
the historical equal seed3/seed4 pair is a separate anchor. This is not the
80/20 new-model blend used by other architecture cards.

## Data and cache boundary

Builder freezes512 distinct fold0 fitting TRAIN components by deterministic
salted component/group hashes before reading labels. Targets t0/t1 are used only
for interior votes on those512 TRAIN groups. Each accepted full22-price code has
modal(class,spread) support>=3/confidence>=.95; class is used only during map
construction, never externally supplied to apply/inference. Preserve all rejected
and conflict votes. Classes are sequence-lattice diagnostics, not allowed oracle
inputs. Interior voting uses nonpositive lattice ticks, spread>=1, raw plateaus
sampled every7rows, then current-code round4/little-endian lookup.

Application reads only current22price features on10535fitTRAIN+1405fitVALID
groups. It never reads evaluation VALID targets/masks/groups. Packed uint8
arrays contain0..11 or255unknown with known=encoded!=255;238800000 payload bytes
(227.74MiB) plus NPY headers, below250MB. Full TRAIN and VALID indexes are source
scoped: equal numeric IDs in different files are legitimate. FIT VALID must be
disjoint from evaluation VALID, and map-fitting components must be disjoint from
evaluation components. Exact groups, row order, map/selection/source/parquet
identities and known counts must be frozen before training. Components used for
selection are split metadata only and never model features.

Reviewer requested strict files/shapes/ranges/duplicates/evaluation rejection;
actor added them to support source and unapplied patch. Loader validation must
precede training cache opens. Final manifest must bind support, patch, proposed
trainer, preregistration, builder, selection, map, all votes, arrays, metadata,
base trainer and exact parent checkpoint. VALID feature-only use is justified
by clean fitting membership; it does not justify VALID target map fitting.

## Independent checks at handoff

`check_spread_aux.py` is an independent harness prepared before actor handoff.
It checks synthetic numerator/denominator values and output gradients against
log-softmax/gather, all-known legacy exactness, all-unknown branch/class
gradient preservation, zero-weight behavior, rejection of changed provenance,
duplicates, dimensions, evaluation membership and invalid self-consistent labels.
It validates actual full cache hashes/ranges and independently replays six raw
current-price groups (first/middle/last from each source) with no targets/masks.
It confirms the base trainer is unpatched and patch passes dryrun. Result will
be recorded separately after final cache/source freeze. These tests do not prove
GPU successful-update parity or FT behavior; root must record those later.

### Handoff result

Independent durable job `c20260930_independent_spread_checks_v1` passed;
`spread_aux_independent_checks_v1.json` preserves six synthetic cases and eleven
failure checks. Loss and gradients match the independent oracle. All-known
legacy and all-unknown branch/class gradients are exactly identical. The
512-component selection was independently rederived;33063 recorded vote sets
reaggregated to6300 accepted codes. This audit did not replay every raw TRAIN
target vote. All full-array hashes/ranges and six raw current-feature group
replays match. The actual nested scope properties were recorded: one core,
MemoryMax2147483648, swap0, wnn-campaign.slice. Analysis3.32s,
peak1755124KiB (1.674GiB), CUDA uninitialized. A harmless tensor-to-scalar
reporting warning is retained in the raw log; numeric checks passed.

`spread_source_commitment_review.json` verifies all19 source commitments;
source-freeze SHA256 `5170aa9ca3bcb7095e5f706064bda3b503db1ec01a02ee6f5d59c25d4d4ab449`.
Final support SHA256 `3aabffc6fc887c0426948e62cbd500679ba3e7a6925e28de9f816fb1da8f9cac`.
Frozen scorer source unchanged. Patch still unapplied. Conditional bounded
screen approval follows these CPU/provenance checks; successful GPU-update
parity, checkpoint parents and both FT-stage records remain required evidence.

No validation checkpoint/blend search. Freeze candidate/control/anchor designs
and provenance before clean-panel scoring; report complete-mask Global WP,
component-bootstrap SD/CI and class/unknown diagnostics with the unchanged
scorer. Positive versus masked comparator and anchor>=.0005 are screen gates,
not public transfer or automatic confirmation permission. Preserve negative
results and scheduler/GPU resource evidence for both arms.
