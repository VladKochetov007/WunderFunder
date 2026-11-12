# B300 wave 2026-11-12

This receiver packet carries PR81's ten implemented mechanisms into a bounded
B300 campaign. The original packet at
[`impact-portfolio-20261111`](../impact-portfolio-20261111/README.md) remains
unchanged. This copy records the Transformer placement amendment and the fit
roster in [`WAVE_20261112.json`](WAVE_20261112.json).

## Current readiness

The canonical PR75 placement profile is documented in
[`qualified_profile/`](qualified_profile/README.md). It compares one B300 at
batch 256 with two B300s at local batch 128, holding the global sequence group
per optimizer update and remaining PR75 recipe fixed. This engineering profile
uses the canonical source without PR81 adapters. Its receiving measurements and
admission decision are recorded by the campaign operator.

For mechanism cells, the packet reuses PR81's isolated Transformer and GRU
adapters. It pins PR75 for cards 01?07 and PR79 for cards 08?10, with existing
prepare, receiver-check, per-cell profile, approval, fit, export, score and
comparison stages. This local packet claims no model execution, receiver check
or quality result. Use [`B300_COMMANDS.md`](B300_COMMANDS.md).

The inherited `IMPLEMENTATION*.md`, `recipe.json`, static review JSON and MIDAS
route previews record PR81's earlier implementation and evidence status. They
remain historical records; the current placement and queue are defined by this
contract, the wave manifest, and the receiver commands.

## Fitting roster

All ten mechanisms remain selected: two new controls and two new treatments per
card at seeds 204101 and 204102, 40 planned fits total. Cards 01?07 use one
B300, batch 256, accumulation 4, PR75 BF16/BF16x9 forward math, FP32 master
parameters and optimizer state, LR `3e-4`, and fixed E4. Cards 08?10 use one
B300, batch 32, accumulation 1, PR79 FP32/AdamW/K16 and fixed E6. No teacher or
pretraining fit is required, and no old result is reused as a control under the
changed Transformer placement.

The first cohort is cards 01?02 at both seeds: eight cells covering two
complete paired screens. Cards 03?04, 05?06, 07?08 and 09?10 follow as complete
two-seed screens in the frozen five-stage order. Card 08 tests a noisy carried
state branch; card 10 tests raw-label tail classes. Both are distinct from PR83's
input view-mask and event screens and remain in the queue. Card 10 has a fixed
train-only tail-prevalence gate.

Every selected cell requires its own accepted 50-update profile. Inspect actual
update time, peak VRAM, host RAM and reload checks before recording profile
approval. A profile does not choose a scientific arm. Fits use the external
shared cutoff, complete per-epoch diagnostics and full official validation.
There are no automatic retries.

## Validation and evidence

The primary protocol is the supplied official **train | valid** file boundary:
10,325 whole fitting sequences and 1,873 independent validation sequences.
Normalizers, PLE knots, teachers, initializers and other learned preprocessing
use training members only. Score all 3,528,273 eligible validation rows with
`global_wp_clip2_v1` after every epoch and at the fixed final E4/E6 endpoint.
Validation has prior exposure and remains development evidence.

Sequence identifiers are shuffled and are not timestamps. The March 5
Predictorium statement supplies the adopted protocol; it does not independently
prove current Connectome timestamps. Additional chronological partitions need
source-qualified order evidence. Keep old folds and frozen recipes labeled with
their original protocol.

The PR81 base head is `650c21e19cdaef4fd9605311098eab45f5316e40`. Runtime pins
are PR75 `8626000a4b965763a70011a0291a891aa6535d7e` for cards 01?07 and PR79
`67c3bd818ea243177eeb1dc789cd3d145eba0e81` for cards 08?10. Receiving
sparse Git worktrees must pass exact runtime admission, data/split checks,
family receiver checks and each selected cell's profile before fitting. A tar
archive without the pinned Git identity cannot pass runtime admission.
