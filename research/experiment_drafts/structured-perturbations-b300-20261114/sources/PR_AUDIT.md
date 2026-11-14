# Prior perturbation evidence and the new panel's boundary

Read-only GitHub/source audit on 14 November 2026. The table pins observed PR
heads; a head identifies code, while PR bodies/comments can change independently.
Numbers below are reported evidence, not measurements reproduced by this packet.
No models, tests, inference, profiles or training were executed for this audit.

## What is already covered

| Source | Observed head | Relevant intervention and evidence | Consequence for this packet |
| --- | --- | --- | --- |
| [PR59](https://github.com/VladKochetov007/WunderFunder/pull/59) | `57f0a99495c04c0bfc4618673586c59b2b8978c5` | Whole-sequence raw-channel dropout, rate 0 versus .05, fitting-only mean imputation. Four GRU256x2 fits proposed; no quality result in the body. | Do not present ordinary channel dropout as new. Temporal staleness preserves past observed values of a coherent input group instead. |
| [PR79](https://github.com/VladKochetov007/WunderFunder/pull/79) | `67c3bd818ea243177eeb1dc789cd3d145eba0e81` | Causal residual FIR K0/K1/K16 before a GRU, retaining history across chunks. | This is learned temporal filtering, not stochastic stale observations or attention-key removal. |
| [PR80](https://github.com/VladKochetov007/WunderFunder/pull/80) | `dfe8708a0927c8e2ac65525b39ddf799e90138b3` | Successor panel already includes interlayer GRU dropout .1, K64 and increased width. | No new generic interlayer-dropout proposal. Its historical running-state body is not a live execution observation. |
| [PR81](https://github.com/VladKochetov007/WunderFunder/pull/81) | `650c21e19cdaef4fd9605311098eab45f5316e40` | Card08 implements noisy carried-state GRU fork regularization: every fourth eligible chunk, offset128, Gaussian noise .01 times hidden RMS, 32 suffix steps and a consistency term; clean carry remains authoritative. | Input staleness and temporary attention masks are different intervention locations. Do not rename hidden-state consistency as a new experiment. |
| [PR83](https://github.com/VladKochetov007/WunderFunder/pull/83) | `9f01bb0eb9f104f7788ec02aa11d94d303d390de` | Column view-mask rates .15/.30 already screened on cold Muon GRU. Reported p=.15 F0 standalone delta -0.0014, component interval [-0.0039,+0.0005], correlation .994; pair-minus-control +0.00025 below the declared +0.001 bar. TabM and NorMuon also failed their screens; epoch averaging was near zero. | Preserve the negative. It argues against another undifferentiated column-mask repeat, not against every causal corruption or a Transformer experiment. |
| [PR86](https://github.com/VladKochetov007/WunderFunder/pull/86) | `742aaacc9a3d4094d93d2f897a6cfe4a42f5f9a5` | Reported train/valid adversarial AUC .984 versus shuffled .499 and cross-partition time links in both directions. Leave-regime-out builder/fits remain a proposal in the body. | Shift motivates robust representations, but these perturbations are not demonstrated models of the public/private shift. Official valid is not a new holdout or simply a known later period. |
| [PR88](https://github.com/VladKochetov007/WunderFunder/pull/88) | `d7aca22297bea1c276468007d8371f92380e4c65` | Limited PR81 execution: eight partial Transformer fits, not all 40 planned cells. E3 numerical PLE on seed204101 improved +0.000303410; scheduled E4 was absent. | Do not substitute partial endpoints or treat this as a result of hidden-state perturbation. |
| [PR90](https://github.com/VladKochetov007/WunderFunder/pull/90) | `34b2beaf165eb7a224b792c17b99e41b5fd7384a` | Signed versus additive attention; conditional versus static FFN routing. Prepared ten-cell panel. | Changes the learned map or routing; this packet instead tests training-only removal of available historical keys. No numerical result is borrowed. |
| [PR91](https://github.com/VladKochetov007/WunderFunder/pull/91) | `190d04be8d722daca598db3efd97728c1c29b32d` | Bilinear i0 x i1 versus additive inputs; entmax versus softmax. Independent E4 exports reported 13 November give bilinear deltas -0.000397023/-0.000074732, mean -0.000235878. Entmax has no result in the audited evidence. | i1 interaction is not staleness robustness. Retain the negative rather than counting early epochs as a successful bilinear result. |
| [PR92](https://github.com/VladKochetov007/WunderFunder/pull/92) | `b107418aef553cb4aecc249ad9cf2c86553d5a1d` | B03 since32 x KD implemented; B09 already proposes SAM with weight perturbation radius .05. | No duplicate SAM or since32 x KD screen. |
| [PR93](https://github.com/VladKochetov007/WunderFunder/pull/93) | `68e5b27066c5bc6d34a1e34f4c8b00593a16333b` | Event-clock, residual, tied-input and other backlog cards. Explicitly discusses mirror and online-normalization predecessors and their limitations. | No sign reversal, time reversal or label-preserving mirror is assumed here. |
| [PR97](https://github.com/VladKochetov007/WunderFunder/pull/97) | `f6207b385766f5d5241069dcf117d507f61e2375` | Prepared 6x128 versus 12x64 attention-head comparison at fixed width; no fitting result. | Same native causal Transformer family is reusable, but a different architectural mechanism. |
| [PR98](https://github.com/VladKochetov007/WunderFunder/pull/98) | `fcde99f1dafb2af077d4d950e433e77f479807a4` | Cold Muon GRU teacher mixtures, future-teacher targets, strict-precision correction and actual submission evidence. Reported since32 x KD interaction changes sign. | Neither establishes a causal input corruption benefit nor supplies this packet's controls. Future-teacher targets are not future inputs for this panel. |

## Original source anchors

- [PR81 exact card08](https://github.com/VladKochetov007/WunderFunder/blob/650c21e19cdaef4fd9605311098eab45f5316e40/research/experiment_drafts/impact-portfolio-20261111/cards/08-state-perturbation.md).
- [PR83 view-mask evidence](https://github.com/VladKochetov007/WunderFunder/blob/9f01bb0eb9f104f7788ec02aa11d94d303d390de/research/campaign_2026-10-29/view_mask_r1/README.md). The PR reports these folders copied from producer `0ea63b982`; this audit does not regenerate them.
- [Legacy M04 idea inventory](https://github.com/VladKochetov007/WunderFunder/blob/7e4074314a62d0907c9608f9220cf72958708a2d/research/unified/ideas/legacy/muse.txt) already lists affine augmentation, masking, jitter, residual shifts, block bootstrap, tail crop and causal time warping. These are broad proposals, not evidence that the present arms were run.
- [S38 symmetry diagnostic](https://github.com/VladKochetov007/WunderFunder/blob/7e4074314a62d0907c9608f9220cf72958708a2d/research/unified/ideas/S38_MIRROR_SYMMETRY_DIAGNOSTIC_R1.md) records limited diagnostics and criticism of their interpretation. It does not establish a legal target transformation or refute every population symmetry.
- [Native runtime pin](https://github.com/VladKochetov007/WunderFunder/tree/8626000a4b965763a70011a0291a891aa6535d7e), inherited from PR75. Its recipe is kept fixed; preparation is not numerical acceptance on B300.

## Frozen distinction of the new screen

The new arms are `clean`, `stale_iid`, `stale_burst`, `history_iid` and
`history_burst`, each on seeds208401/208402. Ten new fits comprise two related
six-fit idea screens sharing the same two fresh clean controls. There are no
teacher/student fits, no imported old controls and no granted runtime allocation.

Input arms replace exactly4992 of20000 raw i1 rows per whole training sequence
with the last observed value of the same sequence; row0 is untouched. The burst
arm selects39 of156 full128-row blocks. The independent arm selects the same
number of eligible rows. This is a fixed25%-approximately corruption budget
(4992/20000 = 24.96%), not equality of information loss: adjacent selected blocks
may merge, and bursts have much older effective observations. Burst-minus-IID
therefore compares two realistic missingness policies, not a pure temporal
topology effect at matched age. Return realized hold-age/run-length summaries.
The current i0, targets, loss mask and loss weights remain canonical; receiver
checks must verify the raw feature-group interpretation and causal assembly.

Attention arms temporarily remove floor(25% of eligible historical cached keys)
before softmax. The latest32 cached keys and the whole current chunk retain
their native causal eligibility. An independent exact-size subset is compared
with a uniformly rotated contiguous block in the eligible-key index ring. Both
give the same marginal removal chance by eligible key age. A wrapped block is
two chronological intervals, not always one actual outage. Marginal masks do
not equalize softmax attention mass or information loss. Cached tensors are
unchanged; removed keys may be accessible on a later forward call. This is
training-time attention access corruption, not a missing-history simulator.

Neither family adds parameters or serving operations. Clean inference and
parameter-schema parity remain claims to check on the receiver, not results
established by reading the code. Unknown target invariance is explicit: less
information can make valid labels harder to predict, and an adverse endpoint is
a meaningful result. No target-based perturbation, time reorder, future row or
future-label gradient is used to create inputs.

## Interpretation and independent design review

The fixedE4 clean-valid comparison uses the native PR75 recipe, identical
initialization for shared parameters, data/split/normalization, ordering,
optimizer and schedule, with separate checkpointed perturbation RNG. Full
official valid has already been inspected many times. Two fitting seeds are
two repetitions on one exposed split, not two independent folds.

The predeclared allocation trigger is mean WP gain at least+0.001 against clean
and positive gain for both seeds. A positive mean structured-minus-IID contrast
supports prioritizing the structured policy; it does not establish a mechanism
independent of staleness age, realized attention mass or training variance.
Report both seeds, both targets, the four versus-clean contrasts and the two
structured-versus-IID contrasts even when negative. Selection of a better epoch
or changing the corruption strength after inspecting grades is a new experiment.

Component bootstrap must use the same sampled source-qualified validation
components across all ten cells and pool sufficient metric statistics before
computing the official nonlinear WP. The implemented readout reports per-seed
intervals and the descriptive mean of two paired seed deltas. If a mean-delta
interval is added in a separately declared analysis, average the paired seed
deltas within each common bootstrap replicate. Such an interval would still be
conditional on these fitted seeds and the exposed split; it would omit fitting-seed
and distribution-shift uncertainty. No Holm or formal p-values are produced here. Four
versus-clean tests can receive Holm adjustment only with a documented valid
bootstrap null-tail construction. Counting positive uncentered bootstrap deltas
is not automatically a calibrated p-value. Report the two mechanistic contrasts
as exploratory unless their testing family is explicitly enlarged in advance.
Holm correction does not repair earlier adaptive reuse of official valid or
establish a population-level discovery from two fitting seeds.

Source-bound receiver checks must cover full/chunked causal behavior, prefix
invariance, row0/sequence reset, exact corruption counts, clean/eval identity,
mask nonemptiness, unchanged cache contents, equal initial common parameters,
and uninterrupted versus resumed perturbation masks. These checks and actual
per-configuration B300 profiles precede bounded, owner-assigned fitting; they
were not run locally. Existing pipelines and running jobs remain untouched.
