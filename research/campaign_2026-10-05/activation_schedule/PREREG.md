# E476: activation × optimizer × learning-rate protocol

## Scope and admission

This is a source-only draft. It must not be used to launch fits until the root
reviews the source, content-binds the inputs and commands, runs the actual-data
CUDA shape probe, and seals its own scorer. The implementation imports the
immutable E474 H64 two-layer GRU and activation classes and E470 fit-only cohort,
causal features, normalization, weighted objective, and finite guards. It does
not inspect any test row group or test outcome during fitting.

The fixed 2 × 2 × 3 factorial has 12 cells: `base` (GELU at embedding and
head) or `sine_embed` (GELU + 0.1 sin(x) at embedding only); AdamW or MuonAdamW;
and one of three deterministic LR protocols. All cells use seed 271828, the
same 24 fit row groups and shuffle, 4 sequences × 256 rows per update, 474
updates, E470 target-absolute-weighted MSE with `need_prediction` mask, global
gradient clip 1.0, weight decay .01, and betas (.9,.999). No dropout, model
placement, capacity, initialization, or split changes are introduced. E474's
live baseline count is 61,058 trainable parameters in every cell.

MuonAdamW applies Muon only to `gru.weight_ih_l0`, `gru.weight_hh_l0`,
`gru.weight_ih_l1`, and `gru.weight_hh_l1`, each split into 3 GRU gate blocks.
Input projection, head parameters, and biases remain on AdamW. The E470 loss
and all masks are unchanged. State resets at each B4 batch and is carried and
detached across its 79 chunks, exactly as E474 specifies.

## LR traces and comparisons

The full 474-element traces are written before any optimizer update to
`lr_traces.json`, with exact values, arithmetic sum/mean, and hashes. The
protocols are: constant .001; 47 updates linearly inclusive from .0001 to
.001 followed by 427 cosine updates inclusive from .001 to .0001; and constant
at the arithmetic mean of the exact 474-value E476 warm-cosine trace. The trace
file records all values, sum/mean, hashes, and the unchanged E474 source trace
for reference. E474's source trace is constant .001; it defines the
architecture/initialization source here, not the mean-LR arm. The E476 constant
mean is about .00055 and has the same arithmetic LR sum as the warm-cosine
arm. Thus warm-cosine versus constant-mean compares LR shape at matched
arithmetic LR area, not matched optimizer-update area. Constant-peak compares
against a larger average LR and tests the higher LR budget/amplitude protocol.
All 12 cells are distinct recipes.

At sampled zero-based updates 0, 46, 236, and 473, save pre-clip global gradient
norm and fit loss plus per-family parameter-delta RMS after the optimizer step
(input projection, recurrent core, hidden head, output head). Telemetry describes
training dynamics only; it is not a selection rule. No evaluation-based
adaptation is allowed.

## Contrasts and interpretation

Retain the complete 12-cell matrix. Report activation and optimizer contrasts
for each LR label. The warm-cosine versus constant-mean comparison estimates a
shape effect at matched arithmetic LR area, while constant-peak compares a
higher mean LR protocol. Any activation × optimizer result is conditional on
the one fixed schedule/seed/cohort and does not establish a generally superior
optimizer or activation. This exposed cohort is exploratory; exposed scores
must not be used for cell selection.

The preflight loads exactly the 24 fit groups and makes one real fit-only
B4×256 update for every factorial cell. Its CPU invocation is part of source
validation. The optional `--device cuda --preflight-only` invocation is a
callable shape/update probe only; it has not been run by this author and does
not authorize or start full fits. A future full run must finish all 12
preregistered cells before root-owned grading.
