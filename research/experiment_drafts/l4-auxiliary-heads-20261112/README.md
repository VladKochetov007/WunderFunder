# PR82 #04: structure and next-change auxiliary heads

## Question and comparison

Does joint supervision for current outcome structure and the next target-change
event improve the causal GRU's primary weighted-Pearson validation score?
The treatment adds a 19-logit structural head (spread bin, branch, and sequence
class) and a 2-output next-change head to the normal 2-output forecast head.
The treatment coefficients are 0.3 for each objective. Both heads share the
GRU's causal recurrent representation. The screen compares two fresh matched
seeds against same-source raw K16 controls. Because the treatment enables both
heads together, this screen does not identify either head's separate effect.

The structural objective follows PR49: three unweighted row-level cross-entropies
are summed, weighted by the training mask, and divided by selected training rows.
The next-change objective follows PR49: squared error for `(log1p(rows_to_change) -
3) / 2` plus binary cross-entropy for the sign of the change, weighted by
`fit_mask & has_later_change` and divided by those selected rows. Each denominator
is reduced across ranks and all chunks in an optimizer window before backward.
This preserves the existing fixed-window DDP scaling and does not let an empty
tail-event subset create a zero-valued validation metric.

## Data and training contract

Configs are based on the clean PR80 K16 recipe at commit
`6b10dfe71fb0ebec9b0fc76788e2de0fd13d1dba`, with identical same-source controls
so the model source and all non-treatment settings match within each seed.
Each fit starts with a fresh GRU and fresh normalizer. Only the 10,325 whole
official training sequences contribute to normalization or any objective labels;
no validation group contributes to fitting, initialization, or label ancestry.
The full 1,873-sequence chronological validation set (3,528,273 scored rows) is
evaluated after every epoch with the unchanged two-target predictor ABI. This
validation set is exposed development data. `seq_ix` is not chronological.

The recipe keeps six epochs, GRU256x2, causal K16 input stem, AdamW, strict FP32,
`warmup_cosine_v2` with 0.1 floor, all rows after warmup for fitting, all-epoch
checkpoint retention, and a fixed E6 endpoint. No teacher or initializer is
used. The canonical endpoint and every epoch monitor export only the original
two forecast outputs; auxiliary logits remain training-only. An L4 profile for
each of the four exact config/source cells is required before ROOT can admit fits.

## Validation performed

- Focused equation tests compare structural targets and next-change labels with
  the PR49 definitions, then verify the legacy loss arithmetic independently.
- A tiny CPU backward check confirms gradients reach the forecast and both
  auxiliary head slices. Prediction export is checked to retain shape `[rows, 2]`.
- Existing sequence loop, since-event, and canonical GRU/Transformer tests pass.
- `scripts/check_training_structure.py` passes the module/function budgets.
- No native profile or scientific fit was launched by this packet builder.

The pinned source, configs, split identity, and fixed CLOCK_R1 digest are recorded
in the companion artifact packet. ROOT owns native profiles, admission, launch,
resource monitoring, and canonical readout.
