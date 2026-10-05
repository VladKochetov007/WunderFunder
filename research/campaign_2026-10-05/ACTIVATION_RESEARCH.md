# Activation research: evidence and discriminating experiments

The public anchor remains the complete seed4 combo at **0.6660**. The user’s
new milestone is **0.7000 honest expected public score**. None of the activation
pilots establishes a public gain. These models use an exposed diagnostic panel
and much less capacity/data than the submitted predictor.

The requested [Mining Generalizable Activation Functions](https://arxiv.org/abs/2602.05688)
motivated GELU-plus-sine and GELU-times-sinc candidates. Our first exact PyTorch
GELU implementation differs from the paper's default approximate JAX GELU;
E473 tested that approximation separately. Sources and transfer caveats are in
[the literature notes](activation_optimizer/LITERATURE.md).

## Completed evidence

| Study | Completed fits | Question | Finding |
|---|---:|---|---|
| E470/E472 | 48 | Eight activations × two optimizers × three seeds | Sine-minus-GELU is positive at all six seed/optimizer comparisons, but rankings and optimizer advantages change with seed. |
| E473 | 10 | Sine curvature versus linear addition, scaling, and approximate GELU | GELU+0.1x matches or exceeds GELU+sine at one seed; the Muon difference is +0.00181 with a positive conditional interval. |
| E471 | 6 | Bounded candidate curvature inside the GRU × optimizer | Negative perturbation helps AdamW; positive perturbation helps Muon at one seed. Both signs are retained for replication. |

All 64 fits finished without a new OOM. Failed parity checks and the subsequent
shared FP32 policy are preserved as incidents, rather than removed from the
record. Official pooled and per-target scores were recomputed from raw finite
selected predictions. Component intervals condition on fitted models and do
not include seed or search uncertainty.

## Why these are different mechanisms

Near zero, `GELU(x)+0.1sin(x)` and `GELU(x)+0.1x` have the same value and slope.
Their first sine-residual discrepancy is cubic. Thus a gain confined to small
preactivations need not reflect useful periodic structure. E473 provides an
actual local control; activation-occupancy plots provide context, not proof of
the cause. Larger models and different preactivation distributions could still
make curvature useful.

The recurrent candidate is `g(u)=u+alpha*sin(pi*u)` for `u=tanh(v)`. It fixes the
endpoints and stays monotone for the tested `alpha=±0.1`, since
`g'(u)=1+alpha*pi*cos(pi*u)>0`. Positive alpha increases the central slope and
reduces the endpoint slope; negative alpha reverses that redistribution. This
does not establish stable gradients through a complete recurrent trajectory.
Reset/update sigmoid gates and the reset-after hidden-bias convention stay
unchanged. The optimizer crossover is an observation, not a diagnosed cause.

## Registered or prepared continuation

| Study | Fits | Change isolated |
|---|---:|---|
| E475 | 12 | Both recurrent perturbations and both optimizers at two additional seeds. |
| E474 | 12 | Sine versus matched linear slope in the input embedding, output head, or both, at two seeds on the incumbent's two-layer topology scaled to H64. |
| E476 | 12 | Two activations × two optimizers × three LR protocols: constant peak, warmup/cosine, and constant at the latter trace's arithmetic mean. |
| E477 | 12 | GELU, parallel half-width GELU/LeakyReLU readouts, or same-channel mixture, with and without head dropout, at two seeds. |

E474/E476/E477 are prepared source studies until independently reviewed,
sealed, and passed through real-device probes. Prepared arms must never count
as completed evidence. E475 began serial GPU replication after its unchanged
shape/policy had passed the preceding real-device gate and its new seed/order
wrapper passed independent CPU checks. The continuation totals 112 fits if
all four additional studies are admitted and finish; 64 were complete when
this overview was written.

Matching LR sums does not match actual optimizer update magnitudes. Matching
parameter counts does not match functional capacity or regularization. Fixed
gradient/update RMS telemetry helps describe these differences without
adapting the recipes to exposed test results. Mixed readout/dropout interaction
intervals must jointly resample the same components and recompute each pooled
correlation; global WP is not a weighted sum of class correlations.

The local worker remains below 5 GB VRAM with a one-worker GPU limit and an
aggregate 12 CPU / 30 decimal GB RAM cap, with headroom and zero swap. Existing
device-hour budgets remain in force. Admission stops at 07:00 UTC / 10:00 Kyiv;
already admitted workers then finish gracefully. Every result, source hash,
resource receipt, and failure belongs in the campaign ledger and plots in its
study's `eda/` directory and the live marimo notebook.
