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
| E471/E475 | 18 | Bounded candidate curvature inside the GRU × optimizer ×three seeds | Positive alpha improves Muon at allthree seeds; negative alpha hurts Muon at allthree. AdamW effects vary. |
| E474 | 12 | Sine/slope change localized to embedding versus head | Embedding effects reverse across seeds; head effects near-neutral. All14 registered contrasts retained. |

All106 officially graded fits finished without a new OOM;118 fits have completed overall, including12 awaiting independent grades. Failed parity checks and the subsequent
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
| E476 | 12 | Two activations × two optimizers × three LR protocols. Allscoresfinished: activationrank depends on schedule; warmup/cosinewinsone-seedAdamWcomparisons. |
| E477 | 12 | GELU, parallel half-width GELU/LeakyReLU readouts, or same-channel mixture, with and without head dropout, at two seeds. |

E475 and all36 E474/E476/E477 fits have finished. E474 has12officialscores and14jointcontrasts; E476 has12officialscores and22contrasts; E477 independent grades are pending. All6 E478 larger recurrent scaling fits finished; its independent CPU grade finished with6scores/3contrasts. The current completed-fit total is118, with106officiallygraded. E479 registers16new learned-shape comparisons before implementation: two scalar bounded amplitudes, one per GRU layer, fixed/learned controls, two initializations and both optimizers at two seeds. These are designed arms, not completed evidence.

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

Precision clarification: common FP32/IEEE controls bind recurrent forward/backward arithmetic and the usual CUDA/cuDNN flags. The existing Muon optimizer deliberately converts its Newton–Schulz orthogonalization matrices to BF16 on CUDA (`src/wnn/optim.py:orthogonalize`). That unchanged optimizer implementation is common to the Muon arms; descriptions of FP32 recurrent controls must not imply every internal optimizer operation is FP32.
