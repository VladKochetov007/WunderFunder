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
| E470/E472 | 48 | Eight activations × two optimizers × three seeds | Sine-minus-GELU positive at all six seed/optimizer comparisons; optimizer rankings change with seed. |
| E473 | 10 | Sine versus linear addition, scaling, approximate GELU | Matched linear addition ties or beats sine at one seed; no periodic-mechanism claim. |
| E471/E475 | 18 | Bounded candidate shape × optimizer × three seeds | Positive alpha improves Muon at all three seeds; AdamW effects vary. |
| E474 | 12 | Embedding versus head placement | Embedding effect reverses across seeds; head effects near-neutral. |
| E476 | 12 | Activation × optimizer × LR protocol | First-seed AdamW warm-cosine advantage; failed replication in E480. All 22 registered contrasts retained. |
| E480 | 24 | Two additional seeds for the full schedule factorial | All eight new warm-cosine-minus-constant-mean effects negative; six intervals wholly negative. All 44 new contrasts retained. |
| E477 | 12 | Parallel/same-channel GELU–LeakyReLU mixture × dropout | Effects reverse with seed; all12 official scores and18 contrasts retained. |
| E478 | 6 | Candidate shape in H64 two-layer topology | All three point deltas positive; two conditional intervals positive, one crosses zero. |
| E479 | 16 | Learned per-layer candidate amplitudes | Learned-minus-fixed effects are small and vary by seed/optimizer; fixed positive shape remains favorable for Muon. All28 contrasts retained. |

Overall212 scientific fits have completed;all212 have independent official grades. Each
completed activation fit finished without a new OOM. Actual attempt receipts,
failed probes and rejected grader checks remain in the record. Component
intervals condition on fitted models and exclude training-seed/search
uncertainty. Independent export checks are recorded for E474/E476/E478. A separate shifted-raw-sum implementation also reproduced every global score and contrast for E477/E480 to within3.4e-15 (checks/activation_export_reproduction.json).

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

| Study | New fits planned | Current state |
|---|---:|---|
| E479 | 16 | Learned per-layer bounded candidate amplitudes; independent source/CPU/CUDA probe passed; all16 full fits independently graded and all28 global contrasts reproduced with a separate numerical method. |
| E480 | 24 | Full activation/optimizer/schedule factorial at two additional seeds; all24 full fits independently graded; separate raw-export reproduction matched all24 scores/all44 contrasts. |
| E481 | 18 | All18 fits independently graded and all30 contrasts reproduced from raw exports. Sine and cubic both help Muon at all three seeds; every cubic-minus-sine Muon interval crosses zero. |
| E482 | 6 | Same six seed271828 recipes on two Kaggle T4s; first private kernel failed worker argument forwarding before GPU probes; repaired v2 completed both native GPU probes and all six fits on distinct T4s; all six independently graded and all ten contrasts reproduced with separate shifted raw sums (max error1.15e-14). |
| E483 | 18 | Fresh-seed shape × recurrent Muon momentum; source and fit-only CPU proof independently reviewed; actual CUDA probe and all18 full fits completed cleanly; independent grader v1 failed its source-map contract before row loading, and both grader contract incidents are retained; v3 graded all18 fits and root reproduced all27 contrasts (max1.12e-14). All nine sine-minus-baseline intervals are positive; momentum ranking reverses with seed. |
| E486 | 0 | Registered CPU-only frozen E483 checkpoint training-shape × inference-shape counterfactual:36 score cells/54 contrasts; blind critic completed; implementation underway. Requires exact18 diagonal predictions and bootstrap reproduction before accepting off-diagonal effects. Zero new scientific fits. |
| E485 | 6 | Registered H128 four-epoch baseline/sine trajectories at three fresh seeds; actual CUDA probe failed native-Muon update peak gate before full fitting (1.7153% RMS difference but6.85e-5 peak exceeds preregistered2e-5). No tolerance widening;0/6 full fits admitted; failure preserved. |
| E484 | 12 | Causal context-conditioned GELU/LeakyReLU readout; blind critique resolved in design v2; actual CUDA probe v1 caught a CPU/GPU initialization-check mismatch before first update; v2 independently reviewed, actual CUDA probe passed all12 cells and all twelve full fits completed cleanly; all12 independently graded and all16 global/32 target contrasts reproduced separately (max3.54e-14). No consistent context-gate advantage across two seeds. |

The registered activation program currently contains218 full fits:212 completed (212 graded), with six duration trajectories not admitted after their native-step probe failed. E485 saves18 checkpoints from six continuous four-epoch trajectories; snapshots are not separate fits. Partial checkpoints from an active matrix are reported separately until its completion receipt validates. Do not count one-update probes or source-only packets as completed fits. E477 has now completed its corrected independent grade. Its
first export grade failed before predictions because its provenance check
expected all40 cohort nodes in a saved training-only24-node receipt. Versioned
corrections and root review preserve that incident; no model result was
inferred from it. Broader learned/contextual mixtures and activation-aware
compression remain separate mechanisms. Additional primary literature is in
[adaptive activation reading](activation_optimizer/ADAPTIVE_ACTIVATION_READING.md).

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
