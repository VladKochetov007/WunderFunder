# Muse Spark 1.3 fresh-context model/scale reframe

Role: independent proposal generator. Use the entire task below; do not inspect the repository, call tools, use web search, or write code. This is ideation only. The prompt intentionally omits current candidate results and branch arguments.

## Research objective

We need to improve a sequence-based supervised time-series competition model enough to pursue top-1 and a public score above 0.700. Treat that as a serious target, not a limit imposed by today's small gains. The data consists of grouped sequences with temporal structure and two prediction channels. The official score is a fixed global weighted squared error with predictions/labels clipped to [-2, 2], original-label magnitude weights and a fixed scored-row mask.

All weights must be trained independently; teammate checkpoints cannot initialize or teach our models. Supplied competition data and applicable data/model rules remain fixed. Training and research need not obey submission-size, CPU-only inference, latency, quantization, architecture-size, or packaging limits. A large or non-deployable model may be useful as an independently trained teacher or research instrument if it can later be distilled into an admissible model. Local compute is bounded and only available until 10:00 Europe/Kyiv today; after that, GPU/TPU experiments run on Kaggle. Kaggle GPU/TPU compute is authorized within campaign quota and admission. Do not presume new paid compute.

## Random public-literature source

The reproducible sensor-derived seed is `8607148522861116446`, made from the first 64 bits of SHA-256 of this machine's `sensors` output. Python `random.Random(seed).choice` selected item 2 from the ordered three-source public pool:

1. Oren et al. (2019), “Distributionally Robust Language Modeling,” https://arxiv.org/abs/1909.02060
2. Holland & Haress (2021), “Learning with risk-averse feedback under potentially heavy tails,” https://proceedings.mlr.press/v130/holland21b.html
3. Singh et al. (2020), “Improving Robustness via Risk Averse Distributional Reinforcement Learning,” https://proceedings.mlr.press/v120/singh20a.html

The selected paper studies CVaR when losses and gradients may be heavy-tailed. It develops a CVaR estimator under finite-variance and local quantile regularity assumptions, then robustly selects among candidate models produced by stochastic-gradient processes; it includes a regression application. This does not establish that our score or gradients are heavy-tailed.

## Assignment

Return eight distinct experiment cards spanning all of these areas:

1. At least two transfers from the paper's robust selection/CVaR estimator or heavy-tail analysis.
2. At least three architecture/model-family ideas, including a plausible scaling-law experiment. Consider convolution/state-space/attention/hybrid recurrent models, multi-resolution, latent-state or probabilistic/ensemble teachers. Avoid name-dropping: state mechanism, data fit and compute ladder.
3. At least two ideas that explicitly exploit relaxing deployment constraints during training (larger teacher, longer context, more precision/ensemble, auxiliary modalities only if rules permit), with a path to an admissible student or a clear reason to keep it research-only.
4. At least one idea that combines a weaker/older model as a complementary teacher, residual, representation or disagreement signal rather than discarding it by standalone score.

For every card specify: mechanism family; source basis and analogy; exact intervention and a minimal 2–3 rung scaling ladder where applicable; training-data/split requirements; predicted observable (no invented score gains); cheapest falsifier; main overfit/leakage or compute risk; expected resource class (local CPU/GPU before cutoff or Kaggle GPU/TPU after); output artifact; and what result would justify the next rung. Do not claim source results transfer automatically. Keep the objective as the official weighted global metric; tail/group metrics may diagnose or regularize, not silently replace it. Distinguish rule constraints from submission-engineering constraints and do not propose prohibited data or reused teammate weights.
