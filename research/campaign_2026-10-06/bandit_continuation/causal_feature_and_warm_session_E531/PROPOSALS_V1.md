# Causal feature discovery and warm research sessions

Status: registered proposals, not fitted models or measured gains. Basis: user requests for TSFEL/tsfresh-like missing heuristic features, persistent warmed-up sessions and faster diverse experiments. Controller-authored; no external model proposal is claimed. Scientific recipes and resource admission remain separate.

## Existing basis and overlap

`src/wnn/features.py` already supplies causal differences, event flags, recency and last-event price deltas (99 event features beside 112 raw channels). `src/wnn/research_features.py` supplies volume movement/path summaries in `volume28` and `volume28_current`. This audit covers those two modules, not every historical experiment. H235 already proposes learned EMA banks; H244 robust fast/slow statistics; H222 sliding DFT; H229 causal scattering; H247 spectral adaptation triggers. Reuse and extend those nodes instead of renaming them as discoveries.

Primary references inspected on 2026-10-08: [TSFEL configuration and feature families](https://tsfel.readthedocs.io/en/latest/descriptions/get_started.html), [tsfresh rolling windows](https://tsfresh.readthedocs.io/en/latest/text/forecasting.html). The transferable components are configurable feature subsets and trailing-window construction, not whole-sequence feature vectors. Their usefulness here is a hypothesis. Current-row inputs may be used only when the callback supplies them; never include future rows. Frequency means row/event-index frequency unless physical sampling time is established.

## Hypothesis cards

| ID | Direction and first discriminating test | Dependencies and possible descendants |
|---|---|---|
| H-248 | Warm bounded research cohort: compare cold-per-job and shared verified-data sessions on identical recipes. Measure upload/hash/decode/startup time, useful fit time, RAM/page cache and dollars per completed artifact. | H213/E528 input reuse. Immutable job manifests, exact data/stat binding, isolated model/optimizer/RNG/state per job. Success permits distinct approaches on two GPUs sharing read-only decoded input. |
| H-249 | Trailing-window distribution shape beyond existing event deltas: small declared channel subset and window ladder, dispersion/asymmetry/tail/occupancy summaries. Compare raw/event baseline, existing fast/slow statistics and added window-shape features with matched model/update budget. | H244/H235. Start with bounded CPU reference and streaming parity, then one own teacher fit. A useful teacher may contribute to H230 even if its standalone score is lower. |
| H-250 | Causal lag and cross-channel coupling: signed change alignment, lagged autocorrelation and cross-group correlations on declared windows. Compare event-only control, lag-zero and lagged variants. | H249 and existing event features. Train-only channel/lag selection; conditional low-rank readout H233 or two-stream teacher is a distinct interaction successor. |
| H-251 | Bounded nonlinear morphology: change burstiness, zero-crossing/run structure and selected complexity summaries from TSFEL/tsfresh-style candidates. Compare against distribution-only features and equal-dimensional random nuisance channels. | No blanket all-feature extraction. Start on a tiny deterministic subset, cap per-channel runtime/state, retain expensive features as teacher-only candidates with compression path. |
| H-252 | Causal feature regime context × complementary own teacher: retain raw context plus innovations and window statistics, and compare the full fixed-student endpoint against each parent and a fixed-mixture control. | H238/H249/H250/H230. Measure contribution to residuals, targets or blend rather than reject on standalone ranking. Freeze interaction and ancestry before selecting weights. |

Spectral/scale extension belongs to H222/H229, with H247 as an adaptation interaction. First test a compact trailing-window spectral bank against existing sliding-DFT/scattering implementations where available; do not assume those proposals have been fitted. Keep event-clock and row-clock variants explicit.

## Causality, validity and cost checks

Prefix isolation, sequence reset, chunk-boundary equivalence and batch/streaming numerical parity precede model fits. Fit feature selection/normalization on the fitting groups only. Exposed validation stays exposed; new feature selection increases selection exposure. Record source/config/channel/window hashes and feature computation costs. Do not change the official metric or evaluation clipping.

Use TSFEL/tsfresh as bounded discovery/reference tools, with explicit worker limits (initially one CPU worker). Avoid materializing every rolling window across all rows/channels. Stream or shard caches; record RAM, page-cache and state size. Save actual EDA plots to this directory's `plots/` through marimo-pair when EDA is performed; no fabricated or empty plot evidence.

## Persistent session contract

Warm data means reusable identity-checked read-only data, not warm weights or reused optimizer state. Each job has a frozen recipe/source hash, own outputs and terminal receipt. Admit only within remaining runtime, quota, host memory and aggregate paid budget. Reuse a content digest only when the verified producer/consumer binding remains valid; replace or rehash changed inputs. Failed jobs preserve diagnostics and cannot contaminate successors.

Vast reuse requires our durable creation receipt, explicitly admitted cohort successors and a bounded idle allowance; never adopt another instance. Kaggle production may use a predeclared multi-experiment notebook rather than assume notebooks can receive arbitrary new commands. Avoid indefinite paid waiting: collect and tear down our instance when no useful admitted successor is available. The current frozen H217 two-fit cohort already amortizes staging; this new direction adds systematic timing and safe decoded-data/session reuse without silently rewriting that packet.

## Bandit allocation and stopping

H248 is useful systems work enabling scientific throughput, not a competition-score observation. H249–H251 enter protected independent exploration; H252 is an explicit interaction. Numerical priors and expected public gains are null until measured. No confidence interval is an automatic kill switch. Defer an exact expensive feature subset for opportunity cost while retaining cheaper, teacher-only or interaction descendants. After an informative terminal result, update measured quality/cost and select a distinct mechanism or concrete child; do not cycle on uninformative feature counts or seeds.

Next owners: controller freezes one bounded reference recipe; independent critic verifies causality/cost; actor owns implementation through fit/export; grader compares frozen endpoints. Provider diversity and directed/random-literature scouts may propose further cards with recorded passages and selection seeds. They do not retrospectively create evidence for these controller proposals.
