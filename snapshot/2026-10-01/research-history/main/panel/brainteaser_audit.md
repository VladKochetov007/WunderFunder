# Brainteaser audit: what obvious-in-hindsight structure are we missing? (2026-09-26)

Lens: I treat the competition as a puzzle the organizer set. For each silent assumption I checked
evidence against it. Tier B (cached valid predictions, LB points, code reading). Nothing was trained.
Every compute check ran through `scripts/limited`, single process, 1 core. The first two checks ran
at WNN_CPU_PCT 8 and the last at 6. After the load-down request I started nothing new. Scripts are in the session scratchpad
under `brainteaser/` (`lbfit.py`, `tvals.py`, `unit.py`, `maskpos.py`, `checks2.py`) and in
`post.py`. Incident: my first `chains.py` was written to the shared scratchpad root and may have
overwritten an older scout script of the same name. `chains.npz` is untouched.

Verdicts: **holds** / **broken** / **unknown**. Every metric number below is global pooled WP on full
valid, lambda = weight on unscored required rows (lambda 0 = true mask).

## 1. Metric fidelity

| # | assumption | verdict | evidence |
|---|---|---|---|
| 1.1 | The LB uses the global pooled metric, not the per-sequence METRIC.md one | holds | Valid vs LB for the three pre-fine-tune LB models (below). Per-sequence offsets disagree (+0.004 raw baseline, +0.019 x0.25, +0.017 ab18). Global offsets are consistent. Per-sequence predicts the x0.25 gain at +0.042; the LB gave +0.057 and global lambda 0 gives +0.056 |
| 1.2 | The LB scores the valid-like mask (lambda 0) | unknown (lambda ~0.02 fits) | Levels: baseline raw 0.6171 / **0.5928** / 0.5660 at lambda 0 / .02 / .05 vs LB **0.5935**. x0.25 0.6728 / 0.6525 / 0.6289 vs 0.6503. ab18 0.6757 / 0.6540 / 0.6286 vs 0.6488. Best-fit lambda is 0.019 / 0.023 / 0.026. All rows (lambda 1) gives 0.44-0.50, so the LB is not all-rows. A mask difference and a harder test are indistinguishable from levels alone (H1) |
| 1.3 | `gmetric.py` = published formula (clip, then w=\|clip y\| per target; float64 sums; degenerate side -> 0; mean of t0/t1; rows = is_scored AND step>=99) | holds | Code read. `tests/test_gmetric.py` checks against a literal transcription. `decide.py` uses the same sums. On valid the first scored step is >= 99 in every sequence |
| 1.4 | Block eligibility / per-block exclusions no longer matter | holds | Under the old rule, 1 of 1873 valid blocks was ineligible. The global metric has no eligibility rule |
| 1.5 | The row selection is independent of position in the sequence | holds | Scored rate is 0.093-0.098 in every step bucket (99-200 ... 19900-20000). Mean \|y\| is 0.60-0.62 and mean t0 -0.20..-0.24, flat. Per class the rate is flat too. `step_in_seq` carries nothing |
| 1.6 | wMSE is the right row loss for the pooled metric | holds (exact) | Maximizing pooled weighted Pearson with w=\|y\| pointwise gives p*(x) = E[\|y\|y\|x]/E[\|y\|\|x] up to an affine map. That is exactly the minimizer of \|clip y\|-weighted MSE on clipped targets (`nn.weighted_mse_loss`). Class-3 rows with t=0 (t0 36%, t1 34% of scored rows) are handled automatically |

## 2. Inference-path fidelity

| # | assumption | verdict | evidence |
|---|---|---|---|
| 2.1 | Packaged native engine = research predictions | holds | Stream vs batch correlation is 0.99997 on required rows (E-143). Native fp32 scores 0.687942 vs GPU autocast 0.687818 (E-076). `fast_activations=True` is inside that tolerance. `solution_multi.py` steps every member on every row, warm-up included, resets on a seq_ix change, and casts to float32 |
| 2.2 | fp16 autocast does not distort research models | **broken for the baseline member only** | `pred_base_rnn_valid` (autocast) differs from the ONNX baseline by up to **8.5** raw on some rows. E-157 reported 0.049; that was not the maximum. WP x0.25 is 0.6723 vs 0.6728, so the errors are rare outliers. E-157/E-158 baseline-member fine-tunes were trained **and** evaluated through this noisy forward pass. A fine-tuned baseline head shipped to the fp32 engine would carry train/serve skew. The main GRUs are unaffected (no +16 shift trick) |
| 2.3 | Training weights rows as the pooled metric does | **broken (effect untested)** | `weighted_mse_loss` divides by `w.sum()` of each (32-sequence x 2000-row) chunk. The optimized objective is sum_c L_c/W_c, not sum L / sum W. Each batch-chunk gets unit total weight. Low-weight chunks, including the first chunk (which carries the 99 warm-up rows), are upweighted. This is mostly random reweighting and probably small (H3) |
| 2.4 | Zero initial state costs little (no burn-in deficit) | holds | ab30_eps_aux WP by step: 99-300 **0.709**, 300-1000 0.688, 1000-3000 0.677, 3000+ 0.683 (weight shares 1/3/10/86%). A learned h0 has no headroom (agrees with ml_lead P4) |
| 2.5 | Runtime is safe for the private set | unknown | The public run takes 36.3 min of 60. The Final Dataset's size is undisclosed (rules: "separate, hidden dataset"). ml_lead already proposes <= 45 min. See H5 |

## 3. Data assumptions

| # | assumption | verdict | evidence |
|---|---|---|---|
| 3.1 | Train vs valid are "different instruments" (E-094, findings) | **likely broken: different periods of the same 4 instruments** | Every valid sequence of a class has exactly one target quantum: 398/347/431/697 sequences at 0.1254/0.2413/0.4685/0.6569. Train classes are identified by the same four quanta (`wnn.data.TARGET_QUANTA`). One tick normalisation per class across both splits points to one instrument per class. The adversarial drivers (a2/a3/a5/a6 levels, volume-code means) are activity levels that drift over time |
| 3.2 | Valid is a random subset of chunks | broken (it is time-blocked) | Chain links train<->valid: **90** of 9191. Random chunk assignment (f = 0.15) would give 2f(1-f) x 9191 ~ 2,300. Mixed-chain patterns: TV 43, VT 37, TVT 5. So valid is contiguous blocks interleaved with train periods, not simply later. Consequence in H2: chain-grouped folds (median chain size 3) can split one period across folds |
| 3.3 | No ordering or metadata leak | holds | Parquet key-value metadata is only ARROW:schema, with no column statistics. Of same-split chain links, 0.04% go to row_group +-1 (median distance 2,475). seq_ix differences are random (sign 50/50). Using neighbours of other sequences at test time would violate the rules anyway |
| 3.4 | Organizer hints on the test set | partial | data_overview: the test has "the same format and sequence structure as valid"; the mask is hidden and "applied after inference". Nothing on class mix or mask rule. Timeline: deadline Nov 15, winners Dec 1. That two-week gap allows a forward-collected Final Dataset (unverified) |
| 3.5 | Mask semantics | holds (as already known) | Causal and activity-driven (E-087/E-122, physicist F6). Position-independent (1.5). Only weakly label-dependent (E-103) |

## 4. Target construction (item 4)

- **Lattice (holds, new detail):** all targets are integer multiples of the class quantum. The class-3
  loser is -3q = -1.9706, which sits just inside the clip. t0+t1 is -1q on 77% of class-3 scored rows
  and -2q on 18%. The quanta are uniform across sequences (3.1), so the organizer normalized each
  instrument once and not per sequence or rolling window. No hidden per-sequence scale is left to recover.
- **Clock (holds):** targets change on 54-65% of i0 price-code rows and on **0.000%** of
  trade-only, i1-only or a-only rows in every class (physicist F7). A real-time horizon would change
  targets on non-price rows as the window slides past future events, and a trade clock would change them on trade rows.
  So the clock is i0 price events, and E-079 (97.5% on moved rows, 99.5% on bid-vs-ask touch rows)
  is not a misidentified time horizon.
- **Rounding (holds):** E-079 uses `np.rint` in reconstructed tick coordinates with an arbitrary origin.
  Banker's rounding would make exact half-tick ties depend on price parity, but 99.5% touch accuracy
  means ties are rare. No parity free lunch.
- **Classes 0-2 (unknown, low value):** an exact law would not change modelling. The remaining uncertainty is the
  direction of future moves, and exact online class-3 labels already failed as a recalibration signal (E-080).

## 5. Free-lunch candidates (item 5), cached predictions, ab30_eps_aux (train-only) unless noted

| candidate | result | verdict |
|---|---|---|
| Rescale to exploit the clip (x1.25 / x1.5 / x2) | -0.0001 / -0.0016 / -0.0091 at lambda 0; same at .02 | exhausted (the x0.25 baseline trick was the only one) |
| Causal smoothing within quiet stretches (EMA reset at each i0 book event), alpha .7/.5/.3 | -0.0003 / -0.0011 / -0.0035, same sign on all 4 chain folds; cumulative stretch mean -0.0375 | refuted: the GRU's within-stretch updates are information, not jitter |
| Per-class calibration of the GRU | weighted slope y~p 0.95-1.00 in every class/target; class means match (e.g. class 3 t0 -0.749 vs -0.757) | calibrated; agrees with E-109 and skeptic H-calib |
| Per-class rescale of the organizer baseline (tick-unit hypothesis) | raw slopes 0.31-0.40, near-uniform across classes (tick units would differ ~5x); class offsets +0.66..+0.83 at x0.25, mostly one global offset | refuted: same targets, scale-free objective (agrees with ml_lead M2) |
| Shrink predictions on low-propensity rows (valid blend 0.65 GRU + 0.35 base x0.25) | prop_v1 < 0.01 (38.6% of required rows, 1.3% of scored): constant gives -0.0022 / -0.0075 / -0.0139 at lambda 0 / .02 / .05; half-shrink -0.0005 / -0.0017 / -0.0031 | no free lunch (predictions there carry signal), but a strong **probe** (H1) |
| Warm-up / position / seq_ix / row order | see 1.5, 2.4, 3.3 | none |
| Deterministic parts (level -sp, loser -(sp+2)q, class id) | already decoded by the model (aux spread/class heads 100% in class 3, E-075) | used |
| Chain neighbours (known history or future of a test chunk) | would use rows of other sequences | forbidden by the rules; not pursued |

## 6. Ranked "missed structure" hypotheses

**H1. Measure the test's effective lambda with one LB probe (decision value for the private pick).**
Why: the LB level matches valid at lambda ~0.02 (1.2). The skeptic's five LB deltas also favour 0.02-0.05.
Mask-specialised levers (fine-tune, true masks, p^2) and robust levers (floor, WiSE) rank oppositely
across that range. Test: upload the incumbent with required-row predictions set to a constant
wherever a causal propensity is below a threshold. Offline predicted deltas (valid, blend analog):
prop_v1 < 0.01 gives -0.0022 at lambda 0 and -0.0075 at lambda 0.02. prop_v1 < 0.02 gives -0.0055 vs
-0.0130. The 0.005-0.0075 gap is more than 10x the same-parent LB noise (~0.0004). So a single upload
separates "valid-rule mask on a harder test" from "softer mask". The package needs a causal gate:
prop_v1 in the native engine (~1-2 h), or a threshold on raw a3 (single-feature AUC 0.83 in class 3;
its offline deltas still need computing, ~2 min, 1 core). Expected effect: 0 direct. It settles which
of two final families to pick, where we estimate +-0.001-0.003 for the private set.

**H2. Chain folds may not be period folds, so valid-trained levers may be optimistic.** Valid is
time-blocked (90 vs ~2,300 cross links, 3.2), while the linker yields 3,289 chains with median size
3. Sequences from one period can therefore land in different chain folds and share their regime.
That inflates fold estimates for anything fitted on valid (fine-tune, true masks, LP-FT, valid data
dividend), and the private set may be a new or forward period (3.4). Cheapest test (CPU, ~10 min):
cluster valid sequences into periods, e.g. with a looser linker threshold plus sequence activity
statistics from `cache/seqstats_valid.parquet`. Then recompute the E-124/E-148 fold deltas from
cached `seqsums_*` with period-grouped folds. If the fine-tune gain shrinks by more than half, weight
the private pick toward the robust family. Expected: no LB gain. It protects about 0.001-0.002 on
private.

**H3. Pooled (not per-chunk) loss normalization (2.3).** Replace `/ w.sum()` per chunk with
a constant: the running mean of W over the first ~200 batches, or the dataset mean. The gradient then
follows the pooled metric's weighting across batches and chunk positions. Cheapest test: one fold-0
run of the incumbent recipe, same seed, compared at lambda 0 and 0.02. The same-seed control removes
most seed noise. Expected +0.0000..+0.0008 (a random reweighting of CV ~20-30% is rarely large). It
also removes the upweighting of the warm-up-diluted first chunk.

**H4. Redo baseline-member research in fp32 (2.2).** E-157's "baseline FT adds ~nothing" and E-158's
base_headft were trained and scored through a forward pass with rare errors up to 8.5 raw. Cheapest
test: disable autocast for `base_rnn` only (or keep that member fp32), rerun the head-only FT on
folds 0/1 (~0.1 GPU-h), and score with the native fp32 engine. Expected 0..+0.001 on the 0.35-weight
member. Rule for now: never ship a baseline-derived member trained under autocast without an fp32
valid check.

**H5. The private "Final Dataset" is an unverified assumption (size, period, mask).** Timeline:
deadline Nov 15, winners Dec 1. The data may be forward-collected. If it has more rows than the
public set, a 36-min package risks the 60-min cap and a zero score. Cheapest test: ask the organizers
on Discord for the Final Dataset's row count and whether it is a later period (zero compute). Until
then keep finals <= ~40-45 min on the public run. Pick one mask-specialised and one robust
candidate, guided by H1/H2. Expected: avoids a catastrophic tail. On valid it is worth 0.

Not ranked (checked, no headroom): clip/scale, smoothing, burn-in, position, per-class calibration,
baseline tick-unit rescale, low-propensity shrink, rounding parity, ordering/metadata leaks.
