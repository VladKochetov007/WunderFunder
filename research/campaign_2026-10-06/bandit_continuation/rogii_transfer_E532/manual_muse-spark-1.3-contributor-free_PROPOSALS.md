# Rogii top-1 transfer proposals — Wunder Connectome Challenge 2 (E532, manual ideation)

- Tool / provider / model: OpenCode harness, `opencode/muse-spark-1.3-contributor-free` (Muse Spark)
- Date (UTC): 2026-10-07
- Source read in full: `writeups/rogii_top1.txt` (Ruby / w5833946, ROGII wellbore 1st-place writeup, lines 1–528, including comments section)
- Skills used: `.agents/skills/research-reframe/SKILL.md`, `.agents/skills/conduct-creative-research/SKILL.md`
- Task framing: Wunder Connectome Challenge 2 — supplied sequences of 112 anonymous numerical input channels, two regression targets, deterministic causal row-by-row inference with per-sequence reset. Official evaluation is fixed `global_wp_clip2_v1`: fixed scored/prediction mask, ±2 clipping, original clipped-label magnitude weights. Training-target clipping is a separate experimental variable. Train our own weights; independently implement any transferred method; no teammate code or weights. Only permitted supplied data plus allowed public pretrained sources. Large research teachers allowed with an explicit compression/serving path. Do not assume geological semantics or invariants in anonymous channels.
- Independence note: worked without reading other ideators' outputs in `rogii_transfer_E532/`. No fits run, no budgets/evaluators changed, no credentials accessed.
- Priors: all numerical priors and public forecasts are NULL until measured. A confidence interval crossing zero is not an automatic rejection; follow the campaign uncertainty rule (bounded follow-up on information value, not auto-kill).
- Causal rule applied throughout: no future inputs at inference. Bidirectional smoothing and symmetric non-causal U-Net structure are not served; where the source uses them, the card specifies a causal substitute and/or a separately justified training-only teacher.

## Source components mined (for traceability)

1. 2D alignment-grid formulation + smoothed cross-entropy + Huber on expected path + `mean(prob × gap)` penalty (lines 6–23).
2. ConvNeXt 2D U-Net, BatchNorm > LayerNorm with BF16 guard, average-pool + interpolation > learnable resampling (lines 24–33).
3. Typewell calibration via visible-region (TVT,GR) binning + interpolation + blending (lines 35–41).
4. Horizontal-well per-bin summaries: mean, nan-rate, std, slope, last-minus-first, quadratic coefficients, quadratic residual RMSE, visible TVT mean (lines 42–54).
5. Interaction features `|a−b|` and coordinate `z_diff` (lines 55–63).
6. Particle-filter features as inputs; accumulated level beats differences when PF context present (lines 64–75).
7. Particle filter proper: low-probability large jumps, calibrated GR, FFBSi smoothing, diverse cfg-profile blending, bin-level (64-sample) updates; standalone ~7.4 RMSE (lines 76–88).
8. XY-neighbor local-linear model, WLS, anisotropic/regularized details marginal; standalone ~11.4 RMSE; CV +0.3 RMSE but public-LB degradation attributed after testing five neighborhood-quality statistics to inconsistent labels; same stats reused in ensemble (lines 89–122).
9. Augmentations: Z-shift with block-bootstrap path + smoothed-noise GR regeneration + rare fault jumps; GR affine `a·GR+b`; plus reverse path, MD stretch, 2D channel masking, GR residual shift, tail crop, typewell-axis sequential masking, GR jitter, PF rotate/shift anti-shortcut (lines 123–156).
10. Conditional ensemble: separate no-XY + z_diff models for ~10% unreliable wells; fixed general/no-XY weights; diversity from features/augmentation/training; weak singles retained for ensemble gain (lines 157–170).
11. Negatives: Transformers < CNNs here; joint sim+real > two-stage pretraining; PF channels helped weak single (7.7→6.7) but not best single, kept for diversity; fancy loss weightings, bad-label removal, and size/resolution scale-up did not help (lines 171–195).

What explicitly does NOT transfer without reinvention: TVT/Z/MD/GR semantics, `TVT+Z` invariants, geographic neighbor geometry, typewell-vs-horizontal layout, 2D spatial U-Net over (345×400) alignment grids, and any use of future rows at inference. Cards below transfer only structure: distribution targets, calibration protocol, jump belief, augmentation operators, conditional blending, geometry summaries, and small-component/training-protocol details, each re-grounded in anonymous causal sequences.

---

## H1 — Smoothed ordinal-distribution head with expectation readout (classification + Huber)

- Local ID: `ROGII-E532-M01`
- Source section/component: alignment-grid CE with exponentially smoothed target distribution normalized along typewell axis + Huber loss on expected path + `mean(prob × gap)` penalty (writeup lines 6–23).
- Structural analogy: ROGII replaces point regression with a 1D distribution over an ordered outcome axis, smooths the one-hot target, and reads out the expectation for the point metric while keeping an auxiliary regression loss. The anonymous-sequence analogue is a per-row, per-target ordinal distribution over the clipped prediction range, with expectation as the served point prediction. Ordering and smoothing are the transfer; grid geometry and geology are not.
- Intervention:
  1. Fix served range to official clipping interval ([−2, 2]) for bin edges; training-target clipping remains a separate declared variable (train on clipped vs. unclipped targets is an explicit sub-arm, evaluation never changes).
  2. Add a causal per-row distribution head for each of the 2 targets: K ordered bins over [−2, 2] (K is a recipe hyperparameter, e.g. two levels tested, not a prior), linear logits from the recurrent state, no future context.
  3. Target distribution = Gaussian/exponential smoothing kernel centered at the (training-clipped) label, normalized over bins; loss = cross-entropy plus Huber on the bin-expectation plus an optional magnitude-aware penalty `mean(prob × bin_distance_or_weight)` using only training-declared weights (never alters official eval weights).
  4. Serve the expectation; keep the existing point head during the transition as a paired control and as a possible residual path.
  5. Weak-standalone clause: even if the distribution head ties or loses standalone, retain it as a teacher target and as a residual/ensemble member; endpoint comparison decides.
- Assumptions: ordinal structure of the clipped target range is informative; smoothing radius and K interact with label noise and metric magnitude weights; backbone capacity is held fixed so head/loss effects are attributable.
- Leakage/causality risks: none from the head itself if inputs are causal rows only. Risks enter only if bin edges, smoothing kernel, or magnitude weights are fit on held-out labels or on future rows; fit all such constants on the declared fitting split(s) only, recompute from scratch per admitted recipe, and keep official eval clipping/weights frozen.
- Predicted observable: lower official-metric variance across seeds and/or improved weighted Pearson on jump/magnitude-heavy rows vs. pure point regression, with better-calibrated uncertainty (wider predicted distributions where errors are larger). Direction is a hypothesis, not a forecast; null or negative standalone is informative.
- Cheapest bounded test and matched control: same causal recurrent backbone, same data splits/seeds, same training-clip declaration; control = point-regression head only; candidate = distribution + expectation + Huber with one K and one kernel width. Compare (a) standalone official-metric delta on the frozen local comparison and (b) the complete combined endpoint when the candidate's expectation (or residual over control) is blended with the control under fixed weights. Predeclare seeds and the exact comparison slice before running.
- Exact-recipe falsifier: on the frozen recipe (declared K, kernel, clip, seeds, splits), the distribution head shows no improvement in (i) standalone official metric, (ii) seed-variance/robustness, (iii) teacher value (distilling its soft targets into the control yields no gain), or (iv) ensemble residual value when added with fixed weights — then the exact recipe is falsified for that backbone/split. Falsification does not reject ordinal heads in general.
- Approximate cost tier: S (one backbone × two heads × predeclared seeds; CPU-feasible distribution-cache pass; no invented timings).
- Parent/interaction links: sibling to heterogeneous recurrent-teacher distillation (soft-distribution teacher is a new teacher family, not another seed of the same teacher); combines with M05 (blend weights), M06 (quadratic-residual as heteroscedastic feature for distribution width), M07 (joint-training protocol for synthetic data). Not a paraphrase of causal TCN / wider GRU / block memory / online dynamics / rolling heuristics — it changes target representation and loss, not backbone width or window.
- Level-2 follow-ups: (a) if teacher value positive but standalone neutral, distill soft distributions into a small served student and test the student endpoint; (b) if width correlates with error, add a width-gated blend (links M05); (c) if K/kernel sensitive, test a two-scale ordinal head (fine + coarse) with a consistency loss.

## H2 — Causal per-sequence reference calibration with blended fallback

- Local ID: `ROGII-E532-M02`
- Source section/component: typewell GR calibration by aggregating (TVT,GR) into TVT bins on visible region, interpolating back, blending with original (lines 35–41).
- Structural analogy: ROGII builds a per-well reference profile from trusted visible rows and blends it with raw inputs so the model sees both absolute and shape deviations. The anonymous analogue is per-sequence, strictly causal adaptive calibration: statistics from already-seen rows of the same sequence map current rows to a reference frame, blended with raw channels. Binning/interpolation/blending is the transfer; TVT/GR roles are not assumed.
- Intervention:
  1. For each sequence, maintain causal prefix statistics only (no future, reset per sequence): running mean/variance and optionally running quantiles or binned mappings per channel or per channel-group, updated row-by-row.
  2. Emit calibrated channels `x_cal = (x − ref) / scale` alongside raw `x` and the residual `x − ref`; blend factor α is a declared recipe constant or a tiny causal gate from prefix length/missingness (not from future).
  3. Channel selection is agnostic: apply to all 112 or a variance/missingness-selected subset with an explicit ablation; do not designate semantic roles.
  4. Serve causally: at row t only rows ≤ t of the same sequence may affect the calibration; first rows use a global fallback until the prefix is long enough (predeclared warmup rule).
- Assumptions: per-sequence level/scale drift exists and prefix statistics estimate it without leaking the label; blending preserves information the backbone would otherwise have to relearn; warmup rule handles short prefixes.
- Leakage/causality risks: calibrating with full-sequence statistics, fitting bin edges/scales on evaluation labels, or carrying state across sequences would leak. Require per-sequence reset, causal prefix only, constants from fitting split only, and a determinism/reset test.
- Predicted observable: improved stability across sequences with level/scale shift and reduced error on late-sequence rows where prefix estimates have converged; raw+calibrated should beat raw-only or calibrated-only if blending matters.
- Cheapest bounded test and matched control: same backbone/head; control = global normalization only; candidate A = causal running calibration; candidate B = candidate A blended with raw (α fixed). All else fixed including seeds/splits. Evaluate standalone and as a teacher/residual in the full endpoint.
- Exact-recipe falsifier: blended causal calibration shows no standalone, teacher, or residual-ensemble gain over the global-normalization control on the frozen splits/seeds and no reduction in cross-sequence variance — then that calibration recipe (window/warmup/α/channels) is falsified. Does not reject all adaptive normalization.
- Approximate cost tier: XS–S (data-prep plus one backbone pair; calibration itself CPU-feasible; no invented timings).
- Parent/interaction links: interacts with online feature dynamics and causal rolling heuristics as a distinct normalization-first cousin (test calibration × rolling-feature presence explicitly); feeds M05 reliability gating (prefix length/missingness as reliability signals); combines with M04 affine augmentation (calibration should reduce brittleness to level/scale jitter).
- Level-2 follow-ups: (a) if warmup rows dominate errors, learn a causal α gate from prefix diagnostics; (b) if a channel subset drives gains, test group-wise calibration and cross-channel residual interactions `|x_i − ref_j|` as M06-style features; (c) if calibration helps teachers only, keep it teacher-side and serve the student without it.

## H3 — Heavy-tail jump belief state with block-binned updates and training-only smoother teacher

- Local ID: `ROGII-E532-M03`
- Source section/component: particle filter with low-probability large jumps, bin-level (64-sample) updates, FFBSi smoothing, diverse cfg-profile blending, calibrated likelihood (lines 76–88); plus PF-heatmap inputs and accumulated-level-beats-differences note (lines 64–75).
- Structural analogy: ROGII maintains an explicit belief over alignment paths that permits rare large jumps, updates at coarser-than-raw resolution, smooths backward for better training targets, and blends diverse dynamics. The anonymous analogue is a causal jump/regime belief scalar (or small vector) maintained alongside the recurrent state: heavy-tail jump proposals, updates every B rows rather than every row, a non-causal smoother used only to create training-time regime/jump labels, and diverse dynamics configs blended. Belief mechanics transfer; motion/observation models do not.
- Intervention:
  1. Define a minimal latent jump/regime process over rows: mostly small drift plus a small-probability large jump (mixture proposal with predeclared probabilities; exact values are recipe constants, not priors about the data).
  2. Forward (served) filter is strictly causal: per-row or per-block update from causal features only, state reset per sequence, deterministic given seed (fixed particle/random seed handling for deterministic serving; or closed-form approximation if particles threaten determinism).
  3. Block binning: update the belief every B rows (e.g. two B values tested) and hold/interpolate between updates; compare against per-row updates.
  4. Training-only smoother: run a backward smoother offline on fitting data to produce jump/regime soft labels; distill into the causal belief head and/or the main backbone. The smoother never runs at inference.
  5. Feed the belief (level + jump probability) as auxiliary inputs alongside raw channels; test accumulated-level vs. difference encoding explicitly per the source note.
  6. Diverse blend: 2–3 dynamics configs (different jump probs/block sizes) whose beliefs are blended; weak configs are retained as ensemble/belief members if they add conditional diversity.
- Assumptions: rare large shifts exist in inputs and/or targets; coarse-block updates denoise without missing decision-relevant jumps; smoother labels carry information beyond what forward-only training finds; determinism can be preserved.
- Leakage/causality risks: smoother labels must come only from fitting splits and must never use evaluation labels or leak into the served path; belief must reset per sequence and use no future rows; particle randomness must be fixed/seeded or replaced by a deterministic approximation for serving. Diagnose by (i) reset test, (ii) future-mask test, (iii) teacher-ancestry check (smoother split membership recorded).
- Predicted observable: concentrated gains on jump/transition rows and reduced overshoot/ringing after shifts; smoother-distilled causal head beats forward-only head on transition slices even when full-average deltas are small; block-binned beats per-row on noisy stretches.
- Cheapest bounded test and matched control: same backbone; control = backbone with no belief inputs and no smoother distillation; candidate = backbone + causal belief inputs + smoother-distilled auxiliary loss, one B and one jump-mix. Predeclare transition-row slice definition (from inputs/labels on the comparison split, before seeing results) and evaluate both full-metric and slice-metric plus the full blended endpoint.
- Exact-recipe falsifier: the frozen belief recipe shows no gain as (i) standalone input, (ii) smoother-distilled teacher, or (iii) blended ensemble member on both full and predeclared transition slices across predeclared seeds — then that dynamics recipe is falsified. Does not reject belief-state mechanisms generally.
- Approximate cost tier: M (belief implementation + smoother-label pass + 2–3 dynamics configs × predeclared seeds; still bounded single-backbone scale; no invented timings).
- Parent/interaction links: child of heterogeneous recurrent-teacher distillation (new smoother-teacher source) and of block memory (block-binned updates as an explicit alternative mechanism — test belief × memory interaction rather than assuming additivity); supplies jump probability to M05 gating and M01 distribution width; consumes M06 residual-RMSE as a candidate likelihood input.
- Level-2 follow-ups: (a) if block-binning wins, sweep B × jump-prob as an explicit interaction node; (b) if smoother teaches but forward belief unused at inference, distill to a stateless jump-probability head and drop particles; (c) if diverse configs disagree usefully, keep the weakest config as a dedicated transition specialist in the M05 blend.

## H4 — Shape-forcing affine + block-bootstrap + anti-shortcut augmentation family

- Local ID: `ROGII-E532-M04`
- Source section/component: Z-shift via block bootstrap on real differences + smoothed-noise regeneration + rare fault jumps; GR affine `a·GR+b`; plus reverse path, MD stretch, 2D channel masking, residual shift, tail crop, axis masking, jitter, PF rotate/shift anti-shortcut (lines 123–156).
- Structural analogy: ROGII's two strongest augmentations force reliance on shape over absolute level (affine) and on plausible path variability including rare jumps (resampled paths), while corruptions prevent shortcut reliance on helper channels. The anonymous analogue is a channel-agnostic augmentation family: per-channel affine jitter, block-bootstrap resampling of differences/paths, residual-shift within sequences, structured masking/cropping, and deliberate corruption of helper/teacher channels. Operators transfer; geological path semantics do not.
- Intervention (each toggleable, tested incrementally):
  1. Per-channel affine: `x' = a·x + b` with small predeclared ranges, applied consistently within a sequence (and optionally consistently across helper channels) to force shape reliance.
  2. Block-bootstrap path synthesis: resample blocks of consecutive row-differences from real fitting sequences to synthesize new drift paths; add rare large offsets with small probability; reconstruct levels by cumulative sum. For inputs, regenerate by residual reattachment (analogue of GR regeneration): keep a real residual `(x − reference)` and reattach to the synthetic path rather than inventing channel semantics.
  3. Residual shift: circularly shift within-sequence residuals to break absolute-position memory while preserving dynamics.
  4. Structured masking: random channel masking, contiguous time masking, tail cropping, and helper-channel corruption (analogue of PF rotate/shift): perturb or drop rolling/teacher/belief channels during training so the backbone cannot shortcut through them.
  5. Time-warp/stretch analogue of MD stretching: small monotonic resampling of row spacing applied causally consistently (no future leakage; labels transformed identically).
  6. Reverse-path treated as high-risk optional arm only: if tested, the reversed sequence is still processed strictly forward with reset; labels reversed identically; report separately and do not bundle with the core family.
- Assumptions: invariances to level/scale, local timing, and helper-channel presence hold approximately; synthetic paths stay within the support of real dynamics; augmentation is joint with real data (per the source negative on two-stage pretraining — see M07).
- Leakage/causality risks: augmentation must never mix fitting statistics into held-out sequences, transform inputs and targets inconsistently, or use future rows to construct current training inputs in a way the served model cannot replicate. Resampling pools, affine ranges, and corruption rates are fit/declared on the fitting split only; augmentation is training-time only; inference rows are unaugmented and causal.
- Predicted observable: reduced brittleness to level/scale shift and to helper-channel dropout; smaller degradation when teacher/rolling features are corrupted or removed at test time; improved transition-row behavior when rare-jump synthesis is on. Standalone full-average gains may be near zero while robustness/teacher value is positive.
- Cheapest bounded test and matched control: same backbone/head/splits/seeds; control = no augmentation (or current baseline augmentation only); candidate = control + one operator at a time (affine first, then residual-shift, then helper-corruption), then the minimal winning bundle. Each step reports standalone metric, corruption-robustness slice (helper dropout at test), and full-endpoint blend value.
- Exact-recipe falsifier: on the frozen recipe, the augmentation bundle shows no standalone gain, no corruption-robustness gain, no teacher/student gain, and no ensemble residual value across predeclared seeds — then that bundle is falsified. Individual operators are falsified individually before rejecting the family.
- Approximate cost tier: S–M (data-operator code once, then one backbone per operator toggle × predeclared seeds; no invented timings).
- Parent/interaction links: augments every other card (M01 needs smoothing-kernel × augmentation interaction; M02 calibration × affine is an explicit interaction node; M03 rare-jump synthesis × belief dynamics is an explicit interaction node); protocol governed by M07 joint-training rule. Distinct from generic "more augmentation" by naming exact operators and anti-shortcut corruption.
- Level-2 follow-ups: (a) if helper-corruption helps, promote corrupted-teacher training as the default for all distillation; (b) if block-bootstrap paths help transitions only, distill a transition-specialist teacher; (c) if affine ranges matter, gate augmentation strength by M05 reliability stats (strong augment where unreliable).

## H5 — Reliability-gated conditional blend with fixed fallback weights

- Local ID: `ROGII-E532-M05`
- Source section/component: conditional ensemble for unreliable-XY wells (~10% fallback to no-XY + z_diff), fixed general/no-XY weights, five neighborhood-quality statistics reused as ensemble inputs, weak singles retained for ensemble gain (lines 105–122, 157–170).
- Structural analogy: ROGII routes or reweights a small set of diverse models by explicit reliability diagnostics with frozen weights, and keeps individually weaker models because the blend wins. The anonymous analogue is a causal reliability gate over 2–4 fixed diverse members (different features/augmentation, not different seeds of the same recipe), with frozen blend weights and a fallback member trained without the least reliable inputs. Routing diagnostics transfer; geographic quality stats do not.
- Intervention:
  1. Fix members before seeing blend results: e.g. Member A generalist (all features), Member B fallback (excludes the least reliable helper/belief channels, adds raw-level fallback analogous to z_diff), optionally Member C transition specialist (M03/M04-derived). Membership and the fact of a fallback are predeclared.
  2. Define causal reliability stats from prefix only: sequence position, prefix length, missing/NaN rate, running variance/novelty vs. fitting distribution, recent jump/belief rate (M03), calibration warmup state (M02). No future, no evaluation labels, reset per sequence.
  3. Frozen blend: predeclare weight vectors (e.g. general vs. fallback weights in reliable vs. unreliable regimes, threshold predeclared or fit on fitting split only). No per-evaluation tuning of weights. Report the exact weight table.
  4. Weak-model clause: members are not pruned for losing standalone; the scored object is always the complete combined endpoint with fixed members/weights. Distinguish general-regime and fallback-regime metrics.
- Assumptions: unreliability is detectable causally; the fallback member covers the unreliable regime; frozen weights prevent blend overfitting; diversity comes from features/augmentation/training, not seeds.
- Leakage/causality risks: reliability stats using future rows or evaluation distributions, weights tuned on the comparison slice, or member selection after peeking at blend scores would invalidate the claim. Freeze members, stats, thresholds, and weights on fitting data; run a determinism/reset check; record full ancestry.
- Predicted observable: the frozen blend beats every individual member on the full official metric and dominates on the predeclared unreliable slice; removing the weakest standalone member hurts the blend (conditional value despite weak standalone).
- Cheapest bounded test and matched control: controls = each member standalone plus an equal-weight average; candidate = predeclared reliability-gated frozen blend. Same splits/seeds for members; blend weights frozen before terminal evaluation. Evaluate full endpoint determinism, CPU/serving shape, and regime slices.
- Exact-recipe falsifier: the frozen gated blend fails to beat the best single member and the equal-weight average on the frozen comparison, and ablation shows the weakest member contributes nothing in either regime — then that member/gating/weight recipe is falsified. Does not reject conditional blending generally.
- Approximate cost tier: M (2–3 member fits already budgeted elsewhere can be reused only if ancestry matches; otherwise bounded member cohort × predeclared seeds; blending/eval itself XS; no invented timings).
- Parent/interaction links: integration node for M01–M04 and M06–M07 (each new component enters through this frozen-blend protocol); sibling to existing blend/distillation work but distinct by reliability gating + fallback-member design and frozen-weight discipline. Explicitly tests weak-model-as-member rather than pruning by standalone rank.
- Level-2 follow-ups: (a) if gating helps, learn the gate as a tiny causal head supervised by fitting-split regime labels, then refreeze; (b) if fallback dominates early rows, add a position-gated two-member micro-blend as a serving-cheap default; (c) if blend wins only via one regime, split into two served endpoints with a documented router.

## H6 — Quadratic-regime summaries with residual-RMSE novelty detector and auxiliary loss

- Local ID: `ROGII-E532-M06`
- Source section/component: horizontal-well per-bin quadratic fitting coefficients + fitting residual RMSE + slope vs. last-minus-first disagreement (lines 42–54).
- Structural analogy: ROGII summarizes each coarse bin by a quadratic fit and treats the residual RMSE as information (regime complexity/change), with slope vs. endpoint-difference as a shape-consistency check. The anonymous analogue is causal trailing-window quadratic summaries per channel (or pooled): least-squares quadratic coefficients, residual RMSE, slope, and last-minus-first, with RMSE/disagreement as jump/novelty signals and as auxiliary supervision. The quadratic-residual-as-detector is the transfer, not the binning grid.
- Intervention:
  1. Over a trailing causal window W (one or two predeclared W values), compute per selected channel: linear slope, last-minus-first, quadratic `(a, b, c)`, residual RMSE, plus std and NaN rate. All strictly causal (rows t−W+1..t), reset per sequence.
  2. Feed a compact subset (pooled RMSE, disagreement `|slope − (last−first)/W|`, top-coefficient magnitudes) to the backbone; keep raw channels intact.
  3. Add an auxiliary head predicting next-row residual RMSE and/or a transition flag derived from fitting data only; auxiliary loss weight is a declared constant. This auxiliary head may serve as a teacher even if the features alone do not win standalone.
  4. Explicitly test pooled vs. per-channel summaries to bound serving cost.
- Assumptions: local quadratic misfit detects regime change/novelty in anonymous channels; disagreement between slope and endpoint difference flags jumps; auxiliary supervision transfers to the main targets without assuming which channels are informative.
- Leakage/causality risks: windows reaching into future rows, statistics pooled across sequences, or transition flags derived from evaluation labels would leak. Require causal windows, per-sequence computation, flags from fitting splits only, and a window-causality unit test.
- Predicted observable: gains concentrate on transition/novelty rows; residual-RMSE auxiliary loss improves jump-row error and M03 belief quality; pooled summaries capture most of the gain at a fraction of serving cost.
- Cheapest bounded test and matched control: same backbone; control = backbone with existing rolling features but no quadratic summaries and no auxiliary loss; candidate = control + pooled quadratic summaries + RMSE auxiliary head. Evaluate full metric, predeclared transition slice, and teacher value of the auxiliary head.
- Exact-recipe falsifier: the frozen (W, channel-subset, aux-weight) recipe shows no standalone, slice, teacher, or ensemble-residual gain across predeclared seeds — then that recipe is falsified. Does not reject geometric summaries generally.
- Approximate cost tier: XS–S (feature code + one backbone pair × predeclared seeds; pooled variant is serving-cheap; no invented timings).
- Parent/interaction links: refines causal rolling heuristics and online dynamics with a specific quadratic-misfit mechanism (test presence × M02 calibration and × M03 belief explicitly); supplies likelihood/novelty inputs to M03 and width/gating signals to M01/M05. Not a wider-GRU or generic rolling-feature paraphrase — the falsifiable claim is that residual RMSE + slope/endpoint disagreement carries the signal.
- Level-2 follow-ups: (a) if pooled RMSE suffices, compress to a 2–3 scalar novelty bus for serving; (b) if auxiliary head teaches, distill it into the student without serving the features; (c) if W-sensitive, test a two-scale (short/long) quadratic pair as an interaction node.

## H7 — Parameter-free multi-scale pooling + joint-training and anti-shortcut protocol [UNCONVENTIONAL SMALL-COMPONENT TRANSFER]

- Local ID: `ROGII-E532-M07`
- Source section/component: average-pool + interpolation beats learnable down/upsampling; BatchNorm beats LayerNorm with BF16 guard; joint sim+real training beats two-stage pretraining; PF channels help weak singles but not the best single (kept for diversity); GR-penalty `mean(prob × gap)` as a small gain (lines 24–33, 157–195).
- Structural analogy: ROGII's headline (ConvNeXt U-Net) mattered less than boring components: parameter-free resampling, normalization placement with a precision guard, joint (not staged) use of synthetic data, keeping weak-but-diverse signals, and a tiny magnitude-aware penalty. The anonymous analogue is a serving-cheap causal multi-scale average-pool pyramid plus a frozen training protocol. Small components transfer; the 2D U-Net and backbone scale do not.
- Intervention:
  1. Causal pooling pyramid: for selected channels, maintain trailing average-pools at 2–3 dyadic scales using only past rows (causal, reset per sequence), upsample by repetition/interpolation to row rate, and concatenate as features. No learned down/upsampling weights. Compare directly against a learned strided/conv resampling control of equal receptive field.
  2. Normalization check: test normalization placement variants applicable to the recurrent backbone (e.g. pre/post, batch vs. layer/group equivalents) with an explicit precision guard (BF16/clipping guard analogue) and a NaN/loss-spike log — the claim is placement + guard, not "bigger model".
  3. Joint-training rule: any synthetic/augmented data (M04) is trained jointly with real data from the start at a declared mixing ratio; a two-stage pretrain-then-finetune run is the explicit control, per the source negative.
  4. Magnitude-aware micro-penalty: `mean(pred_mass × distance_or_magnitude_weight)` analogue of the GR penalty, using only training-declared weights; never touches official eval.
  5. Weak-signal retention rule: any component that loses standalone but adds blend/teacher value under M05 is retained with a fixed small weight rather than pruned — protocol, not tuning.
- Assumptions: multi-scale averaging denoises without learning spurious resampling; normalization/precision interacts with recurrent dynamics; joint training prevents synthetic-distribution lock-in; tiny penalties shape without dominating.
- Leakage/causality risks: pooling windows must be causal with reset; normalization statistics must not pool evaluation sequences; synthetic mixing ratios must be declared before terminal eval; penalty weights must not be the official eval weights repurposed as training signal without declaration. Standard reset/future-mask/determinism checks apply.
- Predicted observable: parameter-free pyramid matches or beats learned resampling at lower serving cost; joint training beats staged on both full and transition slices; normalization + guard reduces loss spikes; micro-penalty gives a small robust gain; retained weak signals help the M05 blend despite neutral standalone.
- Cheapest bounded test and matched control: same backbone; controls = (i) learned resampling, (ii) staged synthetic training, (iii) no micro-penalty; candidates flip one protocol element at a time. Each reports standalone, spike/NaN rate, and M05 blend value.
- Exact-recipe falsifier: on frozen scales/ratios/weights/seeds, the parameter-free pyramid fails to match learned resampling with no cost/robustness advantage, joint fails to beat staged, and the micro-penalty shows no standalone/teacher/blend value — then that protocol recipe is falsified. Small components are falsified individually.
- Approximate cost tier: XS–S (mostly config/data-protocol changes on one backbone; pooling is CPU-cheap; no invented timings).
- Parent/interaction links: governs M04 synthetic use; cost-reduction partner for M06 (pooling vs. quadratic summaries); stabilizer for M01 distribution training (normalization/precision) and M03 belief inputs. Explicitly distinct from "scale up model/resolution" (source negative) — this card predicts the cheap boring component wins.
- Level-2 follow-ups: (a) if the pyramid wins, test pyramid + quadratic-RMSE (M06) as a minimal serving-cheap feature bus; (b) if joint wins, sweep mixing ratio as a bounded interaction node; (c) if the micro-penalty wins, test it as a distillation regularizer for the M01 student.

---

## Ranked diverse top three (by long-term usefulness × information gain)

1. **M01 — ordinal-distribution head.** Touches every future model via target representation; cheapest falsifier; doubles as a teacher factory for distillation and as calibrated uncertainty for gating. Highest long-term leverage per unit cost.
2. **M03 — jump belief with block binning + smoother teacher.** Only card that directly models rare large shifts with an explicit causal/smoother separation; tests a mechanism none of the existing recurrent/TCN/memory arms isolates; feeds M01 width and M05 gating. Highest information gain about regime dynamics.
3. **M05 — reliability-gated frozen blend.** The integration endpoint that makes weak-model value measurable instead of pruned; converts M01–M04/M06–M07 from isolated deltas into a served system with regime slices and frozen weights. Highest long-term serving usefulness.

Diversity note: the three span target/loss (M01), state/belief (M03), and system integration (M05). M04 (augmentation), M02 (calibration), M06 (geometry), M07 (protocol) remain alive as bounded siblings/interaction nodes, not rejected.

## Compact integration table

| ID | Family | Source detail | Intervention in one line | Matched control | Scored endpoint | Cost | Combines with |
|---|---|---|---|---|---|---|---|
| M01 | target/loss | smoothed CE + expected-path Huber | per-row smoothed ordinal head, serve expectation | point head only, same backbone | standalone + teacher + fixed-blend residual | S | M05 blend, M07 joint, M06 width |
| M02 | normalization | visible-region calibration + blend | causal prefix calibration + raw blend, warmup fallback | global norm only | standalone + teacher + variance slice | XS–S | M04 affine, M05 reliability |
| M03 | belief/state | jumps + 64-binning + FFBSi + cfg blend | causal jump belief, block-B updates, smoother distills | no belief, no distillation | full + transition slice + blend | M | M06 likelihood, M01 width, M05 gate |
| M04 | augmentation | Z-shift/block-bootstrap + affine + anti-shortcut | affine + bootstrap paths + residual shift + helper corruption | no-aug baseline, one operator at a time | standalone + robustness slice + teacher | S–M | M07 joint rule, M03 jumps |
| M05 | ensemble/system | conditional no-XY blend + quality stats + weak members | frozen reliability-gated blend with fallback member | best single + equal-weight avg | complete combined endpoint only | M | all cards enter here |
| M06 | geometry/features | quadratic coeffs + residual RMSE | causal quadratic summaries + RMSE aux head | rolling feats, no quadratic/aux | full + transition + teacher | XS–S | M03 input, M01 width |
| M07 | protocol/efficiency | avgpool>learned, joint>staged, BN guard, micro-penalty | causal pool pyramid + joint rule + guard + micro-penalty | learned resample, staged, no penalty | standalone + spike rate + blend | XS–S | M04 synth, M06 bus, M01 stab |

## Reproducibility and guardrails for any actor picking these up

- Freeze: backbone, splits, seeds, training-clip declaration, K/kernel or B/jump-mix or W/α, augmentation pool and mixing ratio, member list and blend weights — before terminal comparison.
- Always report: standalone official-metric delta, predeclared slice deltas (transition/unreliable/early/late), teacher/student value, and full-endpoint blend value with fixed weights. Do not add local deltas to any public score.
- Causality/determinism checks per recipe: per-sequence reset, future-mask, CPU determinism, teacher-ancestry/split-membership log.
- Preserve negatives and exposed-data history: previously inspected folds stay exposed; new agents/seeds do not make them fresh. Record selection exposure and keep a confirmation plan.
- Serving path: every teacher/belief/augmentation arm names its student/compression route (distilled head, stateless jump head, pooled novelty bus, fallback micro-blend) so research scale beyond serving limits still converges to a valid deterministic ZIP.
