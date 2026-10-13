---
kind: review
status: complete
created_utc: 2026-10-13T08:36:46.699119+00:00
updated_utc: 2026-10-13T08:36:46.699119+00:00
hypothesis_ids: [H-STRATEGY-20261012-01, H-STRATEGY-AVERAGING-20261013, H-238, H-STRATEGY-COMPACT-20261013]
experiment_ids: [E-STRATEGY-SWA-F0-20261013-R1, E-STRATEGY-INNOVATION-F0-S3-20261013-R2]
evidence_paths:
  - research/campaign_2026-10-13/strategy_execution_r1/FIRST_WAVE_COST_AND_EFFECT_R1.json
  - research/campaign_2026-10-13/strategy_execution_r1/first_results_analysis_r1/RESULTS_R1.json
  - research/campaign_2026-10-13/strategy_execution_r1/COVERAGE.json
---

# What the first strategy wave bought, and what should follow

## Question or claim

Which parts of our October12 strategy and the imported Claude review have paid for themselves, and how should the next allocation change?

**Claude's review has supplied the better immediate prioritization correction. Our earlier review remains the better scientific blueprint. Neither has yet delivered a demonstrated stronger complete submission.** The strongest new result on cost is fixed checkpoint averaging. The expensive amplitude experiment mainly tells us to reduce attention to that exact knob. These are allocation judgments from a small first wave, not a ranking of the authors or proof that whole mechanism families work or fail.

The user requested this reflection during authorized execution. It reprioritizes the existing39-item campaign, preserving the request to investigate every actionable idea and the top-1/public>.700 objective. It does not add paid spending or close healthy work.

## Evidence and interpretation

### Delivered model comparisons

| Experiment | Matched result | Incremental occupied-worker time | What it earned |
|---|---|---|---|
| Teacher amplitude .8125→1 before training clipping, F0/s3 | .681194969 versus .681155360; delta+.000039609 | 89.96minutes for six-epoch fit and export | A valid, narrow test; weak practical incentive for more coefficient tuning now |
| Fixed epoch3/4/5 parameter mean, F0/s3 | .681198870 versus .681155360; delta+.000043510 | 95.94seconds for export | Cheap candidate for further reuse/confirmation |
| Same averaging rule, F0/s4 | .681315877 versus .681258422; delta+.000057455 | 95.89seconds for export | Second student/initializer seed points the same way |

The two averaging effects have mean+.000050483. This is the mean of paired individual-model scores, **not a measured ensemble score**. Both seeds share one teacher cohort and the same exposed F0 panel. Averaging uses sunk checkpoint-training cost; it does not make training future models free. Its two new exports cost approximately28times less occupied-worker time than the amplitude fit/export for a similar-sized point effect. Worker wall time is a lane-cost proxy, not measured dollars or billed GPU utilization.

The actual paired component analysis uses468 evaluation sequences,141 components,808,863 selected rows and2,000 resamples. It reproduces the existing official grader to1e-12 before resampling:

| Comparison | Conditional95% interval for delta |
|---|---|
| Amplitude/s3 | [−.000225, +.000301] |
| Averaging/s3 | [−.000502, +.000481] |
| Averaging/s4 | [−.000597, +.000665] |

All cross zero. None includes fitting, teacher-cohort, adaptive-selection or future-transfer uncertainty. Therefore these are small, uncertain leads; no new public score, final endpoint winner or robust improvement is established. The cheap averaging arm can still justify a bounded follow-up because it costs little and adds no served network.

The actual bootstrap SDs range from.000135 to.000314, well below the historical scratch-bank.0022 comparison SD. This illustrates why Claude's universal.0064 cutoff is unsuitable. It does **not** make the measured small gains convincing: their relevant intervals still cross zero. Use the uncertainty of the actual paired experiment, cost and selection exposure together.

### Useful work without a quality result

- Both gradient-credit profiles passed40/40 optimizer updates at micro500/update2000, detach500 or2000. They occupied about15seconds each, with peak worker memory4.45GB/3.58GB and observed VRAM1.97GB/1.99GB. The shapes are feasible; gradient-credit quality remains untested.
- Causal moments/innovations pass the earlier full20,000-row CPU/streaming/ONNX parity checks. Independent review caught a real homogeneous-cache consumer mismatch before GPU launch. R2 reuses our executed narrow validator, and the independent Nemotron delta review passes. No representation quality result exists yet.
- The mirror diagnostic completed, but its interpretation needed correction: a within-sequence marginal shift does not refute population symmetry, and the sign-only negative control cannot meet its binary threshold. It earns caution about unjustified augmentation, not a scientific family rejection or a score gain.
- Three input-defined stress-panel families are frozen. The new paired scoring consumer has correctness tests, but these panels have not yet established a better robust complete endpoint. The full C bridge and three remaining panel families are outstanding.
- Historical auxiliary review located seven completed histories and scoped negatives. This supports avoiding exact repeated failures; the saved future compute is not quantified.

### Delivery that did not pay for itself yet

The F3 Kaggle run completed its actual T4 profile and then failed before fit admission. It produced no checkpoint or prediction. Approximately1,132seconds of its preparation were full train/valid Parquet hashing. A cloud profile and a corrected packet are useful operational assets, but they are not scientific returns.

The amplitude CP5 finished at01:20:53UTC and its export began at08:05:08UTC: a6.74hour handoff gap. Native-agent usage exhaustion interrupted ownership while the GPU job itself survived. A workflow that requires a live LLM turn for every reviewed successor remains too fragile. The elapsed gap is documented; other non-campaign GPU activity is not measured.

Too much effort still went to duplicated packets, mismatched CLI copies and validator/schema repair. Some safeguards caught genuine failures, but the work must now buy reusable delivery rather than another bespoke receipt sequence for each arm. No number of files or passed preparation checks compensates for missing endpoint comparisons.

## Comparing the reviews

| Question | Earlier public/private strategy | Claude first-principles plan |
|---|---|---|
| Immediate low-cost moves | Reuse predictions/checkpoints, then three mechanism screens | Explicit averaging and compact-member priorities; this supplied the best new cost-adjusted lead |
| Valid scientific comparisons | Strong: complete endpoints, exact targets, clean ancestry, fixed clocks, exposure and serving | Useful emphasis on repetition, but historical noise thresholds and several causal claims were overgeneralized |
| Focus and operational critique | Correctly identified backlog and end-to-end delivery, but retained a very large menu | Sharper about churn, underused ensembles and time spent on tiny knobs |
| Representation upside | Keeps distinct causal representations, memory and teacher-transfer paths alive | Too categorical about stopping several untested families |
| Statistical and schema restraint | Avoids physical-feature/target-law assumptions and a fresh-holdout fiction | Mirror/target-law suggestions require the corrections already recorded; four folds do not reset exposure |
| Objective and runtime | Retains the user's objective and validates actual serving endpoints | Runtime emphasis is useful; a cross-sequence callback switch needs causal/reset compliance, and the public>.700 ceiling claim is unsupported |

The imported branch contains a static verdict/plan, not its promised full review or completed experiment cards. Credit it with the ideas and criticism actually present. Likewise, our strategy gets no empirical credit for unrun innovations, cross-fitting, path areas or robustness mechanisms.

**The better operating policy is the earlier review's comparison discipline with Claude's cheaper-first ordering and stronger delivery focus.**

## Decision and next allocation

### Immediate sequence

1. **Finish harvesting the existing runs.** Preserve the three grades, uncertainty, source bindings and actual cost. Check eligible retained trajectories on another source-qualified fold before admitting new fitting just to test averaging. Any different checkpoint/horizon rule is registered separately and frozen before its outcomes. Then measure the selected average's actual Q8 quality and fixed-blend contribution. No F0-only submission selection.
2. **Run the representation pair next.** The innovations-versus-moments control is the most useful ready test of a different information representation. Complete both actual291-input profiles, then the paired fixed-CP5 fits. It can change the model's usable causal state rather than merely rescale almost identical teacher targets. Its numerical effect remains unknown. F3 confirmation uses clean inputs when delivered; missing F3 teachers are not a scientific negative.
3. **Keep gradient credit as the next independent mechanism.** Its resource uncertainty is now cheap and resolved. Run one matched detach500/2000 pair, keeping update/schedule/weighting clocks fixed. Do not begin a grid of horizons or attach optimizer changes.
4. **Complete bounded F3 delivery.** R4 is a reviewed conservative-memory successor, reusing the real device profile. Deliver checkpoint plus fit-train predictions, collect, measure actual charge/export time, then allocate remaining exports/seeds against refreshed quota. Do not launch a broad teacher-width ladder while the existing comparison pipeline lacks required teachers.

### Next exploitation and exploration

- **Compact ensembles become the first new design after the ready pair.** Compare H256×1/H192×2 with the current H256×2 pipeline at actual Q8 CPU time and package bytes. Resolve compatible independently trained parent construction first. More cheaper members must earn conditional blend value; equal parameter counts or an old H128 experiment do not settle it.
- **Propensity0/1/2 gets a bounded matched test.** It changes the objective materially and has conflicting historical evidence. Reuse source-qualified controls and test another contrasting fold before fine weight tuning. Do not combine it with a new representation until the interaction is an explicit arm.
- **Reserve one high-upside exploration slot.** Causal cross-fitted targets are a stronger teacher question than another arbitrary width/alpha grid, but need clean inner ancestry and substantial delivery cost. Privileged, relational or path-memory ideas remain eligible; choose one on a discriminating hypothesis and reusable artifacts, not all at once.
- **Defer more amplitude fine-tuning and public-score regression.** The tested amplitude has a tiny uncertain return for90minutes; the historical public regressor underperformed its simple baseline. Reopen on concrete contradictory evidence or a changed mechanism. These are allocation decisions, not universal refutations.

### Rules for spending the next wave

Keep at most two new full-fit recipe families active in a wave, each with an exact end-to-end owner. Preserve all39 items in the register; give them explicit future tests or scoped deferrals instead of opening39 simultaneous workstreams. Before adding a new arm, deliver the previous fit's checkpoint, predictions, ordinary score, conditional uncertainty and relevant blend/serving evidence.

Scale confirmation in stages: use a contrasting exposed fold and paired fitting seed; advance at most two complete finalists to the existing four-fold/two-repetition endpoint comparison, filling missing valid cells rather than refitting good controls. Do not copy the scratch-bank power threshold onto every mechanism, or require every interval to exclude zero. Expensive scale-out needs stronger information value than a nearly free checkpoint transformation.

Use one exercised recipe/export interface and the existing durable sequence tools for reviewed successors. Record phase costs; hash immutable inputs once per kernel when a verified version/path identity can safely support reuse. Preserve the underlying validation rather than repeatedly scanning the same35GB. Store only the minimal source snapshot/config differences needed for execution. No new controller framework is required.

Keep the current strongest supported clean complete endpoint as the measured fallback, explicitly labeling any mismatch to C. Finish the exact clean C bridge in parallel. Final release uses complete CV, actual Q8/runtime/ZIP measurements and the accepted.001 ordinary-CV robustness tradeoff. Public feedback confirms transfer; it does not choose blend weights or supply a forecast by adding local deltas.

## Limits and source records

This reallocation is based on the first delivered model wave and indexed historical evidence. It has not tested the whole39-item plan. No universally best recipe, private robustness improvement or new public score is claimed. CPU/RAM/single-worker/no-paid-capacity rules remain; native Luna roles are unavailable after their current usage-limit failure, so root owns delivery and uses authorized verified external read-only criticism where available.

- [Measured first-wave costs/effects](../../campaign_2026-10-13/strategy_execution_r1/FIRST_WAVE_COST_AND_EFFECT_R1.json) and [paired analysis/reproduction](../../campaign_2026-10-13/strategy_execution_r1/first_results_analysis_r1/README.md).
- [Amplitude packet and grades](../../campaign_2026-10-13/strategy_execution_r1/target_amplitude_f0_s3_r1/README.md), [averaging packet and grades](../../campaign_2026-10-13/strategy_execution_r1/trajectory_average_f0_r1/README.md), [gradient-credit profiles](../../campaign_2026-10-13/strategy_execution_r1/gradient_credit_f0_s3_r1/README.md).
- [Innovation R2](../../campaign_2026-10-13/strategy_execution_r1/innovation_f0_s3_r2/README.md) and [independent delta review](../../campaign_2026-10-13/strategy_execution_r1/reviews/innovation_f0_s3_r2/README.md).
- [Earlier strategy](CHALLENGE2_PUBLIC_PRIVATE_STRATEGY_20261012.md), [Claude plan](FIRST_PRINCIPLES_REVIEW_PLAN_20261012.md), and [source-check assessment](../ideas/FIRST_PRINCIPLES_EXPERIMENTS_20261013.md).
- [Objective](../../../specs/research-objective.md), [ADR0009](../../../specs/adr/0009-uncertainty-is-not-a-kill-switch.md), [current coverage](../../campaign_2026-10-13/strategy_execution_r1/COVERAGE.json).
