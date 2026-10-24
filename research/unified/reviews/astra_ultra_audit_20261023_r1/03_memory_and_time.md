---
kind: review
status: complete
created_utc: 2026-10-23T16:49:47Z
updated_utc: 2026-10-23T16:49:47Z
hypothesis_ids: [H-ASTRA23-03-A, H-ASTRA23-03-B, H-ASTRA23-03-C, H-ASTRA23-03-D]
experiment_ids: []
evidence_paths:
  - research/unified/reviews/astra_ultra_audit_20261023_r1/CONTRACT.md
  - research/campaign_2026-10-21/next_cycle_r1/i1/runtime/GRADE_R1.json
  - research/campaign_2026-10-21/next_cycle_r1/i1_warm_history_extension_r1/PREFLIGHT_F0_RESULT_R3.json
  - research/campaign_2026-10-21/next_cycle_r1/i1_warm_history_extension_r1/PREFLIGHT_F1_ORDER_RESULT_R1.json
  - research/campaign_2026-10-21/next_cycle_r1/WARM_REMAINDER_COST_DEFERRAL_R1.json
  - research/campaign_2026-10-14/cpu_student_frontier_r1/STATE_DECAY_DIAGNOSTIC_R1.md
  - src/wnn/training/forward_backward.py
  - src/wnn/training/online_relationship.py
  - src/wnn/training/age_readout_model.py
---

# Memory and time: preserve useful state before replacing its clock

## Scope and judgment

The strongest missing distinction is between **retaining history, learning which history matters, and extracting that information at prediction time**. Current evidence does not establish inadequate memory duration as the central bottleneck. It supports a small complementary history endpoint, an already persistent recurrent state, and several unresolved ways to access or train that state.

This is a tier-C, source-based review under the [audit contract](CONTRACT.md). I inspected small text receipts and source, and browsed primary literature. I ran no fits, tests, imports, profiles, data/array reads, exports or provider jobs. I did not inspect sibling reports. All proposed tests below are unexecuted and require an explicit research resume and separate resource admission. The October 23 stop remains binding; historical admissions are evidence, not authority.

## Five findings

### 1. History complementarity is real within the measured panel; a memory-deficit explanation remains unidentified

The [six-fit grade](https://github.com/VladKochetov007/WunderFunder/blob/c3e208b387ead0ec967f9abc4b14275eb8ba94c5/research/campaign_2026-10-21/next_cycle_r1/i1/runtime/GRADE_R1.json) records the fixed 90%B/10%C mixture gaining +0.00035953 on F0/1001 and +0.00027145 on F2/1002. Conditional component intervals are [−0.00001187,+0.00070893] and [+0.00008223,+0.00043398]. C alone loses −0.00353711 and −0.00390342. These are cold H512×1 fits, three epochs, fixed CP2, 10% fitting-component membership; they do not measure strong-parent adaptation or production benefit.

The [progression](https://github.com/VladKochetov007/WunderFunder/blob/c3e208b387ead0ec967f9abc4b14275eb8ba94c5/research/unified/ideas/I1_HISTORY_IDEA_PROGRESSION_20261022_R1.md) correctly identifies two competing explanations. C activates random input columns that B masks, changing the initial function and optimization trajectory. Also, the i1-removal A model gives an even larger fixed mixture gain on F2 (+0.00054446), although it is flat/slightly negative on F0. Thus useful diversity does not uniquely identify recovered temporal information. The F0 geometry diagnostic weakens clipping-only and scalar-shrinkage explanations, not optimization diversity.

**Disposition: retain the fixed history contribution; narrow its explanation.** Longer context is one hypothesis, alongside easier access to summaries and different regularization. Two fold/seed cells do not disentangle seed, fold or training variability. All folds remain exposed. A short-budget loss of C alone could also reflect slower optimization of the enlarged active input basis.

The actual history scales also matter: [source](../../../../src/wnn/i1_history.py) fixes half-lives of 32 and 512 rows. In a mature 512-row EMA, only about 6.7% of kernel mass lies more than 2,000 rows back (`2**(-2000/512)`). This is a mathematical property of the proposed filter, not a measured attribution. Its gain could primarily concern accessible summaries *within* the current credit horizon, rather than recovery of much longer context.

### 2. Forward memory survives truncation, while credit and training-state consistency may not

The [state-decay diagnostic](../../../campaign_2026-10-14/cpu_student_frontier_r1/STATE_DECAY_DIAGNOSTIC_R1.md) perturbed one clean-F0 H256×2 checkpoint on one fitting sequence. Prediction-difference RMS from the row-zero perturbation remained 0.001373 at rows 1,000–4,999; the second layer retained more perturbation than the first. This refutes the universal claim that 99 warmup rows erase initialization. It does **not** establish that the persistent direction stores useful target information, characterize other parents, or justify retaining every old observation.

The current [standard forward path](../../../../src/wnn/training/forward_backward.py) carries hidden values but detaches the returned state each chunk. The [runner](../../../../src/wnn/training/runner.py) then updates parameters before processing the next window. The warm recipe uses 2,000-row chunks. Consequently:

- Losses after a detach cannot directly teach the earlier state construction through that boundary, even if its effect persists.
- The next chunk starts with state generated under earlier parameter values. This is ordinary TBPTT behavior, not a confirmed bug; its magnitude and quality impact here are unmeasured.
- Extending the gradient horizon by changing chunk length alone also changes update cadence, kernel shape and loss aggregation. It is not a clean memory experiment.

The repository already contains a carefully separated [S14 detach-500 versus detach-2000 proposal](../../../campaign_2026-10-13/strategy_execution_r1/gradient_credit_f0_s3_r1/README.md), with fixed 500-row microchunks and 2,000-row updates. Its profiles completed; scientific fits remained unrun in that packet. Credit assignment is therefore an unfinished existing question, not a newly discovered architecture family.

For the [prefix initializer](../../../campaign_2026-10-14/cpu_student_frontier_r1/INITIALIZER_IMPLEMENTATION_R1.md), a residual injected before row 99 can influence late predictions while receiving training credit only from its connected initial chunk. A weak initializer result under that regime would not settle full-sequence conditioning. Conversely, injecting useful information repeatedly can create a persistent nuisance or sequence-identity shortcut. Neither longer persistence nor larger state norm is a quality objective.

### 3. Initial-function preservation is now checked; preservation during learning is not

The warm extension zeroes the 156 appended projection columns and preserves the inherited normalization. Actual-parent [F0](https://github.com/VladKochetov007/WunderFunder/blob/c3e208b387ead0ec967f9abc4b14275eb8ba94c5/research/campaign_2026-10-21/next_cycle_r1/i1_warm_history_extension_r1/PREFLIGHT_F0_RESULT_R3.json) and [F1](https://github.com/VladKochetov007/WunderFunder/blob/c3e208b387ead0ec967f9abc4b14275eb8ba94c5/research/campaign_2026-10-21/next_cycle_r1/i1_warm_history_extension_r1/PREFLIGHT_F1_ORDER_RESULT_R1.json) receipts report maximum output error 0.0 on their selected complete fitting sequences, for both control and history children. They also check paired initialization, statistics and sampler identities. Earlier notes saying these gates remain pending are historical.

This removes the intended *initial* function confound on those checks. It does not ensure that fine-tuning leaves the useful recurrent representation intact: all trainable inherited parameters can move after the first update. A history channel may contain complementary information while the adaptation damages stronger existing features. Zero initialization protects time zero, not the fitted endpoint.

The [remaining-panel receipt](https://github.com/VladKochetov007/WunderFunder/blob/c3e208b387ead0ec967f9abc4b14275eb8ba94c5/research/campaign_2026-10-21/next_cycle_r1/WARM_REMAINDER_COST_DEFERRAL_R1.json) preserves one completed F0 control and its export, with three fits deferred for cost. There is no paired warm quality result. **Retain this scientific question; do not call its delivery failure a history failure.** A frozen-state residual experiment is a structurally different alternative: it asks whether history can help without moving the original state dynamics. It cannot replace the original frozen panel retrospectively.

### 4. Event time can be useful without deleting row-time dynamics

For a recurrent transition, equal consecutive raw inputs do not imply an identity update: generally, `F(x,h) != h`. Repeating x can advance a latent estimate, decay uncertainty or implement elapsed-row dynamics. Moreover, the 211-feature stream contains clocks that change even when raw112 is unchanged. The [H-396 feasibility review](../H396_EVENT_TRIGGERED_GRU_FEASIBILITY_20261023_R1.md) confirms that the current native path always advances state and that a gate still needs current features, projection and readout. A skip count therefore proves neither preserved semantics nor runtime savings.

The source analogy is weaker than it first sounds. [Skip RNN](https://arxiv.org/pdf/1708.06834), inspected in its introduction and model equations, **trains** a skip mechanism with the network and can penalize its update budget. H-396 instead proposes an input-novelty rule applied to an already row-trained checkpoint. This changes the transition operator under a distribution the checkpoint was not trained to handle. Failure would narrow that retrofit, not event-time learning generally.

Likewise, GRU-D's observation intervals describe missing measurements. Here an unchanged field is an observed value; its age is time since a change, not time since observation. The [existing field-age note](../../ideas/EVENT_STATE_AND_METRIC_OBJECTIVE_REFRAME_20261015_R1.md) already preserves this distinction and reports an input-only eight-sequence resolution check. Those clocks differ from group clocks, but predictive usefulness is unmeasured.

**Modify event-time exploration:** retain the original row-time GRU, and expose a separate event-time summary with elapsed-row age. This preserves both clocks and lets a task-trained readout determine their value. Current [DualClockModel source](../../../../src/wnn/nn.py) provides an architectural precedent within the repository; a proposal should reuse its semantics where appropriate rather than claim the family is absent. A new narrow sidecar still needs its own parity and serving work.

### 5. The current cheap heads cannot tell us what the parent already knows

The [online residual head](https://github.com/VladKochetov007/WunderFunder/blob/c3e208b387ead0ec967f9abc4b14275eb8ba94c5/src/wnn/training/online_relationship.py) consumes the 419 supplied features. Parent predictions are added after the head; they are not conditioning inputs. The [age head](https://github.com/VladKochetov007/WunderFunder/blob/c3e208b387ead0ec967f9abc4b14275eb8ba94c5/src/wnn/training/age_readout_model.py) similarly maps 323 features to a correction without the parent's hidden state or predictions as regressors. Yet the parent's own [forward](../../../../src/wnn/nn.py) reads both recurrent output and current encoding.

This is a deliberate inexpensive model class, not a correctness defect. Its limitation matters: an age effect could depend on a latent regime already encoded in h. A head that cannot inspect h may miss that interaction, or learn a misleading average correction. Conversely, a stronger readout of existing h might recover most of the benefit attributed to new memory. Neither possibility has been discriminated.

The [online panel](../../ideas/ONLINE_RELATIONSHIPS_20261022_R1.md) remains incomplete and ungraded. Even a future C-over-B gain would not establish nonstationarity: the warmup-frozen arm has only 83 matured outcomes and fewer genuine regressors, while continual updating also reduces estimation error in a stationary relationship. Future comparison needs an equal-information stationary estimator, not only a very short warmup control.

The [future-input auxiliary](https://github.com/VladKochetov007/WunderFunder/blob/c3e208b387ead0ec967f9abc4b14275eb8ba94c5/research/unified/ideas/FUTURE_INPUT_DYNAMICS_AUXILIARY_20261021_R1.md) could train useful state, but forecasting group mean changes and log energies is not intrinsically aligned with the two targets. Its [source](../../../../src/wnn/training/dynamics_auxiliary.py) includes 16/256-row horizons and all valid horizon rows; a successful auxiliary forecast could prioritize predictable activity that contributes little to official weighted target score. Its cost deferral supplies no negative scientific evidence. Retain it as an alignment hypothesis, with target improvement required; do not treat input predictability as proof that causal market dynamics have been learned.

## Seven questions that would change allocation

1. How much useful prediction depends on recurrent context older than 99, 512 or 2,000 rows, after the existing causal feature clocks are held identical?
2. Is the persistent state-decay direction target-relevant across components, or mostly harmless initialization sensitivity? Which layers and positions account for the difference?
3. Does a small correction conditioned on current parent h improve the endpoint before any extra history is supplied? Does history still add value conditional on h?
4. Does history adaptation lose quality by changing the original recurrence, despite passing initial parity? Would freezing it retain the complementary signal?
5. Does error track row distance, event count or the disagreement between them? Is a per-field clock transferable across activity regimes, or a proxy for familiar sequence identity?
6. Do matched detach horizons change late target quality at fixed optimizer cadence, or is the apparent benefit explained by state refresh, loss normalization or more updates?
7. For online relationships and future-input auxiliaries, do improvements survive controls for estimator maturity and predictable input nuisance, and reach the fixed target endpoint?

## Proposed hypothesis cards

All cards have status **not_tested**. They are alternatives, not a cumulative execution queue. Future uncertainty summaries must keep component resampling separate from fitting variability; F0/F1 parent comparisons also change ancestry. No public-score delta is predicted.

### H-ASTRA23-03-A — measure useful recurrent context before lengthening it

**Family/change:** state diagnostics; compare full-prefix state with states reconstructed from finite suffixes, keeping the full causal engineered feature stream identical. Related: [initial-state decay](../../../campaign_2026-10-14/cpu_student_frontier_r1/STATE_DECAY_DIAGNOSTIC_R1.md) and [history progression](https://github.com/VladKochetov007/WunderFunder/blob/c3e208b387ead0ec967f9abc4b14275eb8ba94c5/research/unified/ideas/I1_HISTORY_IDEA_PROGRESSION_20261022_R1.md).

**Assumption/mechanism:** useful dependence on older recurrent context should appear as a target-quality penalty when that context is removed, not merely as hidden-state distance. **Predicted observable:** the gap shrinks with increasing suffix length; a residual gap after 2,000 rows identifies useful longer recurrent dependence in this endpoint.

**Cheapest test/control:** two pinned clean parents, zero new fits/teachers. Before target access, choose eight source-qualified components per fold using a fixed hash ordering, and landmarks 4,096/8,192/16,384. Compare normal state with zero state introduced K=99/512/2,000 rows before each landmark; score the following 256 rows. Only recurrent state is intervened upon; feature clocks, normalization, inputs and numerical backend remain identical. Record prediction differences, both target scores on the identical official-mask subset, and recovery across that window.

**Falsifier:** negligible score/prediction difference after K=512 or 2,000 weakens the claim that this parent requires older recurrent state on this diagnostic sample. It does not prove older input history has no recoverable signal. **Cost/artifacts:** one streamed full pass plus nine bounded suffix branches per sequence; no full hidden-state cache. Actual runtime/RAM are unmeasured and require admission. Reuse parent/checkpoint/split identities. **Ancestry/exposure/serving:** exposed diagnostic subsets, no endpoint promotion; all full-prefix information in engineered features remains available by design. No serving change.

### H-ASTRA23-03-B — separate readout access from missing history

**Family/change:** frozen-state readout; three equal-width, zero-output residual arms with block masks: A=current inputs; B=current inputs plus current parent recurrent output; C=B plus the frozen 156-channel history. Related: [online heads](../../ideas/ONLINE_RELATIONSHIPS_20261022_R1.md) and [function-preserving warm history](../../../campaign_2026-10-21/next_cycle_r1/i1_warm_history_extension_r1/README.md).

**Assumption/mechanism:** useful state already exists but the correction needs access to it; history may then provide an additional accessible summary. **Predicted observable:** B−A measures the value of exposing h to this head; C−B measures incremental history conditional on that access. Require every corrected endpoint to be compared with the unchanged parent.

**Cheapest test/control:** three new heads per cell on two exposed fold/seed cells: **six fits total**, zero GRU/teacher fits, two reused unchanged parent endpoints. Match head width, initialization, optimizer, fitting rows and checkpoint. Freeze parent dynamics throughout. **Falsifier:** B adds nothing and C does not improve over B/parent at the tested capacity; alternatively, C helps A-like access but not B, favoring easier decoding over missing memory. This is not an information-theoretic sufficiency test.

**Cost/artifacts:** share parent forwarding across the three heads; stream per sequence or cost a disk cache explicitly. Existing caches lack h, so this is not a free cache-only experiment. **Ancestry/exposure:** fit-side parent residuals remain in-sample; hold paired memberships fixed and report that mismatch. **Serving:** native h is available internally but needs a supported hook; added dense work and reset parity remain unmeasured.

**Counterevidence to preserve:** the [event-state reframe](../../ideas/EVENT_STATE_AND_METRIC_OBJECTIVE_REFRAME_20261015_R1.md) reports older E-415 frozen-state age heads losing to the untouched parent on 6,158 sampled F0 rows. That four-clock screen is not this conditional 156-history comparison, but it prevents presenting frozen-state correction as a new or already successful family.

### H-ASTRA23-03-C — a second clock without state deletion

**Family/change:** event-time sidecar; compare no extra summary, row-time EMA summaries, and event-time EMA summaries, all attached to an unchanged parent through matched residual heads. Related: [H-396](../H396_EVENT_TRIGGERED_GRU_FEASIBILITY_20261023_R1.md), H-231 and existing dual-clock source.

**Assumption/mechanism:** variable event density dilates a row-based history scale. **Predicted observable:** event-time summaries help especially when row and event ages disagree, while preserving quiet-row parent behavior. Freeze an input-only event definition and ties; calibrate event half-lives from fitting event rates to match the row summaries' typical span, and retain elapsed-row age.

**Cheapest test/control:** three matched heads per cell, **six total fits**, zero GRU/teacher fits, two reused parents. Use identical readout access and width across arms. **Falsifier:** event summaries fail against both row summaries and parent, or their gain is confined to familiar activity rates and reverses across cells. **Cost/artifacts:** O(channels) streaming summaries and small heads; parent feature generation still occurs every row. Actual full-path cost is unknown. **Ancestry/exposure:** fitting-only clock calibration, exposed components, no physical-time claim. **Serving:** preserve all existing state/reset semantics; only sidecar state is new. This tests event representation, not inference speed.

### H-ASTRA23-03-D — complete a clean temporal-credit comparison

**Family/change:** gradient-credit allocation; a strict-FP32 successor of [S14](../../../campaign_2026-10-13/strategy_execution_r1/gradient_credit_f0_s3_r1/README.md), with fixed 500-row microchunks, 2,000-row optimizer windows, and detach at 500 versus 2,000.

**Assumption/mechanism:** useful signal crosses the shorter detach boundary and is trainable at the available fidelity. **Predicted observable:** longer credit improves target/late-window quality at identical parameter initialization, forward context, update count, loss denominator and LR schedule. **Cheapest test/control:** two matched arms on two fold/seed cells, **four new scientific fits**, zero new teacher fits. Reuse permitted parent/teacher artifacts; historical FP16 profiles are not strict-FP32 controls.

**Falsifier:** no reliable directional endpoint advantage at comparable cost, despite confirmed gradient-path differences. This narrows 500-versus-2,000 credit, not all long-memory architectures. **Cost/artifacts:** four continuations plus exports; retain up to four microchunk graphs in the longer arm and reprofile memory. Reuse controlled-loss code after checking compatibility with the selected current objective. **Ancestry/exposure/serving:** report all paired deltas and parent lineage; exposed robustness evidence. Serving topology is unchanged. State staleness remains common to both arms and cannot explain their contrast by itself.

## Primary literature: inspected claims and transfer limits

- [Tallec and Ollivier, *Unbiasing Truncated Backpropagation Through Time*](https://arxiv.org/pdf/1705.08209), §§1–3 and equations 8–9: truncation cuts recurrent gradient terms while hidden values may persist. This supports distinguishing credit from memory. It does not show that ARTBP's variance/cost tradeoff benefits this task; no ARTBP fit is recommended here.
- [Tallec and Ollivier, *Can recurrent neural networks warp time?*](https://arxiv.org/pdf/1804.11188), §1 and equations 3–7: gates can represent changes of time scale. This concerns a model class's representational capability, not invariance of a particular fitted model to skipped rows. Reinitializing a strong parent's gates would destroy its starting function.
- [Campos et al., *Skip RNN*](https://arxiv.org/pdf/1708.06834), §3, equations 1–6: a learned update/copy rule and budget objective motivate conditional recurrence. They do not validate H-396's post-training novelty threshold or establish a local speedup.
- [Che et al., *GRU-D*](https://www.nature.com/articles/s41598-018-24271-9), retrieved methods and limitations passages: missingness masks, intervals and trainable decay can be informative. This supports a timing representation question; unchanged observed fields are not missing values. The paper explicitly limits benefit when missingness is uninformative.

## Best next test, counterarguments and decision

**Proposed and unexecuted: H-ASTRA23-03-A, the bounded finite-context replay.** It provides a useful decision before another architecture or head panel: does old recurrent context improve the current predictor on the sampled targets, rather than merely perturb its outputs? Freeze its landmarks, component identities, official-mask subset and backend before scoring. No broad threshold sweep or automatic follow-up is proposed.

If short reconstruction recovers the predictor, deprioritize generic longer-memory claims and consider readout/representation access. If old state is useful, preserve it while comparing a sidecar or matched credit treatment. Either outcome leaves open the possibility that the parent fails to encode additional useful history.

The strongest objection is that resetting h creates off-distribution states, so a loss can reflect poor recovery rather than a necessary old signal. Increasing K, measuring recovery, and keeping engineered clocks identical help characterize this limitation but do not eliminate it. A second objection is that frozen-state heads cannot create a representation the parent never learned; their failure cannot reject end-to-end history. A third is operational: even zero-fit replays can be expensive, and existing paired warm-history work already has sunk preparation value. Root should compare actual complete costs after a resume, rather than automatically preferring the new probe.

The immediate deliverable is this review. No new empirical claim, scientific admission, submission recommendation or experiment continuation follows from it.
