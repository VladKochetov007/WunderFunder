# E526 — source-grounded reframe of review_2

Status: ideation only. No code, fit, score, or allocation was changed. The source brief was `review_2.txt` (SHA-256 `7d501713557cb2206e64574278285c738387bb7a913b7203690b980d9c219d90`, 523 lines). This was an independent Codex review; the requested separate native Luna/max role could not be started because all four collaboration slots were occupied. No external model call was substituted.

## What is new and what is already represented

- **H-232 already covers the wide-teacher × live event-feature question.** It compares the own GRU384 raw-only view with its full causal event-feature view. A width × feature *interaction* is a distinct extension only if it adds a matched compact-width pair and estimates the difference-in-differences; it must remain linked to H-212 and H-232.
- **H-137/H-138 and E-434 already cover activity/event-age conditioning of memory.** E-434 is still `planned_implementation_required`, not a negative fit. Do not register another “event activity gates retention” arm under a new name.
- **H-148 is the closest parent for learned decay or diagonal memory.** E-497 has a small CPU-validated prototype but no trainer integration, resource profile, or scientific fit. A bounded multi-half-life EMA and its learnable-decay control can be a concrete child/refinement; fixed-query pooling is a separate state/readout mechanism.
- **H-113/H-149 are post-state spline/readout directions.** A monotone transform *before* a structured input contrast changes a different stage. H-003’s still-probing rank-Gauss hypothesis makes it essential to retain raw values and avoid pretending transformed values are physical volumes/prices.
- **E-088/E-089 and H-034 are historical evidence against naive additive heuristic/propensity inputs.** They used older model/objective/evaluation recipes, so they are relevant controls rather than decisive evidence for the current recipe. They do not test a conditional bilinear correction, scale selection, or a structured pre-transform; they are not a blanket family rejection.
- **H-212 has a wide raw teacher but no matched compact-width result.** H-230 is a separate heterogeneous-student recipe. Neither establishes that width × features interacts or that a teacher helps one fixed student.

## Candidate cards

### E526-A — Low-rank conditional correction from hidden state and causal context

**Mechanism and source.** TimeXer separates endogenous and exogenous representations and uses directed interaction; Feature-aware Modulation conditions feature transforms on temporal context. Those papers motivate separating “what the recurrent state says” from “how a small causal context changes its interpretation.” This low-rank correction is our adaptation, not either paper’s model. The conditional-information notation in review_2 is a motivation, not something to infer from a score.

**Intervention.** Freeze an independently trained own GRU and add a small correction from projections of its hidden state and a preregistered causal feature vector: hidden projection, context projection, and their elementwise product feed a two-target residual readout. Initialize the final correction to zero so the starting prediction exactly equals the frozen model; clip only at the existing official inference point.

**Controls and test.** On identical train/evaluation groups compare: frozen anchor; same-size correction from hidden state only; additive hidden-plus-context correction; the interaction correction; and a calibration-only correction. Fit only on the declared training groups and assess on disjoint groups. Keep E-088/E-089 as relevant evidence for plain feature concatenation. Compare complete-path latency and memory, not parameter count alone.

**Observable / falsifier.** The interaction earns follow-up only if it improves the fixed official `global_wp_clip2_v1` endpoint over both the hidden-only and additive controls with stable group-level effects, while retaining causal-prefix/reset behavior. If it only matches the additive control, the interaction mechanism is unsupported for this recipe. Illustrative 32-wide projections over a 256 hidden state and 64 context values imply about 10.4k linear MAC/row before feature construction; measure the real path.

**Lineage / cost.** New conditional-readout direction, related to H-232 but not its wide-teacher input-view comparison. Reusing cached frozen states could make screening cheaper than a full retrain; a complete inference profile is still required. Numerical prior remains unknown.

### E526-B — Monotone scalar calibration before a structured contrast

**Mechanism and source.** The NeurIPS numerical-embedding paper shows piecewise-linear and periodic scalar embeddings can be useful modeling choices in tabular networks. LOBench documents that feature-wise z-scoring can break relative price-level order in its actual LOB data. Neither establishes benefit for Challenge2’s channels. In particular, the LOBench result is only a warning when the relation being modeled is known and preserved; it does not license assigning physical meaning to anonymous channels.

**Intervention.** For a small, preregistered set of verified channels, learn a bounded monotone piecewise-linear scalar map and form same-channel temporal deviations or explicitly declared learned contrasts. Preserve raw inputs and a separate missingness mask. Call cross-channel quantities “learned contrasts,” not volume, spread, or order-book ratios, unless the data schema proves those semantics.

**Controls and test.** Compare fixed raw-space contrasts; calibrated values with an unstructured matched-size readout; calibrated structured contrasts; and a parameter-matched ordinary nonlinear feature map. Share parameters across channels only where the schema justifies equivalence. Audit causal prefix, sequence reset, sentinel handling, and numerical range before any score.

**Observable / falsifier.** A useful signal must beat both the fixed-contrast and matched nonlinear controls on the same frozen groups; improvement only over an unparameterized baseline would not identify calibration as the cause. If channel semantics cannot be established, keep only per-channel operations and do not evaluate physical ratios.

**Lineage / cost.** Distinct from post-hidden-state H-113/H-149. Small input-map parameters but extra per-row transforms; numerical prior unknown.

### E526-C — Bounded multi-timescale causal EMA state (refinement under H-148)

**Mechanism and source.** TimeMixer motivates representing fine and coarse temporal structure separately. The proposed online EMA bank is a smaller recurrent implementation of that general multiscale idea; it is not the paper’s architecture and does not inherit its forecasting results.

**Intervention.** Maintain a tiny bank of causal means over preregistered channels, with half-lives spanning short through longer row horizons. Compare fixed half-lives with learned, bounded half-lives; only after that contrast is informative, test a causal context-conditioned mixture over the bank. Reset every bank at each sequence boundary and preserve state across chunk boundaries.

**Controls and test.** Match hidden/state dimension and output head across fixed and learned variants. Include a single-decay control, current activity versus stale/shuffled activity, and a static branch where feasible. Test row-by-row/chunk parity, prefix invariance, reset and gradients through the actual training truncation length. Long forward memory is not evidence of a long gradient horizon.

**Observable / falsifier.** The learned bank must improve over its fixed bank and single-decay controls, not merely over the original GRU. If learned decays collapse to one timescale, gradients fail, or complete-path cost erases the gain, defer this exact variant without rejecting H-148 or all memory mechanisms.

**Lineage / cost.** Child/refinement of H-148, cross-linked to H-137/H-138 and E-434 only for the later activity-conditioned mixture. Four scales × sixteen channels require 64 scalar states per sequence; actual compute, gradients, and batching must be profiled. No use of real elapsed-time semantics is assumed from row index.

### E526-D — Fixed-query discounted pooling over recurrent states

**Mechanism and source.** The supplied review describes attention pooling from an attached paper, but the repo has no PDF, authors, DOI, or resolvable exact-title primary citation for “Transformer for Forecasting High-Frequency Financial Time Series.” Treat that paper-specific account as unverified. The proposed streaming numerator/denominator recurrence is a standalone hypothesis: fixed learned queries assign content weights to GRU states, with an exponential discount so state remains bounded and causal.

**Intervention.** Project each current hidden state to a small vector; for a few fixed queries update discounted weighted-state numerators and scalar denominators. Append their normalized summaries to a small readout. Because the query is fixed, old weights need not be recomputed. A query conditioned on the current state would not have that recurrence and is out of scope for this card.

**Controls and test.** Use identical decay and head for (1) zero query, which reduces to ordinary discounted averaging, (2) frozen random query, and (3) learned query. Include a no-pooling recurrent baseline. Check stable denominator/numerics, exact sequence reset and chunk parity. Keep query count low and report extra per-row work and state.

**Observable / falsifier.** Learned content selection must outperform zero-query pooling, not just the no-memory baseline. If random and learned queries are equivalent, learning the query is unsupported. The attached paper cannot be used as empirical support until its identity is recovered.

**Lineage / cost.** New recurrent-state pooling direction adjacent to, but distinct from, H-148’s learned diagonal/residual memory. With query count K and projected width d, the recurrence is O(Kd) per row and holds O(Kd) numerator plus O(K) denominator state.

### E526-E — Parameter-efficient ensemble of small readouts on one frozen recurrent trunk

**Mechanism and source.** TabM’s primary ICLR paper studies parameter-efficient ensembling of MLPs using BatchEnsemble plus paper-specific changes. A few readout members over one recurrent trunk is an adaptation, not TabM itself. Shared parameters save storage; they do not make the K member computations free.

**Intervention.** Keep one own-trained causal GRU trunk and train a small fixed number of member-specific readout adapters over the same hidden/context representation. Average member predictions with weights fixed before evaluation; do not tune member weights on the exposed evaluation fold.

**Controls and test.** Compare one ordinary readout at matched parameter budget, K shared-trunk members, and K independent readouts at matched training/inference budget. Record member disagreement only as a diagnostic; do not call it calibrated uncertainty without a calibration study. Compare both score and full-path latency/VRAM.

**Observable / falsifier.** Follow only if the shared-member endpoint improves the official metric over one matched readout and its benefit survives the fixed averaging rule. Stop this exact low-cost candidate if gains vanish under the control or inference cost is unacceptable; no claim about ensemble families follows.

**Lineage / cost.** New readout-optimization direction, not blending old weak model outputs. Requires K branch computations at inference and its own actual profile. Numerical prior unknown.

### E526-F — Preserve raw regime context beside causal normalized innovations

**Mechanism and source.** TimeBridge explicitly separates short-horizon non-stationarity treatment from long-horizon dependency preservation. Feature-aware Modulation studies temporal context changing feature scales/skew. Those results motivate a test, not the exact streaming normalizer below.

**Intervention.** For selected channels maintain previous-row-only means and stable variances. Give a small branch raw value, normalized innovation, and slow mean/scale context; preserve the original raw path. Exclude missing sentinels from statistics, model missingness separately, initialize from train-only statistics, and reset state per sequence.

**Controls and test.** Compare raw only; raw plus moments; and raw plus moments plus normalized innovation at the same head capacity. Freeze the channel set, update schedule, epsilon and warm-start before scores. Verify no current/future target data enters the estimates and no new output de-normalization is applied.

**Observable / falsifier.** If raw-plus-moments alone helps as much as adding normalized values, the benefit is regime context, not normalization. If normalized innovations fail against that control, drop only this representation. Results on exposed folds remain developmental.

**Lineage / cost.** A normalization-context refinement adjacent to H-232’s input-view change; it isolates preserved slow context versus standardized local deviation. Adds causal per-channel state and per-row arithmetic; actual profile is mandatory.

### E526-G — Width × feature-view factorial and fixed-student teacher test

**Mechanism.** H-232 is a single-width feature-view interaction; H-212 asks about wide raw teacher capacity versus a compact raw basis. A matched 2×2 design could test whether width helps exploit the added causal feature view, rather than attributing the answer to only the wide cell.

**Intervention.** If a later allocation supports it, preregister compact/wide widths × raw/full causal feature views with the same own training recipe, targets, optimizer, update/EMA clocks, epoch rule and seeds. Estimate the difference-in-differences of the feature effect across widths. Then, in a separate declared experiment, distill each source into the *same fixed student* and compare against the matched raw-teacher/student control; source score alone is not teacher value.

**Controls and test.** Four matched cells are needed for the interaction; identical seeds do not estimate seed variance, so replicate seeds if the allocation supports it. Keep target treatment fixed. Freeze the student and grader before inspecting source scores. Report measured training and inference costs; do not add parent score gains.

**Observable / falsifier.** A feature gain at wide width alone is not an interaction. The diff-in-diffs must be positive on a declared development endpoint with paired group/chain uncertainty; teacher utility is a separate fixed-student endpoint. A weak standalone teacher remains eligible as a component if its fixed-student contribution is useful.

**Lineage / cost.** Explicit interaction node linked to H-212 and H-232; no numeric prior. Cost is at least four teacher fits before seed replication, plus separate fixed-student comparisons, so this is not the cheap first test.

## Primary-source checks and limits

The checks below use publisher/conference pages or the authors’ arXiv record, not secondary summaries.

| Source | Confirmed source fact | What it does not establish here |
|---|---|---|
| [TimeXer, NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/0113ef4642264adc2e6924a3cbbdf532-Abstract-Conference.html) | Exogenous and endogenous variables are treated distinctly, with patch self-attention, variate cross-attention and global endogenous tokens. | A bilinear GRU correction or gain on this task. |
| [Feature-aware Modulation, NeurIPS 2025 paper](https://proceedings.neurips.cc/paper_files/paper/2025/file/ddc20f403d0d296c87940393838d5971-Paper-Conference.pdf) | The paper proposes temporal-context-conditioned feature modulation, including changing scale/skew properties. | Transfer from its temporal-table benchmarks or its model to anonymous Challenge2 inputs. The official abstract page returned 503 during this check; the linked official proceedings PDF was available in the primary search index. |
| [Numerical feature embeddings, NeurIPS 2022](https://proceedings.neurips.cc/paper_files/paper/2022/hash/9e9f0ffc3d836836ca96cbf8fe14b105-Abstract-Conference.html) | Piecewise-linear and periodic numerical embeddings are studied as design choices and reported to help on their tabular benchmarks. | That a learned transform before a handcrafted contrast will help here. |
| [LOBench, arXiv v1](https://arxiv.org/html/2505.02139v1) | Its LOB data section reports that feature-wise z-score can break cross-level price ordering and create large relative deviations. | Challenge2 has those same channel semantics, or that all per-channel normalization is harmful. |
| [TimeMixer, ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/a7ac8a21e5a27e7ab31a5f42a0117bdb-Abstract-Conference.html) | It mixes decomposed temporal scales in its MLP forecasting model. | An EMA bank is equivalent to TimeMixer or inherits its empirical results. |
| [TabM, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/c1ba41c694834aeef91ae161711d4939-Abstract-Conference.html) | It studies parameter-efficient MLP ensembling based on BatchEnsemble and added modifications. | A shared recurrent trunk/readout design has the same efficiency or accuracy. |
| [TimeBridge, ICML 2025](https://proceedings.mlr.press/v267/liu25cb.html) | It treats short-term non-stationarity and long-term dependencies with distinct attention mechanisms. | The proposed causal running moments are the paper’s method or transfer to this task. |
| [Scaling laws for large time-series models, arXiv](https://arxiv.org/abs/2405.13867) | Reports power-law scaling for large decoder-only time-series models trained over heterogeneous data and large ranges of model/data/compute. | A scaling law for GRU128/256/384 on this dataset. |
| [Scaling paradox in large TS models, ICML 2026](https://proceedings.mlr.press/v306/qiu26q.html) | Finds underused layers and non-monotonic scaling in much larger TS foundation-model families. | That a wide Challenge2 GRU is harmful; use it as a reason to measure width × features rather than extrapolate. |
| [Inference-compute frontier for LOB, arXiv 2026](https://arxiv.org/abs/2606.25986) | A preprint on FI-2010 reports structural compute and measured latency have different empirical relationships. | A latency estimate for the current GRU or Challenge2 hardware. |

The exact primary source for the “Transformer for Forecasting High-Frequency Financial Time Series” named in review_2 was not present in the repository and did not resolve by exact-title search. Its multi-clock and pooling description is therefore retained as an unverified literature lead, not evidence. All folds mentioned in existing local work are exposed; any future local screen is development evidence under the unchanged official `global_wp_clip2_v1` scorer, never a public forecast. No numerical priors are assigned.

## Actual method and model provenance

This was a bounded review of the complete 523-line source note, current H-137/H-138/H-148/H-212/H-232 records, E-434/E-497/E-088/E-089 evidence, the HD-1–HD-8 slate, and H-232’s current card. Primary sources were checked via the URLs above. Current agent: Codex API reviewer; exact model build is not exposed in the tool manifest. No code from the neighboring actor was copied or adapted. No native Luna or external model call was executed.
