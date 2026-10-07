# Fresh weak-model complementarity scout

Role: read-only ideator; no files or experiments were changed by the scout.
Native provider: gpt-6-luna, reasoning effort max. The source assignment and
random draw are documented in [SOURCE.json](SOURCE.json) and
[RANDOM_PASSAGE.md](RANDOM_PASSAGE.md). Directed primary source: Krogh and
Vedelsby, *Neural Network Ensembles, Cross Validation, and Active Learning*,
NIPS 7 (1994), pp. 231–238; see [DIRECTED_SOURCE.md](DIRECTED_SOURCE.md).

The original proposal cards below were frozen before independent review. They
are hypotheses, not experiment results or numerical priors.

## Card 1: Cross-fitted residual satellite

- **Intervention:** Freeze a complete anchor endpoint and an independently trained, ancestry-audited weak candidate outside E-514's pool. Generate sequence-group out-of-fold predictions for both. Train a compact causal corrector to predict the anchor's label residual, using allowed input history and the weak candidate's output or hidden features. Add the corrector to the anchor with a coefficient fixed using fitting groups only; evaluate the complete endpoint under `global_wp_clip2_v1`.
- **Source connection:** Krogh–Vedelsby motivates testing whether a candidate's disagreement actually corresponds to lower ensemble error. The library passage suggests that recent state may forecast when a next-step contribution is reliable. Here, the candidate must predict anchor residuals after conditioning on the anchor; low correlation alone is not evidence.
- **Assumptions and failure points:** The weak model must carry residual information that the anchor does not. It may instead add noise, or the corrector may simply relearn the anchor from the same inputs. All candidate and anchor training, checkpoint selection, and residual targets need group-safe cross-fitting. Any use of clipped labels or residual transforms is a declared training choice; only the official scorer defines the endpoint.
- **Prediction and cheapest test:** First measure whether the cross-fitted candidate features predict anchor residuals on disjoint groups; then compare complete-endpoint WP on an already exposed group as development evidence only. Match against a same-capacity corrector without candidate features and one with shuffled candidate features. Falsify if the candidate features do not improve residual prediction and the complete endpoint fails to beat those controls or collapses to zero correction.
- **Artifact and cost:** A hash-bound cross-fitted prediction/feature cache, ancestry record, residual recipe, and paired endpoint report. Requires candidate prediction exports plus one small corrector fit; defer resource estimates until E-508 establishes an actual-device profile.
- **Tree and status:** New direction under `family:teacher` or `family:ensemble`, related to `arm:teacher-crossfit` and `arm:heterogeneous-teacher-compression`. Do not reopen E-513's invalid cohort or E-514's pool. **Deferred** until profile recovery and a new ancestry-audited candidate exist.

## Card 2: Disagreement-guided student training

- **Intervention:** Use cross-fitted anchor–candidate disagreement to identify training examples or sequence windows where the student needs more updates. Train the student from supplied labels, not teacher pseudo-labels, with bounded sampling and importance correction so the intended training distribution is explicit. The weak model leaves no output component in the served endpoint.
- **Source connection:** The primary paper also discusses using ensemble ambiguity to select examples for labeling. Here all supplied examples are already labeled, so the analogy is to allocate optimization effort where models disagree—not to active learning or to treating disagreement as correctness. The momentum passage's state-dependent, next-step decision is a loose analogy for prioritizing input contexts whose predictive difficulty may change over time.
- **Assumptions and failure points:** Disagreement must mark examples where extra updates help the student, rather than irreducible noise or harmless model variation. Sampling can distort the official objective if correction weights are mishandled. Compute may be spent on disagreement without improving either target.
- **Prediction and cheapest test:** Before fitting, check on cross-fitted outputs whether disagreement strata predict anchor residual/error under the official row mask and target weighting. Compare the highest-disagreement stratum with matched random strata. If that diagnostic is positive, run an equal-compute student comparison: disagreement-guided schedule versus uniform sampling, with a matched sham schedule. Falsify if disagreement does not predict relevant residual error or the complete student endpoint does not beat uniform and sham controls.
- **Artifact and cost:** A cross-fitted disagreement manifest, corrected sampling schedule, and matched-compute result. The initial diagnostic is CPU-only; a fit adds one paired student budget and should wait for E-508 profiling.
- **Tree and status:** New direction under `family:optimization` or `family:objective`, interacting with `arm:teacher-crossfit` and possibly `arm:dependence-regime-gate`. **Deferred**; a diagnostic can be proposed after profile recovery, but no source cache or compute is authorized by this scout assignment.

## Card 3: Input-state reliability gate for a new specialist

- **Intervention:** Train a new weak specialist on a predeclared, causally observable regime using its own weights. From sequence-group out-of-fold predictions, train one small continuous gate to predict whether the specialist's correction helps the anchor from current and lagged input features. Serve `anchor + gate(input_state) × specialist_correction` with fixed gate parameters. This uses neither E-514's pool, a scalar lambda grid, nor its conditional-average rule; it requires a new specialist and per-example residual supervision.
- **Source connection:** The Krogh–Vedelsby result says disagreement can help only insofar as it reduces prediction error under its stated squared-loss setting. The library passage motivates forecasting predictor reliability from recent state, analogous to changing a next-step position when forecast volatility changes. The analogy is about state-dependent trust, not about transferring a market strategy or using target history.
- **Assumptions and failure points:** A causal feature state must predict specialist advantage after conditioning on the anchor. The gate may overfit exposed groups, learn target-specific noise, or act as an expensive blend that contributes nothing. At inference it cannot update from test labels; official callback inputs contain only current and earlier features.
- **Prediction and cheapest test:** On cross-fitted specialist predictions, test whether weighted anchor residual contribution varies predictably with the predeclared input state. If it does, compare the single learned gate with a constant gate, a shuffled-state gate, and the unchanged anchor on an exposed outer group. Falsify if the learned gate fails to beat the constant and shuffled controls, selects negligible correction, or worsens the complete official endpoint.
- **Artifact and cost:** New specialist checkpoint and ancestry manifest, cross-fitted gate targets, fixed gate, and complete endpoint report. One specialist fit plus gate fitting and inference export; **deferred** until E-508 profile recovery and a distinct candidate pool.
- **Tree and status:** Candidate child of the weak complementarity direction, with interaction links to teacher cross-fit and heterogeneous teacher compression. **Proposed, untested, null prior, no launch authorization.**

## Scout priority

Keep E-508 profile recovery and the own-teacher path first; E-511 remains an uncertain separate comparison; heterogeneous teacher compression stays untested behind profile recovery; direct weak-model blending remains low and deferred. A new candidate with cross-fitted, target-aligned residual contribution that beats matched controls would justify moving residual or specialist work ahead of direct blending. None of these cards supports a numeric public-score forecast.
