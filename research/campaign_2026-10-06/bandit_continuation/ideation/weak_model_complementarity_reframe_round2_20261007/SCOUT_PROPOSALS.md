# Fresh weak-model complementarity reframe — proposal cards

**Role:** no-fit independent ideator. **Native provider/model:** Codex
subagent, `gpt-6-luna`, reasoning effort `max`. **Research-reframe mode:**
fresh-context; proposals were committed before reading competing ideas.
Selection and prompt provenance are in [SOURCE.json](SOURCE.json) and
[PROMPT.md](PROMPT.md). These are hypotheses, not measurements or numerical
priors.

The scout used only the assigned NIPS paper and the random library draw. The
NIPS ambiguity analysis is grounded in squared-error decomposition; it does not
establish gains on `global_wp_clip2_v1`. The randomly selected library page is
from a VC-dimension exercise on polynomial classifier classes. It is used only
to suggest bounded-capacity interaction features, not as evidence that those
features will help this task. [Krogh and Vedelsby (1994)](https://papers.nips.cc/paper/1001-neural-network-ensembles-cross-validation-and-active-learning)

## Card 1: Ambiguity-ranked training allocation

**Mechanism family:** data and gradient allocation.

Train a small set of independent, diverse models and use their out-of-fold
prediction variance to rank training sequences or time points. Refit the strong
anchor with a fixed update budget but allocate more updates to high-ambiguity
examples. Keep uniform sampling and matched random reweighting as controls.
The weak models are absent at inference; this changes training allocation,
not output blending.

Krogh and Vedelsby define ensemble ambiguity as output variance around the
ensemble mean and use it to identify candidate inputs for active learning. Here
labels are already available, so allocating gradient budget by ambiguity is an
extrapolation. Positive evidence would require out-of-fold ambiguity to enrich
for anchor errors on separate groups, followed by a fixed-budget refit beating
uniform and matched-random allocation. Falsify if the enrichment is absent or
the guided refit fails either control. Generate each ambiguity score without
fitting on the entire sequence group it scores; reserve group-safe evaluation
groups outside model fitting and allocation choices. Cost: two or three compact
cross-fit score models plus one anchor refit. Preserve fold membership,
score-generation ancestry and the fixed allocation rule. Keep below active
own-baseline/raw-teacher recovery; open only after repeatable group-level error
enrichment and a matched-compute gain.

## Card 2: Cross-fitted latent residual repair

**Mechanism family:** nonlinear representation fusion.

Hold an anchor and an independently trained weak model fixed. Generate their
causal hidden representations on groups excluded from both models' fitting,
then train a small correction head to predict the anchor residual from the
anchor state, companion representation and current legal features. The final
output is the anchor plus a learned correction, clipped for scoring. This
changes the prediction function through a fitted nonlinear correction; it is
not a scalar or convex blend.

The NIPS paper relates member disagreement to ensemble error and discusses
cross-validation as a way to estimate member errors. The transfer hypothesis is
that a weak model's representation carries group-safe information about the
anchor residual. Measure that conditionally rather than assuming low correlation
is useful. Compare the companion-feature corrector with an anchor-only
corrector, shuffled-companion control and unchanged anchor. Falsify if the
companion representation adds no held-out residual information or the complete
endpoint fails the controls. Keep base-model fit groups, correction-fit groups
and evaluation groups separate; exclude evaluation groups from all fitting and
calibration. Cost: cross-fitted representations from two models, one small
corrector fit and two model evaluations per callback. Preserve the feature
contract, correction head and ancestry manifest. Keep below current recovery;
consider a representation-specific child under residual-correction only after
out-of-group information is demonstrated.

## Card 3: Causal memory-reset event expert

**Mechanism family:** state management; unconventional interaction.

Train an independent event expert on the current row and bounded causal
summaries. Its output estimates whether retaining the anchor's recurrent history
is harmful after a transition. On a deterministic, hysteresis-limited trigger,
reset or attenuate the anchor's internal state. The expert changes the anchor's
memory path; it does not gate or blend the output.

The NIPS paper reports that high ambiguity can flag regions with larger average
error, while low ambiguity does not certify safety. That motivates looking for
localized failure regions, but does not itself justify state resets. Require
group-safe evidence that reset helps. On held-out groups, compare no reset,
learned reset, matched-rate sham reset, and an offline oracle-reset diagnostic.
Positive evidence means official pooled WP improves with benefits concentrated
around detected transitions. Falsify if the learned trigger does not predict
reset benefit or loses against the sham/unchanged anchor. The causal trigger
uses current features and causal rolling summaries only, never future rows or
labels. Generate reset/retain traces on groups excluded from event-expert
fitting; train and evaluate the trigger on further disjoint groups. Cost: causal
replay, one compact event-expert fit and small inference overhead. Preserve
reset semantics, deterministic replay traces and hysteresis settings. Keep
below recovery; open a new interaction branch only if an oracle reset diagnostic
shows repeatable held-out group benefit and an expert predicts it.

## Card 4: Degree-bounded interaction sidecar

**Mechanism family:** explicit representation with controlled capacity.

Train an independent low-degree polynomial sidecar on allowed current-row
features or causal summaries. Feed its basis activations or compact
representation into a retrained recurrent student. Do not average the sidecar's
prediction with the anchor; make it part of a learned temporal predictor.

The selected library page asks readers to reason about polynomial classifier
capacity through a Dudley representation and contrasts bounded-degree classes
with the unrestricted polynomial class. The transferable idea is to bound the
interaction basis explicitly; the source does not establish task relevance.
First test whether cross-fitted sidecar activations predict anchor residuals on
held-out groups. Then compare the student with the sidecar against a matched
parameter-budget input-only student and fixed random-basis control. Falsify if
the sidecar adds no out-of-group residual information or fails to improve the
complete official endpoint. Select the fixed interaction list and degree on
training groups; keep evaluation groups out of sidecar, student and calibration
fits. Cost: one compact sidecar fit and one student fit with two inference
components. Preserve basis definition, degree, feature list and ancestry. Keep
below own-baseline/raw-teacher recovery. Open a branch only after out-of-group
residual signal and matched-student gains.

## Bandit placement

All four proposals remain below the active own-teacher recovery arm. No
numerical priors or public-score expectations are assigned. Card 1 concerns
training allocation, Card 2 nonlinear residual representation, Card 3 recurrent
state intervention, and Card 4 bounded explicit interactions. The latter three
should first have low-cost legality/alignment checks; do not spend on model fits
until candidate ancestry, group-level prediction generation, and a suitable
evaluation split are established. No experiments or files were created by the
ideation subagent; this root-authored note preserves its returned proposals.
