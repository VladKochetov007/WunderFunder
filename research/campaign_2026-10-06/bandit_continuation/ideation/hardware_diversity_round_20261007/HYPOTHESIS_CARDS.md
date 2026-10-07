# Hardware-shaped teacher and component hypotheses — 2026-10-07

This is an ideation slate, not an allocation or fit authorization. The first
four proposals in `NATIVE_IDEAS_PRE_GROK.md` were hashed before the independent
Grok 4.7 call. The final slate keeps two distinct native proposals and all six
Grok proposals. Native proposals about a semantic group encoder and causal
scattering remain preserved in the pre-Grok record; this round does not rank
them or claim they are inferior. IDs below are local to this ideation artifact.

The objective is useful, independently trained model/component experiments
per actual accelerator-hour and complementary teacher signals. Challenge2 rows
must be processed causally, state resets at each sequence, and outputs are
required from step 99 onward. The official `global_wp_clip2_v1` mask, two-target
pooled weighted Pearson calculation, ±2 metric clipping, and clipped-label
magnitude weights remain fixed. Training-target treatment is a separate recipe
choice and requires its own audit. Do not use teammate weights. Every existing
validation fold is exposed; a chain-excluded fold comparison is development
evidence and cannot forecast public performance.

Hardware evidence shapes proposals but is not a speed claim for any of them:
the E522 local profile measured 49.53 sequences/s at batch 16/zero loader
workers, 50.49 at batch 32/zero workers (+1.94% with much higher VRAM), and
51.72 at batch 16/one worker (+4.42%). A two-worker profile was killed at its
scoped RAM cap. Cloud v6 spent 795.47 s hashing train, 188.75 s hashing valid,
then repeated 748.68 s of train hashing before timing out pre-CUDA. A reviewed
same-worker verified-digest handoff is being integrated; any future arm must
preserve the source identity checks and receive separate controller admission.
Profile each proposed complete input/feature/update path on the actual target
hardware before estimating useful experiments/hour.

## HD-1 — Sparse echo-state reservoir with streaming readout

**Lens and sources.** Randomized dynamical systems. Jaeger, “The ‘Echo State’
Approach to Analysing and Training Recurrent Neural Networks,” GMD Report 148
(2001; corrected author-hosted erratum version 2010), Sections 2–3:
[author PDF](https://www.ai.rug.nl/minds/uploads/EchoStatesTechRep.pdf). The
source fact is that it studies recurrent states determined by input history and
readout learning by linear regression. As an unconventional implementation
transfer, the preregistered random user-library draw selected Vertox’s
“TWAP & VWAP — More important than you think,” PDF page 4, for its recursive
numerator/denominator bookkeeping pattern; the exact 21-word excerpt and draw
receipt are in [`RANDOM_LIBRARY_PASSAGE.md`](RANDOM_LIBRARY_PASSAGE.md). That
popular article is not evidence about this task or its labels.

**Intervention.** Draw a frozen sparse reservoir from one recorded seed (initial
candidate width 32 or 64; width is fixed before scoring), update its state
causally from a low-rank projection of the 112 inputs, and learn only a two-row
linear readout from scratch. Stream bounded weighted readout statistics in
chunks; never materialize the complete hidden-state matrix. The recurrence and
readout are our own initialized/fitted model, with no teacher or teammate
weights. The relation to the random passage is computational: keep the
numerator/denominator or Gram sufficient statistics incrementally rather than
recomputing/storing the past.

**Mechanism and observable.** Fixed nonlinear dynamics may map transients and
decays differently from a learned GRU, while eliminating BPTT and optimizer
state. Measure complete updates or predictions per second, peak memory, and
the model’s standalone score and residual relation to the frozen baseline. The
expected direction is more completed work per measured hour; no gain is
promised.

**Smallest discriminating test.** First test prefix invariance, chunk-state
parity, reset, seed replay, and streamed-readout parity against a direct tiny
synthetic calculation. If those pass, profile a single width on actual hardware
against the already-profiled compact GRU shape. Only after recipe review, fit a
prespecified seed on the complete-chain-excluded development split and grade
against the same fixed anchor. Complementary distillation is a later separately
allocated DAG, not inferred from a low standalone score.

**Falsifier and risks.** Stop if the full path is not faster or requires more
memory than the matched GRU, if state/prediction parity fails, or if the
preregistered endpoint shows neither standalone utility nor a distinct
teacher signal. Recurrent updates may still serialize; streaming matrix
statistics may dominate compute. Keep exact training clipping/weights and
normalization in an independently audited recipe; never substitute the
official metric calculation.

**Artifact.** Frozen reservoir seed/matrices, state-update definition, readout
objective, split/membership hashes, timing/resource receipt, checkpoint,
row-aligned prediction/provenance, and exact anchor comparison.

## HD-2 — Liquid microstate with input-dependent retention

**Lens and sources.** Continuous-time dynamical systems/control. Hasani et al.,
“Liquid Time-constant Networks,” AAAI 2021, model construction and
time-constant discussion: [official proceedings](https://ojs.aaai.org/index.php/AAAI/article/view/16936).
For the bounded computational form, Hasani et al., “Closed-form Continuous-time
Neural Models,” Nature Machine Intelligence 2022: [publisher page](https://doi.org/10.1038/s42256-022-00556-7)
and [author preprint](https://arxiv.org/abs/2106.13898). The papers motivate
continuous-time state dynamics and closed-form approximations; they do not
establish a Challenge2 benefit.

**Intervention.** Use a small stable hidden state with learned bounded decay
rates and an input-dependent update/retention term evaluated in closed form,
without an ODE solver. Initialize the activity input to recover a fixed row
clock. Freeze three matched variants: fixed decay, current-row innovation
activity, and stale activity. Derive activity from current and preceding
normalized features only; do not infer wall-clock time from sequence row count.

**Mechanism and observable.** The useful memory horizon may vary with how much
the observed representation changed. Test whether aligned activity beats both
fixed and stale activity at equal state width, with complete-update wall time
and VRAM as co-primary resource observations. This is distinct from another
event-stepping GRU and from a fixed diagonal recurrence, but its gating adds
compute.

**Smallest discriminating test.** Test bounded states, row-by-row versus
chunked state, causal prefix invariance, and reset. Then profile exact full
updates at one small width. If admitted, compare all three variants on the same
complete-chain-excluded development split under a frozen recipe and anchor.

**Falsifier and risks.** Falsify the activity mechanism if aligned activity
fails to beat stale activity, learned rates saturate to a constant, or the
added gate erases any resource advantage. Uniformly indexed rows may not encode
elapsed real time; no such interpretation is assumed. Require a full optimizer
profile before a fit.

**Artifact.** Cell equations/config, initialization/seed, three matched recipe
commands, prefix/reset/parity tests, actual-device profile, membership and
target hashes, checkpoint and fixed-anchor comparison.

## HD-3 — Sliding-DFT state on trade channels

**Lens and source.** Streaming signal processing. Jacobsen and Lyons, “The
Sliding DFT,” *IEEE Signal Processing Magazine* 20(2), 74–80 (2003),
especially the per-sample bin recursion and stabilization discussion:
[IEEE DOI](https://doi.org/10.1109/MSP.2003.1184347). The source discusses
successive-bin updates and cases where only a subset of DFT bins is needed.

**Intervention.** Maintain a small frozen set of sliding spectral bins for the
documented trade-price and trade-volume channels of both instruments. Reset
all state per sequence and emit features causally at every row. Fit a tiny
readout from scratch. Use one preregistered set of windows/bins and a matched
raw delay-window ridge on those same channels; do not tune bins against the
screen. Use a numerically checked implementation and measure drift through the
full 20,000-row sequence.

**Mechanism and observable.** A handful of multiply-add recurrences may expose
periodic or quasi-periodic structure with constant state, providing a cheap
teacher different from a learned hidden-state trajectory. Measure full-pass
throughput, feature drift, and whether the spectral readout beats the same
channel/delay control on the exposed development screen.

**Smallest discriminating test.** Synthetic sinusoids, impulses and resets
against a direct finite-window DFT; then a 20,000-row actual-device profile.
Only if numerical and resource checks pass, fit the fixed candidate and control
on the same chain-excluded groups.

**Falsifier and risks.** Falsify if the bin features drift beyond a frozen
tolerance, do not beat the delay control, or feature-update cost outweighs
saved training cost. Unit-circle recurrences can be numerically fragile; keep
double-precision reference tests and do not add damping after looking at a
score. The named channel groups are only input schemas, not proof of regular
market periodicity.

**Artifact.** Exact channel indices, bins/window, arithmetic precision,
recurrence, reset test, numerical drift report, readout objective, split hashes,
timed profile, checkpoint and predictions.

## HD-4 — Input-only extended dynamic mode decomposition coordinates

**Lens and source.** Nonlinear control/system identification. Williams,
Kevrekidis, and Rowley, “A Data-Driven Approximation of the Koopman Operator:
Extending Dynamic Mode Decomposition,” *Journal of Nonlinear Science* 25,
1307–1346 (2015), Sections 2.2–3: [author preprint](https://arxiv.org/abs/1408.4408).
The source describes fitting a finite Koopman approximation from snapshot pairs
and a dictionary; its Markov-process interpretation requires assumptions that
are not established for this dataset.

**Intervention.** Choose one fixed 64-wide random Fourier dictionary of current
input rows. On training groups only, fit the finite input-evolution operator
from successive snapshot pairs. At prediction time, transform the current
dictionary vector with that frozen operator; fit a separate two-target readout
on training labels only. The next row is used only to fit the operator from
training-only input pairs. Compare with the identical dictionary and readout
without the fitted evolution transform.

**Mechanism and observable.** If dictionary observables make the input dynamics
approximately linear, their eigencoordinates may provide a compact basis that
differs from label-trained recurrent states. Record input one-step residuals
and complete-pass cost, then compare the readout to its dictionary-only
control. No Markov property or useful forecast is assumed.

**Smallest discriminating test.** Accumulate operator matrices over training
rows only, solve one 64-dimensional regularized least-squares problem, report
condition numbers and input residuals, and compare both readouts on the same
chain-excluded screen. Use no validation input pairs to fit the operator.

**Falsifier and risks.** Falsify if the operator is ill-conditioned, the
residual is not lower than its frozen identity control, or transformed
coordinates fail to improve the matched screen. Snapshots may not be Markov;
eigenvectors can be unstable under small data changes. Keep the random map
fixed and provenance-bound. A 64-wide operator is compact, but full input IO
still needs actual timing.

**Artifact.** Dictionary seed and scale, pair membership, regularized operator,
eigensystem, training-label readout, condition/residual report, split hashes,
profile, predictions.

## HD-5 — Random Fourier features on a short causal window

**Lens and source.** Randomized algorithms/kernel approximation. Rahimi and
Recht, “Random Features for Large-Scale Kernel Machines,” NeurIPS 2007,
Section 2: [official proceedings](https://papers.nips.cc/paper/3182-random-features-for-large-scale-kernel-machines).
The paper motivates randomized low-dimensional maps whose inner products
approximate a chosen shift-invariant kernel, followed by linear learning.

**Intervention.** Concatenate the last four rows of inputs (with only causal
history; initial padding is fixed) and map that window to 128 frozen Fourier
features from one recorded draw and kernel scale. Fit a two-target weighted
ridge readout from scratch. Primary dimension and scale must be registered
before scoring; a linear ridge on the identical window is the matched control.
Any separate dimensionality/resource ladder is descriptive and cannot be
selected on the exposed screen.

**Mechanism and observable.** A low-dimensional random kernel map may capture
short nonlinear interactions while training only a compact head, making a
useful fast teacher. Measure whole-pass update/readout speed and memory, then
whether it beats the raw-window linear control on the same exposed screen.

**Smallest discriminating test.** Causal-window and seed-replay synthetic
checks; profile memory-bounded chunks; then one fixed candidate/control pair on
the same chain-excluded development partition. Maintain row order and exact
target mask at grading.

**Falsifier and risks.** Falsify if the random map does not beat the linear
window control or if random-draw variation dwarfs any preregistered
architecture contrast. Materializing all row features would be excessive;
stream the readout statistics and predictions. A four-row model intentionally
cannot represent longer temporal state, so poor long-memory behavior is a
known boundary rather than a reason to silently increase the window.

**Artifact.** Window/padding, random seed and frequency/bias arrays, scale,
readout regularization/objective, chunk semantics, split/row hashes, resource
profile, checkpoint and predictions.

## HD-6 — Instrument-wise invariant pooling of numbered channels

**Lens and source.** Set functions / inductive bias. Zaheer et al., “Deep Sets,”
NeurIPS 2017, Theorem 2 and Sections 2–3:
[official paper PDF](https://papers.nips.cc/paper/2017/file/f22e4747da1aa27e363d86d40ff442fe-Paper.pdf).
The theorem characterizes permutation-invariant set functions through shared
per-element transformations and pooling. It does not say the numbered
Challenge2 channels are exchangeable.

**Intervention.** Per instrument, pool documented price-like channels and
volume-like channels separately through a shared tiny scalar map. Keep
trade-price, trade-volume and the eight `a*` channels on indexed paths; append
explicit instrument/type identifiers. Compare with a same-width index-aware
map over the exact same fields. Do not permute channels during fitting or
assume indices encode sorted price depth.

**Mechanism and observable.** If numbered channels within the selected group
are arbitrary labels, pooling may discard nuisance index dependence and
compress row width before a small causal state/readout. Test pooling versus
index-aware control, input-width cost, and the invariant arm's predictions
under a post-fit within-group permutation.

**Smallest discriminating test.** Verify data-column assignment. On synthetic
inputs, confirm that the pooled model is invariant only within its declared
groups, while instrument IDs and `a*` values remain distinguishable. Then
profile both maps and compare them under one chain-excluded screen.

**Falsifier and risks.** Falsify if pooling loses to the matched indexed model
or within-group permutation changes the pooled outputs. The source theorem
applies only if the permutation-invariance assumption is valid; the dataset
documentation explicitly does not guarantee sorted depth, but does not prove
exchangeability either. Keep this arm assumption-led and do not generalize it
to all features.

**Artifact.** Channel grouping table, map/embedding sizes, training seed and
weights, permutation test, membership hashes, full-path profile, checkpoint,
predictions.

## HD-7 — Gaussian past–future information bottleneck

**Lens and sources.** Information theory / predictive compression. Creutzig,
Globerson, and Tishby, “Past-future information bottleneck in dynamical
systems,” *Physical Review E* 79, 041925 (2009):
[APS primary source](https://doi.org/10.1103/PhysRevE.79.041925). For the
Gaussian projection solution, Chechik, Globerson, Tishby, and Weiss,
“Information Bottleneck for Gaussian Variables,” *JMLR* 6, 165–188 (2005),
Theorem 3.1 and Section 7.1: [JMLR paper](https://jmlr.org/papers/v6/chechik05a.html).
The source facts concern predictive past/future compression and Gaussian
variables; the Challenge2 distribution is not established as Gaussian.

**Intervention.** Form an eight-row causal input window; project it through one
fixed low-dimensional random map; accumulate train-only covariances for this
past representation and the two targets using a declared clipped-label and
weight formula. Solve a small Gaussian-IB generalized eigenproblem at a fixed
registered compression parameter. A same-rank training-input PCA/readout is
the control. Any use of official label weights in covariance estimation is a
new training-objective choice requiring preregistration; the official scorer
remains untouched.

**Mechanism and observable.** The projection chooses a compact representation
for label relevance instead of input variance alone. Measure retained rank,
conditioning, complete-pass cost, and whether it beats the equal-rank PCA
control on the same exposed development fold.

**Smallest discriminating test.** Train-only streamed covariances, one
preselected compression parameter, stable eigenvalue/conditioning checks, and
the equal-rank PCA control. If nonlinear/non-Gaussian mismatch is too large,
do not escalate to model selection on the exposed fold.

**Falsifier and risks.** Falsify if the IB projection does not beat PCA at
equal rank, is numerically unstable, or requires a larger materialized window.
The Gaussian model and target-weighted covariance extension are assumptions,
not source results; keep them explicit. Fit on allowed training groups only,
and retain exact metric masking/clipping/weighting only in the grader.

**Artifact.** Window/random-map provenance, exact covariance and clipping
formula, compression parameter, eigenvalues/projection, PCA control, split and
target hashes, conditioning/profile receipt, checkpoint and predictions.

## HD-8 — Time-augmented depth-2 path signature

**Lens and source.** Rough paths / ordered interaction geometry. Chevyrev and
Kormilitzin, “A Primer on the Signature Method in Machine Learning,” 2016,
Sections 1.2–1.3 and 2.3: [arXiv primary preprint](https://arxiv.org/abs/1603.03788).
The source defines iterated-integral features and Chen's concatenation identity;
it also discusses applications to financial streams. Reparameterization
invariance is a property of signatures, not evidence that row-time can be
discarded here.

**Intervention.** Project each 112-channel row to four frozen seeded
Rademacher coordinates, append known within-sequence normalized row position,
and maintain the depth-2 signature of this five-coordinate causal path using
Chen's identity. Reset at sequence start, persist state across chunks, and fit
a small two-target ridge head. Compare with the same projected path summarized
by an eight-row delay map, including the identical row-position coordinate.
The time augmentation deliberately preserves rate information.

**Mechanism and observable.** The second-level terms encode signed temporal
interactions/areas that a matched delay model's linear readout may not express
as efficiently. Five dimensions through level two give 30 nonconstant
signature coordinates (31 if a constant coordinate is appended), a fixed-size
state. Test whether they improve the matched screen and whether the second-level
block carries fitted weight.

**Smallest discriminating test.** Synthetic two-channel paths with known
order/area, direct versus Chen updates, row-by-row versus chunk equivalence,
causal prefix and reset checks. Then compare one frozen candidate against its
delay control and profile a complete pass before fitting any broader depth.

**Falsifier and risks.** Falsify if the signature does not beat the matched
delay control, if the level-two block contributes no stable signal, or if
numerical/update cost erases the compact-state advantage. Do not increase
projection width or signature depth after looking at scores; dimension grows
rapidly. The time coordinate is allowed only as known sequence position and
must never carry sequence identity or future data.

**Artifact.** Projection seed, path embedding, signature levels/update rule,
known-time definition, ridge objective, split/target hashes, parity and reset
tests, measured profile, checkpoint and row-aligned predictions.

## Shared admission boundary

These are eight non-ranked hypotheses with null public-gain priors. The
cheapest gate is implementation/parity and measured complete-update
throughput/memory on the actual device; no speed gain is inferred from paper
claims or small kernels. A fit would require a new fixed recipe, independent
review, current resource/quota admission, complete-chain exclusion, exact
provenance, and an unchanged independent official grader. Any screen on an
already exposed fold is development evidence only. No combination of these
cards is assumed additive; a teacher/student interaction must be registered
and evaluated as its own complete endpoint.
