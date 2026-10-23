# Independent specification review: per-field age readout screen

**Review type:** source-only pre-implementation critique  
**Status:** conditional on a frozen recipe and root admission; no fits or scores reviewed.  
**Reviewed inputs:** root's task brief, the canonical age feature source, the prior online-panel cache/grader contract, and the historical per-field-age proposal. I found no frozen new age-screen recipe in the repository or task disk when this review began, so findings below bind to the design in the assignment and must be reconciled against the final recipe before admission.

## Decision

The six-fit design can answer a narrow question: whether these fixed, zero-output residual readouts improve the frozen parent endpoint when 112 appended channels encode either group-level or per-field recency. It cannot establish that per-field age contains new causal information, estimate seed variance, or establish a native submission. The panel is a conditional, exposed-fold screen.

The evaluator surface is well chosen if it scores complete candidate predictions against the exact fold-matched parent and reports paired component uncertainty within each fold. F−A and F−G are the feature-representation contrasts; A/G/F−parent measure the complete residual endpoints. Keep those claims separate.

## Required freezes before fit admission

1. **Cell identity and source lineage.** Freeze exactly six new readout fits: A/G/F × (F0, seed1001; F1, seed1002), one parent per fold, and one arm-shared fit-membership digest per fold. Bind the identical complete-sequence 10% component cache sidecars and hashes from the old online panel. Also bind the exact feature transform, trainer/readout source map, parent prediction cache, split manifest, checkpoint, optimizer, and target/row-weight policy. Do not infer identity from a filename or mutable `recipe.json`.

2. **A/G/F control semantics.** A must append exactly112 zero columns; G and F must append the canonical 112-wide group_age and field_age streams. All three must have the same 323-input layout, network topology, parent, initialization, optimizer, schedule, batch, epoch and checkpoint. Keep the frozen parent normalization exactly on the original 211 features; use the root-proposed fixed [0,1] age encoding without fitting age statistics, consistently for G/F. A receives112 literal zeros. Initialize the shared 323→64 hidden layers identically across arms and zero-initialize only the two-output residual projection; no parent input projection is expanded. Record an exact parent-output parity check on a feature-only fixture. The historical B300 field-age source defines group clocks by seven fixed raw ranges and field clocks by exact per-column changes, both `log1p(age)`, capped at10,000 and reset at row zero. Root's current proposal instead scales the fixed log-age encoding by `log1p(10_000)` into [0,1], with no learned age statistics. Freeze one exact implementation and pin it by hash; do not call the normalized proposal the existing `research_features` implementation unless its code actually matches.

3. **Objective and data membership.** Reuse one frozen canonical row-weight array per fit cache, not recomputed per arm. Bind WMSE denominator/reduction order, any target clipping, the batch-size-8192 treatment of sequence tails, update count, 3-epoch fixed CP2 selection, AdamW settings, gradient clipping, and strict-FP32 determinism. Require every arm in a fold to consume exactly the same complete fit sequences, row order, masks, labels and row weights. The readout contributes no GRU or teacher fit.

4. **Outer-fold parent ancestry.** Require each parent checkpoint/provenance chain to exclude that fold's evaluation groups and each parent prediction array to cover only the exact ordered evaluation groups (F0:468; F1:457). The old cache may also contain selected fitting groups with targets; those are training inputs only and must never be scored. Record that parent predictions for fit groups are **in-sample** because the parent saw those groups during fitting. That creates train/evaluation feature-distribution mismatch for residual learning. It does not invalidate a clean outer evaluation if the parent excluded the evaluation fold, but it weakens transfer claims and is not cross-fitted parent evidence.

5. **Score and uncertainty.** Use unchanged `global_wp_clip2_v1`, with its fixed score mask, ±2 prediction clipping and label-magnitude weights. For each fold report parent/A/G/F endpoints; F−A, F−G, A−parent, G−parent and F−parent. Report overall WP and its paired dependency-component bootstrap interval (2,000 draws, fixed seed); both target correlations and the four fixed sequence-position bins are conditional point deltas. Keep every interval fold-local: seed and fold are aliased (and may also be device-aliased), so no pooled interval estimates seed or device effects. Do not make a fresh-holdout or public-score claim; both folds are exposed and no multiplicity adjustment is specified.

6. **Coverage before labels.** For every one of the six candidate arrays and two parent arrays, first validate ordered row-group references against the split, exact complete group count, shape `[n_groups, 20000, 2]`, little-endian FP32 C order, exact NPY payload length, matching sidecar/recipe/source/checkpoint/provenance/parent-cache hashes, and all-finite prediction values. Finish all eight artifact checks before opening the official target cache. Then verify target-cache identity against the split and align targets, masks and component IDs by ordered group IDs. Fail closed on any missing, extra, duplicate or reordered row group. The previous grader's NPY header parser, streaming SHA, official score wrapper and Welford component bootstrap provide reusable primitives.

7. **Device and release limits.** The latest root plan runs all six fits locally on one selected device/runtime after its real-shape profile, then sends only the grade bundle to Kaggle CPU. Keep each fold's A/G/F triplet source/runtime matched; if all six share the same local device, device is not confounded with fold. Record the actual runtime/device after profile. A CPU-only Kaggle grade does not change candidate predictions and must be described as an independent scoring environment, not a training replication. The historical per-field-age proposal says the native C++ path does not yet implement the new feature modes. A positive head grade therefore cannot promote a submission: later release work needs a native causal transform, sequence reset/chunk parity, deterministic callback parity, package-size and runtime checks.

## Findings by severity

- **Admission blocker — missing frozen grade recipe/admission receipt.** The task brief supplies enough to critique the design, but not the final artifact hashes, row orders, output paths, split pins, exact fit-cache sidecars or grade authorization. The adapter must reject absent/unadmitted recipe and receipt before touching predictions or labels.
- **Resolved by root clarification, pending recipe/source pins — age scaling and common initialization.** The current proposal uses a fixed [0,1] log-age encoding, no fitted age statistics, the original 211 normalization unchanged, A=112 zeros, identical shared-hidden initialization across the three heads, and a zero-output 0.1 residual projection. The parent prediction bypasses the new readout; no parent input projection is extended. This addresses the main normalization confound if all arm source hashes and initial-function parity match. The existing `research_features` implementation emits unnormalized `log1p(age)`, so do not cite it as identical until the new pinned code proves equivalence.
- **High — in-sample parent-fit prediction mismatch.** Preserve this as a model-procedure limitation in the report; do not imply cross-fitted residual learning.
- **High — exact cache/weight identity.** The existing online screen's cache contract includes sequence order, row weights and source-qualified membership. Reuse only complete caches with sidecars and a root local full-payload verification receipt that prove the same 10% component selection for all three arms. Kaggle need not receive the 32 GB sequence payload if the independent grader binds the local digest receipt to its recipe and sidecars; label this as an attested local cache check, not an independent Kaggle rehash. Candidate and parent prediction payloads must still be independently hashed before labels.
- **Release blocker, not a fit blocker — no native field-age implementation.** Do not call an improved residual endpoint a valid submission until the callback implements the same causal transform/reset semantics.

## Interpretation of the contrasts

- **F−A:** per-field age representation versus a same-width zero-channel residual-readout control. This combines feature values and resulting optimizer geometry; it is not an information-theoretic comparison.
- **F−G:** fine per-field clocks versus seven group clocks repeated across their raw-field ranges. Both are 323-wide, but G has repeated coordinates and lower representation rank. A gain is evidence for this fixed representation/optimization contrast, not for 105 additional independent signals.
- **A/G/F−parent:** complete endpoint utility of each residual readout, including the shared 0.1 residual scale. A can improve the parent through generic nonlinear readout capacity even though its appended channels are zero. Do not equate F−A with F−parent.

No prediction arrays, official labels, target masks, or score outputs were opened for this review.

## Correction after the implementation source review

The earlier draft of item 2 incorrectly said to append zero input-projection
weights. That is not part of this recipe: the frozen parent endpoint is reused
unchanged, each new head has a 323→64 hidden layer initialized identically
across A/G/F, and only its two-output residual projection is zero-initialized.
The recipe was frozen later than this review's initial pass; the grade-admission
receipt remains a separate gate for the future independent score.

## Post-freeze implementation check

I inspected the actor's source packet at
`/home/vlad/development/wunder/research_disk/b300-fast-20261023/b300_scout/source`
against the frozen recipe. The feature assembler preserves the original 211
parent-normalized inputs and adds the fixed log-age basis; the three arms share
initial hidden weights and differ only in appended age values. The readout
starts at the exact cached parent output, uses cached canonical row weights,
and evaluates from raw features plus cached parent predictions. Fit and
evaluation row order, the incomplete final 8,192-row batch, fixed epoch-2
selection after three epochs, and the target-free evaluation column slice are
implemented explicitly. I found no scientific implementation mismatch in
those inspected paths. This was a source review; no real cache, candidate
prediction, target, or grade values were read.

The profile CLI path still checks recipe, mode, and device but does not require
the same `status=admitted` gate as `fit-fold`. Because that profile fits three
heads on a small real fit subset, root should issue a specific profile receipt
or strengthen the CLI guard before launching it. The fit path additionally
needs the final source-hash map and six-cell shared-device admission already
required by the coordinator. The independent grade adapter and synthetic
prelabel checks are recorded in the sibling `../grader/README.md`; neither
produces a quality result without the root receipt.
