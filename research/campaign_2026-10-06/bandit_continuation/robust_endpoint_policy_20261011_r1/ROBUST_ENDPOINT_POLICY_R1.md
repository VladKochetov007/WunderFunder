# Robust complete-endpoint policy — R1

Independent analysis; controller owns integration/admission. Basis: MAIN
`d2c82811c53f047585678dad52624136f117b5e9`, plus the hash-pinned live sources in
[SOURCES_R1.json](SOURCES_R1.json). No fits, provider calls, spending, queue edits or MAIN edits.
Latest delegated scope permits Sol analysis only; implementation preferences and
resource authority are unchanged. Support commands used the supplied one-core,
700 MB, 120-second wrapper. Verification is exposed-development discovery and
serving correctness; public transfer remains uncertain. ADR0032 permits packaging
now, including explicitly provisional competition choices.

**Current evidence preference: the H388 teacher4/5 Q8 pair, conditional on serving
margin; keep the H390 student3/teacher5 Q8 pair as its main challenger.** This is
an evidence-fidelity ordering, not a demonstrated final-score ordering. Neither
exact all-data endpoint has a held-out grade. The latter's apparent F1 advantage
uses a materially different student and cannot settle their rank. Recommend one
winning endpoint after the bounded comparisons; preserve other packages as
artifacts. At a genuine unresolved tie, choose the endpoint with cleaner recipe
correspondence and adequate measured runtime margin, or explicitly release one
provisional choice under ADR0032. Do not imply that packaging must await public
confirmation.

## What the evidence supports

- F1 has 457 evaluated groups / 149 components and extensive selection exposure.
  The 52-component complement is a blend diagnostic from that same exposed fold.
  It is not new CV. Its bootstrap conditions on the fitted predictions.
- The three H384 teacher seeds have mean single score .6706805; equal trio
  .6725416, versus best single .6714970. Pair average .6720749 and trio-minus-pair
  average .0004667 describe this one seed set. The three overlapping pairs are
  not three independent training replications.
- Teacher4/5 scores .6723361, prototype student3/teacher5 .6725743, and prototype
  student-half/teacher-trio-half .6726401. The quartet adds only .0000985 over the
  trio. Those selected small differences do not establish robust superiority.
  The teacher-trio complement interval versus teacher5 is [-.004893,.007197].
- Source quality has measured value: T3 .6725416 exceeds weaker-source T6
  .6718337 and matched-count diverse D3 .6718487. Those exact additions did not improve the frozen endpoint,
  not heterogeneous teaching generally. Raw cross-family residual correlations
  partly reflect scale: after calibration-scale alignment, .993–.995 overlaps
  same-family values. Shrunk covariance weights and scale fitting supplied no
  worthwhile complement advantage. Keep fixed equal coefficients.
- H388 matches F1 teacher recipe, seed and CP2. H390 changes the student's parent
  and `since_weight` 0→32; its targets come from own cleanfold F1 strong teachers,
  not the H388 all-data teachers. Both final families fitted train+valid. Their
  valid predictions cannot validate final generalization. H382's nearby folds
  test other propensity/epoch contrasts, not these exact complete endpoints.
- Exact delivered teacher4/5 and student3/teacher5 Q8 ZIPs passed Python3.11
  reset/replay/finiteness checks. Sizes are 17,272,496 and 12,914,019 bytes. This
  establishes serving behavior, not held-out score or guaranteed portal duration.

## Freeze four complete policies for useful comparisons

Here `Tj` means H388 Muon-all H384 e3 seed j at CP2; `Sj` means H390 homogeneous
H256 student seed j, R3, at CP5. Exact available sources are bound by canonical
build receipts in the source manifest. Evaluate cleanfold recipe counterparts;
do not score final all-data outputs as if they were held out.

| Policy | Fixed complete endpoint | Evidence weight and rank-changing observation |
|---|---|---|
| A, primary reference | `.5 T4 + .5 T5`, Q8 | Strongest exact recipe/seed correspondence; exposed F1 mixture evidence. Demote if matched served endpoint loses to B/C across the declared panel or runtime margin fails. Canonical ZIP SHA `837b0fb3c71b22062747d7e1e3bfd2241cb6c5cd41edac908745dd36d5469410`. |
| B, family challenger | `.5 S3 + .5 T5`, Q8 | Prototype F1 .6725743 is analogy only. Promote if the actual changed-parent/since32 student improves this complete served mixture versus A and versus its student control. Canonical ZIP SHA `d83723831277cf7327ab84ba781a2f2ebe883aca40cd1f7bc6a222e30be28284`. |
| C, compact student pair | `.5 S3 + .5 S4`, Q8 | Same-family final pair lacks an exact cleanfold sibling-pair grade; the public student improvement is contextual evidence. Promote if cleanfold complete-pair mean beats A/B, or tied quality has materially safer serving. Current corrected ZIP SHA `3420ea5e6318e15b6340582125959ed7489983b17338f399c748965cb42c4a4a`. |
| D, transport/ensemble interaction | `(T3 + T4 + T5)/3`; FP16 transport for dense GRU/`head.0.weight`, cast to F32 by the existing loader, Q8 runtime | Prefer this fixed transport test before Q4. A bounded research endpoint, not a ready submission. It preserves the Q8 inference path while admitting more strong teachers if actual bytes fit. Test exact ZIP size/runtime and official complete-blend score against A; reject this exact policy if enlarged ensemble benefit does not compensate transport/quantization loss. |

The H390-pair ten-group precision panel found Q8−FP32 −.0000709 and Q4-grid−Q8
−.0018013. That is evidence against replacing that same pair with Q4. It neither
measures teacher-trio compression loss nor rejects an enlarged compressed blend.
The relevant comparison is `score(D served) − score(A served)`, including all
effects. Do not add separately measured member gains/losses or assume smaller
storage makes this int8 backend faster. The prototype family quartet's .0000985
increment does not make a fourth member compulsory or justify coefficient search.
It remains eligible for one fixed conditional contribution test if transport makes
it admissible. Ordinary three-teacher F32 storage with Q8 runtime exceeds 20 MB
at current payload sizes. The controller's latest message reports measured
wide-pair FP16-transport ZIP reduction 17.272→9.591 MB and a proposed
three-teacher-plus-student raw payload around 18.43 MB. These new measurements
were provided by the controller, not independently reproduced in this analysis;
their exact receipts and delivered-ZIP quality remain required. D therefore uses
Q8 runtime/FP16 transport first. Q4 remains a bounded alternative only when its
complete endpoint earns its place after all losses and actual CPU cost.

## Cheapest discriminating test and selection rule

First grade fixed F1 recipe counterparts of A and D through their delivered
callbacks, official mask/clip/weights, on all eligible complete sequences. Existing
teacher checkpoints suffice; no fitting is needed. Freeze hashes before scoring.
This quickly falsifies D if transport removes its conditional value or it fails
size/runtime. Keep FP32-versus-served results separately from recipe gains.

For final recipe ranking, use cleanfold A/B/C counterparts on F1 and the prepared
F0 pipeline, then a second source-qualified fold when feasible. B/C need the
actual H390 parent, since32, checkpoint and target-production recipe reconstructed
within each fold. All teacher/parent/propensity ancestry must exclude the evaluated
groups; F1 teachers cannot be reused for an F0 held-out claim. Freeze seed IDs
and coefficients rather than pick each fold's best seed. Evaluate every endpoint
on identical membership. Additional disjoint seed sets test seed sensitivity;
repeated machines or overlapping pairs do not add independent n.

Rank by a prespecified mean of matched fold-level endpoint contrasts, averaging
distinct seed sets within fold. Report all fold/seed deltas, target scores and
whole-component intervals; report fitting variation and transfer uncertainty
separately. Preserve the complete attempted set. Use exposure and instability
to describe confidence and allocate bounded follow-up, without a universal
zero-crossing rejection or arbitrary F1 shrinkage factor. A/B's present ordering
is falsified by a clean, exact-recipe served comparison favoring B; a source-
qualified C win similarly changes the choice. Missing evidence stays missing.

The public-gap review's section11 withdrew its numeric public-offset forecast,
Gaussian winner's-curse correction, .5 shrinkage rule and .003 kill threshold.
Its observed transfer disagreement motivates these tests, not a universal ban on
scratch members. Human uploads remain the gate; no numeric public expectation
is assigned and no local delta is added to a public score.

## Confirmed provenance-index defect

`SOURCE_ENDPOINT_MAP_R1.json` records H388 seed4 CP SHA ending `f9f0fcc`.
Bounded hashing of the actual `model_ep2.pt` gives
`592c51fb9f8dd306c469f1b09cd20d57d06c54ce2a72d1e04ab00ac50f9a0fcc`, matching
canonical `BUILD_RECEIPT_R3.json`. Append a correction/source-binding receipt;
preserve the historical map. This is a selector-index typo, not evidence that
the canonical delivered teacher pair contains the wrong checkpoint.
