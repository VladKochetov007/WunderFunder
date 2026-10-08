# H233 readout grader source review

**Verdict:** source and fixture checks pass, with one explicit root-admission
check for the frozen parent sidecar before any score is run.

I reviewed grader SHA `c084c12ea7ad3c9af3725a532c1aad3fd76e50a015b02931e2216cea53f7e19e`
against preregistration SHA
`e59c74763f764b3b57893d8873ae64ea687d57c9c1579b5ad3450c8e0d0f1b5a`.
The preregistration fixes epoch 1, the exposed fold-0 evaluation set of 468
valid groups, all four readout modes, primary comparisons against the trained
hidden-only readout, secondary comparisons against the frozen C1 parent at
weights 0, 0.1, 0.25, 0.5, and 1, and a 2,000-replicate component bootstrap.

The grader binds root admission to the exact preregistration and grader hashes,
requires the complete artifact set, and verifies each admitted artifact hash.
It also binds the frozen recipe and split manifest, fixed epoch-1 checkpoint,
parent checkpoint and prediction, target cache, and all four export arrays.
The split/provenance helper compares exact source-qualified fit groups and
rejects both held-out source-qualified groups and connected-component overlap.
The export metadata must declare `split=valid`, epoch 1, the frozen checkpoint
and parent, all four prediction hashes, both mask hashes, and a passing parent
anchor-parity gate.

The candidate export is required to contain the full ordered valid-target row
list, with 1,873 unique row groups. The group mask must equal the frozen 468
evaluation groups; the selected scoring mask must equal the original scored
mask after warmup for those groups and be all false for the 1,405 valid
fit-development groups. The grader joins rows through `("valid", row_group)`
keys and uses the actual full-export row positions. The focused reorder fixture
checks nonmatching physical row orders, while duplicate identities and mask
drift are rejected.

Scoring delegates to the shared E540 `score_pair` helper and the frozen campaign
scorer. It blends raw predictions before the scorer applies official ±2
clipping, uses the official weighted pooled moments, and pairs component
bootstrap replicates against the corresponding zero-weight control. Every
predeclared mode and weight is reported; the output marks exposed discovery,
disables weight selection, and makes no public forecast. The direct-metric
fixture checks reordered rows and distinguishes raw blending from premature
member clipping.

The primary readout-vs-hidden-only comparison uses the same exported trunk
path. Parent comparisons are secondary and are limited by the exporter's
`max_absolute_error <= 1e-3` anchor-parity tolerance, not bitwise precision
identity. The preregistration and report both disclose that limit, so those
comparisons should not be described as precision-matched.

One admission boundary remains: the grader hashes the parent sidecar and
prediction, fixes the prediction SHA, and uses the sidecar's row groups, but it
does not itself require `parent_sidecar.split == "valid"` or require that the
parent sidecar list the complete 1,873-group valid export. Before authorizing
scoring, root admission should inspect the exact admitted sidecar and verify
its valid split, parent checkpoint/run linkage, and full row-group identity.
If root admission does not already enforce those semantics, add explicit
grader checks before scoring. This is not a prediction-data finding; no
candidate or parent prediction array was opened in this review.

Bounded validation: the two focused test files pass **9 tests** in 0.27 s;
Ruff passes for the grader, fixtures, and imported scoring helpers. Tests use
synthetic fixtures only. No candidate output, fit result, or score was read or
computed. The review supports source-level mask, membership, alignment, metric,
and preregistration fidelity only; root retains actual terminal artifact
admission and candidate grading.
