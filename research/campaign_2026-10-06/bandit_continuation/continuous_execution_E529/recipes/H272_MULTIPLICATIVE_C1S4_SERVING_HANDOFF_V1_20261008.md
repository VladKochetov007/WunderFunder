# H272 multiplicative head on C1 seed-4: source and serving handoff

This is a source-only proposal. It does not authorize profiling, fitting,
exporting predictions, grading, provider work, blend selection, or packaging.
Root retains source review and fresh-compute admission. The machine-readable
contract is
[`H272_MULTIPLICATIVE_C1S4_FOLLOWUP_V1_20261008.json`](H272_MULTIPLICATIVE_C1S4_FOLLOWUP_V1_20261008.json).

H272's root-grade result gives a bounded reason to continue the multiplicative
direct-label branch on another own parent: standalone WP was `0.6817322281`
against the frozen parent at `0.6816181422`; the fixed raw blends at readout
weights `0.1`, `0.25`, `0.5`, and `1.0` were all positive on the exposed fold.
Their paired 141-component bootstrap intervals include zero. Root's direct
`gmetric.reference_global_wp` check matched within `1e-12`. This is previously
exposed fold-0 discovery evidence, not a public-score forecast or public-best
claim. The exact result and scorer hashes are pinned in the JSON.

The best available cleanfold seed-4 parent is
`c20261003_combo_s4_f0@ep3`, checkpoint
`0203b109a1f25f3c82794ad3684b83fe4a4bc815f4b5c1bc4bc1ea450dba0a8d`.
It is a plain GRU with 256 hidden units and two layers, 211 input features, a
512-wide `[last recurrent state, projected input]` representation, and 23
outputs. Its source-qualified fit set is 10,535 train groups plus 1,405
fit-development valid groups; the fixed evaluation set contains 468 valid
groups. Use `(source, row_group)` for identity. The train and valid namespaces
reuse integer values, so integer-only overlap is not a valid leakage test.

The current public-best package references
`c20261004_combo_s4_final@ep3|c20261004_combo_head_s4_final@ep0`. The base
checkpoint `c20261004_combo_s4_final@ep3` has the same seed, model tensor
shapes, and source-code identity as the cleanfold seed-4 parent, but it was
trained on a different full-data teacher/propensity lineage. Its 12,111 groups
include 374 of the 468 exposed fold-0 evaluation groups, as does the paired
head according to the corrected membership audit. Therefore the cleanfold
parent is **family/seed matched, not exact recipe-ancestry matched**. Keep the
current package intact as the fallback. The full-data head would be a separate
deployment-only artifact: train it on all 12,111 groups, score nothing, and
resolve the missing `prop_v1` membership provenance before any admission.

The proposed screen changes only the frozen parent/readout interaction. Reuse
the H272 direct-label objective: primary pair target `clip(y, -2, 2)`, original
clipped-label magnitude times the same split's propensity-derived main row
weight, `prop_power=2`, `prop_eps=0.03`, and `valid_true_mask=0.03`. The whole
parent stays in eval/frozen mode; no readout teacher columns are loaded. Train
one multiplicative rank-32 head, initialized from the same seed-3 H272
`ConditionalReadoutHeads` construction order, with only that branch trainable.
Use AdamW `lr=0.001`, `wd=0.01`, constant schedule, batch 8, 2,000-row chunks,
zero data-loader workers, two epochs, max-norm clip 1.0, and fixed epoch 1
selection. The cleanfold screen has 29,840 updates; the full-data deployment
head has 30,260. H272 measured 1,941 seconds for 29,840 updates using its
four-head run; linear scaling to 30,260 is only a rough setup-excluding estimate,
not admission evidence. Do not reduce the update budget to fit a timeout.

Before root review, the new runtime must add parent-specific hash and
provenance guards rather than reuse H272's seed-3 parent constants. Validate
the actual `model_ep3.pt` producer identity separately from the `model.pt`
alias. Keep H272's fixed blend set: parent at alpha 0, candidate at alpha 1,
and raw preclip mixes at alpha `0.1`, `0.25`, and `0.5`; report all of them.
The fold-0 grade must use only the frozen 468-group mask and official
`global_wp_clip2_v1`, with no score-based epoch or weight selection. The
all-data head receives no local grade.

The native implementation looks feasible but is not yet validated. E416's
`StreamFeatures` already generates the causal 99-column context, and
`FastRnn` internally forms the 512-wide state/projection concatenation before
its parent output head. A 22,722-parameter FP32 contextual head adds 90,888
weight bytes and roughly 22,720 multiply-accumulates per row, about 2.3% of a
plain 256x2 GRU's estimated matrix work. This is an architecture estimate;
there is no Rust parity, ZIP-size, RSS, or end-to-end runtime measurement yet.

Before any ZIP, implement the readout in the native per-row path using the
already computed stream context and expose the parent representation without
changing recurrent update order. Test Rust/PyTorch readout parity, then full
sequence parity including warmup, causal ordering, sequence reset and repeat
determinism. Measure exact ZIP size below 20 MB, evaluator RSS at or below 4 GB,
and representative one-CPU runtime below 60 minutes against the existing
public-best package. Do not replace that package until root reviews the entire
served endpoint against it. JSON pins the reviewed Rust/Python entry points and
the H272 result, fit/export receipts, source review, checkpoints and
membership audit.
