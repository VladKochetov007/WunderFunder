# H273: distinguish feature diversity from loss and seed diversity

Status: proposed; numerical prior and measured effect unknown.

Basis: the inspected own `raw_wmse` source checkpoints declare `raw_only=false`. The family's name therefore does not establish raw-feature diversity. See the [method comparison](../../reviews/MAXIM_READY_PAIR_METHOD_COMPARISON_20261008_R2.md).

Mechanism: a restricted input representation can make different errors even when its standalone accuracy is weaker. It may improve a heterogeneous teacher target, student representation or served blend by supplying complementary information.

First comparison: independently train genuine raw-feature WMSE teachers using the existing source-family architecture and training recipe, changing only the declared input view. Preserve the current full-feature teachers and compare equal, fixed family coefficients. A separate interaction compares each family's contribution to the complete student/blend endpoint. Do not select families from standalone score alone.

Recipe owner must freeze supplied-data membership, preprocessing, objective weighting, clipping, optimizer, schedule, checkpoint selection, row order and seed before admission. Use independent weights; inspect teammate methods only. Match group exclusion through teacher and student ancestry for the exposed cleanfold comparison. All-data production training is a separately identified deployment step.

Scientific evaluation: official metric on the frozen qualified evaluation groups; report exposure. Compare prediction residuals and family contributions, then evaluate the complete endpoint. Save diagnostic plots in this lineage's `plots/` directory through marimo-pair. Public feedback confirms transfer rather than acting as the sole selection criterion.

Follow-ups depend on evidence: feature-restricted family mixtures, residual specialists, disagreement-conditioned targets, or different causal feature summaries. A weak standalone teacher does not falsify conditional value. A lack of useful diversity under the fixed replacement justifies changing representation or interaction before repeating seeds.

Execution priority: after the ready stronger-parent dependency, with a runnable frozen recipe prepared during existing fits. H230 delivery actor owns recipe preparation; root owns resource admission and independent interpretation. No fit is authorized by this hypothesis card.
