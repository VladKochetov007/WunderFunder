# Canonical tiedlead capability publication

The opt-in tiedlead adapter is independently reviewed and published at commit **b104b0ad36c05d1d5aa13f949cd7d3aff16af318** on [experiment/tl3-tiedlead-capability-20261018](https://github.com/VladKochetov007/WunderFunder/tree/experiment/tl3-tiedlead-capability-20261018). [Publication](PUBLICATION.json) binds every one of the 30 changed Git blobs and eight exact metadata copies. The branch is separate from the canonical B source and the propensity branch; no campaign source was merged or staged.

The adapter assembles 112 raw + 99 event + 52 causal lead channels, shares one 104→128→64 encoder across raw-scale instrument/lead branches, and resets differences at every encode call while carrying GRU state. It supports checkpoint reconstruction and canonical feature-only prediction/evaluation. The [source manifest](SOURCE_MANIFEST_V1.json) names code-branch files, not files to reinterpret against main. The [engineering descriptor](recipe.json) records the fixed profile and missing work.

[Validation V2](VALIDATION_V2.json) records four focused tiny CPU tests, Ruff and the training structure check. [Independent review V2](INDEPENDENT_REVIEW_V2.json) checked all 16 implementation/test/doc hashes and five design references. These are engineering results, with no actual-data/GPU profile, fitted model, quality effect or historical execution equivalence. Same-head family-zero and direct alpha-zero objective capabilities, scientific integration and native ZIP/ONNX serving remain pending.

Two local metadata errors are preserved transparently: the original untracked validation V1 bytes were overwritten before capture ([erratum](VALIDATION_ERRATUM_V1.md), [stored V1](VALIDATION_V1.json)); V2 is authoritative. The [first review](INDEPENDENT_REVIEW_V1.json) had a future timestamp; V2 corrects it using the observed clock without changing source or conclusions. Neither correction changed source code or tests.

The [PR body](PR.md) is prepared only. There is no created PR URL after the earlier access failures; the push response's pull/new link is prospective. No active browser was used. Root owns the subsequent objective integration, bounded actual runtime profile and scientific preregistration; this publication grants no fitting admission.

[Independent publication check](INDEPENDENT_PUBLICATION_CHECK_V1.json) reconciles all 30 changed commit paths, 21 source/design pins and eight exact metadata copies. It verifies local Git objects; the root-recorded remote ref check is distinguished from an independent network query.
