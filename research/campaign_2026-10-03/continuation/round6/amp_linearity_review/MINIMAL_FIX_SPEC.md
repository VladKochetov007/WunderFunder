Separate-version correction: metadata-only JSON approval projection
==================================================================

Preserve the frozen amp_linearity_audit author files. The algebra is not the
blocker. Actual source fit_panel.py constructs pre.metadata.cached_propensity
and scored_mask as tensors; direct equality with an approval JSON dictionary
cannot represent their identities faithfully. Tiny source-schema fixture rejects
before descriptive calculation. Do not read the actual arrays to work around it.

Define one explicit, versioned JSON projection used identically when constructing
controller approval and when inspecting the prestate. Retain all scalar/string/
JSON-container metadata, including evidence kind, panel hash, row IDs, teacher
bindings, departure, s and auxiliary positions. For the two tensor-valued metadata
fields, use ONLY canonical metadata descriptors: exact tensor class, shape, dtype,
CPU storage/device, and any required layout/detachedness metadata. No .tolist(),
numeric equality, finiteness scan or numerical hash of prestate arrays.

Bind the approved exact pre-file SHA256 from the independent measurement review
as a declared external identity; preserve the explicit no-local-pre-rehash limit.
Do not replace the comparison with dropping arbitrary metadata or comparing only
the evidence-kind label. Unexpected fields/types must fail closed rather than
being silently skipped. Approval schema must document the projection and compare
it to the prestate projection; direct pre.metadata==JSON equality must disappear.

Minimal tests use tiny source-shaped dictionaries only: scalar-only positive
control, tensor metadata descriptors surviving JSON roundtrip and permitting the
same arithmetic, wrong descriptor shape/dtype rejection, and a sentinel that
throws on tensor-valued metadata numerical operations yet allows shape/dtype
inspection. Keep fixed two normal cases and ordinary/default prereqs. Approval
still comes externally from the controller and independent measurement review;
synthetic fixtures cannot authorize reading real gradients.

Keep FP64 order main,ema,struct,next, existing coefficients/scales applied once,
chunk limit, None participation and RMS denominator, undefined zero-reference
relative norm, and all descriptive-only outputs unchanged. Do not add scientific
acceptance limits, default-floor allowances, conflict/update/WP interpretations,
new cases or replacement selection. Freeze and independently review the corrected
version BEFORE actual captured-gradient analysis.
