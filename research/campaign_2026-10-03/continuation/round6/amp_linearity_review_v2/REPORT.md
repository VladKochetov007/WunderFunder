Independent v2 AMP linearity instrument review — narrow PASS
==========================================================

Verdict: PASS_DESCRIPTIVE_FIXED_TWO_CASE_INSTRUMENT_ONLY. The real-metadata JSON
blocker is corrected; no additional blocker was found for the frozen instrument
and already prepared two normal cases. Actual gradient analysis remains UNRUN by
this reviewer and requires explicit controller authorization. No receipt or GPU
work occurred; prior author/reviewer artifacts remain unchanged.

Correction and independent tests
--------------------------------
The only diagnostic changes are importing the public canonical_metadata helper,
requiring versioned metadata_identity instead of raw approval metadata, and
comparing the shared projection. Numerical functions/main are AST-identical to
the original reviewed instrument. FP64 main/ema/struct/next sum minus whole
unscaled PRECLIP, coefficients/scale, None support, canonical RMS denominator,
zero-reference undefined relative norm, groups/scopes and chunking are unchanged.

Tiny real-source-shaped tensor metadata survives JSON roundtrip and permits the
same independently checked residual arithmetic. A sentinel permits only Tensor
metadata getters and throws on numerical operations; projection and serialization
pass. Wrong shape/dtype/attachedness, scalar/extra provenance, unexpected tensor
field, cycles, raw old metadata and projection version reject. Typed bool versus
int identities differ. The completed tiny tests were preserved, not rerun after
the controller constructor change. Body0.0064s, peakRSS525176KiB.

Descriptors bind class/shape/dtype/CPU device/layout/gradient attachment, not tensor
values: equal descriptors with different invented values intentionally project
identically. This is honest metadata identity. Contents remain bound by the exact
independently verified pre-file SHA; there is no local pre-input rehash, numerical
propensity/scored-mask comparison, list conversion or silent field omission.
Projection budgets bound depth16/nodes4096/containers1024/text1MiB. Converted
FP64 workspace remains the previously verified8MiB ceiling, below32MiB; CPU
process limit remains2GiB. Actual gradient schema/finiteness compatibility is still
untested and must retain fixed-case rejection without replacement.

Controller preparation integration
-----------------------------------
In the tiny review process, analyzer and preparation helper import the EXACT same
canonical_metadata function from frozen metadata.py. No helper main executed.
The permitted prepared fixed_case_metadata_v2.json is JSON-only and remains marked
METADATA_ONLY_NOT_ANALYSIS_AUTHORIZATION; analyzer approval guard rejects it alone.

Its two normal names, paths and pre/post/components SHA declarations match the
earlier independent streamed-byte verification; file stats remain unchanged.
Real tensor descriptors, native approved groups/IDs/parameter identities, source
hashes and measurement-review STATE identity are consistent. No actual capture
file content or gradient was opened by this review. Exact fixed JSON hash and
projection source hash are in integration_bindings.json.

The preparation helper has a dormant branch that hashes numerical inputs if
real_trajectory.json is absent. That branch is OUTSIDE this metadata-only scope.
The existing trajectory is preserved and hash-bound; controller reports it was
unchanged during preparation. Use the prepared JSON, and do not rerun that helper
after deleting/replacing the trajectory. This PASS does not authorize its fallback.

Final controller constructor guard
----------------------------------
The updated approve_amp_audit.py was reviewed and re-pinned BEFORE terminal freeze.
It verifies the final review manifest, matching review/integration verdict and
exact four candidate hashes, then rejects changes to fixed metadata, measurement
STATE, preparation helper, existing trajectory and output manifest. Fixed paths/
capture declarations are also checked; approval is exclusive-create and includes
candidate, review and fixed-metadata bindings. Recent observed completion, healthy
watch, barriers and deadline slack remain controller admission prerequisites.

Only the new rejection expressions were evaluated against tiny fake-hash maps:
matching bindings pass and each substituted artifact/verdict/candidate rejects.
No approval constructor, real SHA operation or capture access occurred in those
predicate tests. Their separate results preserve the earlier tiny retest output.
The constructor hash is refreshed in integration_bindings.json. Controller must
verify that hash and every external reviewed binding before issuance/launch.

Bindings and scope
------------------
candidate_sha256.json is EXACTLY the flat four-name inventory PREREG.md,audit.py,
metadata.py,tests.py, included in the final review manifest. Projection hash:
57daa24568eab2ff6d447379af21506fadb026f161716cd76d815aa5a9e580ea.
Independent measured-capture STATE hash:
17a006c34ac4fb6be4d866f37c457d0d9b02dc4173bd5970f8ec55a187c58d6c.
All declared frozen author/source/prior critic and preserved manifest hashes match.

Only an explicitly authorized descriptive residual readout on these TWO fixed
normal cases is within scope. There is NO discrepancy acceptance threshold or
default-floor allowance. Zero or nonzero residual establishes neither a scientific
decomposition bound nor conflict, optimizer-update attribution, WP improvement,
teacher causality or a scientist-arm result. Controller must retain failed cases
and choose a fresh output without overwriting frozen files. Final STATE/HASHES
bind this terminal review; they are not themselves an actual-analysis receipt.
