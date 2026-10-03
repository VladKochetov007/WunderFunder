Independent AMP linearity instrument review — BLOCKED as frozen
===============================================================

One exact actual-schema blocker prevents admission of the required real normal
case. The saved v4 trajectory/parity result is unaffected. No actual captured
gradient/prestate tensor was loaded or inspected in this review; no GPU, replay,
autograd, author edits or receipt occurred.

Exact blocker
-------------
audit.py parameter_map directly compares pre['metadata'] != case['metadata'].
The hash-bound actual adapter fit_panel.py:196–197 puts cached_propensity and
scored_mask tensors inside capture_metadata; Capture.before preserves them in
pre.metadata. Approval is loaded from JSON, which cannot directly serialize
those tensors. A JSON list/descriptor is not the same Tensor-bearing dictionary.

Our independent tiny fixture proves both boundaries: json.dumps(Tensor-bearing
metadata) raises TypeError; converting the two tensors to JSON lists makes the
frozen discrepancy() raise ValueError('Approved source/backend/case metadata
differs'). The identical scalar-only JSON metadata control passes. This uses
invented 1x2 tensors and source inspection, not actual gradient data. The
controller independently observed the real Tensor schema via metadata-only mmap
and retained its serialization failure/partial metadata; we did not read that
capture or failure artifact. Direct Tensor comparisons may also produce ambiguous
boolean errors; our measured JSON-list witness reports the exact ValueError above.

This is a deterministic schema/admission problem, not an observed scientific
residual, corruption or invalidation of the already approved measured capture.
The runner would retain a rejected fixed real case rather than replace it, but
cannot complete the proposed two-case descriptive measurement as frozen.

What passes
-----------
The diagnostic's algebra has the intended descriptive meaning: fixed FP64 order
sum(main,ema,struct,next) minus whole-loss UNCLIPPED/unscaled saved preclip.
Captured coefficients/scale are not applied again; previous whole/default parity
supports that reference identity but supplies no component-decomposition bound.

Independent Python oracle for a tiny active parameter plus an unused parameter
matches error L2=sqrt(2), maxabs1, RMS=sqrt(2/6), reference norm3 and relative
sqrt(2)/3. Unused elements explicitly enter the canonical RMS denominator; zero
reference has undefined/null relative, with no epsilon. A deliberately large
residual remains descriptive and does not fail a magnitude threshold. Per-name,
serialized-group, Muon/Adam and fixed-name-scope aggregation is descriptive;
it does not attribute updates or establish conflicts, acceptance or WP benefit.

The code streams post/components hashes and uses trusted CPU mmap-only loading;
pre hash remains an independently approved declaration, not a local input-byte
scan. Approval guard precedes capture path access. The guard checks declared
review/controller flags and receipt hash syntax; authenticity and exact approved
case paths remain an external controller trust prerequisite, not proved by toy
fixtures. No actual case receipt or permission was fabricated here.

CHUNK=131072; slicing precedes FP64 conversion, with no full-gradient concatenation
or parameter-sized converted array. Conservative declared live-array ceiling is
8MiB, below32MiB. Code includes finite-value/support/name/shape/group rejection and
fixed-case exclusion without replacement. CPU fixture body0.0089s, peakRSS527024KiB
(Torch import dominates), below2GiB. Source contains no model/CUDA/autograd calls.
Actual gradient contiguity/dtype/finiteness compatibility remains untested.

Required separate correction
----------------------------
MINIMAL_FIX_SPEC.md requires an explicit JSON metadata-only projection: retain
frozen scalar/JSON provenance fields and bind tensor-valued metadata by exact
shape/dtype/storage/layout descriptors plus independently approved pre-file SHA.
Do not read numeric propensity/scored values, convert them to lists, rehash raw
prestate inputs, silently drop arbitrary fields or weaken case identities.
Alternatively select explicitly frozen required JSON fields, while those excluded
tensor values remain covered by the external exact file identity. The projection
and approval schema must be explicit on both sides. Tiny JSON-roundtrip and
metadata-only sentinel controls must pass in a new preserved version before
actual-gradient admission. Keep all algebra, scope, cases and resource limits.

Preservation and verdict
------------------------
All declared author frozen artifact hashes and FROZEN_SPEC match. Both original
instruments' prior manifest identities match their recorded preservation entries;
their scientific findings were not read or changed. The schema-source adapter
hash matches the independently approved v4 candidate. Author files, captures,
receipts and prior reviews remain untouched. Own runnable tests, results,
protected_verification.json, STATE.json and HASHES.json bind terminal evidence.

Verdict: BLOCKED_REAL_METADATA_JSON_IDENTITY_SCHEMA. No actual diagnostic receipt
for this frozen instrument. A corrected instrument may later be reviewed ONLY
for descriptive residual measurement; no scientific acceptance bound, conflict,
optimizer-update attribution, WP claim or scientist-arm authorization follows.
