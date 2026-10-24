---
kind: review
status: source_fixture_pass
created_utc: 2026-10-24T12:02:07Z
updated_utc: 2026-10-24T12:02:07Z
hypothesis_ids: []
experiment_ids: []
evidence_paths:
  - membership_repair_r1/SOURCE_PINS_R1.json
  - membership_repair_r1/TEST_RECEIPT_R1.json
  - membership_repair_r1/test_membership_repair_fixture.py
---

# Independent source review: portable subset membership repair

## Finding

The repair is narrowly aligned to the producer's stored schema. The frozen raw
fit-subset JSON stores `selected_members_sha256` at its top level. The normal
producer loader wraps that raw manifest and exposes its fields under a
`manifest` key; the cache metadata producer reads the wrapped field and writes
it as `subset_members_sha256` to the fit and evaluation sidecars. The portable
consumer reads the raw JSON directly, so its old nested lookup caused the
version-3 failure even though the source subset and cache sidecars agreed.

The revised consumer reads the raw top-level value, requires a lowercase
64-character SHA256 string, and keeps exact equality checks against both fit
and evaluation sidecars. The existing recipe, subset-file, split, fold,
source-qualified row-reference, parent-prediction, and ancestry checks remain
in place. The `module_hashes` import moved inside the full-cache validation
function that uses it; that preserves its full-cache check and permits the
metadata path to load without Torch.

## Independent metadata-only check

I independently called the pinned fixture functions under the 150 MB,
one-CPU, 60-second `scripts/limited` scope. Both folds passed against the
source-produced sidecars and frozen subset manifests. The fit-side group
membership digest recomputed from the subset's `(split, row_group)` members
matched the manifest and both cache sidecars. Fold 0 had 1,236 fit refs and
468 evaluation refs; fold 1 had 1,243 fit refs and 457 evaluation refs. The
fixture then changed a fit-side membership digest and confirmed validation
rejected it. All six source-module hashes matched the repair's pin manifest;
Torch remained unimported.

The check did not open cache arrays, labels, or masks, rebuild caches, run a
profile microfit, or establish remote import/runtime behavior. It verifies
the schema/equality path only. The original Kaggle version-3 error remains
preserved as execution evidence; this review does not change that outcome or
authorize another provider run.

## Trace

- [Repair packet and diagnosis](membership_repair_r1/README.md)
- [Source pins](membership_repair_r1/SOURCE_PINS_R1.json)
- [Producer-backed fixtures and equality test](membership_repair_r1/test_membership_repair_fixture.py)
- [Actor's bounded test receipt](membership_repair_r1/TEST_RECEIPT_R1.json)
- [Version-3 terminal receipt](TERMINAL_RETRY_R1.json) and its
  [checksum errata](TERMINAL_RETRY_R1_ERRATA.json)

