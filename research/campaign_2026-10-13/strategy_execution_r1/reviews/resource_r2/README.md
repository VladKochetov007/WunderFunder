# Resource admission review R2

This folder independently reviews commit `f6550d6c1fd7318d48c9381db781fb341db28912`, the headroom correction on top of `b185e7048785e1c6db60303dbd228c8ce803bd3a`. It confirms rejection when parent use, unused admitted-worker reservation, and a new task bound total 18 GB; confirms the exact 17 GB boundary and no double-counting; and checks current-attempt exclusion plus fail-closed handling for missing, queued, or vanished worker cgroups. The focused resource and allocator tests pass.

The fresh, zero-active first local GPU attempt is conditionally admissible after root performs the supported resume of the historical shared runtime. Dynamic subtraction from an active worker's sampled memory has a non-atomic snapshot edge: until stronger measurement is available, overlapping support admission must use the full active-worker reservation in its manual parent-memory check (or wait for that worker to finish). The legacy queue remains stopped. This review contains no model-quality evidence and no GPU, Kaggle, Vast, or billing action.

- [Independent review](INDEPENDENT_RESOURCE_REVIEW_20261013.md) records concrete findings, evidence, and launch disposition.
- [Machine-readable receipt](INDEPENDENT_RESOURCE_REVIEW_20261013.json) records commit identities, limits, checks, and remaining condition.
