# E-514 result metadata correction

This addendum preserves a filename-reference typo in the frozen [result](E514_RESULT.json). At `source_manifest`, the result names `SOURCE_MANIFEST_E514.json` (V1), while `source_manifest_sha256` contains `8de4dc6c1e2a9023510d3edbb3a46771579470bfb859ac50f451f7baa269d6fd`, the SHA-256 of [SOURCE_MANIFEST_E514_V3.json](SOURCE_MANIFEST_E514_V3.json). V3 binds the corrected notebook, prediction sources, chain partition, and prior manifests used by the completed comparison.

The V3 digest and source gate are correct. This is a result metadata reference error; it does not change prediction identity, ancestry, row order, metric computation, or scores. The result JSON remains unchanged so its original bytes and manifest hash stay reproducible. The [independent evidence review](INDEPENDENT_EVIDENCE_REVIEW_20261007.md) records the same finding.
