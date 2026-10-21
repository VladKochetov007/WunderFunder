# Score acceptance helper V4

V4 is the reviewed helper shape used in the final score acceptance chain. Its fixed-model checkpoint field is exactly the scorer-emitted `{bytes, relative_path, sha256}` object; it does not add `loaded_sha256`. Accepted export authority continues to bind the loaded checkpoint identity.

- [V4 helper source review](HELPER_SOURCE_REVIEW_V4.json)
- [Exact fixed-model delta proof](HELPER_V4_FIXED_MODEL_DELTA_PROOF.json)
- [Focused actual-metadata fixture source](test_score_result_acceptance_v4_actual_metadata.py)
- [ROOT score acceptance and exact-stop record](../../records/first_cell_evaluation_v1/score_v4/README.md)

The review and fixture scope is metadata/source validation; neither makes a scientific quality claim.
