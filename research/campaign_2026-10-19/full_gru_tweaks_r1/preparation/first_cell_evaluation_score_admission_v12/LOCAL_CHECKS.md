# V12 local checks

- Ran the exact cache-schema proof with the maintenance virtualenv Python: PASS_V12_ACTUAL_CACHE_SCHEMA_AND_EXACT_INVERSE.
- Ran the V12 focused cache-schema fixture with the maintenance virtualenv Python: 3 tests passed. The fixture uses frozen C metadata receipts and synthetic stat responses for array paths only.
- Ruff check and format check passed for the new proof and fixture.
- Ruff check on REMOTE_SUBMISSION_V12.py reports 23 E402/F811 findings from the inherited second import block at body line 218. The V12 body is an exact V11 copy except for the single cache predicate, so this inherited block was preserved rather than cleaned up.
- No old suites, model code, source/model files, B resources, payload arrays, SSH, unit, or admission actions were accessed or used.
- Independent exact-body source review remains required before any new admission.
