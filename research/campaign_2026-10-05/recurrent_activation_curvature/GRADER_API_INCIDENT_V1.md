# E481 grader API failure before any test-data read

The full18-fit matrix finished successfully. The frozen v1 independent grader
did not pass its actual fit-only receipt check. Root and the separate reviewer
both executed `validate_attempt` followed by `validate_fit_manifest` against
the completed runtime artifacts. Attempt validation passed; manifest validation
failed at line412 with:

```text
AttributeError: module 'fit_activation_optimizer' has no attribute 'file_hash'
```

The grader calls `e470.file_hash(DATA_PATH)`, but the frozen imported helper has
no such public API. The grader already defines its own `file_sha256` function.
This is an API/integration error, not a scientific result or a training failure.
No evaluation loader, predictions or score arrays were opened by either failed
fit-only check. No model fits are discarded or repeated.

Root's bounded systemd helper `wnn-e481-complete-fit-root-verifier` exited1
after9.003s with303.3MB peak RAM, no GPU. It used1CPU/2GiB/no swap and had
CUDA visibility disabled. The original source remains immutable at SHA256
`52b91f9a3eaead30c71ef824feed8370b5a1e339ad7f8005c509b115b903c4d5`.
The six synthetic tests and prior conditional source review did not exercise
this actual module API call. They were insufficient evidence for full readiness.

The correction must use a separate v2 source identity and validate the actual
completed18-fit proof without monkeypatching either imported helper or grader.
Add an executable fit-only path and regression coverage for the real API calls.
An independently reviewed new grading spec must bind v2 plus the actual
complete proof before any scoring. Preserve v1 and this incident; do not
report a clean v1 pass or count this validation helper as an experiment fit.
