# Conditional first-cell score admission V5

V5 preserves V4 and closes two deterministic source blockers. The original AllThree fit LAUNCH.json has no top-level invocation_id; V5 now checks the exact pinned receipt bytes, validates the submission and post-submission envelopes, strictly decodes nested base64, and binds the InvocationID property from systemctl_after_submission.stdout_b64 to the known AllThree invocation. It also validates the run, unit, captured state, cgroup, and representation before the score preflight continues. The renderer now opens the already-reviewed EXPORT_SOURCE_REVIEW_V2.json filename that matches its existing V2 digest and status pin.

The new fixture reads only the two small historical fit receipt JSON files. It executes only the AST-extracted V5 receipt helper with synthetic stat rows. It checks the actual receipt pair, rejects a wrong nested invocation even when a misleading top-level field contains the expected ID, and confirms a wrong top-level field cannot override the correct nested property. No model, dataset, cache, prediction array, B host, or remote service is read.

SCORE_ADMISSION_V5_INVERSE_PROOF.json proves the body and renderer are exact reconstructions from V4 with only the receipt identity repair and documented V5/path labels. The plan leaves actual export acceptance/exact-stop pins null and records that corrected exporter V3 and scorer V3 pins are still required. The prior actual exporter V1 launch failed before checkpoint or prediction payload reads; this V5 packet is not renderable or admissible.

Independent source review is pending. This packet authorizes no B query, staging, launch, scoring, or quality claim.
