# Sparse-mask sample edge case and exact recovery

E415 cache extraction failed before producing a cache: a prespecified sequence
had no selected scored rows after seeded within-stratum sampling. The correct
contribution is empty; the extractor instead raised ValueError. Its failure
closed global admission and interrupted the independent E412 GPU fit with
SIGTERM. All failed attempts, source manifests and logs remain unchanged.
No OOM occurred in either worker or the parent cgroup.

The fix will retain the original sampled rows and report every zero-contribution
sequence explicitly, without target-based resampling. Regression tests precede
a new attempt. No E415 scientific result is available from the failed attempt.

E412 completed epochs0 and1 and saved the full raw model, EMA, optimizer,
scheduler, scaler, RNG and loader state. Its partial epoch2 work is discarded.
The immutable recovery snapshot is SHA256
`f4f7a6368b64dd943a1d8747eab73bf17cd706de871ec68fef344590d99888bf`.
Every saved source hash matches the current trainer/package. The same recipe,
data, sampling and schedule resumes at epoch2 under fresh run identity
`c20261004_combo_s3_final_r1`, because the runner refuses to overwrite prior
attempt outputs. The resumed head/packaging must reference this new parent.

Evidence: `checks/e412_recovery_epoch2.json`,
`specs/combo_final_resume1_new_identity.sealed.json`, both E412 attempt logs,
and `runner/logs/e415-feature-cache-a-v1.log`. Acknowledgement binds the current
pause and both diagnosed attempts; no legacy pause marker was cleared.
