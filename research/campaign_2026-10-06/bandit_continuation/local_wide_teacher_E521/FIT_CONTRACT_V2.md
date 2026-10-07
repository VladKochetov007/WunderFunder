# E521 bounded scientific fit, version2

Authority: explicit user local-GPU reopening in runtime/LOCAL_GPU_RESUME_20261007.json.
Root owns this independent scratch fit and export handoff; independent grader owns comparison.
Frozen recipe: RECIPE.json conditional_fit_command, batch16, GRU384x2, seed3,4epochs,
fixed zero-based ep3, no fit-time evaluation. Width is the only scientific change from
E520 GRU128 seed3 after inactive source additions are audited; record any source mismatch.
Use absolute project .venv/bin/python path without resolving its symlink. No save-training-state
flag added. Train/evaluation groups, loss, feature encoding and clipping stay as registered.

B16 actual full-shape profile passed exit0,8timed batches:2.6225GB observed totalVRAM,
4.3004GB scopedRAM,14.279GB aggregateRAM and zero OOM increments. B16 median batch
1113.66ms projects4*746 batches to3323.2s before modest loading/checkpoint overhead.
Fit wall bound7200s;3CPU,6.5GB scopedRAM,4.3GB allocator cap,5GB whole-deviceVRAM;
aggregate6CPU/20GBRAM,18GBmemory-high,no swap, one durable GPU lock.
Use existing bandit_resources admission/heartbeat/source-hash checks and external timeout.

Scientific endpoint remains exposed fold0, full two-target official global_wp_clip2_v1,
same seed3 raw GRU128 E520 fixedep3 control. No local result is a public forecast.
If matched control is not yet available preserve candidate export pending its completion.
No public upload or serving promotion is authorized from this fit alone.
Independent recipe review must PASS before launch. Fit success requires checkpoint and
terminal receipt; export/evaluation are separate owned follow-up phases.

Admission revision:7GB reservation was refused before launch because parentRAM11.2524GB plus7GB exceeds18GB working ceiling. Use6.5GB scoped cap, retaining2.1996GB headroom above measured4.3004GB B16 profile. Parent18GBhigh/20GBhard and GPU limits stay fixed; no fit ran in refused attempt.
