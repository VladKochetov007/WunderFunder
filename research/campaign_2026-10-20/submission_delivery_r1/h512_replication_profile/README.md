# Conditional local H512 cost profile

Status: **NOT_ADMITTED**. The [profile plan](PROFILE_PLAN_R1.json) derives the
wide-arm command from the frozen PR25/R01 recipe and rebinds it to the existing
local producer, canonical F0 manifest and seed1003. It preserves the cold
211-input H512×1, label-only, strictFP32, B8/C2000, six-epoch schedule with
fixedCP5, AdamW1e-4, OneCycle, deterministic seeded batches, no pinning/fast
loader flag, no auxiliary terms and no fit-time validation. The finite profile
is one warm-up plus three measured batches (40chunk updates), not a scientific
six-epoch fit or quality result.

The [independent PR results review](../../../unified/reviews/B300_PR_RUN_UPDATE_REVIEW_20261019_R1.md)
records12available exposed-fold width/depth pairs. Wide wins9/12 and averages
0.68409111 versus0.68331379 for deep. This favors a conditional wide-model
replication if NLL has little endpoint value; it does not guarantee a local
win. Actual PR25 quality seeds are1003–1005. The original seed1001 command was
a profile/staged recipe, so changing seed/source/paths is explicitly derived.

A metadata-only independent review rejected the assumed4h+4h fold/production
cost. The short B300 profile extrapolates to about5.8h and6.1h, before setup
and delivery, and is not local timing. This local profile must determine
whether a complete endpoint fits the remaining15-hour budget. It admits zero
new fits and may only run after the current single-GPU chain drains.

The new execution source68ff6829 differs from R01's a266b036; source equivalence
is not claimed. The plan pins original recipe/command bytes and names both
revisions. Before any profile, verify source/data/split identity, defaults,
resource headroom and the shared GPU lock. A later H512 fit needs its own exact
recipe, cost reservation and frozen official comparison. Replacing TL3's plain
student with H512 could preserve three models, but ZIP size, CPU runtime and
blend contribution are unmeasured.

## Actual local profile completed

[Admission](PROFILE_ADMISSION_R1.json), [raw log](profile_r1.log), and [result](PROFILE_RESULT_R1.json) bind the derived source and actual data checks. Service4efb14e7231e4252ad407af755c6c3cd completed successfully in15.17seconds:40/40finite optimizer updates, no AMP, zero process swaps, cgroup peak3.602GB. Median batch wall1.89304seconds projects4.707hours for six fold epochs; this is a short-profile estimate, not full-fit timing. No scientific fit has yet been admitted. Two metadata preflight stops (footer-length interpretation and residual inactive cache) occurred before model launch and are retained in admission.

[Aggregate resource enforcement](AGGREGATE_LIMIT_ENFORCEMENT_R1.json) records the runtime parent change to the latest 5 GB task allocation, two CPU cores and zero swap, after cache reclaim. This enforces the existing user contract and does not alter model updates.

[Reload precedence repair](AGGREGATE_LIMIT_RELOAD_REPAIR_R2.json) supersedes the first runtime enforcement method: a service daemon reload restored a higher-priority legacy20GBsetting, briefly allowing5.87GBcache-inclusive usage (2.32GBanonymous). Inactive-file reclaim reduced usage below3.8GB; a later-named runtime slice drop-in survives reload and re-enforces5GB. OOM/kill counters remained0 and task swap remained0. No model process restarted.
