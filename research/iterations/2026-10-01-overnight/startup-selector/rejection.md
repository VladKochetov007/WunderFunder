# Startup selector: rejected at source review

Status: **UNALLOCATED_NOT_DELIVERY — REJECTED_REPRODUCIBILITY_REQUIREMENT**.

The proposed selector would measure generated full-callback latency at construction, choose either all four unchanged V1 members with equal quarters or the three h128 members with equal thirds, then keep that choice for the entire actual input stream. The provisional discussion used three 4,096-callback trials per profile, a maximum-trial estimate, a 1.15 margin and a 50-minute projected budget. These were preparation ideas only: no implementation, package, calibration, inference, timing pilot, quality readout or training was executed.

The repository's official-documentation copies are decisive:

- `docs/faq.md:49–50`: “Must my solution be deterministic?” — “Yes. Repeated runs on the same input must produce identical predictions.”
- `docs/submission_guide.md:146`: “Confirm identical predictions across repeated runs.”

Clock-dependent selection can cross the threshold on repeated runs because of scheduling, startup state or load. Keeping a profile fixed after construction does not remove that dependence. The two profiles produce different forecasts, so the proposed selector does not robustly satisfy the stated reproducibility requirement. Root stopped the branch before implementation or empirical allocation; no threshold changes, hysteresis, retries or substitute selectors are authorized.

The historical V1 wall-clock governor has the same reproducibility concern. Its prior recorded public score and runtime do not establish compliance for this proposed artifact, nor would they transfer to its changed composition. The already packaged fixed3 model remains a separate fixed-composition option, subject to its existing ISA and runtime qualifications.

Only this rejection note and its source-binding JSON were created in `startup-selector/`. Existing ZIPs, native sources, checkpoints and forecasts were not changed or read for this branch. The official-documentation wording was inspected locally, not freshly verified online. No claim of a new output, quality result or deliverable is made.
