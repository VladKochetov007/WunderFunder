# ADR0031: Cross-fold ranking and top-tier submission release

Status: Accepted. Date: 2026-10-11.

## Context

The user explicitly requires diversified, reliable cross-validation rather than reliance on fold 1, and asks to receive only the best submissions after the current exploration results land. The selected scratch model's fold-1 gain did not transfer to public scoring. Additional seeds on the same exposed fold measure training variation but cannot by themselves measure fold dependence. The user also plans an exploitation phase using the competition's 60-minute serving headroom, potentially with Q4 compression.

## Decision

Rank recipes and complete fixed mixtures using multiple source-qualified folds and distinct fitting seeds. Report the per-fold/per-seed table, paired comparisons to relevant controls, selection exposure, component uncertainty and training variation separately. Absolute scores on different folds are not interchangeable; use matched contrasts and fixed fold aggregation, not the maximum observed score. Reuse completed matching controls; machine reruns of one seed do not add independent evidence.

Every held-out comparison excludes its evaluated groups from all fitting, teacher, initializer, propensity and learned-combiner ancestry. A fold-1-trained teacher or parent cannot teach/init a held-out fold-0/2/3 model merely because the student excludes those groups. Existing folds remain exposed. Cross-validation measures robustness without claiming pristine confirmation.

After the current exploration results are collected and interpreted, give the user only the strongest supported deterministic submissions. Do not recommend additional fold-1 variants while waiting or supply a long list of weaker fallback ZIPs. Existing ZIPs and public receipts stay frozen. Missing cross-fold evidence is a concrete delivery dependency, not a reason to invent a public forecast or reject a mechanism universally.

Freeze a small final endpoint set before all-data exploitation. Use the strongest supported source recipes and useful conditional diversity; independently train final weights, retain provenance, and test actual quantized endpoint quality/reset/runtime. Exploit Q8 headroom first. Calibration-aware Q4 or mixed precision is eligible when it improves the complete endpoint after compression losses and actual CPU cost. The 60-minute/20-MB envelope does not imply that every additional model is useful or that half the bits halves runtime.

The user has not stopped healthy research, changed provider/model preferences, raised the $70 owned-Vast cap or authorized rentals/top-ups. Remaining prepaid credit and provider expiry must bound starts more tightly when necessary, with collection time reserved. Continue the same controller and durable monitoring; record next-stage compute admissions separately from this selection policy.

## Consequences

Kaggle H382's current three-arm fold-0 seed-3 comparison is a useful first robustness test, not the complete multi-fold/multi-seed requirement. Before final recommendation, audit existing clean cross-fold artifacts and prepare targeted missing confirmations for the shortlisted recipes/blends. Finish already-admitted scientific arms. The archived unstarted idea backlog is not a prerequisite to finishing this bounded exploration wave.

[ADR0030](0030-local-evidence-selects-submissions.md) remains in force: research and local validation select submissions; public scoring confirms transfer. [ADR0009](0009-uncertainty-is-not-a-kill-switch.md) still distinguishes allocation uncertainty from a universal kill rule.

Evidence and delivery state: [current wave and exploitation record](../../research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/IDEA_STATUS_AND_EXPLOITATION_PLAN_20261011_R1.md), [public-gap review](../../research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/review_public_gap_20261011/REVIEW_PUBLIC_GAP.md), and [current state](../current-state.md).
