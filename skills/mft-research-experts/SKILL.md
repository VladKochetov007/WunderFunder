---
name: mft-research-experts
description: "Run research orchestration for data quality, factor geometry, hypothesis validation, and incident forensics. Use when you need SHIP/KILL/ITERATE decisions with strict validation. Triggers: mft-strategist, data-sentinel, factor-geometer, skeptic, forensic-auditor, research pipeline, hypothesis validation, post-mortem."
---

# MFT Research Experts

Use this skill for end-to-end quantitative research orchestration and validation.

## Role Map

- `mft-strategist` -> `agents/mft-strategist.md`
- `data-sentinel` -> `agents/data-sentinel.md`
- `factor-geometer` -> `agents/factor-geometer.md`
- `skeptic` -> `agents/skeptic.md`
- `forensic-auditor` -> `agents/forensic-auditor.md`

## Workflow

1. Read `README.md` and the active research contract; distinguish prediction, trading and causal-effect claims.
2. Inspect the applicable data audit before fitting or grading. Use `data-sentinel` for new data or unresolved issues; reuse a provenance-matched audit for unchanged data.
3. If the user requested one role explicitly, spawn that role with `agent_type` and targeted scope.
4. If no role was specified, use `mft-strategist` to assign only the necessary roles within the contract's worker, compute and review-cycle limits.
5. Use `factor-geometer` before `skeptic` only when factor exposure or risk geometry is relevant to the claim. Prediction tasks may go directly to predictive validation.
6. For incident or regression investigations, include `forensic-auditor`.
7. Synthesize a scoped SHIP/KILL/ITERATE recommendation with evidence status and uncertainty. SHIP means the registered gate passed, not automatic deployment; KILL rejects the tested candidate or claim, not all versions of its mechanism.
