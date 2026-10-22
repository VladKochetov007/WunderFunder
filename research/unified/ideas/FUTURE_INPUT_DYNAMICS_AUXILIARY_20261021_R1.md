---
kind: idea
status: implementation_pending_validation
created_utc: 2026-10-21T00:00:00Z
updated_utc: 2026-10-21T00:00:00Z
hypothesis_ids: [H-FUTURE-INPUT-DYNAMICS-20261021-R1]
experiment_ids: [E-FUTURE-INPUT-DYNAMICS-20261021-R1]
evidence_paths:
  - research/experiment_drafts/dynamics-aux-20261021/README.md
---

# Future-input dynamics auxiliary head

The accepted October 21 research cycle allocates a four-fit cold screen for a
training-only auxiliary head on the small H512×1 i1-control backbone. It
predicts 52 summaries of future raw-input change for 13 canonical feature
groups and horizons16/256, with fit-only target standardization and mean Huber
loss beside the unchanged WMSE. Future observations are labels only. The
matched weight-zero control allocates the same initialized head. Both arms use
F0/seed1001 and F2/seed1002; the exact frozen memberships, equations, CLI, and
test command are recorded in the [implementation packet](../../experiment_drafts/dynamics-aux-20261021/README.md).

The component implementation reuses the canonical modular trainer. Both
serving exporters strip the auxiliary head and normalization buffers, while
training checkpoints retain them for provenance and restoration. Four focused
CPU tests and static checks pass. The isolated branch now has literal
fit→full-fold-predict→score plans for all four runs, a 10-warmup/100-measured
F0 aux005 profile plan, an exact-hash F2 manifest copy, and a fixed paired
full-fold grader that checks head/target-statistics identity, complete NPY
coverage, per-target and position-bin deltas, and the unsearched 90/10 mixture.
The profile and fits remain pending. This packet supports the cold
screen only; strong-parent continuation is currently rejected and requires a
separate extension. No target-only or public gain is claimed.
