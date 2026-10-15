---
kind: review
status: reviewed-for-conditional-root-admission
created_utc: 2026-10-15T12:29:01Z
updated_utc: 2026-10-15T12:29:01Z
hypothesis_ids: []
experiment_ids: []
evidence_paths:
  - /home/vlad/development/wunder/worktrees/f3-teacher-export-r2-20261015/research/campaign_2026-10-15/f3_teacher_export_r2/README.md
  - /home/vlad/development/wunder/worktrees/f3-teacher-export-r2-20261015/research/campaign_2026-10-15/f3_teacher_export_r2/RESOURCE_AND_QUOTA_PLAN_R2.json
  - /home/vlad/development/wunder/worktrees/f3-teacher-export-r2-20261015/research/campaign_2026-10-15/f3_teacher_export_r2/SOURCE_BINDING_R2.json
  - /home/vlad/development/wunder/worktrees/f3-teacher-export-r2-20261015/research/campaign_2026-10-15/f3_teacher_export_r2/OFFLINE_PREFLIGHT_R2.json
  - /home/vlad/development/wunder/worktrees/f3-teacher-export-r2-20261015/research/campaign_2026-10-15/f3_teacher_export_r2/collect_terminal_r2.py
  - /home/vlad/development/wunder/worktrees/f3-teacher-export-r2-20261015/research/campaign_2026-10-15/f3_teacher_export_r2/tests/test_f3_teacher_export_packet_r2.py
---

# F3 CP2 export outer-wall and collection review R2

## Question or claim

Does the frozen F3 seed-3 H384 epoch-2 export packet bind a total runtime that fits the reported remaining quota, and can a complete-export claim survive transfer validation?

## Evidence and interpretation

The initial R1 packet was held because its bootstrap allowed 43,200 seconds while the latest reported quota had only 26,627.389 seconds remaining. R2 commit `65755f814b460d45fa6ba08c7924e4ba836d5bfd` fixes the actual outer process deadline at 21,600 seconds. The generated bootstrap, resource plan, source binding, offline preflight, collection plan, collector, and root-admission receipt checks agree on that limit; the worker cap remains 18,000 seconds, leaving a total 3,600-second startup/finalization envelope. Its worker resolver shrinks the worker timeout if bootstrap work consumes that envelope. Fresh typed quota of at least 21,600 seconds and separate root admission remain mandatory; the earlier quota snapshot is stale and grants no launch authority.

The R2 builder and offline preflight also pin the corrected failure diagnostic, `runner completed outside the 21,600-second kernel-process budget`. Verified hashes: builder `ae36caf61d7875c985174f3e46d2a4a8f4ecbe7605ee47821fbd3b48cb920789`, source binding `d767d0d59f1bf36227a128d2cc5a66d12d195803df861c7593b05469d4e3014c`, preflight `c8e46805a3f139071f347ed4e52fb44158758dd4c15e078876404a0d6177793d`, bootstrap `c4acf68854dad83fb55a980c0961641b82a8e2a6f594d064d0eaea37e2a4edbe`, archive `07a0299427b33260f250a6a5dc2cc958f36a084c30d5685126e442ca57593902`, metadata `86b537cc6b60bf5c596f10a720b18795aaeeb982882c578ef58351b220f7fcce`, and collector `b7ad2f306ab49d8ae2b839e9cca5189c05d0583a953f46a7e902d0fb7e7acbdc`.

The export source remains the frozen predictor and verified CP2/provenance/fold-3 membership. The triplet is causal FP32 prediction export only, with no fit, label-based grading, or selection. R2 collector verifies root admission/push pins, downloaded lengths and hashes against the runner manifest and sidecars, CP2/provenance/split/source hashes, all three stage receipts, exact array geometry/dtype and bounded mmap finiteness. A complete payload flag is gated on validation; a failed validation is preserved in the terminal receipt before the collector raises. The 2.5 GB download cap and 4 GB physical-disk floor cover the 2,204,800,000-byte prediction payload plus metadata/log reserve.

## Decision and limits

Conditional operational GO for the exact R2 packet after fresh typed quota is at least 21,600 seconds and root records exact private READY versions and admission. This is not provider authorization from the reviewer, evidence that prediction runtime fits 18,000 seconds, a teacher score, or a quality claim. No provider call, export, fit, or grade was performed for this review. The actor-reported focused tests and offline preflight were not rerun here.

## Source records

Actor packet: `/home/vlad/development/wunder/worktrees/f3-teacher-export-r2-20261015`, commit `65755f814b460d45fa6ba08c7924e4ba836d5bfd`. R1 remains preserved as a distinct superseded packet.
