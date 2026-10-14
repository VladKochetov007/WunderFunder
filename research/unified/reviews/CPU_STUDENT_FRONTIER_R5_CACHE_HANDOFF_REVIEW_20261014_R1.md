---
kind: review
status: complete
created_utc: 2026-10-14T02:47:57Z
updated_utc: 2026-10-14T02:47:57Z
hypothesis_ids: [H-STUDENT-LADDER-20261014]
experiment_ids: []
evidence_paths:
  - research/campaign_2026-10-14/cpu_student_frontier_r1/recipes/R5_SEQUENCE_STATIC_PREFLIGHT_R2.json
  - research/campaign_2026-10-14/cpu_student_frontier_r1/recipes/CPU_STUDENT_FRONTIER_F0_S3_H256L1_R5_SEQUENCE.json
  - scripts/reclaim_student_frontier_cache.py
---

# F0 student-frontier R5 cache-handoff review

## Question or claim

Does the sealed R5 successor preserve the admitted per-shape fit/profile/export/grade steps and gate each file-cache handoff on source-bound terminal evidence and unchanged resource limits?

## Mechanism and method

Read-only review of the clean `student-frontier-ops-r5-20261014` worktree at commit `e49a7b6ae6c2bcca36cda1d129f39ac624adabda`. Reviewed the handoff helper, builder, static checker, seven generated plans, actual R4 base plans, static receipt, tests, and root `bandit_resources.py`/`bandit_sequence.py` controller behavior. Independently compared the eight preserved source steps in all seven R5 plans with their R4 counterparts under a 500 MB/one-CPU cap. No handoff, GPU job, fit, export, grade, or provider operation was run.

## Evidence and interpretation

**Bounded GO:** the packet preserves the eight original per-shape worker steps byte-for-byte and adds only one guarded 400 MB/one-CPU handoff after each validated profile and after each validated grade. All seven plans are single-shape. The H256×1 plan reuses its existing attempt ID; later shapes remain separately admitted. The independent step comparison passed for all seven shape pairs.

The static receipt pins the R5 helper SHA256 `6b776d6b324b3fa6fa8b69fd4d9beb567ed76d83b2fa2d07144d0d44123e7ccc`, builder `5cfa28cd49d4d2acbb21f19e91dc0bed91aa675f28b5577be21fe41477b5be47`, checker `864f65e741d0515b2af1c3f995ecef8c35a0f068781981bd6b1e026fd4a5fcc2`, and receipt `bca823bd593aab5f90329681a55dd3d287fc47f13740c699029397e64d5fd7a8`. It pins the root admission, profile contract, and seven plan hashes. The R5 report records five focused helper tests and the static preflight passing; those tests use a mocked cgroup and establish guard logic, not real kernel reclaim behavior.

Before reclaim, the helper checks exact plan/contract hashes and terminal profile evidence, including the custom predictor hash, output size and prediction digest, resource observers, observed VRAM, and OOM deltas. The after-grade path also requires terminal source-bound export and official fold-0 grade evidence. It acquires the same controller and GPU locks used by the launcher, rejects active or unresolved GPU attempts, and rechecks the current campaign parent identity and 5-core/18 GB-high/20 GB-max/no-swap limits. It requests reclaim only when the parent is above 8.5 GB and the file-stat budget allows it, then reports `ready` only if the observed parent current plus the next 6 GB export or 5 GB fit reserve is at most 17 GB. It does not delete artifacts, change caps, retry a phase, or select by score.

## Limits and validity

No live write to `memory.reclaim` was attempted, so this review admits only a first root-controlled handoff attempt under fresh resource observation; it does not establish actual reclaimed bytes, downstream export capacity, fit completion, or quality. The current host is on cgroup v2 kernel `7.2.7-200.fc44.x86_64`; the campaign `memory.reclaim` file exists and current systemd limits match the frozen policy, but interface write behavior remains untested here.

The helper caps its request using `memory.stat:file` and asks for `swappiness=0`, but Linux documents that `memory.reclaim` may over- or under-reclaim and that `swappiness` uses VM swappiness semantics rather than a strict file-only selector. The helper records the post-action `file` charge but its `ready` test does not enforce `after.file >= 6 GB`; the documented 6 GB retained-file floor is therefore a request target, not a hard guarantee. This can change cache warmth, not persisted data or model semantics. [Linux kernel cgroup v2 documentation](https://docs.kernel.org/admin-guide/cgroup-v2.html#memory).

The `handoff_attempt_sha256` is computed while the helper's controller attempt is still running; the supervisor later updates that JSON with terminal status and final measurements. Treat this field as a live-record snapshot rather than the final attempt digest. The terminal controller record remains separately available. The helper blocks concurrent GPU attempts but not already-running CPU support work; root must account for concurrent CPU reservations and accept any cache-cold slowdown before running it. The following `bandit_resources` step still rechecks actual capacity, and post-grade handoff success does not itself admit the next shape.

## Decision and next step

Source and recipe review passes for one root-admitted execution of the matching handoff step. After execution, interpret only its terminal controller record and observed before/after cgroup values; do not infer a guaranteed 6 GB file-cache floor. Keep the next export/fit under its separate existing `bandit_resources` capacity gate and keep every later shape under root's per-shape admission.

## Source records

- [R5 recipe and commands](../../campaign_2026-10-14/cpu_student_frontier_r1/recipes/README.md#r5-scoped-file-cache-handoffs-for-the-remaining-f0-seed-3-shapes)
- [Seven-plan static preflight receipt](../../campaign_2026-10-14/cpu_student_frontier_r1/recipes/R5_SEQUENCE_STATIC_PREFLIGHT_R2.json)
- [R5 handoff implementation](../../../scripts/reclaim_student_frontier_cache.py)
- [Root sequence and resource controllers](../../../scripts/bandit_sequence.py), [resource controller](../../../scripts/bandit_resources.py)
