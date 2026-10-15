---
kind: review
status: conditional-source-go
created_utc: 2026-10-15T12:44:00Z
updated_utc: 2026-10-15T12:44:00Z
hypothesis_ids: []
experiment_ids: []
evidence_paths:
  - /home/vlad/development/wunder/worktrees/student-quality-alpha0-dataset-r3-20261015/research/campaign_2026-10-15/student_quality_tradeoff_r1/ALPHA0_INPUT_BINDING_VALIDATION_R3.json
  - /home/vlad/development/wunder/worktrees/student-quality-alpha0-dataset-r3-20261015/research/campaign_2026-10-15/student_quality_tradeoff_r1/ALPHA0_DATASET_BYTE_BINDING_R3.json
  - /home/vlad/development/wunder/worktrees/student-quality-alpha0-dataset-r3-20261015/research/campaign_2026-10-15/student_quality_tradeoff_r1/ALPHA0_STAGE_INPUT_GATES_R3.json
  - /home/vlad/development/wunder/worktrees/student-quality-alpha0-dataset-r3-20261015/research/campaign_2026-10-15/student_quality_tradeoff_r1/verify_alpha0_input_binding_r3.py
---

# Matched warm-parent alpha0 input-binding review R3

## Question or claim

Does R3 bind the unchanged three-stage alpha0 continuation to the intended full train/valid bytes and A9 parent checkpoints while keeping fitting authority outside the input-gate packet?

## Evidence and interpretation

The frozen source is actor commit `5ee81e57f116d1c514883845e6d23ecbf2852ab5`. I verified hashes for the R3 validation receipt (`bc901f12880696239902cabeb0aff8972afcc1a9288f346c96c96363005ee271`), one-time data-byte receipt (`8fa68124ac67114f4d86aa4d1200bde84133aee09318181bf9e5a23a86ce7bcd`), stage-gate plan (`db09391b3275b734f5baab8705b4d6dab44ec59bdaa2625128b32e477ed297c8`), checker (`446e37f16ac66cb29024cc5d30977bd499b0ec68b1272602b5bdbc18dc53a6f0`) and materializer (`1afe3c57953ad6227ea45e4db317107c6d87e24ee9171eb754e0cc01c0959792`).

The byte receipt reports one full streaming pass matching the earlier official pins: train 29,240,219,011 bytes, SHA-256 `4a7756449a4bcf272cf715bb405a2152987041605f6c1cce650243ab2abf4e81`; valid 5,615,509,128 bytes, SHA-256 `2861ee30914065702626a3ac033d80d0563cdacaea50c667fc828c2eeb418782`. It also rehashes the P1/P2/P3 A9 endpoint checkpoints once against provenance. The reported cgroup peak is 44,154,880 bytes under a 299,999,232-byte cap, with no swap, at 1 CPU. Each P1/P2/P3 stage smoke reports matching dataset/checkpoint identity, absent alpha0 outputs, and no repeated full-data hash. I reviewed source and receipt metadata only; I did not reread the 34.9 GB data payload or rerun tests.

The three immutable R2 fit contracts and worker argv hashes remain pinned; R3 adds separate data/checkpoint bindings and bounded pre-fit gate commands. Each gate explicitly says it authorizes neither fit nor launch. The external conditions require independent review, successful H384 complete-sequence export/grade, and a fresh root resource admission for each stage. Thus this packet changes input evidence and launch gating, not the alpha0 scientific treatment.

## Limits and validity

The A9 P1 attempt receipt records successful `train+valid` fitting but lacks raw Parquet hashes and attempt-time inode/ctime. R3 therefore cannot prove the exact bytes read by A9 from that receipt alone. It documents that limitation and triangulates pre-A9 official full-file pins from the portable registry, E520 spec and H390 start receipt, then binds the currently streamed bytes and subsequent stage identities. This supports the planned shared-warm-parent continuation contrast with an explicit ancestry limitation; it is not a new independent initialization or a pristine evaluation.

## Decision and next step

Conditional source/data GO for this exact R3 preflight packet. It is not permission to fit: wait for the H384 complete-sequence chain, then recheck current input identities and obtain fresh root resource admission before each stage. No fit, launch, grade, or quality result was performed or authorized by this review.

## Source records

Actor packet: `/home/vlad/development/wunder/worktrees/student-quality-alpha0-dataset-r3-20261015`, commit `5ee81e57f116d1c514883845e6d23ecbf2852ab5`. The full-stream receipt identifies the official pre-A9 data-pin sources and preserves the earlier receipt revisions and failures.
