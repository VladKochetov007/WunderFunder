# F3 terminal collection and result handoff — readiness note

**Status: read-only preparation; no terminal checkpoint payload was read, no remote operation was run, and no admission is granted here.** This note binds to the frozen seven-step F3 plan and the already completed stage1 receipts. It identifies the smallest next engineering task: prepare and review terminal-specific root admission/capture wrappers around the existing collector/verifier, then use the exact commands below only after fresh root gates.

## Current evidence and ownership

The plan is `../f3_beleriand_preparation_r1/F3_BELERIAND_PLAN.json`, SHA-256 `b303146cf9db05e0a3bddf8f245d3d9c9b880d055ce7e97ebe5dd13510b5f4a2d8`. Its native zero-based sequence is: index 0 wide stage1, 1 deep stage1, 2 wide continuation, 3 deep continuation, 4 wide export, 5 deep export, 6 paired grade. There is **no separate terminal-checkpoint step**. Wide terminal support is expected after index 2 completes while index 3 deep continuation is current; deep terminal support may be after index 3 while index 4/5 export, index 6 CPU grade, or the finished controller is current. A root wrapper must bind to the actual current sequence pointer/attempt and exact populated descendants each time; it must not assume that a GPU worker is present or treat an unknown descendant as idle.

The latest local observation I inspected is `../f3_beleriand_execution_r1/POST_STAGE1_PROGRESS_20261018T2010Z.json` (observed 2026-10-18 20:06:13 UTC): index 2 wide continuation was running with the expected native attempt. It is a historical local snapshot, not live status. Both stage1 archive/verification pairs now exist at the canonical paths below. Wide stage1 collection and verification receipts are `../receipts/RANK01_WIDE_F3_S1001_STAGE1_COLLECTION.json` (`2af09d13ea5d36c90f84c52e1b6e3b3606f6af143f27280bd1e39d5b29e7cfdd`) and `../receipts/RANK01_WIDE_F3_S1001_STAGE1_VERIFICATION.json` (`2a274447676b657f30280631e30a1ff3a67acdec02497b2355486d84a789aeb4`); deep receipts are `../receipts/RANK01_DEEP_F3_S1001_STAGE1_COLLECTION.json` (`99595b13eaa0f7fc7dac54ac74e09f6b55551e4085219d6dba7e05a95af5266b`) and `../receipts/RANK01_DEEP_F3_S1001_STAGE1_VERIFICATION.json` (`223725c151ec40f6f26e71332b1cf1b0f6fbe26a2c749ce6421f0b49d06a3efa`). Both verifications report `next_epoch=2`, 29,760 completed updates / 2,976 batches, and matching expected fold3 membership identities. Structural acceptance receipts are `../receipts/F3_WIDE_STAGE1_ACCEPTANCE_V1.json` and `../receipts/F3_DEEP_STAGE1_ACCEPTANCE_V1.json`; the deep receipt-file binding correction is `../receipts/F3_DEEP_STAGE1_ACCEPTANCE_V1_CORRECTION_V1.json`. These accept stage1 structure only; they do not establish either terminal checkpoint, resume execution, score, or quality.

Use the canonical support packet, not the stage1 root wrappers as terminal wrappers:

- `../f3_checkpoint_support_r1/PREPARATION_MANIFEST.json`: `7e4fce0d7232a487a769e12d611050bc608f9cc7fcea300bc2063525abfa38dc`; independent review `../f3_checkpoint_support_r1/INDEPENDENT_REVIEW.json`: `de709897397d7cd775d2aba84fb71f8f7d91504802814f5bcfc26e48e203763b`.
- `collect_f3_checkpoint.py`: `f937f4634f6c0e4b3a49a9808d8a344a1d5244469d397047c50530f10c6413be`; `verify_f3_checkpoint.py`: `08021054e0ddd4225bbee7a3781aaee505538e5a2d164f3732c15f507c9c3a2b`; dependency `f3_checkpoint_common.py`: `b24f637128c699eccf7714bbb94a0edad14b4cc77d0d00f9d0ae98f824da3fa8`. Root must rehash the full manifest allowlist before staging.
- Existing deep stage1 V2 wrappers are `../f3_beleriand_execution_r1/support/launch_stage1_v2.py` (`75205a7f20951d65dcace65fe3140c38c51eacbc2c48baefbb87ca9e9bca4dc9`) and `capture_stage1_v2.py` (`48092e6664e69571b9a5f2777b1564f998408ba9de9cdbe20ff69ca8f4088ada`), review `ROOT_STAGE1_SUPPORT_REVIEW_V2.json` (`de709897397d7cd775d2aba84fb71f8f7d91504802814f5bcfc26e48e203763b`). They are deep-stage1-specific and are not authorized or sufficient for terminal capture.
- Result collector packet: `../f3_result_collection_r1/PREPARATION_MANIFEST.json` `0fbff60fd77d02bf80c8fcff51aa7ea8d90b479e6b7947eabd514096f515036a`, review `INDEPENDENT_REVIEW.json` `deeb5a04bbd0f197697c337077647d44a9a3a267fbcb2c694ab004623ba2729a`. It remains prepared, not admitted.

## Existing terminal helper commands

The helper interfaces and paths below are from `../f3_checkpoint_support_r1/README.md` and each plan step. They are not launch/admission commands. Use B's campaign root and Python exactly as pinned:

```sh
B=/home/grinchenkoeg/computer-vision-homework-20261018
SUPPORT=$B/research/campaign_2026-10-18/priority25_execution_r4/f3_checkpoint_support_r1
PLAN=$B/research/campaign_2026-10-18/priority25_execution_r4/f3_beleriand_preparation_r1/F3_BELERIAND_PLAN.json
PYTHON=/home/grinchenkoeg/WunderFunder/.venv/bin/python
SRC=$B/source-a266b036
REGISTRY=$B/support-source/terminal-v1/PORTABLE_SOURCES.json
ARCHIVES=$B/support-source/terminal-v1/PORTABLE_ARCHIVES.json
FOLD=$B/splits/fold3.json
```

After wide continuation's exact plan attempt has finished successfully, collect wide terminal to its plan-declared snapshot and receipt:

```sh
"$PYTHON" "$SUPPORT/collect_f3_checkpoint.py" --arm wide --phase terminal \
  --campaign-root "$B" --plan "$PLAN" \
  --stage1-snapshot "$B/checkpoint_snapshots/rr18_r01_wide_f3_s1001/stage1_epoch1" \
  --output "$B/receipts/RANK01_WIDE_F3_S1001_TERMINAL_COLLECTION.json"
```

Then verify wide terminal, explicitly binding both same-arm and peer stage1 snapshots/verification receipts:

```sh
"$PYTHON" "$SUPPORT/verify_f3_checkpoint.py" --arm wide --phase terminal \
  --snapshot-dir "$B/checkpoint_snapshots/rr18_r01_wide_f3_s1001_e6/epoch5_terminal" \
  --source-root "$SRC" \
  --collection-receipt "$B/receipts/RANK01_WIDE_F3_S1001_TERMINAL_COLLECTION.json" \
  --plan "$PLAN" --fold-manifest "$FOLD" \
  --data-registry "$REGISTRY" --archive-manifest "$ARCHIVES" \
  --stage1-snapshot-dir "$B/checkpoint_snapshots/rr18_r01_wide_f3_s1001/stage1_epoch1" \
  --stage1-verification "$B/receipts/RANK01_WIDE_F3_S1001_STAGE1_VERIFICATION.json" \
  --peer-stage1-snapshot-dir "$B/checkpoint_snapshots/rr18_r01_deep_f3_s1001/stage1_epoch1" \
  --peer-stage1-verification "$B/receipts/RANK01_DEEP_F3_S1001_STAGE1_VERIFICATION.json" \
  --output "$B/receipts/RANK01_WIDE_F3_S1001_TERMINAL_VERIFICATION.json"
```

After deep continuation finishes successfully, use the corresponding deep output path and swap own/peer stage1 inputs:

```sh
"$PYTHON" "$SUPPORT/collect_f3_checkpoint.py" --arm deep --phase terminal \
  --campaign-root "$B" --plan "$PLAN" \
  --stage1-snapshot "$B/checkpoint_snapshots/rr18_r01_deep_f3_s1001/stage1_epoch1" \
  --output "$B/receipts/RANK01_DEEP_F3_S1001_TERMINAL_COLLECTION.json"

"$PYTHON" "$SUPPORT/verify_f3_checkpoint.py" --arm deep --phase terminal \
  --snapshot-dir "$B/checkpoint_snapshots/rr18_r01_deep_f3_s1001_e6/epoch5_terminal" \
  --source-root "$SRC" \
  --collection-receipt "$B/receipts/RANK01_DEEP_F3_S1001_TERMINAL_COLLECTION.json" \
  --plan "$PLAN" --fold-manifest "$FOLD" \
  --data-registry "$REGISTRY" --archive-manifest "$ARCHIVES" \
  --stage1-snapshot-dir "$B/checkpoint_snapshots/rr18_r01_deep_f3_s1001/stage1_epoch1" \
  --stage1-verification "$B/receipts/RANK01_DEEP_F3_S1001_STAGE1_VERIFICATION.json" \
  --peer-stage1-snapshot-dir "$B/checkpoint_snapshots/rr18_r01_wide_f3_s1001/stage1_epoch1" \
  --peer-stage1-verification "$B/receipts/RANK01_WIDE_F3_S1001_STAGE1_VERIFICATION.json" \
  --output "$B/receipts/RANK01_DEEP_F3_S1001_TERMINAL_VERIFICATION.json"
```

Terminal verification should confirm the actual checkpoint against the expected `next_epoch=6`, 89,280 optimizer updates / 8,928 batches, exact six-epoch histories, source/data/membership identities, and same-arm continuation ancestry. Stage1 verification already binds fold3 SHA `21c7fb52e8e69f38126362cbc5dc60dbc46539041d649b9206ec35e0f21ada1f`, counts 10,539 fit-train / 1,368 fit-valid / 505 evaluation groups / 145 evaluation components, and ordered digests; terminal helpers must rebind the actual staged inputs rather than copy forward these claims.

## Root work still required; create-only outputs

No terminal-specific admission/launch/capture wrappers are present in the stage1 V2 packet. Prepare/review separate terminal wrappers that:

1. Freshly rehash the exact plan and all helper pins, re-read the current sequence plus matching native attempt and finished exit/result, and fail closed on unknown/extra descendants. Wide-terminal support must account for the expected deep continuation worker at index 3 if it is active. Deep-terminal support must account for whichever exact plan step (index 4/5 export, index 6 grade, or finished controller) is current; no foreign or unrelated unit is to be stopped.
2. Refresh parent CPU/RAM/swap/OOM, physical scratch/disk, GPU ownership/headroom, and current worker/support reservations before each separate collector and verifier. Preserve the aggregate two-CPU budget. The reviewed envelopes are collector 0.25 CPU / 512 MB / 120 s and verifier 0.5 CPU / 1.5 GB / 120 s, hidden CUDA, zero swap, physical TMP. The native controller is 0.1 CPU / 512 MB; an active GPU attempt is 1 CPU / 8 GB (7.2 GB high), one worker, with the frozen 4 GB allocator cap and selected GPU UUID `GPU-a50d0c3b-d28f-a2a7-8b43-3e02b543fc0c`. Capture raw stdout/stderr/status/receipt before stopping only the exact exited support unit. The support caps do not displace the current worker/controller reservations.
3. Check every output below absent immediately before use and create exclusively; preserve partial/collision evidence, do not retry into it. The canonical support helper's terminal archive and JSON paths are fixed by the plan:
   - Wide snapshot `checkpoint_snapshots/rr18_r01_wide_f3_s1001_e6/epoch5_terminal`; wide collection and verification receipts `receipts/RANK01_WIDE_F3_S1001_TERMINAL_COLLECTION.json` and `receipts/RANK01_WIDE_F3_S1001_TERMINAL_VERIFICATION.json`.
   - Deep snapshot `checkpoint_snapshots/rr18_r01_deep_f3_s1001_e6/epoch5_terminal`; deep collection and verification receipts `receipts/RANK01_DEEP_F3_S1001_TERMINAL_COLLECTION.json` and `receipts/RANK01_DEEP_F3_S1001_TERMINAL_VERIFICATION.json`.
   - The terminal wrapper packet is missing. For each `ARM` in `WIDE` and `DEEP`, the proposed local root-receipt names are `RANK01_{ARM}_F3_S1001_TERMINAL_COLLECTOR_ADMISSION_V1.json`, `..._COLLECTOR_LAUNCH_V1.json`, `..._COLLECTION_EXECUTION_V1.json`, `..._COLLECTION_STOP_V1.json`, `..._VERIFIER_ADMISSION_V1.json`, `..._VERIFIER_LAUNCH_V1.json`, `..._VERIFICATION_EXECUTION_V1.json`, and `..._VERIFICATION_STOP_V1.json`. Replace `{ARM}` with one literal arm name; the ellipsis means the same exact `RANK01_{ARM}_F3_S1001_TERMINAL` prefix. These are proposed names, not evidence of a completed collision check. Stage1 V2 files are already used and must not be overwritten.
4. Keep native plan later outputs create-only and bind their final attempt/capture evidence: export attempt records `runtime/attempts/computer-vision-homework-r01-wide-f3s1001-export-ep5-v1.json` and `runtime/attempts/computer-vision-homework-r01-deep-f3s1001-export-ep5-v1.json`; prediction outputs `data/cache/pred_rr18_r01_wide_f3_s1001_e6_valid.npy`, `data/cache/pred_rr18_r01_wide_f3_s1001_e6_valid.json`, `data/cache/pred_rr18_r01_deep_f3_s1001_e6_valid.npy`, and `data/cache/pred_rr18_r01_deep_f3_s1001_e6_valid.json`; grade attempt `runtime/attempts/computer-vision-homework-r01-f3s1001-grade-v1.json` and paired grade `receipts/RANK01_F3_S1001_GRADE.json`. The plan specifies fixed epoch 5, evaluation-only FP32 prediction, and grade metric `global_wp_clip2_v1` with 10,000 paired bootstrap resamples. Do not declare these finished from plan text.
5. After terminal verifications, both exports, grade, and the original sequence's finished status/capture exist, fill `receipts/F3_TERMINAL_EVIDENCE_SEAL_V1.json` with actual observed receipt paths/byte counts/SHA values and two independent PASS reviews as required by the result collector README. Its create-only published outputs are `f3_result_collection_r1/records/F3_S1001_RESULT_R1/` and `receipts/RANK01_F3_S1001_RESULT_COLLECTION_R1.json`. The R1 collector also requires local/B collisions checked before admission. No terminal hashes, scores, or result-receipt values are predicted here.

The plan's terminal snapshot paths are consequences of the two continuation steps, not new sequence steps. Existing stage1 receipts and reviews are not substitutes for the terminal root admission/capture work or fresh resource/ownership checks. This note does not itself authorize execution or result collection.
