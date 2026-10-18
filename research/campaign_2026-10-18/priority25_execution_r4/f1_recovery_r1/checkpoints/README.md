# F1 recovery checkpoint helpers (V3 preparation)

This directory is a local-only V3 derivative for collecting and verifying the **wide F1/seed1001 terminal checkpoint** after the exact V2 continuation attempt was admitted. It keeps the original F1 plan as the scientific contract, binds the separate recovery plan's single attempt-ID substitution in two plan fields, and binds its not-started rejection record. It accepts the existing V2 stage1 verification receipts as continuation parents. The deep arm remains on its original V1 continuation attempt and existing V2 checkpoint helpers.

The frozen inputs are:

- Base plan `../../f1_beleriand_fallback/F1_BELERIAND_PLAN.json`, SHA-256 `96cfb67b3447866651ae04638166f17dccb9729b3d4a9340f1b72b8d7b089f96`.
- Recovery plan `../F1_RECOVERY_PLAN.json`, SHA-256 `f1aa897f0be902e114935f0ac9044b0616e2f3e3a889991651d028823938a111`. Its only semantic changes are the `continuation_wide` attempt ID and the matching `--attempt` value in the launch vector; the training command and its `--contract` still bind the base plan.
- Rejected V1 attempt `../REJECTED_ATTEMPT_V1.json`, SHA-256 `c1f506e3e4015903b3a067a5f7a34814694916a73d928a989a9b67fac09e22f0`. It records `not_started` / `admission_failed` for the CPU reservation; V3 refuses a rejection that records any start or a different command/contract.

The helper calls accept the original F1 plan path with `--plan`. This packet supports only wide terminal collection and verification. Wide continuation intermediate-checkpoint support is not implemented: the existing V2 collector binds the old attempt ID and must not be used for the new wide V2 attempt. Both stage1 checkpoints and the deep terminal remain on their existing V2 path. For the wide terminal, the V3 common module separately rehashes all three frozen inputs and changes the in-memory attempt spec to `computer-vision-homework-r01-wide-f1s1001-ep25-v2`. It does not edit either plan. The collector writes a recovery-binding object into its create-only receipt; the verifier checks that object, the exact V2 attempt, stable copied artifacts, fold1/source/registry identities, counters, FP32 tensors/Adam state and stage1 ancestry.

The terminal verifier accepts stage1 parent receipts only when they have the exact V2 verifier SHA and complete helper/dependency map recorded in `PREPARATION_MANIFEST_V2_R2.json`. The receipt's original F1 plan, fold, source, registry, run, attempt and state-SHA checks remain in place. A modified V3 verifier is not treated as the producer of those earlier V2 receipts. The output terminal verification receipt uses the distinct `priority25.f1_checkpoint_verification.recovery_v3` schema.

## Future commands

These are interfaces only; no V3 checkpoint collection or verification has run. Collection must be launched inside the already reviewed Beleriand systemd support envelope, after a fresh headroom check, and all output paths must be new:

```text
python collect_f1_checkpoint.py --arm wide --phase terminal \
  --campaign-root /home/grinchenkoeg/computer-vision-homework-20261018 \
  --plan <B>/research/campaign_2026-10-18/priority25_execution_r4/f1_beleriand_fallback/F1_BELERIAND_PLAN.json \
  --stage1-snapshot <B>/checkpoint_snapshots/rr18_r01_wide_f1_s1001/stage1_epoch1 \
  --output <new-wide-terminal-collection-receipt.json>
```

The separate CPU-only verifier should receive the collected terminal snapshot and receipt, the exact source-a266 tree, fold1 manifest, portable source registry/archive inventory, and both V2 stage1 snapshots/verification receipts. Run it under the previously required 50% CPU, 1.5 GB memory, no-swap, 120-second, CUDA-hidden support envelope with parent headroom checked before entry. Use create-only receipt paths.

## Validation and limits

The seven local unit tests exercise the exact allowed retry, reject changed trainer/launch commands and changed rejection identity, keep deep on attempt V1, and require the exact V2 stage1 verifier/dependency hashes. Ruff and byte-compilation pass. No checkpoint, remote output, prediction, grade, training result, or measured resume parity is included. The actual wide continuation is an independently admitted root operation; this helper packet does not prove its completion or quality.

See `PREPARATION_MANIFEST_V3.json` for exact helper/input hashes and validation scope.
