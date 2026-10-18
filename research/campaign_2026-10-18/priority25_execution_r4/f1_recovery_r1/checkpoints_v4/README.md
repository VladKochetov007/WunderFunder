# F1 wide-terminal checkpoint verifier (V4)

This is a new, isolated terminal-verifier derivative for the wide F1/seed1001 recovery attempt. It corrects a static V3 lineage bug found before any V3 terminal verification was executed. V3 checked that the peer stage1 receipt was produced by the exact V2 verifier and dependency set, then redundantly compared that receipt's V2 verifier SHA with the current recovery verifier SHA. Those hashes must differ. V4 binds the peer receipt to the exact V2 stage1 verifier SHA while retaining the peer run, attempt, dependency, state-hash, pair-contract and membership checks.

The collection implementation is byte-identical to V3. The root operator must collect the terminal snapshot using the original V3 collector path, then run this V4 verifier against that collection receipt. The copied collector file is present because the verifier hashes its sibling helper set; its bytes must continue to match the V3 collector receipt. V4 emits schema `priority25.f1_checkpoint_verification.recovery_v4` so its result cannot be confused with V3.

The original F1 plan remains the scientific contract. The V4 common and evidence helpers preserve the existing exact recovery-plan and no-start rejection bindings. Both stage1 receipts must have the complete original V2 helper/dependency identities. Wide terminal recovery is the only supported recovery scope; wide continuation intermediate collection is unsupported. The deep terminal remains on its existing V2 path.

## Validation

The stdlib regression test loads the actual wide and deep V2 stage1 verification receipts. A synthetic checkpoint-state fixture verifies that the valid V2 peer receipt passes despite a distinct current recovery-verifier SHA, and that a peer receipt relabeled with the recovery SHA is rejected by the V2 identity check. The other copied tests cover the frozen retry-plan binding. Nine local stdlib tests and Ruff checks pass. The regression does not load Torch or actual checkpoint tensors.

No terminal checkpoint has been collected or verified with V4 by this preparation packet. It makes no training, GPU-resume-parity, metric, convergence, or quality claim. See `PREPARATION_MANIFEST_V4.json` for exact helper and input hashes.
